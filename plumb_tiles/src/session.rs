//! The tile session ([`Amx::with_tiles`]) and the operations it unlocks ([`Tiles`]).
//!
//! Every tile instruction is an `asm!` block. None of them is `pure`: the tile registers are
//! invisible to the compiler, so the only thing keeping "load T0, then store T0" in that order
//! is that the compiler never reorders or drops side-effecting asm blocks. The memory options
//! then say what each instruction does to memory the compiler can see: loads are `readonly`
//! (pending stores to the source are completed first), stores (and `STTILECFG`) claim
//! nothing, and zero/compute/release are `nomem`. No tile instruction touches the stack or
//! RFLAGS.

use core::arch::asm;
use core::cell::Cell;
use core::fmt;
use core::marker::PhantomData;
use core::mem::MaybeUninit;

use crate::reg::TileReg;
use crate::{Amx, AmxBf16, AmxFp16, AmxInt8, ROW_BYTES, ROWS, tile_span};

/// The LDTILECFG descriptor: palette 1 with all eight tiles at 16 rows x 64 bytes.
#[repr(C, align(64))]
struct TileConfig([u8; 64]);

static CONFIG: TileConfig = TileConfig(palette1_all_16x64());

const fn palette1_all_16x64() -> [u8; 64] {
    // Byte 0 palette id, byte 1 start_row (0: no interrupted load to resume), bytes 2..16
    // reserved; colsb[16] as u16 LE at 16..48 and rows[16] at 48..64. Only tiles 0..7 exist,
    // and palette 1 requires the fields of the others to stay zero.
    let mut b = [0u8; 64];
    b[0] = 1;
    let mut t = 0;
    while t < 8 {
        b[16 + 2 * t] = ROW_BYTES as u8;
        b[48 + t] = ROWS as u8;
        t += 1;
    }
    b
}

std::thread_local! {
    /// Whether this thread is inside `with_tiles`. Tile configuration is per thread, so a
    /// nested session would reconfigure (and on exit release) the tiles of the outer one.
    /// This flag only knows this copy of the crate; `live_palette` covers everything else.
    static IN_SESSION: Cell<bool> = const { Cell::new(false) };
}

impl Amx {
    /// Runs `f` with the tiles configured, and releases them afterwards.
    ///
    /// On entry `LDTILECFG` sets all eight tiles to [`ROWS`] x [`ROW_BYTES`] and zeroes them;
    /// on exit `TILERELEASE` returns the tile state to its initial (unconfigured) state, also
    /// when `f` panics. Releasing matters beyond tidiness: while tiles are configured the OS
    /// saves and restores 8 KiB of tile data on every context switch of this thread.
    ///
    /// # Panics
    ///
    /// If this thread is already inside `with_tiles` (from any token), or its tiles are
    /// configured by anything else: another copy of this crate in the same program, or another
    /// AMX library that is mid-use or didn't release them. Starting would overwrite that
    /// configuration and the closing `TILERELEASE` would pull the tiles from under its owner,
    /// so the session refuses before touching anything. Other threads have their own tile
    /// state and may run sessions concurrently.
    ///
    /// # Example
    ///
    /// ```
    /// use plumb_tiles::{Amx, T0, ROW_BYTES, TILE_BYTES};
    ///
    /// let Some(amx) = Amx::try_new() else { return };
    /// let src: Vec<u8> = (0..TILE_BYTES).map(|i| i as u8).collect();
    /// let mut dst = vec![0u8; TILE_BYTES];
    /// amx.with_tiles(|t| {
    ///     t.load::<T0>(&src, ROW_BYTES); // 16 packed rows of 64 bytes
    ///     t.store::<T0>(&mut dst, ROW_BYTES);
    /// });
    /// assert_eq!(src, dst);
    /// ```
    #[inline]
    #[track_caller]
    pub fn with_tiles<R>(self, f: impl FnOnce(&mut Tiles<'_>) -> R) -> R {
        let _session = Session::open(self);
        // `f` only ever sees `&mut Tiles`, and its result type can't name that borrow, so the
        // handle cannot outlive `_session`, whose drop releases the tiles.
        f(&mut Tiles {
            _session: PhantomData,
            _per_thread: PhantomData,
        })
    }
}

/// Configured tiles. Dropping it releases them.
struct Session {
    _per_thread: PhantomData<*mut ()>,
}

impl Session {
    #[inline]
    #[track_caller]
    fn open(amx: Amx) -> Self {
        // Both checks run before LDTILECFG, so a refused session leaves the configuration it
        // found untouched. The flag alone would miss a second copy of this crate (each copy
        // has its own thread_local), which is why the hardware state is read as well.
        let nested = IN_SESSION.replace(true);
        if nested || live_palette(amx) != 0 {
            tiles_in_use(nested);
        }
        // SAFETY: the Amx token proves the CPU has AMX-TILE with a palette 1 that accepts this
        // descriptor and that the OS saves tile state for this thread. LDTILECFG reads the 64
        // descriptor bytes and nothing else.
        unsafe {
            asm!("ldtilecfg [{cfg}]", cfg = in(reg) CONFIG.0.as_ptr(),
                 options(nostack, readonly, preserves_flags));
        }
        Session {
            _per_thread: PhantomData,
        }
    }
}

/// The palette id of this thread's live tile configuration (`STTILECFG` byte 0): 0 while the
/// tiles are released, whoever configured them otherwise.
#[inline(always)]
fn live_palette(_: Amx) -> u8 {
    let mut live = MaybeUninit::<TileConfig>::uninit();
    // SAFETY: the Amx token proves AMX-TILE and OS-enabled XTILECFG state, so STTILECFG is
    // defined, configured or not; it writes exactly the 64 bytes of `live` (all zero when
    // unconfigured). The block isn't readonly, so the compiler treats `live` as written.
    unsafe {
        asm!("sttilecfg [{cfg}]", cfg = in(reg) live.as_mut_ptr(),
             options(nostack, preserves_flags));
        live.as_ptr().cast::<u8>().read()
    }
}

impl Drop for Session {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: TILERELEASE is defined whenever AMX-TILE is, configured or not, and touches
        // no memory.
        unsafe { asm!("tilerelease", options(nostack, nomem, preserves_flags)) };
        IN_SESSION.set(false);
    }
}

#[cold]
#[inline(never)]
#[track_caller]
fn tiles_in_use(nested: bool) -> ! {
    if nested {
        // The flag belongs to the outer session; leave it set.
        panic!(
            "plumb_tiles: nested with_tiles on one thread (tile state is per thread; one session at a time)"
        )
    }
    // `open` set the flag for this attempt; no session of ours is running, so clear it again.
    IN_SESSION.set(false);
    panic!(
        "plumb_tiles: with_tiles on a thread whose tiles are already configured (by another copy \
         of plumb_tiles or another AMX library); they must be released first"
    )
}

/// The configured tiles of one [`with_tiles`](Amx::with_tiles) session.
///
/// Every tile is [`ROWS`] rows of [`ROW_BYTES`] bytes and starts the session zeroed. Methods name
/// the tile with a type parameter ([`T0`](crate::T0)..[`T7`](crate::T7)).
///
/// # Compile-time checks
///
/// The `TDP*` methods take their three tiles as type parameters and reject a repeated one
/// (the instruction would raise `#UD`) when the call is monomorphized:
///
/// ```compile_fail,E0080
/// use plumb_tiles::{Amx, T0, T1};
/// let amx = Amx::try_new().unwrap();
/// let int8 = amx.int8().unwrap();
/// amx.with_tiles(|t| t.dpbssd::<T0, T0, T1>(int8)); // tmm0 as C and A
/// ```
///
/// `Tiles` is neither `Send` nor `Sync`: the tile registers belong to the thread that
/// configured them.
///
/// ```compile_fail,E0277
/// fn needs_send<T: Send>() {}
/// needs_send::<plumb_tiles::Tiles<'static>>();
/// ```
/// ```compile_fail,E0277
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<plumb_tiles::Tiles<'static>>();
/// ```
pub struct Tiles<'s> {
    /// The session this handle belongs to; reserved so later views of an AMX session (such
    /// as AMX-AVX512 row reads) can borrow from it. ACE's palette-2 session gets its own
    /// handle type instead (see the crate docs' ACE roadmap).
    _session: PhantomData<&'s mut ()>,
    _per_thread: PhantomData<*mut ()>,
}

impl fmt::Debug for Tiles<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tiles").finish_non_exhaustive()
    }
}

/// Panics unless `len` bytes hold [`ROWS`] rows of [`ROW_BYTES`] at `stride`.
///
/// The arithmetic must not wrap: the instruction adds the stride register as a *signed* index,
/// so a huge stride that wrapped to a small span would really address memory below the slice.
#[inline(always)]
#[track_caller]
fn check_span(len: usize, stride: usize) {
    match tile_span(stride) {
        Some(span) if span <= len => {}
        _ => out_of_bounds(len, stride),
    }
}

#[cold]
#[inline(never)]
#[track_caller]
fn out_of_bounds(len: usize, stride: usize) -> ! {
    match tile_span(stride) {
        Some(span) => panic!(
            "plumb_tiles: tile access out of bounds: {ROWS} rows of {ROW_BYTES} bytes at stride \
             {stride} span {span} bytes, but the buffer has {len}"
        ),
        None => panic!(
            "plumb_tiles: tile access out of bounds: {ROWS} rows at stride {stride} span more \
             than usize::MAX bytes (buffer has {len})"
        ),
    }
}

// The raw instructions. Callers have checked the span and hold a session.

#[inline(always)]
unsafe fn tileloadd<T: TileReg>(src: *const u8, stride: usize) {
    // SAFETY (caller): src + r*stride .. +ROW_BYTES is readable for r < ROWS; tiles configured.
    unsafe {
        asm!("tileloadd tmm{t}, [{src} + {stride}*1]", t = const T::N,
             src = in(reg) src, stride = in(reg) stride,
             options(nostack, readonly, preserves_flags));
    }
}

#[inline(always)]
unsafe fn tileloaddt1<T: TileReg>(src: *const u8, stride: usize) {
    // SAFETY (caller): as tileloadd.
    unsafe {
        asm!("tileloaddt1 tmm{t}, [{src} + {stride}*1]", t = const T::N,
             src = in(reg) src, stride = in(reg) stride,
             options(nostack, readonly, preserves_flags));
    }
}

#[inline(always)]
unsafe fn tilestored<T: TileReg>(dst: *mut u8, stride: usize) {
    // SAFETY (caller): dst + r*stride .. +ROW_BYTES is writable for r < ROWS; tiles configured.
    unsafe {
        asm!("tilestored [{dst} + {stride}*1], tmm{t}", t = const T::N,
             dst = in(reg) dst, stride = in(reg) stride,
             options(nostack, preserves_flags));
    }
}

/// The three operands of a TDP instruction must be different tiles (`#UD` otherwise).
const fn distinct<C: TileReg, A: TileReg, B: TileReg>() -> bool {
    C::N != A::N && C::N != B::N && A::N != B::N
}

macro_rules! tdp {
    ($(#[$doc:meta])* $name:ident, $token:ty, $mnemonic:literal) => {
        $(#[$doc])*
        ///
        /// `C`, `A` and `B` must be three different tiles; naming one twice is a compile error
        /// (see [Compute](crate#compute)).
        #[inline(always)]
        pub fn $name<C: TileReg, A: TileReg, B: TileReg>(&mut self, _: $token) {
            const {
                assert!(distinct::<C, A, B>(),
                        concat!("plumb_tiles: ", $mnemonic, " needs three different tiles"))
            };
            // SAFETY: the token proves the instruction exists and the session configured the
            // tiles with shapes the TMUL limits accept (checked by detection); it touches only
            // tile registers.
            unsafe {
                asm!(concat!($mnemonic, " tmm{c}, tmm{a}, tmm{b}"),
                     c = const C::N, a = const A::N, b = const B::N,
                     options(nostack, nomem, preserves_flags));
            }
        }
    };
}

impl Tiles<'_> {
    /// Sets every byte of tile `T` to zero (`TILEZERO`).
    #[inline(always)]
    pub fn zero<T: TileReg>(&mut self) {
        // SAFETY: the session configured T; TILEZERO touches no memory.
        unsafe { asm!("tilezero tmm{t}", t = const T::N, options(nostack, nomem, preserves_flags)) }
    }

    /// Loads tile `T` from `src`: row `r` is `src[r * stride..][..ROW_BYTES]` (`TILELOADD`).
    ///
    /// Any stride works, including 0 (every row is the first 64 bytes) and strides below 64
    /// (overlapping rows); there is no alignment requirement.
    ///
    /// # Panics
    ///
    /// If `src` is shorter than [`tile_span(stride)`](crate::tile_span), or that overflows.
    #[inline(always)]
    #[track_caller]
    pub fn load<T: TileReg>(&mut self, src: &[u8], stride: usize) {
        check_span(src.len(), stride);
        // SAFETY: every row lies inside `src` (checked above); the session configured T.
        unsafe { tileloadd::<T>(src.as_ptr(), stride) }
    }

    /// [`load`](Self::load) with the T1 hint (`TILELOADDT1`): the data is not expected to be
    /// reused soon, so the CPU may keep it out of the nearest cache levels. Hints are
    /// implementation-defined; the result is the same as `load`.
    ///
    /// # Panics
    ///
    /// As [`load`](Self::load).
    #[inline(always)]
    #[track_caller]
    pub fn load_t1<T: TileReg>(&mut self, src: &[u8], stride: usize) {
        check_span(src.len(), stride);
        // SAFETY: as load.
        unsafe { tileloaddt1::<T>(src.as_ptr(), stride) }
    }

    /// Stores tile `T` to `dst`: row `r` goes to `dst[r * stride..][..ROW_BYTES]`
    /// (`TILESTORED`). Bytes between rows are not written.
    ///
    /// Rows are written in order, so where they overlap (stride below 64) later rows win.
    ///
    /// # Panics
    ///
    /// If `dst` is shorter than [`tile_span(stride)`](crate::tile_span), or that overflows.
    /// Nothing is written in that case.
    #[inline(always)]
    #[track_caller]
    pub fn store<T: TileReg>(&mut self, dst: &mut [u8], stride: usize) {
        check_span(dst.len(), stride);
        // SAFETY: every row lies inside `dst` (checked above), which we borrow mutably.
        unsafe { tilestored::<T>(dst.as_mut_ptr(), stride) }
    }

    /// [`load`](Self::load) from a `u64` buffer. `stride` is still in bytes.
    ///
    /// # Panics
    ///
    /// If `src` is shorter than [`tile_span(stride)`](crate::tile_span) bytes, or that overflows.
    #[inline(always)]
    #[track_caller]
    pub fn load_u64<T: TileReg>(&mut self, src: &[u64], stride: usize) {
        check_span(size_of_val(src), stride);
        // SAFETY: as load; any bytes of a u64 are readable as u8.
        unsafe { tileloadd::<T>(src.as_ptr().cast(), stride) }
    }

    /// [`load_t1`](Self::load_t1) from a `u64` buffer. `stride` is still in bytes.
    ///
    /// # Panics
    ///
    /// As [`load_u64`](Self::load_u64).
    #[inline(always)]
    #[track_caller]
    pub fn load_t1_u64<T: TileReg>(&mut self, src: &[u64], stride: usize) {
        check_span(size_of_val(src), stride);
        // SAFETY: as load_u64.
        unsafe { tileloaddt1::<T>(src.as_ptr().cast(), stride) }
    }

    /// [`store`](Self::store) to a `u64` buffer. `stride` is still in bytes.
    ///
    /// # Panics
    ///
    /// If `dst` is shorter than [`tile_span(stride)`](crate::tile_span) bytes, or that
    /// overflows. Nothing is written in that case.
    #[inline(always)]
    #[track_caller]
    pub fn store_u64<T: TileReg>(&mut self, dst: &mut [u64], stride: usize) {
        check_span(size_of_val(dst), stride);
        // SAFETY: as store; every bit pattern is a valid u64.
        unsafe { tilestored::<T>(dst.as_mut_ptr().cast(), stride) }
    }

    tdp!(
        /// `C += A * B` on signed x signed bytes, INT32 accumulate (`TDPBSSD`).
        ///
        /// With each tile read as 16 rows of 16 dwords:
        /// `C[m][n] += sum(k < 16, i < 4) A[m][k].i8[i] * B[k][n].i8[i]`, wrapping at 32 bits.
        /// See [Compute](crate#compute) for the matrix (VNNI) layout.
        dpbssd, AmxInt8, "tdpbssd"
    );
    tdp!(
        /// `C += A * B` with signed bytes in `A` and unsigned bytes in `B` (`TDPBSUD`).
        ///
        /// `C[m][n] += sum(k < 16, i < 4) A[m][k].i8[i] * B[k][n].u8[i]`, wrapping at 32 bits.
        dpbsud, AmxInt8, "tdpbsud"
    );
    tdp!(
        /// `C += A * B` with unsigned bytes in `A` and signed bytes in `B` (`TDPBUSD`).
        ///
        /// `C[m][n] += sum(k < 16, i < 4) A[m][k].u8[i] * B[k][n].i8[i]`, wrapping at 32 bits.
        dpbusd, AmxInt8, "tdpbusd"
    );
    tdp!(
        /// `C += A * B` on unsigned x unsigned bytes (`TDPBUUD`).
        ///
        /// `C[m][n] += sum(k < 16, i < 4) A[m][k].u8[i] * B[k][n].u8[i]`, wrapping at 32 bits.
        dpbuud, AmxInt8, "tdpbuud"
    );
    tdp!(
        /// `C += A * B` on BF16 pairs, FP32 accumulate (`TDPBF16PS`).
        ///
        /// `C[m][n] += sum(k < 16, i < 2) A[m][k].bf16[i] * B[k][n].bf16[i]`, `C` as 16 x 16
        /// f32. MXCSR is ignored: rounding is to nearest even, denormal inputs count as zero
        /// and denormal results are flushed to zero.
        dpbf16ps, AmxBf16, "tdpbf16ps"
    );
    tdp!(
        /// `C += A * B` on FP16 pairs, FP32 accumulate (`TDPFP16PS`).
        ///
        /// `C[m][n] += sum(k < 16, i < 2) A[m][k].f16[i] * B[k][n].f16[i]`, `C` as 16 x 16
        /// f32. MXCSR is ignored. Unlike [`dpbf16ps`](Self::dpbf16ps), FP16 denormal inputs
        /// are kept (FP16's range is small enough that their products are normal in f32).
        dpfp16ps, AmxFp16, "tdpfp16ps"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_descriptor_is_palette1_all_16x64() {
        let b = &CONFIG.0;
        assert_eq!(b[0], 1, "palette id");
        assert!(
            b[1..16].iter().all(|&x| x == 0),
            "start_row and reserved bytes"
        );
        for t in 0..8 {
            assert_eq!(
                u16::from_le_bytes([b[16 + 2 * t], b[17 + 2 * t]]),
                64,
                "colsb[{t}]"
            );
            assert_eq!(b[48 + t], 16, "rows[{t}]");
        }
        assert!(b[32..48].iter().all(|&x| x == 0), "colsb of tiles 8..15");
        assert!(b[56..64].iter().all(|&x| x == 0), "rows of tiles 8..15");
        assert_eq!(core::ptr::from_ref(&CONFIG).addr() % 64, 0);
    }

    #[test]
    fn distinct_rejects_any_repeat() {
        use crate::{T0, T1, T2};
        assert!(distinct::<T0, T1, T2>());
        assert!(!distinct::<T0, T0, T2>());
        assert!(!distinct::<T0, T1, T0>());
        assert!(!distinct::<T0, T1, T1>());
    }
}
