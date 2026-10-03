//! Cache-line and memory-ordering operations for [fearless_simd]: the instructions a memory
//! tester (or any bandwidth-bound kernel) needs that a portable SIMD library doesn't have.
//!
//! - [`view`]: split a `&[u64]` (or any element slice) into an unaligned head, a run of aligned
//!   fearless vectors, and a tail. A `&mut u64x8<S>` proves 64-byte alignment, and iterating a
//!   `&[V]` compiles to a pointer walk.
//! - [`nt`]: non-temporal (streaming) stores in a **scope that ends with `SFENCE`**.
//! - [`flush`]: CLFLUSHOPT, including a scope that **flushes everything it wrote** before
//!   anything can read it again.
//! - [`direct`](mod@direct): MOVDIR64B (64-byte direct stores) in a scope that ends with `SFENCE`.
//!
//! # Why scopes take closures
//!
//! NT stores and MOVDIR64B are weakly ordered. stdarch's contract for them is that the writing
//! thread must `SFENCE` before any other access to the memory they wrote, and Rust's own fences
//! don't provide that (`fence(Release)` emits nothing on x86; `fence(SeqCst)` emits a locked
//! `or`). So the scopes borrow the destination mutably and run the fence when they end:
//! nothing can read the memory until it has run. It has to be a closure, as in
//! `std::thread::scope`, not a guard object: `mem::forget` on a guard would end the borrow
//! without the fence. Inside the scope, writers hand out write-only access.
//!
//! The flush scope is different in kind. Caches are coherent, so flushing is never needed for
//! memory safety; it's what makes a verify read come from DRAM. That scope gives ordinary
//! `&mut` access and flushes the whole borrowed range when it ends.
//!
//! # Codegen
//!
//! Everything on a hot path is `#[inline(always)]`. Call the scopes from inside a fearless
//! `#[simd]` function or `kernel!` so the closures inline into the function that has the target
//! features (the scopes' closures are called once, and small writer closures are called in the
//! hot loop). The workspace's `bench/asm_check.py` verifies the emitted loops: the expected
//! instruction, the named width, no calls, no `memset`.
//!
//! # Instructions and toolchains
//!
//! Stable Rust uses `asm!` for CLFLUSHOPT and MOVDIR64B (`asm!` needs no target feature and
//! inlines anywhere; it measured the same as the intrinsic). The `nightly` feature switches the
//! range flush to stdarch's `_mm_clflushopt` (rust-lang/rust#157096) so LLVM can unroll it, and
//! exports `with_clflushopt!` (module `entry`), which builds a fearless kernel entry point whose target
//! features are the level's plus `clflushopt`, so the intrinsic also inlines inside fearless
//! loops. MOVDIR64B has no intrinsic in stdarch yet, so it is always `asm!`.
//!
//! [fearless_simd]: https://crates.io/crates/fearless_simd

#![no_std]
#![cfg_attr(feature = "nightly", feature(clflushopt_target_feature, simd_x86_clflushopt))]
#![cfg(target_arch = "x86_64")]

pub mod cpu;
pub mod direct;
#[cfg(feature = "nightly")]
pub mod entry;
pub mod flush;
pub mod nt;
pub mod view;

pub use direct::{DirectWriter, Line, Movdir64b, as_lines, as_lines_mut, direct};
pub use flush::{Clflushopt, flush_after};
pub use nt::{NtBytes, NtSlot, NtStore, NtWriter, nontemporal};
pub use view::{as_vectors, as_vectors_mut};

/// `SFENCE`: orders this thread's earlier stores, including non-temporal and direct stores,
/// before its later stores.
#[inline(always)]
pub fn sfence() {
    // SAFETY: SSE is part of every x86_64 baseline.
    #[allow(unused_unsafe)]
    unsafe {
        core::arch::x86_64::_mm_sfence()
    }
}

/// `MFENCE`: orders all of this thread's earlier loads and stores, and CLFLUSHOPTs, before
/// its later ones. This is the fence a flush needs before the verify loads.
#[inline(always)]
pub fn mfence() {
    // SAFETY: SSE2 is part of every x86_64 baseline.
    #[allow(unused_unsafe)]
    unsafe {
        core::arch::x86_64::_mm_mfence()
    }
}
