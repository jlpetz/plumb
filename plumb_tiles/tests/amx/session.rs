// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The session: configuration, release (also on unwind), nesting, and per-thread state.

use std::time::Duration;

use plumb_tiles::{Amx, ROW_BYTES, T0, T2, T3, T5, TILE_BYTES};

use crate::{Lcg, amx_or_skip, expected_config, panic_location, panic_message, tile_config};

/// A load/store round trip in a fresh session: proves the thread can use tiles again.
fn round_trip(amx: Amx, seed: u64) {
    let src = Lcg::new(seed).bytes(TILE_BYTES);
    let mut dst = vec![0_u8; TILE_BYTES];
    amx.with_tiles(|t| {
        t.load::<T3>(&src, ROW_BYTES);
        t.store::<T3>(&mut dst, ROW_BYTES);
    });
    assert_eq!(src, dst, "seed {seed}: the tile must round-trip");
}

#[test]
fn session_configures_palette1_and_releases_on_exit() {
    let Some(amx) = amx_or_skip("session_configures_palette1_and_releases_on_exit") else {
        return;
    };
    assert_eq!(tile_config(amx), [0; 64], "released before the session");
    let inside = amx.with_tiles(|_| tile_config(amx));
    assert_eq!(
        inside,
        expected_config(),
        "configuration inside the session"
    );
    assert_eq!(tile_config(amx), [0; 64], "released after the session");
}

#[test]
fn every_session_starts_with_zeroed_tiles() {
    let Some(amx) = amx_or_skip("every_session_starts_with_zeroed_tiles") else {
        return;
    };
    // Every tile non-zero in the first session, so the second one's zeros prove the reset for
    // all eight (a fresh thread's tiles start zeroed anyway).
    let ones = vec![0xFF_u8; TILE_BYTES];
    let mut out = vec![0xEE_u8; 8 * TILE_BYTES];
    macro_rules! each {
        ($t:ident, $op:ident, $buf:expr) => {
            each!($t, $op, $buf, T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5, T6 = 6, T7 = 7)
        };
        ($t:ident, load, $buf:expr, $($reg:ident = $i:literal),*) => {{
            $( $t.load::<plumb_tiles::$reg>($buf, ROW_BYTES); )*
        }};
        ($t:ident, store, $buf:expr, $($reg:ident = $i:literal),*) => {{
            $( $t.store::<plumb_tiles::$reg>(&mut $buf[$i * TILE_BYTES..], ROW_BYTES); )*
        }};
    }
    amx.with_tiles(|t| {
        each!(t, load, &ones);
        each!(t, store, out);
    });
    assert!(
        out.iter().all(|&b| b == 0xFF),
        "the first session loaded every tile"
    );
    out.fill(0xEE);
    amx.with_tiles(|t| each!(t, store, out));
    let dirty: Vec<usize> = (0..8)
        .filter(|i| out[i * TILE_BYTES..][..TILE_BYTES].iter().any(|&b| b != 0))
        .collect();
    assert!(
        dirty.is_empty(),
        "tiles {dirty:?} not zero at session start"
    );
}

#[test]
fn with_tiles_returns_the_closure_result() {
    let Some(amx) = amx_or_skip("with_tiles_returns_the_closure_result") else {
        return;
    };
    let src = vec![0x11_u8; TILE_BYTES];
    let v = amx.with_tiles(|t| {
        let mut row = [0_u8; 2 * ROW_BYTES];
        t.load::<T0>(&src, ROW_BYTES);
        t.store::<T0>(&mut row, 0); // stride 0: all 16 rows into the first 64 bytes
        row
    });
    assert!(v[..ROW_BYTES].iter().all(|&b| b == 0x11) && v[ROW_BYTES..].iter().all(|&b| b == 0));
}

#[test]
fn nested_session_panics_and_the_outer_one_is_released() {
    let Some(amx) = amx_or_skip("nested_session_panics_and_the_outer_one_is_released") else {
        return;
    };
    let msg = panic_message(|| amx.with_tiles(|_| amx.with_tiles(|_| ())));
    assert!(msg.contains("nested with_tiles"), "unexpected panic: {msg}");
    // Through a compute token's `amx()` too: it's the same thread state.
    if let Some(i8) = amx.int8() {
        let msg = panic_message(|| amx.with_tiles(|_| i8.amx().with_tiles(|_| ())));
        assert!(msg.contains("nested with_tiles"), "unexpected panic: {msg}");
    }
    assert_eq!(
        tile_config(amx),
        [0; 64],
        "the unwind released the outer session"
    );
    round_trip(amx, 1);
}

/// The compute tokens' `with_tiles` is the same session (with the `nightly` feature it also
/// enables the compute set): same configuration, released after, and it nests with `Amx`
/// sessions in neither order.
#[test]
fn compute_token_sessions_behave_like_amx_sessions() {
    let Some(amx) = amx_or_skip("compute_token_sessions_behave_like_amx_sessions") else {
        return;
    };
    let mut checked = 0;
    let mut check = |name: &str, open: &dyn Fn(&mut dyn FnMut()), nested: &dyn Fn()| {
        let mut inside = [0_u8; 64];
        open(&mut || inside = tile_config(amx));
        assert_eq!(inside, expected_config(), "{name}: configuration inside");
        assert_eq!(tile_config(amx), [0; 64], "{name}: released after");
        let msg = panic_message(nested);
        assert!(
            msg.contains("nested with_tiles"),
            "{name}: unexpected panic: {msg}"
        );
        let msg = panic_message(|| open(&mut || amx.with_tiles(|_| ())));
        assert!(
            msg.contains("nested with_tiles"),
            "{name}: unexpected panic: {msg}"
        );
        assert_eq!(
            tile_config(amx),
            [0; 64],
            "{name}: released after the unwinds"
        );
        checked += 1;
    };
    if let Some(i8) = amx.int8() {
        check("int8", &|f| i8.with_tiles(|_| f()), &|| {
            amx.with_tiles(|_| i8.with_tiles(|_| ()));
        });
    }
    if let Some(bf16) = amx.bf16() {
        check("bf16", &|f| bf16.with_tiles(|_| f()), &|| {
            amx.with_tiles(|_| bf16.with_tiles(|_| ()));
        });
    }
    if let Some(fp16) = amx.fp16() {
        check("fp16", &|f| fp16.with_tiles(|_| f()), &|| {
            amx.with_tiles(|_| fp16.with_tiles(|_| ()));
        });
    }
    if checked == 0 {
        eprintln!("skip: compute_token_sessions_behave_like_amx_sessions: no AMX compute set");
    }
    round_trip(amx, 4);
}

/// The session's panics point at the caller's line: `#[track_caller]` holds through every
/// layer, including the `nightly` feature's target-feature wrappers.
#[test]
fn session_panics_report_the_callers_location() {
    let Some(amx) = amx_or_skip("session_panics_report_the_callers_location") else {
        return;
    };
    let here = |line| (file!().to_owned(), line);
    let line = line!() + 1;
    let at = panic_location(|| amx.with_tiles(|_| amx.with_tiles(|_| ())));
    assert_eq!(at, here(line), "nested Amx session");
    let line = line!() + 1;
    let at = panic_location(|| amx.with_tiles(|t| t.load::<T0>(&[], 0)));
    assert_eq!(at, here(line), "out-of-bounds load");
    if let Some(i8) = amx.int8() {
        let line = line!() + 1;
        let at = panic_location(|| i8.with_tiles(|_| i8.with_tiles(|_| ())));
        assert_eq!(at, here(line), "nested int8 session");
    }
}

/// A nested attempt must panic before touching the outer configuration: code that catches the
/// panic inside the outer session keeps working tiles.
#[test]
fn caught_nested_panic_leaves_the_outer_session_intact() {
    let Some(amx) = amx_or_skip("caught_nested_panic_leaves_the_outer_session_intact") else {
        return;
    };
    let src = Lcg::new(2).bytes(TILE_BYTES);
    let mut dst = vec![0_u8; TILE_BYTES];
    amx.with_tiles(|t| {
        t.load::<T0>(&src, ROW_BYTES);
        let msg = panic_message(|| amx.with_tiles(|_| ()));
        assert!(msg.contains("nested with_tiles"), "unexpected panic: {msg}");
        assert_eq!(tile_config(amx), expected_config());
        t.store::<T0>(&mut dst, ROW_BYTES);
    });
    assert_eq!(src, dst, "T0 survived the rejected nested session");
}

/// A configuration this crate didn't make (another copy of it, another AMX library) is found
/// from the hardware state and left alone: the session refuses before LDTILECFG, so the
/// foreign tile shapes and data survive, and once they're released sessions work again.
#[test]
fn session_refuses_a_foreign_configuration_and_leaves_it_alone() {
    let Some(amx) = amx_or_skip("session_refuses_a_foreign_configuration_and_leaves_it_alone")
    else {
        return;
    };
    // Palette 1 with only tmm0, at 8 rows of 32 bytes: nothing like the crate's descriptor.
    let mut foreign = [0_u8; 64];
    foreign[0] = 1;
    foreign[16..18].copy_from_slice(&32_u16.to_le_bytes());
    foreign[48] = 8;
    let data = Lcg::new(6).bytes(8 * 32);
    let mut back = vec![0_u8; 8 * 32];
    // SAFETY: the token proves AMX-TILE. The descriptor is valid for palette 1; tmm0's 8 rows
    // of 32 bytes at stride 32 are exactly `data` and `back`. TILERELEASE runs before any
    // session below expects released tiles.
    unsafe {
        std::arch::asm!("ldtilecfg [{}]", in(reg) foreign.as_ptr(), options(nostack, readonly, preserves_flags));
        std::arch::asm!("tileloadd tmm0, [{} + {}*1]", in(reg) data.as_ptr(), in(reg) 32_usize,
                        options(nostack, readonly, preserves_flags));
    }
    let msg = panic_message(|| amx.with_tiles(|_| ()));
    assert!(
        msg.contains("already configured"),
        "unexpected panic: {msg}"
    );
    // Through a compute token as well.
    if let Some(i8) = amx.int8() {
        let msg = panic_message(|| i8.amx().with_tiles(|_| ()));
        assert!(
            msg.contains("already configured"),
            "unexpected panic: {msg}"
        );
    }
    assert_eq!(
        tile_config(amx),
        foreign,
        "the refused session reconfigured the tiles"
    );
    // SAFETY: as above; tmm0 is still configured as 8 x 32.
    unsafe {
        std::arch::asm!("tilestored [{} + {}*1], tmm0", in(reg) back.as_mut_ptr(), in(reg) 32_usize,
                        options(nostack, preserves_flags));
        std::arch::asm!("tilerelease", options(nostack, nomem, preserves_flags));
    }
    assert_eq!(
        back, data,
        "the refused session touched the foreign tile data"
    );
    // The refusal left the per-thread flag clear: sessions work once the tiles are released.
    round_trip(amx, 5);
    round_trip(amx, 6);
}

#[test]
fn panic_inside_session_releases_tiles() {
    let Some(amx) = amx_or_skip("panic_inside_session_releases_tiles") else {
        return;
    };
    let msg = panic_message(|| {
        amx.with_tiles(|t| {
            t.zero::<T0>();
            panic!("boom inside the session");
        });
    });
    assert_eq!(msg, "boom inside the session");
    assert_eq!(
        tile_config(amx),
        [0; 64],
        "TILERELEASE ran during the unwind"
    );
    round_trip(amx, 3);
    // And again: the per-thread guard was reset, not just the hardware state.
    round_trip(amx, 4);
}

#[test]
fn sessions_on_many_threads_are_independent() {
    let Some(amx) = amx_or_skip("sessions_on_many_threads_are_independent") else {
        return;
    };
    std::thread::scope(|s| {
        for i in 0..8_u64 {
            s.spawn(move || {
                for round in 0..50 {
                    round_trip(amx, i * 1000 + round);
                }
            });
        }
    });
}

/// Tile data must survive the thread being switched out mid-session: XCR0 detection claims the
/// OS saves and restores it, and nothing else would test that. More threads than CPUs, two
/// different tiles live per thread, and a sleep plus yields between load and store, so every
/// session is descheduled while other threads use their own tiles on the same cores.
#[test]
fn tile_data_survives_context_switches() {
    let Some(amx) = amx_or_skip("tile_data_survives_context_switches") else {
        return;
    };
    let threads = 4 * std::thread::available_parallelism().map_or(8, |n| n.get());
    std::thread::scope(|s| {
        for i in 0..threads as u64 {
            s.spawn(move || {
                for round in 0..4 {
                    let seed = 1_000_000 + i * 100 + round * 2;
                    let (a, b) = (
                        Lcg::new(seed).bytes(TILE_BYTES),
                        Lcg::new(seed + 1).bytes(TILE_BYTES),
                    );
                    let (mut a2, mut b2) = (vec![0_u8; TILE_BYTES], vec![0_u8; TILE_BYTES]);
                    amx.with_tiles(|t| {
                        t.load::<T2>(&a, ROW_BYTES);
                        t.load::<T5>(&b, ROW_BYTES);
                        std::thread::sleep(Duration::from_millis(1));
                        for _ in 0..20 {
                            std::thread::yield_now();
                        }
                        t.store::<T2>(&mut a2, ROW_BYTES);
                        t.store::<T5>(&mut b2, ROW_BYTES);
                    });
                    assert_eq!(a2, a, "thread {i} round {round}: tmm2");
                    assert_eq!(b2, b, "thread {i} round {round}: tmm5");
                }
            });
        }
    });
}
