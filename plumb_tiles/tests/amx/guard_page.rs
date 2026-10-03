// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tile loads and stores flush against inaccessible pages (Windows: `VirtualAlloc`, then
//! `PAGE_NOACCESS` on the pages either side). Next to a heap `Vec` a load that reads past the
//! checked span goes unnoticed, because the bytes after it are readable; here a single byte
//! outside `tile_span(stride)`, above or below, is an access violation. Every placement also
//! checks the loaded rows and that the store changed exactly its rows.

#![cfg(windows)]

use core::ffi::c_void;
use std::ptr;

use plumb_tiles::{Amx, ROW_BYTES, ROWS, T0, T1, T2, TILE_BYTES, tile_span};

use crate::{Lcg, amx_or_skip};

const PAGE: usize = 4096;
/// Read/write bytes between the guard pages: room for the largest span below.
const DATA: usize = 2 * PAGE;

/// 0 and below 64 overlap rows; 65, 72 and 268 put most rows across a cache-line boundary.
const STRIDES: [usize; 7] = [0, 1, 32, 64, 65, 72, 268];

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_NOACCESS: u32 = 0x01;
const PAGE_READWRITE: u32 = 0x04;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualAlloc(address: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
    fn VirtualProtect(address: *mut c_void, size: usize, protect: u32, old: *mut u32) -> i32;
    fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> i32;
}

/// `DATA` read/write bytes with a no-access page on each side.
struct Guarded(*mut u8);

impl Guarded {
    fn new() -> Self {
        // SAFETY: a fresh private allocation of ordinary 4 KiB pages.
        let base = unsafe {
            VirtualAlloc(
                ptr::null_mut(),
                DATA + 2 * PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!base.is_null(), "VirtualAlloc failed");
        for guard in [0, PAGE + DATA] {
            let mut old = 0;
            // SAFETY: the page lies inside the allocation, and nothing refers to it yet.
            let ok =
                unsafe { VirtualProtect(base.add(guard).cast(), PAGE, PAGE_NOACCESS, &mut old) };
            assert_ne!(ok, 0, "VirtualProtect failed");
        }
        Self(base)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: DATA committed read/write bytes after the first guard page, borrowed for as
        // long as `self` is.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(PAGE), DATA) }
    }

    fn words(&mut self) -> &mut [u64] {
        // SAFETY: as `bytes`; the start is page aligned, so aligned for u64, and every bit
        // pattern is a valid u64.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(PAGE).cast(), DATA / 8) }
    }
}

impl Drop for Guarded {
    fn drop(&mut self) {
        // SAFETY: the allocation from `new`; no borrow of it outlives `self`.
        unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE) };
    }
}

/// One placement: `len` bytes from `start` of the data pages, through the byte methods or the
/// `_u64` ones (then `start` and `len` are multiples of 8). Loads with both hints, stores a
/// different tile back at the same stride, and checks both against fresh random contents.
fn check(
    amx: Amx,
    mem: &mut Guarded,
    stride: usize,
    start: usize,
    len: usize,
    words: bool,
    seed: u64,
) {
    let place = format!(
        "stride {stride}, bytes {start}..{}, u64 {words}",
        start + len
    );
    let before = Lcg::new(seed).bytes(DATA);
    mem.bytes().copy_from_slice(&before);
    let tile = Lcg::new(seed + 1000).bytes(TILE_BYTES);
    let (mut a, mut b) = (vec![0_u8; TILE_BYTES], vec![0_u8; TILE_BYTES]);
    amx.with_tiles(|t| {
        if words {
            let buf = &mut mem.words()[start / 8..][..len / 8];
            t.load_u64::<T0>(buf, stride);
            t.load_t1_u64::<T1>(buf, stride);
            t.load::<T2>(&tile, ROW_BYTES);
            t.store_u64::<T2>(buf, stride);
        } else {
            let buf = &mut mem.bytes()[start..][..len];
            t.load::<T0>(buf, stride);
            t.load_t1::<T1>(buf, stride);
            t.load::<T2>(&tile, ROW_BYTES);
            t.store::<T2>(buf, stride);
        }
        t.store::<T0>(&mut a, ROW_BYTES);
        t.store::<T1>(&mut b, ROW_BYTES);
    });
    let mut want = before.clone();
    for r in 0..ROWS {
        let row = start + r * stride..start + r * stride + ROW_BYTES;
        let got = r * ROW_BYTES..(r + 1) * ROW_BYTES;
        assert_eq!(
            a[got.clone()],
            before[row.clone()],
            "load, {place}, row {r}"
        );
        assert_eq!(
            b[got.clone()],
            before[row.clone()],
            "load_t1, {place}, row {r}"
        );
        // Rows are stored in order, so where they overlap the later one wins.
        want[row].copy_from_slice(&tile[got]);
    }
    assert!(mem.bytes() == want, "store, {place}: wrong bytes written");
}

#[test]
fn tile_ops_stay_inside_the_span_next_to_guard_pages() {
    let Some(amx) = amx_or_skip("tile_ops_stay_inside_the_span_next_to_guard_pages") else {
        return;
    };
    let mut mem = Guarded::new();
    for (i, stride) in STRIDES.into_iter().enumerate() {
        let span = tile_span(stride).unwrap();
        let seed = 4 * i as u64;
        // Ending at the upper guard page, then starting at the lower one.
        check(amx, &mut mem, stride, DATA - span, span, false, seed);
        check(amx, &mut mem, stride, 0, span, false, seed + 1);
        // The u64 slices are whole words, so the tile ends up to 7 bytes short of the upper
        // guard when the span isn't a multiple of 8.
        let up = (DATA - span) / 8 * 8;
        check(amx, &mut mem, stride, up, DATA - up, true, seed + 2);
        check(
            amx,
            &mut mem,
            stride,
            0,
            span.div_ceil(8) * 8,
            true,
            seed + 3,
        );
    }
}
