// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The load kernels (load.rs) on the real CPU: each matches its reference, the check catches a
//! wrong value, and a duty-cycled run stops when told. Kinds the CPU lacks are skipped.

use crate::common::detect;
use crate::load::{Duty, Kind, Load, Stats};
use std::sync::atomic::{AtomicBool, Ordering};

const KINDS: [Kind; 4] = [Kind::Sse2, Kind::Avx2, Kind::Avx512, Kind::Amx];

fn loads() -> Vec<Load> {
    let cpu = detect();
    KINDS
        .iter()
        .filter_map(|&k| {
            let l = Load::new(k, &cpu);
            if l.is_none() {
                eprintln!("skip: no {} on this CPU/OS", k.name());
            }
            l
        })
        .collect()
}

#[test]
fn every_batch_matches_its_reference() {
    for l in loads() {
        for _ in 0..3 {
            assert_eq!(
                l.batch(),
                0,
                "{}: batch differs from the reference",
                l.kind().name()
            );
        }
    }
}

#[test]
fn the_check_catches_a_wrong_value() {
    let cpu = detect();
    for &k in &KINDS {
        let Some(l) = Load::new(k, &cpu) else {
            continue;
        };
        assert_eq!(
            l.with_wrong_reference().batch(),
            1,
            "{}: one wrong value",
            k.name()
        );
    }
}

#[test]
fn a_duty_cycled_run_counts_batches_and_stops() {
    for l in loads() {
        let (stop, stats) = (AtomicBool::new(false), Stats::default());
        std::thread::scope(|s| {
            s.spawn(|| {
                l.run(
                    Duty {
                        on_us: 200,
                        off_us: 200,
                    },
                    &stop,
                    &stats,
                )
            });
            std::thread::sleep(std::time::Duration::from_millis(20));
            stop.store(true, Ordering::Relaxed);
        });
        assert!(
            stats.batches.load(Ordering::Relaxed) > 0,
            "{}: no batches",
            l.kind().name()
        );
        assert_eq!(
            stats.errors.load(Ordering::Relaxed),
            0,
            "{}: errors",
            l.kind().name()
        );
    }
}

#[test]
fn duty_parses_on_off_microseconds() {
    assert_eq!(
        Duty::parse("500:250"),
        Some(Duty {
            on_us: 500,
            off_us: 250
        })
    );
    assert_eq!(Duty::parse("0:5"), None);
    assert_eq!(Duty::parse("500"), None);
    assert_eq!(Kind::parse("avx512"), Some(Kind::Avx512));
    assert_eq!(Kind::parse("avx"), None);
}

/// Load placement: the other hyperthread of each test core, or alternating cores without SMT.
#[test]
fn load_threads_go_beside_the_test_threads() {
    use crate::{load_partner, load_placement};
    let smt = vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]];
    assert_eq!(load_placement(&smt, 2), Some((vec![0, 2], vec![1, 3])));
    assert_eq!(
        load_placement(&smt, 4),
        Some((vec![0, 2, 4, 6], vec![1, 3, 5, 7]))
    );
    assert_eq!(load_placement(&smt, 5), None);
    let single: Vec<Vec<usize>> = (0..4).map(|c| vec![c]).collect();
    assert_eq!(load_placement(&single, 2), Some((vec![0, 2], vec![1, 3])));
    assert_eq!(load_placement(&single, 3), None);
    assert_eq!(load_partner(&smt, 2), Some(3));
    assert_eq!(load_partner(&smt, 3), Some(2));
    assert_eq!(load_partner(&single, 1), Some(2));
    assert_eq!(load_partner(&single, 3), Some(0));
    assert_eq!(load_partner(&[vec![0]], 0), None);
    assert_eq!(load_partner(&smt, 9), None);
}
