//! Scoped measurement of initial public-instance and phase-zero advice commitments.
//!
//! The timer records the wall-clock union of in-flight commitment operations.
//! This avoids summing worker-thread durations when Halo2 commits in parallel.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct State {
    active: usize,
    started: Option<Instant>,
    elapsed: Duration,
    instance_commitments: usize,
    advice_commitments: usize,
    normalization_batches: usize,
}

static MEASUREMENT_ACTIVE: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<State> = Mutex::new(State {
    active: 0,
    started: None,
    elapsed: Duration::ZERO,
    instance_commitments: 0,
    advice_commitments: 0,
    normalization_batches: 0,
});

/// Owns one proof's commitment measurement. Only one proof may be measured in
/// a process at a time; individual commitment calls may run concurrently.
#[derive(Debug)]
pub struct Measurement {
    finished: bool,
}

/// Per-proof native commitment diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementResult {
    /// Wall-clock union of all measured native commitment operations.
    pub elapsed: Duration,
    /// Number of public-instance commitments included in the measurement.
    pub instance_commitments: usize,
    /// Number of phase-zero advice commitments included in the measurement.
    pub advice_commitments: usize,
    /// Number of affine-normalization batches included in the measurement.
    pub normalization_batches: usize,
}

/// Measures one native initial commitment, including affine normalization
/// when the caller keeps the guard alive through that step.
#[derive(Debug)]
pub struct Guard {
    enabled: bool,
}

fn lock_state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Begin an isolated commitment measurement for one native proof.
pub fn begin() -> Result<Measurement, &'static str> {
    MEASUREMENT_ACTIVE
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| "a Halo2 commitment measurement is already active")?;
    {
        let mut state = lock_state();
        if state.active != 0 {
            MEASUREMENT_ACTIVE.store(false, Ordering::Release);
            return Err("Halo2 commitment timer has unfinished operations");
        }
        state.started = None;
        state.elapsed = Duration::ZERO;
        state.instance_commitments = 0;
        state.advice_commitments = 0;
        state.normalization_batches = 0;
    }
    Ok(Measurement { finished: false })
}

impl Measurement {
    /// Finish the measurement and return the union wall duration.
    pub fn finish(mut self) -> Result<MeasurementResult, &'static str> {
        let result = {
            let state = lock_state();
            if state.active != 0 || state.started.is_some() {
                return Err("Halo2 commitment timer still has active operations");
            }
            MeasurementResult {
                elapsed: state.elapsed,
                instance_commitments: state.instance_commitments,
                advice_commitments: state.advice_commitments,
                normalization_batches: state.normalization_batches,
            }
        };
        self.finished = true;
        MEASUREMENT_ACTIVE.store(false, Ordering::Release);
        Ok(result)
    }
}

impl Drop for Measurement {
    fn drop(&mut self) {
        if !self.finished {
            MEASUREMENT_ACTIVE.store(false, Ordering::Release);
        }
    }
}

/// Start timing a single native commitment call. Nested/concurrent guards are
/// accumulated as a union, so their total cannot exceed elapsed proof time.
pub fn guard() -> Guard {
    if !MEASUREMENT_ACTIVE.load(Ordering::Acquire) {
        return Guard { enabled: false };
    }
    let mut state = lock_state();
    // Recheck after taking the mutex to avoid racing a completed measurement.
    if !MEASUREMENT_ACTIVE.load(Ordering::Acquire) {
        return Guard { enabled: false };
    }
    if state.active == 0 {
        state.started = Some(Instant::now());
    }
    state.active += 1;
    Guard { enabled: true }
}

/// Start timing an actual initial public-instance commitment and count it for
/// diagnostics. The caller keeps this guard through affine normalization.
pub fn instance_guard() -> Guard {
    let guard = guard();
    if guard.enabled {
        let mut state = lock_state();
        state.instance_commitments += 1;
    }
    guard
}

/// Start timing one initial phase-zero advice-polynomial commitment.
pub fn advice_guard() -> Guard {
    let guard = guard();
    if guard.enabled {
        let mut state = lock_state();
        state.advice_commitments += 1;
    }
    guard
}

/// Start timing one batch of affine normalization for initial commitments.
pub fn normalization_guard() -> Guard {
    let guard = guard();
    if guard.enabled {
        let mut state = lock_state();
        state.normalization_batches += 1;
    }
    guard
}

impl Drop for Guard {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        let mut state = lock_state();
        state.active = state.active.saturating_sub(1);
        if state.active == 0 {
            if let Some(started) = state.started.take() {
                state.elapsed += started.elapsed();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{advice_guard, begin, guard, instance_guard, normalization_guard};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn concurrent_guards_accumulate_union_wall_time() {
        let measurement = begin().unwrap();
        let first = guard();
        let worker = thread::spawn(|| {
            let _second = guard();
            thread::sleep(Duration::from_millis(30));
        });
        thread::sleep(Duration::from_millis(30));
        drop(first);
        worker.join().unwrap();
        let result = measurement.finish().unwrap();
        assert!(result.elapsed >= Duration::from_millis(25));
        assert!(result.elapsed < Duration::from_millis(58));
        assert_eq!(result.instance_commitments, 0);
    }

    #[test]
    fn guards_outside_a_measurement_are_noops() {
        drop(guard());
        assert!(begin().unwrap().finish().unwrap().elapsed.is_zero());
    }

    #[test]
    fn initial_commitment_categories_are_counted_separately() {
        let measurement = begin().unwrap();
        drop(instance_guard());
        drop(advice_guard());
        drop(advice_guard());
        drop(normalization_guard());
        let result = measurement.finish().unwrap();
        assert_eq!(result.instance_commitments, 1);
        assert_eq!(result.advice_commitments, 2);
        assert_eq!(result.normalization_batches, 1);
    }
}
