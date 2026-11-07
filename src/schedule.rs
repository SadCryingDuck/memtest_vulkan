//! What a test mode runs when: the pattern kinds, access order and load of each part, and the
//! hold near the end, when the load pauses before all memory is checked again.

use crate::{
    cli::{STANDARD_DURATION, TestDuration, TestMode},
    iobuf::{AccessOrder, PatternKind},
};
use std::time::{Duration, Instant};

/// One part of a test: what it writes and how
#[derive(Debug, PartialEq)]
pub struct Phase {
    pub name: &'static str,
    /// kinds used in turn, one per pair of iterations (the pattern, then inverted)
    pub kinds: &'static [PatternKind],
    pub order: AccessOrder,
    /// bursts of load with idle gaps between them
    pub pulsed: bool,
}

const ALL_KINDS: &[PatternKind] = &[
    PatternKind::Address,
    PatternKind::WordToggle,
    PatternKind::Checkerboard,
];

pub const ADDRESS: Phase = Phase {
    name: "address pattern",
    kinds: &[PatternKind::Address],
    order: AccessOrder::Sequential,
    pulsed: false,
};
pub const BUS: Phase = Phase {
    name: "bus-toggling patterns",
    kinds: &[PatternKind::WordToggle, PatternKind::Checkerboard],
    order: AccessOrder::Sequential,
    pulsed: false,
};
pub const RANDOM: Phase = Phase {
    name: "random access order",
    kinds: ALL_KINDS,
    order: AccessOrder::Random,
    pulsed: false,
};
pub const PULSED: Phase = Phase {
    name: "pulsed load",
    kinds: ALL_KINDS,
    order: AccessOrder::Sequential,
    pulsed: true,
};

/// The thorough test's phases, with their share (in percent) of the time before the hold
const THOROUGH: [(&Phase, u32); 4] = [(&ADDRESS, 30), (&BUS, 30), (&RANDOM, 20), (&PULSED, 20)];
/// Phase length of an endless thorough test, which repeats the phases
const ENDLESS_THOROUGH_PHASE: Duration = Duration::from_secs(3 * 60);
/// Testing after the hold's check, before the test ends
const AFTER_HOLD: Duration = Duration::from_secs(15);
const STANDARD_HOLD: Duration = Duration::from_secs(15);

pub struct Schedule {
    mode: TestMode,
    target: Option<Duration>,
}

impl Schedule {
    pub fn new(mode: TestMode, duration: TestDuration) -> Self {
        let target = match duration {
            TestDuration::Limited(target) => Some(target),
            TestDuration::Endless => None,
        };
        Self { mode, target }
    }

    /// The phase running at `elapsed`
    pub fn phase_at(&self, elapsed: Duration) -> &'static Phase {
        match self.mode {
            TestMode::Standard => &ADDRESS,
            TestMode::Bus => &BUS,
            TestMode::Random => &RANDOM,
            TestMode::Pulsed => &PULSED,
            TestMode::Thorough => match self.target {
                Some(target) => {
                    let active = self.pause_at().unwrap_or(target);
                    let mut end = Duration::ZERO;
                    for (phase, share) in THOROUGH {
                        end += active * share / 100;
                        if elapsed < end {
                            return phase;
                        }
                    }
                    THOROUGH[THOROUGH.len() - 1].0
                }
                None => {
                    let phase = elapsed.as_secs() / ENDLESS_THOROUGH_PHASE.as_secs();
                    THOROUGH[phase as usize % THOROUGH.len()].0
                }
            },
        }
    }

    /// How long the load pauses before all memory is checked again. The thorough test holds
    /// longer, to find memory cells that slowly lose their data.
    pub fn hold(&self) -> Duration {
        match (self.mode, self.target) {
            (TestMode::Thorough, Some(target)) => {
                (target / 10).clamp(Duration::from_secs(30), Duration::from_secs(5 * 60))
            }
            (TestMode::Thorough, None) => Duration::from_secs(2 * 60),
            _ => STANDARD_HOLD,
        }
    }

    /// When the hold starts, None for tests too short for one
    pub fn pause_at(&self) -> Option<Duration> {
        let hold_and_after = self.hold() + AFTER_HOLD;
        match (self.mode, self.target) {
            (_, Some(target)) if target < Duration::from_secs(60) => None,
            (_, Some(target)) => target.checked_sub(hold_and_after),
            // endless: after one round of all phases, or near the end of the standard time
            (TestMode::Thorough, None) => Some(ENDLESS_THOROUGH_PHASE * THOROUGH.len() as u32),
            (_, None) => STANDARD_DURATION.checked_sub(hold_and_after),
        }
    }

    /// Testing time after which stopping with Ctrl+C still gives a complete result
    pub fn required(&self) -> Duration {
        match (self.mode, self.target) {
            (TestMode::Thorough, Some(target)) => target,
            // one round of all phases and the hold
            (TestMode::Thorough, None) => {
                self.pause_at().unwrap_or_default() + self.hold() + AFTER_HOLD
            }
            (_, Some(target)) => target.min(STANDARD_DURATION),
            (_, None) => STANDARD_DURATION,
        }
    }

    /// What the test plan says about the test, None for the standard test
    pub fn description(&self) -> Option<String> {
        match self.mode {
            TestMode::Standard => None,
            TestMode::Thorough => Some(format!(
                "thorough test: address pattern, bus-toggling patterns, random access order \
                 and pulsed load in turn, then a {}s hold",
                self.hold().as_secs()
            )),
            mode => Some(format!("{} test: {}", mode.arg(), mode.description())),
        }
    }
}

/// Bursts of load and idle gaps of the pulsed phases, in milliseconds. Varied, so the power
/// and clock changes of the GPU come at different rhythms.
const PULSES: [(u64, u64); 6] = [
    (1000, 500),
    (200, 200),
    (2000, 1000),
    (500, 100),
    (100, 300),
    (300, 700),
];

/// Times the bursts of the pulsed load
#[derive(Default)]
pub struct Pulse {
    burst_end: Option<Instant>,
    step: usize,
}

impl Pulse {
    /// Called after each GPU pass: the idle gap to take now, if the burst is over
    pub fn after_pass(&mut self, now: Instant) -> Option<Duration> {
        let (burst, gap) = PULSES[self.step % PULSES.len()];
        let end = *self
            .burst_end
            .get_or_insert(now + Duration::from_millis(burst));
        if now < end {
            return None;
        }
        // the next burst starts after the gap
        self.step += 1;
        self.burst_end = None;
        Some(Duration::from_millis(gap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minutes(m: u64) -> Duration {
        Duration::from_secs(m * 60)
    }

    #[test]
    fn standard_test_keeps_its_15_second_pause() {
        let schedule = Schedule::new(TestMode::Standard, TestDuration::default());
        assert_eq!(schedule.hold(), Duration::from_secs(15));
        assert_eq!(
            schedule.pause_at(),
            Some(minutes(5) - Duration::from_secs(30))
        );
        assert_eq!(schedule.phase_at(minutes(4)), &ADDRESS);
        assert_eq!(schedule.required(), minutes(5));
        assert_eq!(schedule.description(), None);
        let short = Schedule::new(TestMode::Standard, TestDuration::Limited(minutes(1) / 2));
        assert_eq!(short.pause_at(), None);
        let endless = Schedule::new(TestMode::Standard, TestDuration::Endless);
        assert_eq!(
            endless.pause_at(),
            Some(minutes(5) - Duration::from_secs(30))
        );
    }

    #[test]
    fn thorough_test_goes_through_all_phases_then_holds() {
        let schedule = Schedule::new(TestMode::Thorough, TestDuration::Limited(minutes(15)));
        // a 90s hold, then 15s of testing after the check
        assert_eq!(schedule.hold(), Duration::from_secs(90));
        let pause_at = minutes(15) - Duration::from_secs(105);
        assert_eq!(schedule.pause_at(), Some(pause_at));
        // 13:15 before the hold: 30% address, 30% bus, 20% random, 20% pulsed
        assert_eq!(schedule.phase_at(Duration::ZERO), &ADDRESS);
        assert_eq!(schedule.phase_at(pause_at * 29 / 100), &ADDRESS);
        assert_eq!(schedule.phase_at(pause_at * 31 / 100), &BUS);
        assert_eq!(schedule.phase_at(pause_at * 61 / 100), &RANDOM);
        assert_eq!(schedule.phase_at(pause_at * 81 / 100), &PULSED);
        assert_eq!(schedule.phase_at(minutes(15)), &PULSED);
        assert_eq!(schedule.required(), minutes(15));
        assert!(schedule.description().unwrap().ends_with("then a 90s hold"));
        // the hold stays between 30 seconds and 5 minutes
        let short = Schedule::new(TestMode::Thorough, TestDuration::Limited(minutes(2)));
        assert_eq!(short.hold(), Duration::from_secs(30));
        let long = Schedule::new(TestMode::Thorough, TestDuration::Limited(minutes(120)));
        assert_eq!(long.hold(), minutes(5));
    }

    #[test]
    fn endless_thorough_test_repeats_the_phases() {
        let schedule = Schedule::new(TestMode::Thorough, TestDuration::Endless);
        assert_eq!(schedule.phase_at(minutes(1)), &ADDRESS);
        assert_eq!(schedule.phase_at(minutes(4)), &BUS);
        assert_eq!(schedule.phase_at(minutes(7)), &RANDOM);
        assert_eq!(schedule.phase_at(minutes(10)), &PULSED);
        assert_eq!(schedule.phase_at(minutes(13)), &ADDRESS);
        assert_eq!(schedule.pause_at(), Some(minutes(12)));
        assert_eq!(schedule.required(), minutes(14) + Duration::from_secs(15));
    }

    #[test]
    fn single_type_tests_use_their_phase() {
        for (mode, phase) in [
            (TestMode::Bus, &BUS),
            (TestMode::Random, &RANDOM),
            (TestMode::Pulsed, &PULSED),
        ] {
            let schedule = Schedule::new(mode, TestDuration::default());
            assert_eq!(schedule.phase_at(minutes(2)), phase);
            assert_eq!(schedule.hold(), Duration::from_secs(15));
        }
    }

    #[test]
    fn pulses_alternate_bursts_and_gaps() {
        let start = Instant::now();
        let mut pulse = Pulse::default();
        // the first burst lasts 1s
        assert_eq!(pulse.after_pass(start), None);
        assert_eq!(pulse.after_pass(start + Duration::from_millis(900)), None);
        assert_eq!(
            pulse.after_pass(start + Duration::from_millis(1000)),
            Some(Duration::from_millis(500))
        );
        // then 200ms with a 200ms gap
        let next = start + Duration::from_millis(1500);
        assert_eq!(pulse.after_pass(next), None);
        assert_eq!(
            pulse.after_pass(next + Duration::from_millis(200)),
            Some(Duration::from_millis(200))
        );
    }
}
