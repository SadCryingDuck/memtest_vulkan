//! End-to-end checks on a real Vulkan device: injected memory errors, failures and edge cases must
//! give the right result. They need a GPU, or a software Vulkan driver like Mesa's lavapipe (used
//! by CI), so they're ignored by default. Run them one at a time, since they share the GPU:
//!
//!     cargo test --release --test gpu -- --ignored --test-threads=1
//!
//! MEMTEST_CHECK_DEVICE (GPU number, default 1), MEMTEST_CHECK_SIZE (default 1G) and
//! MEMTEST_CHECK_SECONDS (length of the passing runs, default 4) adjust them.

use std::process::Command;

/// The emulation variables, cleared for every run so the caller's environment can't interfere
const EMULATION_VARS: [&str; 5] = [
    "MEMTEST_VULKAN_EMULATE_WRITE_BUG_ITERATION",
    "MEMTEST_VULKAN_EMULATE_SKIPPED_WRITE_ITERATION",
    "MEMTEST_VULKAN_EMULATE_ALLOCATION_LIMIT",
    "MEMTEST_VULKAN_EMULATE_FAILURE",
    "MEMTEST_VULKAN_MAX_WINDOW_MB",
];

const MODES: [&str; 4] = ["standard", "bus", "random", "pulsed"];

/// Exit codes, as documented in --help
const PASSED: i32 = 0;
const ERRORS_FOUND: i32 = 1;
const WRITES_NOT_APPLIED: i32 = 2;
const NOT_RUN: i32 = 3;

struct Run {
    code: i32,
    output: String,
}

impl Run {
    fn expect(&self, code: i32, texts: &[&str], what: &str) {
        let missing: Vec<&&str> = texts
            .iter()
            .filter(|text| !self.output.contains(**text))
            .collect();
        assert!(
            self.code == code && missing.is_empty(),
            "{what}: expected exit code {code} and {texts:?}, got exit code {} and missing \
             {missing:?} in:\n{}",
            self.code,
            self.output
        );
    }
}

fn setting(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

fn seconds() -> String {
    format!("{}s", setting("MEMTEST_CHECK_SECONDS", "4"))
}

/// Runs memtest_vulkan in batch mode on the checked GPU and size, with `args` and `env` added
fn memtest(args: &[&str], env: &[(&str, &str)]) -> Run {
    let device = setting("MEMTEST_CHECK_DEVICE", "1");
    let size = setting("MEMTEST_CHECK_SIZE", "1G");
    let mut command = Command::new(env!("CARGO_BIN_EXE_memtest_vulkan"));
    command
        .args(["--batch", "--no-log", "-d", &device, "-s", &size])
        .args(args);
    for var in EMULATION_VARS {
        command.env_remove(var);
    }
    for (name, value) in env {
        command.env(name, value);
    }
    let output = command.output().expect("couldn't start memtest_vulkan");
    Run {
        code: output.status.code().unwrap_or(-1),
        output: String::from_utf8_lossy(&output.stdout).into_owned()
            + &String::from_utf8_lossy(&output.stderr),
    }
}

/// A run that ends at the first error found, so the error checks take seconds
fn until_first_error(mode: &str, env: &[(&str, &str)]) -> Run {
    memtest(&["--test", mode, "-t", "120s", "--stop-on-error"], env)
}

#[test]
#[ignore = "needs a Vulkan device"]
fn lists_the_gpus_and_shows_the_version() {
    memtest(&["--list"], &[]).expect(PASSED, &["1: "], "--list");
    memtest(&["-V"], &[]).expect(PASSED, &["GPLv3+"], "-V");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn every_test_passes_on_good_memory() {
    for mode in MODES {
        memtest(&["--test", mode, "-t", &seconds()], &[]).expect(
            PASSED,
            &["Result:     PASSED"],
            mode,
        );
    }
}

#[test]
#[ignore = "needs a Vulkan device"]
fn bit_flips_fail_in_every_test_and_both_halves_of_a_pair() {
    for mode in MODES {
        for iteration in ["3", "4"] {
            until_first_error(
                mode,
                &[("MEMTEST_VULKAN_EMULATE_WRITE_BUG_ITERATION", iteration)],
            )
            .expect(
                ERRORS_FOUND,
                &["Error found", "Result:     FAILED"],
                &format!("{mode} test, bit flip in iteration {iteration}"),
            );
        }
    }
}

#[test]
#[ignore = "needs a Vulkan device"]
fn writes_that_never_land_are_invalid_not_errors() {
    for mode in MODES {
        for iteration in ["3", "6"] {
            until_first_error(
                mode,
                &[("MEMTEST_VULKAN_EMULATE_SKIPPED_WRITE_ITERATION", iteration)],
            )
            .expect(
                WRITES_NOT_APPLIED,
                &["WRITES NOT APPLIED", "Result:     INVALID"],
                &format!("{mode} test, write skipped in iteration {iteration}"),
            );
        }
    }
    // the initial write of the last window
    until_first_error(
        "standard",
        &[("MEMTEST_VULKAN_EMULATE_SKIPPED_WRITE_ITERATION", "0")],
    )
    .expect(
        WRITES_NOT_APPLIED,
        &["Result:     INVALID"],
        "initial write skipped",
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn gpu_failures_end_the_test_instead_of_retrying() {
    // pass 1 is in the initial fill, pass 30 in the test itself
    for failure in ["device-lost", "hang", "panic"] {
        for pass in ["1", "30"] {
            let run = until_first_error(
                "standard",
                &[(
                    "MEMTEST_VULKAN_EMULATE_FAILURE",
                    &format!("{failure}:{pass}"),
                )],
            );
            let what = format!("{failure} in pass {pass}");
            run.expect(NOT_RUN, &["runtime error"], &what);
            assert!(
                !run.output.contains("Retrying with less memory"),
                "{what} must not be retried:\n{}",
                run.output
            );
        }
    }
}

#[test]
#[ignore = "needs a Vulkan device"]
fn errors_found_before_a_crash_still_fail() {
    // An iteration is 2 passes per window, so pass 400 comes well after iteration 3 even with
    // many small windows (like lavapipe's, limited to 128MB)
    memtest(
        &["-t", "120s"],
        &[
            ("MEMTEST_VULKAN_EMULATE_WRITE_BUG_ITERATION", "3"),
            ("MEMTEST_VULKAN_EMULATE_FAILURE", "panic:400"),
        ],
    )
    .expect(ERRORS_FOUND, &["ERRORS FOUND"], "bit flip, then a panic");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn running_out_of_memory_retries_with_less() {
    // 1GB doesn't fit under a 700MB limit, 1GB minus the 400MB retry step does
    memtest(
        &["-s", "1G", "-t", &seconds()],
        &[("MEMTEST_VULKAN_EMULATE_ALLOCATION_LIMIT", "700M")],
    )
    .expect(
        PASSED,
        &["Retrying with less memory", "Result:     PASSED"],
        "allocation limit 700M",
    );
    // nothing worth testing fits under 300MB
    memtest(
        &["-s", "1G", "-t", &seconds()],
        &[("MEMTEST_VULKAN_EMULATE_ALLOCATION_LIMIT", "300M")],
    )
    .expect(NOT_RUN, &["failed even with"], "allocation limit 300M");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn too_little_memory_stops_at_once() {
    memtest(&["-s", "200M", "-t", &seconds()], &[]).expect(
        NOT_RUN,
        &["smaller than the minimum"],
        "--size 200M",
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn thorough_test_runs_every_part_and_finds_data_lost_in_the_hold() {
    // 70 seconds: all four parts, a 30 second hold, then the check after it, where window 2
    // has lost its data. Lost data is a memory error, not a write that never landed.
    let run = memtest(
        &["--test", "thorough", "-t", "70s"],
        &[("MEMTEST_VULKAN_EMULATE_FAILURE", "fade")],
    );
    run.expect(
        ERRORS_FOUND,
        &[
            "Testing with address pattern",
            "Testing with bus-toggling patterns",
            "Testing with random access order",
            "Testing with pulsed load",
            "Pausing the GPU load for 30 seconds",
            "Mode AFTER_PAUSE",
            "Result:     FAILED",
        ],
        "thorough test with data lost in the hold",
    );
    assert!(!run.output.contains("WRITES NOT APPLIED"), "{}", run.output);
}
