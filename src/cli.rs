//! Command line options.

use std::{ffi::OsString, path::PathBuf, time::Duration};

/// The standard test length, long enough to warm up the GPU and catch most unstable memory
pub const STANDARD_DURATION: Duration = Duration::from_secs(5 * 60);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TestDuration {
    Limited(Duration),
    Endless,
}

impl Default for TestDuration {
    fn default() -> Self {
        Self::Limited(STANDARD_DURATION)
    }
}

impl TestDuration {
    /// Value accepted back by `--duration`, used to pass it to the worker process
    pub fn to_arg(self) -> String {
        match self {
            Self::Limited(duration) => format!("{}s", duration.as_secs()),
            Self::Endless => "endless".to_owned(),
        }
    }
}

/// What kind of test runs, chosen with --test or on the start screen
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TestMode {
    /// the address-based pattern, written and read at full speed
    #[default]
    Standard,
    /// all the test types in turn, then a longer hold: for final checks and suspect cards
    Thorough,
    /// bit patterns switching the most memory bus lines at once
    Bus,
    /// all patterns, going through memory in random order
    Random,
    /// all patterns in bursts of load with idle gaps between them
    Pulsed,
}

/// The default duration of the thorough test, which goes through all test types
pub const THOROUGH_DURATION: Duration = Duration::from_secs(15 * 60);

impl TestMode {
    pub const ALL: [Self; 5] = [
        Self::Standard,
        Self::Thorough,
        Self::Bus,
        Self::Random,
        Self::Pulsed,
    ];

    /// The name accepted by --test
    pub fn arg(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Thorough => "thorough",
            Self::Bus => "bus",
            Self::Random => "random",
            Self::Pulsed => "pulsed",
        }
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let lower = text.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|mode| mode.arg() == lower)
            .ok_or_else(|| {
                format!("unknown test '{text}', use standard, thorough, bus, random or pulsed")
            })
    }

    /// What the test does, for the start screen and the test plan
    pub fn description(self) -> &'static str {
        match self {
            Self::Standard => "address pattern at full speed",
            Self::Thorough => "all test types in turn, then a longer hold",
            Self::Bus => "bus-toggling patterns",
            Self::Random => "random access order",
            Self::Pulsed => "pulsed load (experimental)",
        }
    }

    /// The duration used when none is given
    pub fn default_duration(self) -> TestDuration {
        match self {
            Self::Thorough => TestDuration::Limited(THOROUGH_DURATION),
            _ => TestDuration::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum LogTarget {
    #[default]
    Default,
    File(PathBuf),
    Disabled,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    /// GPU number as shown in the list (1-based)
    pub device: Option<usize>,
    /// Internal: index in Vulkan device enumeration, used by the worker process
    pub device_enumeration_index: Option<usize>,
    /// Memory to test, in bytes
    pub size: Option<i64>,
    pub duration: TestDuration,
    /// The duration was given, rather than the test's default
    pub duration_given: bool,
    pub mode: TestMode,
    /// Maximum size of one test window, in megabytes
    pub window_mb: Option<i64>,
    pub log: LogTarget,
    /// End the test at the first error found
    pub stop_on_error: bool,
    pub verbose: bool,
    /// No prompts or key waits
    pub batch: bool,
    /// Only list devices
    pub list: bool,
    /// Internal: run the test in this process, as started by the console process
    pub worker: bool,
}

#[derive(Debug, PartialEq)]
pub enum Command {
    Run(Options),
    Help,
    Version,
}

pub const HELP: &str = "\
Usage: memtest_vulkan [OPTIONS]

Tests GPU video memory (VRAM) for errors using Vulkan compute shaders.
Without options it lists the GPUs and starts the standard 5-minute test after a
short countdown. Before that you can type a GPU number, or press S for settings:
the GPU, the test, the duration, the memory size and stopping at the first error.

Options:
  -d, --device <N>       Test GPU number N as shown in the list (no prompt)
  -s, --size <SIZE>      Memory to test, like 4G or 512M (default: all free VRAM)
  -t, --duration <TIME>  How long to test, like 30s, 10m, 2h or \"endless\"
                         (default: 5m, the standard test; 15m for thorough)
      --test <TEST>      What to test with (default: standard):
                           standard  address-based pattern at full speed
                           thorough  all of the tests below in turn, then a
                                     longer idle hold before a last check
                           bus       bit patterns switching the most memory
                                     bus lines at once
                           random    all patterns in random access order
                           pulsed    all patterns in bursts of load
                                     (experimental)
  -w, --window-mb <MB>   Limit one test window to MB megabytes (128MB steps,
                         default up to 1024). If errors disappear with small
                         windows, like 256, they come from the driver
      --log <FILE>       Log file (default: memtest_vulkan.log in the current folder)
      --no-log           Don't write a log file
  -e, --stop-on-error    End the test at the first error, like when tuning an overclock
  -l, --list             List the GPUs and exit
  -b, --batch            No prompts and no key press at the end, for scripts
  -v, --verbose          Print diagnostic information
  -h, --help             Show this help
  -V, --version          Show the version and license

Press Ctrl+C once to stop testing and show the results, twice to quit at once.

Exit codes:
  0  test passed, no errors
  1  memory errors found
  2  results invalid: test writes didn't reach the memory (likely a driver problem)
  3  the test couldn't run, or stopped because of a runtime error
  4  stopped before the test duration ended (or cancelled), no errors so far
";

/// Parses a memory size like `4G`, `512MB`, `1.5GiB` or plain bytes
pub fn parse_size(text: &str) -> Result<i64, String> {
    let lower = text.trim().to_ascii_lowercase();
    let number_end = lower
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(lower.len());
    let (number, unit) = lower.split_at(number_end);
    let multiplier: f64 = match unit.trim() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => {
            return Err(format!(
                "unknown size unit in '{text}', use like 4G or 512M"
            ));
        }
    };
    let value: f64 = number
        .parse()
        .map_err(|_| format!("invalid size '{text}', use like 4G or 512M"))?;
    let bytes = value * multiplier;
    if !(1.0..9.0e18).contains(&bytes) {
        return Err(format!("size '{text}' is out of range"));
    }
    Ok(bytes as i64)
}

/// Parses a duration like `90s`, `10m`, `2h`, `endless`; a plain number means minutes
pub fn parse_duration(text: &str) -> Result<TestDuration, String> {
    let lower = text.trim().to_ascii_lowercase();
    if ["endless", "infinite", "inf", "forever", "0"].contains(&lower.as_str()) {
        return Ok(TestDuration::Endless);
    }
    let number_end = lower
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(lower.len());
    let (number, unit) = lower.split_at(number_end);
    let seconds_per_unit = match unit.trim() {
        "s" | "sec" | "secs" | "second" | "seconds" => 1.0,
        "" | "m" | "min" | "mins" | "minute" | "minutes" => 60.0,
        "h" | "hour" | "hours" => 3600.0,
        _ => {
            return Err(format!(
                "unknown time unit in '{text}', use like 30s, 10m or 2h"
            ));
        }
    };
    let value: f64 = number
        .parse()
        .map_err(|_| format!("invalid duration '{text}', use like 30s, 10m or 2h"))?;
    let seconds = value * seconds_per_unit;
    if !(1.0..1.0e9).contains(&seconds) {
        return Err(format!("duration '{text}' is out of range"));
    }
    Ok(TestDuration::Limited(Duration::from_secs_f64(seconds)))
}

/// Parses the arguments after the program name
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let args: Vec<String> = args
        .into_iter()
        .map(|arg| {
            arg.into_string()
                .map_err(|arg| format!("argument {arg:?} isn't valid text"))
        })
        .collect::<Result<_, _>>()?;

    // Legacy form `memtest_vulkan <device> [<bytes>]` used by earlier versions for the worker
    // process and by scripts: runs the test non-interactively without a time limit
    if let Some(first) = args.first()
        && let Ok(label) = first.parse::<isize>()
    {
        let mut options = Options {
            worker: true,
            duration: TestDuration::Endless,
            ..Options::default()
        };
        if label > 0 {
            options.device = Some(label as usize);
        } else if label < 0 {
            options.device_enumeration_index = Some((-1 - label) as usize);
        }
        if let Some(bytes) = args.get(1).and_then(|s| s.parse::<i64>().ok()) {
            options.size = Some(bytes);
        }
        return Ok(Command::Run(options));
    }

    let mut options = Options::default();
    let mut duration_given = false;
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        let (name, inline_value) = match arg.split_once('=') {
            Some((name, value)) if name.starts_with("--") => {
                (name.to_owned(), Some(value.to_owned()))
            }
            _ => (arg.clone(), None),
        };
        let mut value = |what: &str| -> Result<String, String> {
            inline_value
                .clone()
                .or_else(|| iter.next())
                .ok_or_else(|| format!("{name} needs {what}"))
        };
        match name.as_str() {
            "-h" | "--help" | "/?" => return Ok(Command::Help),
            "-V" | "--version" => return Ok(Command::Version),
            "-d" | "--device" => {
                let text = value("a GPU number")?;
                let device = text
                    .trim()
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| {
                        format!("invalid GPU number '{text}', use the number from the list")
                    })?;
                options.device = Some(device);
            }
            "--device-enumeration-index" => {
                let text = value("an index")?;
                options.device_enumeration_index = Some(
                    text.trim()
                        .parse()
                        .map_err(|_| format!("invalid index '{text}'"))?,
                );
            }
            "-s" | "--size" => options.size = Some(parse_size(&value("a size like 4G")?)?),
            "-t" | "--duration" => {
                options.duration = parse_duration(&value("a time like 10m")?)?;
                duration_given = true;
            }
            "--test" => options.mode = TestMode::parse(&value("a test like thorough")?)?,
            "-w" | "--window-mb" => {
                let text = value("a size in megabytes")?;
                let mb = text
                    .trim()
                    .parse::<i64>()
                    .ok()
                    .filter(|&mb| mb > 0)
                    .ok_or_else(|| {
                        format!("invalid window size '{text}', use megabytes like 512")
                    })?;
                options.window_mb = Some(mb);
            }
            "--log" => options.log = LogTarget::File(PathBuf::from(value("a file name")?)),
            "--no-log" => options.log = LogTarget::Disabled,
            "-e" | "--stop-on-error" => options.stop_on_error = true,
            "-l" | "--list" => options.list = true,
            "-b" | "--batch" => options.batch = true,
            "-v" | "--verbose" => options.verbose = true,
            "--worker" => options.worker = true,
            _ => return Err(format!("unknown option '{arg}', see --help")),
        }
    }
    if !duration_given {
        options.duration = options.mode.default_duration();
    }
    options.duration_given = duration_given;
    Ok(Command::Run(options))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Options {
        match parse(args.iter().map(OsString::from)) {
            Ok(Command::Run(options)) => options,
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn test_mode_sets_the_default_duration() {
        let thorough = run(&["--test", "thorough"]);
        assert_eq!(thorough.mode, TestMode::Thorough);
        assert!(!thorough.duration_given);
        assert!(run(&["-t", "5m"]).duration_given);
        assert_eq!(thorough.duration, TestDuration::Limited(THOROUGH_DURATION));
        // a given duration wins, wherever it is
        for args in [
            ["--test", "Thorough", "-t", "2m"],
            ["-t", "2m", "--test", "thorough"],
        ] {
            let options = run(&args);
            assert_eq!(options.mode, TestMode::Thorough);
            assert_eq!(
                options.duration,
                TestDuration::Limited(Duration::from_secs(120))
            );
        }
        for mode in TestMode::ALL {
            assert_eq!(run(&["--test", mode.arg()]).mode, mode);
        }
        assert_eq!(run(&["--test=bus"]).duration, TestDuration::default());
        assert!(parse(["--test", "fast"].map(OsString::from)).is_err());
    }

    #[test]
    fn defaults_to_standard_interactive_test() {
        let options = run(&[]);
        assert_eq!(options, Options::default());
        assert_eq!(options.duration, TestDuration::Limited(STANDARD_DURATION));
    }

    #[test]
    fn parses_all_options() {
        let options = run(&[
            "-d",
            "2",
            "--size=1.5G",
            "--duration",
            "2h",
            "-w",
            "512",
            "--log",
            "x.log",
            "-b",
            "-v",
            "-e",
        ]);
        assert!(options.stop_on_error);
        assert_eq!(options.device, Some(2));
        assert_eq!(options.size, Some(1536 * 1024 * 1024));
        assert_eq!(
            options.duration,
            TestDuration::Limited(Duration::from_secs(7200))
        );
        assert_eq!(options.window_mb, Some(512));
        assert_eq!(options.log, LogTarget::File("x.log".into()));
        assert!(options.batch && options.verbose && !options.worker);
    }

    #[test]
    fn keeps_legacy_positional_form() {
        let options = run(&["-2", "4294967296"]);
        assert!(options.worker);
        assert_eq!(options.device_enumeration_index, Some(1));
        assert_eq!(options.size, Some(4294967296));
        assert_eq!(options.duration, TestDuration::Endless);
        assert_eq!(run(&["3"]).device, Some(3));
    }

    #[test]
    fn parses_sizes_and_durations() {
        assert_eq!(parse_size("512M"), Ok(512 * 1024 * 1024));
        assert_eq!(parse_size("4gb"), Ok(4 * 1024 * 1024 * 1024));
        assert_eq!(parse_size("1000"), Ok(1000));
        assert!(parse_size("4X").is_err());
        assert!(parse_size("").is_err());
        assert_eq!(
            parse_duration("90s"),
            Ok(TestDuration::Limited(Duration::from_secs(90)))
        );
        assert_eq!(
            parse_duration("10"),
            Ok(TestDuration::Limited(Duration::from_secs(600)))
        );
        assert_eq!(parse_duration("endless"), Ok(TestDuration::Endless));
        assert!(parse_duration("5 days").is_err());
        let limited = TestDuration::Limited(Duration::from_secs(125));
        assert_eq!(parse_duration(&limited.to_arg()), Ok(limited));
    }

    #[test]
    fn rejects_bad_input() {
        for args in [&["--bogus"][..], &["-d", "0"], &["-d"], &["-w", "-5"]] {
            assert!(parse(args.iter().map(OsString::from)).is_err(), "{args:?}");
        }
        assert_eq!(parse([OsString::from("--help")]), Ok(Command::Help));
    }
}
