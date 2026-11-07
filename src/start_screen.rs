//! The screen before the test: starts the test after a short countdown, or lets the user choose
//! the GPU or change the settings first. Doing nothing runs the default test.

use crate::{
    cli::{STANDARD_DURATION, TestDuration},
    shader::{GB, MIN_WANTED_ALLOCATION},
    tester::format_duration,
};
use crossterm::{
    cursor::{MoveToColumn, MoveUp},
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    queue,
    style::Print,
    terminal::{self, Clear, ClearType},
};
use std::{
    io::{IsTerminal, Write},
    time::{Duration, Instant},
};

/// What the test will do
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// index in the device list
    pub device: usize,
    pub duration: TestDuration,
    /// memory to test, None for all free memory
    pub size: Option<i64>,
    pub stop_on_error: bool,
}

/// A GPU offered on the screen
pub struct DeviceChoice {
    pub label: String,
    /// memory that can be tested by default
    pub testable: i64,
}

pub enum Outcome {
    Start(Settings),
    Quit,
}

/// Durations offered in the settings, the standard test first
const DURATIONS: [TestDuration; 7] = [
    TestDuration::Limited(STANDARD_DURATION),
    TestDuration::Limited(Duration::from_secs(60)),
    TestDuration::Limited(Duration::from_secs(15 * 60)),
    TestDuration::Limited(Duration::from_secs(30 * 60)),
    TestDuration::Limited(Duration::from_secs(60 * 60)),
    TestDuration::Limited(Duration::from_secs(2 * 60 * 60)),
    TestDuration::Endless,
];

/// Memory sizes offered in the settings, below the default amount
const SIZES_GB: [i64; 11] = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64];

pub fn next_duration(current: TestDuration) -> TestDuration {
    match DURATIONS.iter().position(|duration| *duration == current) {
        Some(index) => DURATIONS[(index + 1) % DURATIONS.len()],
        None => DURATIONS[0], // a duration set on the command line
    }
}

/// The memory sizes to choose from: all free memory, then fixed sizes smaller than that
pub fn size_choices(testable: i64) -> Vec<Option<i64>> {
    let mut choices = vec![None];
    choices.extend(
        SIZES_GB
            .iter()
            .map(|gb| gb * 1024 * 1024 * 1024)
            .filter(|&size| size >= MIN_WANTED_ALLOCATION && size < testable)
            .map(Some),
    );
    choices
}

pub fn next_size(current: Option<i64>, testable: i64) -> Option<i64> {
    let choices = size_choices(testable);
    match choices.iter().position(|size| *size == current) {
        Some(index) => choices[(index + 1) % choices.len()],
        None => None,
    }
}

pub fn describe_duration(duration: TestDuration) -> String {
    match duration {
        TestDuration::Limited(d) if d == STANDARD_DURATION => "5:00 (standard test)".to_owned(),
        TestDuration::Limited(d) => format_duration(d),
        TestDuration::Endless => "endless (until Ctrl+C)".to_owned(),
    }
}

/// Short form for the countdown line, like "standard 5-minute test" or "15:00 test"
fn describe_test(duration: TestDuration) -> String {
    match duration {
        TestDuration::Limited(d) if d == STANDARD_DURATION => "standard 5-minute test".to_owned(),
        TestDuration::Limited(d) => format!("{} test", format_duration(d)),
        TestDuration::Endless => "endless test".to_owned(),
    }
}

pub fn describe_size(size: Option<i64>, testable: i64) -> String {
    match size {
        None => format!("all free memory (about {:.1}GB)", testable as f32 / GB),
        Some(size) => format!("{:.0}GB", size as f32 / GB),
    }
}

/// Raw console mode while the screen is shown, restored when dropped
struct RawMode;

impl RawMode {
    fn enable() -> Option<Self> {
        terminal::enable_raw_mode().ok().map(|()| Self)
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
    }
}

/// Fits a line to the console width, so redrawing in place works
fn fit(line: &str) -> String {
    let width = terminal::size().map_or(80, |(columns, _)| columns as usize);
    line.chars().take(width.saturating_sub(1)).collect()
}

enum Key {
    Char(char),
    Enter,
    Backspace,
    Quit,
    Other,
}

/// Waits up to `timeout` for a key press
fn read_key(timeout: Duration) -> std::io::Result<Option<Key>> {
    if !event::poll(timeout)? {
        return Ok(None);
    }
    let Event::Key(KeyEvent {
        code,
        modifiers,
        kind,
        ..
    }) = event::read()?
    else {
        return Ok(Some(Key::Other));
    };
    if kind == KeyEventKind::Release {
        return Ok(Some(Key::Other));
    }
    let ctrl = modifiers.contains(KeyModifiers::CONTROL);
    Ok(Some(match code {
        KeyCode::Enter => Key::Enter,
        KeyCode::Esc => Key::Quit,
        KeyCode::Char('c' | 'q' | 'd') if ctrl => Key::Quit,
        KeyCode::Char(c) if !ctrl => Key::Char(c.to_ascii_lowercase()),
        KeyCode::Backspace | KeyCode::Delete => Key::Backspace,
        _ => Key::Other,
    }))
}

/// Shows the screen. Without an interactive console, starts the test at once.
pub fn show(devices: &[DeviceChoice], settings: Settings, countdown: Duration) -> Outcome {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Outcome::Start(settings);
    }
    let Some(raw_mode) = RawMode::enable() else {
        return Outcome::Start(settings);
    };
    let outcome = run(devices, settings, countdown).unwrap_or(Outcome::Quit);
    drop(raw_mode);
    outcome
}

fn run(
    devices: &[DeviceChoice],
    mut settings: Settings,
    countdown: Duration,
) -> std::io::Result<Outcome> {
    let mut out = std::io::stdout();
    let several = devices.len() > 1;
    queue!(
        out,
        Print(format!(
            "S: settings   Enter: start now   Esc: quit{}\r\n",
            if several {
                format!("   1-{}: choose the GPU", devices.len())
            } else {
                String::new()
            }
        ))
    )?;
    let start = Instant::now();
    let mut counting = true;
    let mut typed = String::new();
    loop {
        let line = if !typed.is_empty() {
            format!("GPU number: {typed}  (Enter to confirm)")
        } else if counting {
            let seconds_left = countdown.saturating_sub(start.elapsed()).as_secs() + 1;
            format!(
                "Starting the {} of GPU {} in {seconds_left} second{}",
                describe_test(settings.duration),
                settings.device + 1,
                if seconds_left == 1 { "" } else { "s" }
            )
        } else {
            format!(
                "Ready: the {} of GPU {}",
                describe_test(settings.duration),
                settings.device + 1
            )
        };
        queue!(
            out,
            MoveToColumn(0),
            Print(fit(&line)),
            Clear(ClearType::UntilNewLine)
        )?;
        out.flush()?;
        if counting && start.elapsed() >= countdown {
            queue!(out, Print("\r\n"))?;
            return Ok(Outcome::Start(settings));
        }
        match read_key(Duration::from_millis(250))? {
            None | Some(Key::Other) => {}
            Some(Key::Quit) => {
                queue!(out, Print("\r\n"))?;
                return Ok(Outcome::Quit);
            }
            Some(Key::Enter) => {
                if !typed.is_empty() {
                    match typed.parse::<usize>() {
                        Ok(number) if (1..=devices.len()).contains(&number) => {
                            settings.device = number - 1;
                        }
                        _ => {
                            typed.clear();
                            continue;
                        }
                    }
                }
                queue!(out, Print("\r\n"))?;
                return Ok(Outcome::Start(settings));
            }
            Some(Key::Backspace) => {
                typed.pop();
            }
            Some(Key::Char(c)) if c.is_ascii_digit() && several => {
                counting = false;
                typed.push(c);
            }
            Some(Key::Char('s')) => {
                queue!(out, Print("\r\n"))?;
                return settings_menu(&mut out, devices, settings);
            }
            Some(Key::Char(_)) => {}
        }
    }
}

/// The settings menu: single keys change the settings, redrawn in place
fn settings_menu(
    out: &mut std::io::Stdout,
    devices: &[DeviceChoice],
    mut settings: Settings,
) -> std::io::Result<Outcome> {
    const LINES: u16 = 4;
    queue!(
        out,
        Print("Settings: press a key to change, Enter to start the test, Esc to quit\r\n")
    )?;
    let mut first_draw = true;
    loop {
        if !first_draw {
            queue!(out, MoveUp(LINES))?;
        }
        first_draw = false;
        let device = &devices[settings.device];
        let lines = [
            format!(
                "  G  GPU                  {}: {}",
                settings.device + 1,
                device.label
            ),
            format!(
                "  D  Duration             {}",
                describe_duration(settings.duration)
            ),
            format!(
                "  M  Memory               {}",
                describe_size(settings.size, device.testable)
            ),
            format!(
                "  E  Stop at first error  {}",
                if settings.stop_on_error { "yes" } else { "no" }
            ),
        ];
        for line in lines {
            queue!(
                out,
                MoveToColumn(0),
                Print(fit(&line)),
                Clear(ClearType::UntilNewLine),
                Print("\r\n")
            )?;
        }
        out.flush()?;
        match read_key(Duration::from_secs(3600))? {
            None | Some(Key::Other | Key::Backspace) => {}
            Some(Key::Quit) => return Ok(Outcome::Quit),
            Some(Key::Enter) => return Ok(Outcome::Start(settings)),
            Some(Key::Char('g')) => {
                settings.device = (settings.device + 1) % devices.len();
                // a fixed size may not fit the other GPU
                if settings
                    .size
                    .is_some_and(|size| size >= devices[settings.device].testable)
                {
                    settings.size = None;
                }
            }
            Some(Key::Char('d')) => settings.duration = next_duration(settings.duration),
            Some(Key::Char('m')) => {
                settings.size = next_size(settings.size, devices[settings.device].testable)
            }
            Some(Key::Char('e')) => settings.stop_on_error = !settings.stop_on_error,
            Some(Key::Char(_)) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: i64 = 1024 * 1024 * 1024;

    #[test]
    fn durations_cycle_from_the_standard_test() {
        let mut duration = TestDuration::default();
        let mut seen = vec![duration];
        for _ in 0..DURATIONS.len() {
            duration = next_duration(duration);
            seen.push(duration);
        }
        assert_eq!(seen.first(), seen.last()); // back to the standard test
        assert_eq!(seen[1], TestDuration::Limited(Duration::from_secs(60)));
        assert!(seen.contains(&TestDuration::Endless));
        // a duration from the command line goes to the standard test
        assert_eq!(
            next_duration(TestDuration::Limited(Duration::from_secs(42))),
            TestDuration::default()
        );
    }

    #[test]
    fn sizes_offered_fit_the_free_memory() {
        // like Intel Arc B580 with 10.7GB testable
        let testable = 107 * GIB / 10;
        assert_eq!(
            size_choices(testable),
            vec![
                None,
                Some(GIB),
                Some(2 * GIB),
                Some(4 * GIB),
                Some(6 * GIB),
                Some(8 * GIB)
            ]
        );
        assert_eq!(next_size(None, testable), Some(GIB));
        assert_eq!(next_size(Some(8 * GIB), testable), None);
        // a size from the command line goes back to all free memory
        assert_eq!(next_size(Some(3 * GIB), testable), None);
        // too little memory for fixed sizes: only all free memory
        assert_eq!(size_choices(GIB / 2), vec![None]);
    }

    #[test]
    fn describes_settings() {
        assert_eq!(
            describe_duration(TestDuration::default()),
            "5:00 (standard test)"
        );
        assert_eq!(
            describe_duration(TestDuration::Limited(Duration::from_secs(3600))),
            "1:00:00"
        );
        assert_eq!(
            describe_test(TestDuration::default()),
            "standard 5-minute test"
        );
        assert_eq!(
            describe_size(None, 107 * GIB / 10),
            "all free memory (about 10.7GB)"
        );
        assert_eq!(describe_size(Some(4 * GIB), 0), "4GB");
    }
}
