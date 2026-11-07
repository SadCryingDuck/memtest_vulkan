//! The screen before the test: starts the test after a short countdown, or lets the user choose
//! the GPU or change the settings first. Doing nothing runs the default test.
//!
//! The settings menu logic is in `Menu`, which turns keys into setting changes and lines to
//! show; `show` only reads the keys and draws the lines, so the logic is testable on its own.

use crate::{
    cli::{STANDARD_DURATION, TestDuration},
    shader::{GB, MIN_WANTED_ALLOCATION},
    tester::format_duration,
};
use crossterm::{
    cursor::{MoveToColumn, MoveUp},
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    queue,
    style::{Attribute, Print, SetAttribute},
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

#[derive(Debug, PartialEq)]
pub enum Outcome {
    Start(Settings),
    Quit,
}

/// Durations offered in the settings, shortest first
const DURATIONS: [TestDuration; 7] = [
    TestDuration::Limited(Duration::from_secs(60)),
    TestDuration::Limited(STANDARD_DURATION),
    TestDuration::Limited(Duration::from_secs(15 * 60)),
    TestDuration::Limited(Duration::from_secs(30 * 60)),
    TestDuration::Limited(Duration::from_secs(60 * 60)),
    TestDuration::Limited(Duration::from_secs(2 * 60 * 60)),
    TestDuration::Endless,
];

/// Memory sizes offered in the settings, below the default amount
const SIZES_GB: [i64; 11] = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64];

/// Longest custom duration that can be typed, in minutes (a week)
const MAX_CUSTOM_MINUTES: u64 = 7 * 24 * 60;

/// Orders durations, endless last
fn duration_rank(duration: TestDuration) -> u64 {
    match duration {
        TestDuration::Limited(d) => d.as_secs(),
        TestDuration::Endless => u64::MAX,
    }
}

/// The next (`forward`) or previous offered duration, wrapping around; a custom duration goes
/// to the nearest offered one in that direction
pub fn step_duration(current: TestDuration, forward: bool) -> TestDuration {
    let rank = duration_rank(current);
    let found = if forward {
        DURATIONS.iter().find(|d| duration_rank(**d) > rank)
    } else {
        DURATIONS.iter().rev().find(|d| duration_rank(**d) < rank)
    };
    match (found, forward) {
        (Some(duration), _) => *duration,
        (None, true) => DURATIONS[0],
        (None, false) => DURATIONS[DURATIONS.len() - 1],
    }
}

/// Memory size choices from the smallest: fixed sizes below the free memory, then all of it
pub fn size_choices(testable: i64) -> Vec<Option<i64>> {
    let mut choices: Vec<Option<i64>> = SIZES_GB
        .iter()
        .map(|gb| gb * 1024 * 1024 * 1024)
        .filter(|&size| size >= MIN_WANTED_ALLOCATION && size < testable)
        .map(Some)
        .collect();
    choices.push(None);
    choices
}

/// The next (`forward`) or previous memory size choice, wrapping around; a custom size goes
/// to the nearest offered one in that direction
pub fn step_size(current: Option<i64>, testable: i64, forward: bool) -> Option<i64> {
    let choices = size_choices(testable);
    let rank = |size: Option<i64>| size.unwrap_or(i64::MAX);
    let found = if forward {
        choices.iter().find(|s| rank(**s) > rank(current))
    } else {
        choices.iter().rev().find(|s| rank(**s) < rank(current))
    };
    match (found, forward) {
        (Some(size), _) => *size,
        (None, true) => choices[0],
        (None, false) => choices[choices.len() - 1],
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
        Some(size) => format!("{}GB", trim_number(size as f64 / GB as f64)),
    }
}

/// 3 for 3.0, 2.5 for 2.5
fn trim_number(value: f64) -> String {
    let text = format!("{value:.1}");
    text.strip_suffix(".0").unwrap_or(&text).to_owned()
}

/// A key press, as the screen understands it
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Key {
    Char(char),
    Up,
    Down,
    Left,
    Right,
    Tab,
    Enter,
    Backspace,
    Esc,
    /// Ctrl+C and the like: always quits
    Interrupt,
    Other,
}

/// Rows of the settings menu
#[derive(Clone, Copy, Debug, PartialEq)]
enum Row {
    Gpu,
    Duration,
    Memory,
    StopOnError,
}

const ROWS: [Row; 4] = [Row::Gpu, Row::Duration, Row::Memory, Row::StopOnError];

/// The settings menu: a highlighted row whose value arrows change, letter shortcuts, and typed
/// custom values
pub struct Menu<'a> {
    devices: &'a [DeviceChoice],
    pub settings: Settings,
    selected: usize,
    /// a custom value being typed for the selected row
    editing: Option<String>,
    /// shown once in the hint line, like a rejected custom value
    message: Option<String>,
}

impl<'a> Menu<'a> {
    pub fn new(devices: &'a [DeviceChoice], settings: Settings) -> Self {
        Self {
            devices,
            settings,
            selected: 0,
            editing: None,
            message: None,
        }
    }

    fn testable(&self) -> i64 {
        self.devices[self.settings.device].testable
    }

    fn row(&self) -> Row {
        ROWS[self.selected]
    }

    fn select(&mut self, row: Row) {
        self.selected = ROWS.iter().position(|r| *r == row).unwrap_or(0);
    }

    /// Changes the value of `row` one step forward or back
    fn step(&mut self, row: Row, forward: bool) {
        let settings = &mut self.settings;
        match row {
            Row::Gpu => {
                let count = self.devices.len();
                settings.device = if forward {
                    (settings.device + 1) % count
                } else {
                    (settings.device + count - 1) % count
                };
                // a fixed size may not fit the other GPU
                if settings
                    .size
                    .is_some_and(|size| size >= self.devices[settings.device].testable)
                {
                    settings.size = None;
                }
            }
            Row::Duration => settings.duration = step_duration(settings.duration, forward),
            Row::Memory => {
                settings.size = step_size(
                    settings.size,
                    self.devices[settings.device].testable,
                    forward,
                )
            }
            Row::StopOnError => settings.stop_on_error = !settings.stop_on_error,
        }
    }

    /// Applies a typed custom value to the selected row
    fn apply_custom(&mut self, text: &str) {
        let Ok(value) = text.parse::<f64>() else {
            self.message = Some(format!("'{text}' isn't a number"));
            return;
        };
        match self.row() {
            Row::Gpu => {
                let number = value as usize;
                if value.fract() == 0.0 && (1..=self.devices.len()).contains(&number) {
                    self.settings.device = number - 1;
                    self.step(Row::Gpu, true);
                    self.step(Row::Gpu, false); // drops a fixed size that doesn't fit
                } else {
                    self.message = Some(format!("There's no GPU {text}"));
                }
            }
            Row::Duration => {
                if value >= 1.0 && value <= MAX_CUSTOM_MINUTES as f64 {
                    self.settings.duration =
                        TestDuration::Limited(Duration::from_secs((value * 60.0).round() as u64));
                } else {
                    self.message = Some(format!(
                        "Type 1 to {MAX_CUSTOM_MINUTES} minutes, or use Left/Right for endless"
                    ));
                }
            }
            Row::Memory => {
                let size = (value * GB as f64) as i64;
                if size >= MIN_WANTED_ALLOCATION && size <= self.testable() {
                    self.settings.size = Some(size);
                } else {
                    self.message = Some(format!(
                        "Type {} to {}GB",
                        trim_number(MIN_WANTED_ALLOCATION as f64 / GB as f64),
                        trim_number((self.testable() as f64 / GB as f64 * 10.0).floor() / 10.0)
                    ));
                }
            }
            Row::StopOnError => {}
        }
    }

    /// Handles a key; returns the outcome once the user starts the test or quits
    pub fn handle(&mut self, key: Key) -> Option<Outcome> {
        self.message = None;
        if key == Key::Interrupt {
            return Some(Outcome::Quit);
        }
        if let Some(text) = &mut self.editing {
            match key {
                Key::Char(c) if c.is_ascii_digit() || (c == '.' && !text.contains('.')) => {
                    text.push(c)
                }
                Key::Backspace => {
                    text.pop();
                }
                Key::Enter => {
                    let text = self.editing.take().unwrap_or_default();
                    if !text.is_empty() {
                        self.apply_custom(&text);
                    }
                }
                Key::Esc => self.editing = None,
                _ => {}
            }
            return None;
        }
        match key {
            Key::Enter => return Some(Outcome::Start(self.settings.clone())),
            Key::Esc => return Some(Outcome::Quit),
            Key::Up => self.selected = (self.selected + ROWS.len() - 1) % ROWS.len(),
            Key::Down | Key::Tab => self.selected = (self.selected + 1) % ROWS.len(),
            Key::Left => self.step(self.row(), false),
            Key::Right => self.step(self.row(), true),
            Key::Char(' ') if self.row() == Row::StopOnError => self.step(Row::StopOnError, true),
            Key::Char(c) if c.is_ascii_digit() && self.row() != Row::StopOnError => {
                self.editing = Some(c.to_string())
            }
            Key::Char(c) => {
                let row = match c.to_ascii_lowercase() {
                    'g' => Row::Gpu,
                    'd' => Row::Duration,
                    'm' => Row::Memory,
                    'e' => Row::StopOnError,
                    _ => return None,
                };
                self.select(row);
                self.step(row, true);
            }
            _ => {}
        }
        None
    }

    /// The menu lines, the selected row's index among them, and the hint line
    pub fn lines(&self) -> (Vec<String>, usize, String) {
        let device = &self.devices[self.settings.device];
        let value = |row: Row, text: String| match (&self.editing, self.row() == row) {
            (Some(typed), true) => format!("{typed}_"),
            _ => text,
        };
        let rows = vec![
            format!(
                "G  GPU                  {}",
                value(
                    Row::Gpu,
                    format!("{}: {}", self.settings.device + 1, device.label)
                )
            ),
            format!(
                "D  Duration             {}",
                value(Row::Duration, describe_duration(self.settings.duration))
            ),
            format!(
                "M  Memory               {}",
                value(
                    Row::Memory,
                    describe_size(self.settings.size, device.testable)
                )
            ),
            format!(
                "E  Stop at first error  {}",
                if self.settings.stop_on_error {
                    "yes"
                } else {
                    "no"
                }
            ),
        ];
        let lines = rows
            .into_iter()
            .enumerate()
            .map(|(i, row)| format!("{} {row}", if i == self.selected { ">" } else { " " }))
            .collect();
        let hint = if let Some(message) = &self.message {
            message.clone()
        } else if self.editing.is_some() {
            match self.row() {
                Row::Gpu => "Type the GPU number, Enter to set it, Esc to cancel",
                Row::Duration => "Type the duration in minutes, Enter to set it, Esc to cancel",
                _ => "Type the memory in GB, like 4 or 2.5, Enter to set it, Esc to cancel",
            }
            .to_owned()
        } else {
            match self.row() {
                Row::Gpu => "Left/Right or type a number to choose the GPU",
                Row::Duration => "Left/Right to change, or type a number of minutes",
                Row::Memory => "Left/Right to change, or type a number of GB",
                Row::StopOnError => "Left/Right or Space: end the test at the first error",
            }
            .to_owned()
        };
        (lines, self.selected, hint)
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
        KeyCode::Char('c' | 'q' | 'd') if ctrl => Key::Interrupt,
        KeyCode::Char(c) if !ctrl => Key::Char(c),
        KeyCode::Up => Key::Up,
        KeyCode::Down => Key::Down,
        KeyCode::Left => Key::Left,
        KeyCode::Right => Key::Right,
        KeyCode::Tab => Key::Tab,
        KeyCode::Enter => Key::Enter,
        KeyCode::Backspace | KeyCode::Delete => Key::Backspace,
        KeyCode::Esc => Key::Esc,
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
    let outcome = run(devices, settings.clone(), countdown);
    drop(raw_mode);
    // if the keyboard can't be read, start the default test rather than quitting:
    // the user may have started it and walked away
    outcome.unwrap_or_else(|e| {
        println!("\nCouldn't read the keyboard ({e}), starting the test.");
        Outcome::Start(settings)
    })
}

/// The countdown line; S or the arrow keys open the settings
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
            None | Some(Key::Other | Key::Tab | Key::Left | Key::Right) => {}
            Some(Key::Esc | Key::Interrupt) => {
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
            Some(Key::Char('s' | 'S') | Key::Up | Key::Down) => {
                queue!(out, Print("\r\n"))?;
                return settings_menu(&mut out, Menu::new(devices, settings));
            }
            Some(Key::Char(_)) => {}
        }
    }
}

/// Draws the settings menu and passes keys to it until the test starts or the user quits
fn settings_menu(out: &mut std::io::Stdout, mut menu: Menu) -> std::io::Result<Outcome> {
    queue!(
        out,
        Print(
            "Settings: Up/Down to choose, Left/Right to change, Enter to start the test, Esc to quit\r\n"
        )
    )?;
    let mut drawn_lines = 0;
    loop {
        let (lines, selected, hint) = menu.lines();
        if drawn_lines > 0 {
            queue!(out, MoveUp(drawn_lines))?;
        }
        for (i, line) in lines.iter().chain(std::iter::once(&hint)).enumerate() {
            queue!(out, MoveToColumn(0))?;
            if i == selected {
                queue!(out, SetAttribute(Attribute::Reverse))?;
            }
            queue!(out, Print(fit(line)), SetAttribute(Attribute::Reset))?;
            queue!(out, Clear(ClearType::UntilNewLine), Print("\r\n"))?;
        }
        drawn_lines = lines.len() as u16 + 1;
        out.flush()?;
        if let Some(key) = read_key(Duration::from_secs(3600))?
            && let Some(outcome) = menu.handle(key)
        {
            return Ok(outcome);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: i64 = 1024 * 1024 * 1024;

    fn minutes(m: u64) -> TestDuration {
        TestDuration::Limited(Duration::from_secs(m * 60))
    }

    #[test]
    fn durations_step_both_ways_and_wrap() {
        let standard = TestDuration::default();
        assert_eq!(step_duration(standard, true), minutes(15));
        assert_eq!(step_duration(standard, false), minutes(1));
        assert_eq!(step_duration(minutes(1), false), TestDuration::Endless);
        assert_eq!(step_duration(TestDuration::Endless, true), minutes(1));
        // a custom duration goes to the nearest offered one in that direction
        assert_eq!(step_duration(minutes(45), true), minutes(60));
        assert_eq!(step_duration(minutes(45), false), minutes(30));
    }

    #[test]
    fn sizes_step_both_ways_and_fit_the_free_memory() {
        // like Intel Arc B580 with 10.7GB testable
        let testable = 107 * GIB / 10;
        assert_eq!(
            size_choices(testable),
            vec![
                Some(GIB),
                Some(2 * GIB),
                Some(4 * GIB),
                Some(6 * GIB),
                Some(8 * GIB),
                None
            ]
        );
        assert_eq!(step_size(None, testable, false), Some(8 * GIB));
        assert_eq!(step_size(None, testable, true), Some(GIB));
        assert_eq!(step_size(Some(3 * GIB), testable, true), Some(4 * GIB));
        assert_eq!(step_size(Some(3 * GIB), testable, false), Some(2 * GIB));
        // too little memory for fixed sizes: only all free memory
        assert_eq!(size_choices(GIB / 2), vec![None]);
    }

    #[test]
    fn describes_settings() {
        assert_eq!(
            describe_duration(TestDuration::default()),
            "5:00 (standard test)"
        );
        assert_eq!(describe_duration(minutes(60)), "1:00:00");
        assert_eq!(
            describe_test(TestDuration::default()),
            "standard 5-minute test"
        );
        assert_eq!(
            describe_size(None, 107 * GIB / 10),
            "all free memory (about 10.7GB)"
        );
        assert_eq!(describe_size(Some(4 * GIB), 0), "4GB");
        assert_eq!(describe_size(Some(5 * GIB / 2), 0), "2.5GB");
    }

    fn devices() -> Vec<DeviceChoice> {
        vec![
            DeviceChoice {
                label: "Intel(R) Arc(TM) B580 Graphics".into(),
                testable: 107 * GIB / 10,
            },
            DeviceChoice {
                label: "AMD Radeon(TM) Graphics".into(),
                testable: 5 * GIB,
            },
        ]
    }

    fn defaults() -> Settings {
        Settings {
            device: 0,
            duration: TestDuration::default(),
            size: None,
            stop_on_error: false,
        }
    }

    /// Feeds keys to a menu, returning the outcome of the last one
    fn press(menu: &mut Menu, keys: &[Key]) -> Option<Outcome> {
        keys.iter().fold(None, |_, key| menu.handle(*key))
    }

    #[test]
    fn arrows_choose_rows_and_change_values() {
        let devices = devices();
        let mut menu = Menu::new(&devices, defaults());
        press(&mut menu, &[Key::Right]); // GPU 2
        press(&mut menu, &[Key::Down, Key::Left, Key::Left]); // 5:00 -> 1:00 -> endless
        press(&mut menu, &[Key::Down, Key::Left]); // all memory -> 4GB (AMD has 5GB)
        press(&mut menu, &[Key::Down, Key::Char(' ')]);
        let outcome = press(&mut menu, &[Key::Enter]);
        assert_eq!(
            outcome,
            Some(Outcome::Start(Settings {
                device: 1,
                duration: TestDuration::Endless,
                size: Some(4 * GIB),
                stop_on_error: true,
            }))
        );
        // Up wraps from the first row to the last
        let mut menu = Menu::new(&devices, defaults());
        press(&mut menu, &[Key::Up, Key::Right]);
        assert!(menu.settings.stop_on_error);
    }

    #[test]
    fn letter_shortcuts_select_and_change() {
        let devices = devices();
        let mut menu = Menu::new(&devices, defaults());
        press(&mut menu, &[Key::Char('d'), Key::Char('M'), Key::Char('e')]);
        assert_eq!(menu.settings.duration, minutes(15));
        assert_eq!(menu.settings.size, Some(GIB));
        assert!(menu.settings.stop_on_error);
        let (lines, selected, _) = menu.lines();
        assert_eq!(selected, 3);
        assert!(
            lines[3].starts_with("> E  Stop at first error  yes"),
            "{lines:?}"
        );
    }

    #[test]
    fn typed_custom_values() {
        let devices = devices();
        let mut menu = Menu::new(&devices, defaults());
        press(&mut menu, &[Key::Down, Key::Char('4'), Key::Char('5')]);
        let (lines, _, hint) = menu.lines();
        assert!(lines[1].ends_with("45_"), "{lines:?}");
        assert!(hint.contains("minutes"), "{hint}");
        press(&mut menu, &[Key::Enter]);
        assert_eq!(menu.settings.duration, minutes(45));

        // memory in GB with a decimal point
        press(
            &mut menu,
            &[
                Key::Down,
                Key::Char('2'),
                Key::Char('.'),
                Key::Char('5'),
                Key::Enter,
            ],
        );
        assert_eq!(menu.settings.size, Some(5 * GIB / 2));

        // too much memory is refused with a message, and nothing changes
        press(&mut menu, &[Key::Char('5'), Key::Char('0'), Key::Enter]);
        assert_eq!(menu.settings.size, Some(5 * GIB / 2));
        // the largest size offered is rounded down, so typing it always fits
        assert_eq!(menu.lines().2, "Type 0.4 to 10.6GB");

        // Esc cancels typing without quitting, Backspace edits
        press(
            &mut menu,
            &[Key::Char('7'), Key::Backspace, Key::Char('3'), Key::Esc],
        );
        assert_eq!(menu.settings.size, Some(5 * GIB / 2));
        assert_eq!(menu.handle(Key::Esc), Some(Outcome::Quit));
    }

    #[test]
    fn typed_gpu_number_drops_a_size_that_does_not_fit() {
        let devices = devices();
        let mut menu = Menu::new(
            &devices,
            Settings {
                size: Some(8 * GIB),
                ..defaults()
            },
        );
        press(&mut menu, &[Key::Char('2'), Key::Enter]);
        assert_eq!(menu.settings.device, 1);
        assert_eq!(menu.settings.size, None); // 8GB doesn't fit the 5GB of GPU 2
        press(&mut menu, &[Key::Char('7'), Key::Enter]);
        assert_eq!(menu.settings.device, 1);
        assert!(menu.lines().2.contains("no GPU 7"));
        assert_eq!(menu.handle(Key::Interrupt), Some(Outcome::Quit));
    }
}
