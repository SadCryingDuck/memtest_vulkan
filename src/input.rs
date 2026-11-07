//! "Press any key" at the end and the result color. The start screen is in start_screen.rs.

use crossterm::{
    event::{self, Event, KeyEvent, KeyEventKind},
    queue,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal,
};
use std::io::{IsTerminal, Write};

/// Reads keys from the console in raw mode; restores the console when dropped
#[derive(Default)]
pub struct Reader {
    /// raw mode was enabled by this reader
    raw_mode: bool,
    /// a color was set, so it's reset when dropped
    colored: bool,
}

impl Reader {
    pub fn set_pass_fail_accent_color(&mut self, failed: bool) {
        if crate::close::close_requested() {
            //color methods not available while closing on windows
            return;
        }
        let mut out = std::io::stdout();
        if !out.is_terminal() {
            return; // no colors in redirected output
        }
        let color = if failed { Color::Red } else { Color::Green };
        if queue!(out, SetForegroundColor(color)).is_ok() {
            self.colored = true;
        }
        let _ = out.flush();
    }

    pub fn wait_any_key(&mut self, immediate_close_on_signal_during_wait: bool) {
        if crate::close::close_requested() {
            //interaction methods not available while closing on windows
            let seconds_wait = 3;
            let mut out = std::io::stdout();
            if write!(out, "Closing in {seconds_wait} seconds: ").is_err() {
                return;
            }
            for i in (1..=seconds_wait).rev() {
                if write!(out, "{i}... ").is_err() {
                    return;
                }
                if out.flush().is_err() {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            println!();
            return;
        }

        if !self.try_prepare_terminal() {
            //terminal input not available, don't perform wait.
            return;
        }
        if immediate_close_on_signal_during_wait {
            crate::close::prefer_immediate_to_graceful();
        }
        // ignore keys pressed before the question was shown
        while matches!(event::poll(std::time::Duration::ZERO), Ok(true)) {
            if event::read().is_err() {
                return;
            }
        }
        let mut out = std::io::stdout();
        if write!(out, "  press any key to continue...").is_err() || out.flush().is_err() {
            return;
        }
        loop {
            match event::read() {
                Err(_) => break,
                Ok(Event::Key(KeyEvent { kind, .. })) if kind != KeyEventKind::Release => break,
                Ok(_) => continue, // key releases, resize and other events
            }
        }
        self.restore_mode();
        let _ = queue!(out, ResetColor, Print("\n"));
        self.colored = false;
        let _ = out.flush();
    }

    /// Enables raw mode to read single keys, if the console is interactive
    fn try_prepare_terminal(&mut self) -> bool {
        if self.raw_mode {
            return true;
        }
        if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
            //terminal input not available, treate it as empty input
            return false;
        }
        self.raw_mode = terminal::enable_raw_mode().is_ok();
        self.raw_mode
    }

    fn restore_mode(&mut self) {
        if self.raw_mode {
            let _ = terminal::disable_raw_mode();
            self.raw_mode = false;
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        if crate::close::close_requested() {
            return; //don't touch console methods while closing
        }
        self.restore_mode();
        if self.colored {
            let mut out = std::io::stdout();
            let _ = queue!(out, ResetColor);
            let _ = out.flush();
        }
    }
}
