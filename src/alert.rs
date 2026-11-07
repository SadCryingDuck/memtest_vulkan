//! Drawing the user's attention to the console, like when the first error is found.

use std::io::{IsTerminal, Write};

/// Rings the console bell and, on Windows, flashes the console's taskbar button until the
/// window is brought to the front. Does nothing when the output isn't a console.
pub fn alert() {
    let mut out = std::io::stdout();
    if !out.is_terminal() {
        return;
    }
    let _ = out.write_all(b"\x07");
    let _ = out.flush();
    #[cfg(windows)]
    flash_console_window();
}

#[cfg(windows)]
fn flash_console_window() {
    use windows_sys::Win32::{
        System::Console::GetConsoleWindow,
        UI::WindowsAndMessaging::{FLASHW_TIMERNOFG, FLASHW_TRAY, FLASHWINFO, FlashWindowEx},
    };
    let window = unsafe { GetConsoleWindow() };
    if window.is_null() {
        return;
    }
    let info = FLASHWINFO {
        cbSize: size_of::<FLASHWINFO>() as u32,
        hwnd: window,
        dwFlags: FLASHW_TRAY | FLASHW_TIMERNOFG,
        uCount: 0,
        dwTimeout: 0,
    };
    unsafe { FlashWindowEx(&info) };
}
