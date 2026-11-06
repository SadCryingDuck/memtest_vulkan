//! Process status, exit codes and Ctrl+C / termination handling.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering::SeqCst};

static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);
static INTERRUPT_COUNT: AtomicU8 = AtomicU8::new(0);
static HANDLER_MODE: AtomicU8 = AtomicU8::new(HandlerMode::Immediate as u8);

/// Status bits, also used as the exit code of the worker process, read by the console process
pub mod app_status {
    pub const SIGNATURE: u8 = 0b0100_0000;
    pub const SIGNATURE_MASK: u8 = 0b1100_0000;
    pub const INITED_OK: u8 = 0b00_0001; //first test passes ran successfully
    pub const RUNTIME_ERRORS: u8 = 0b00_0010; //memory errors were found
    pub const RUNTIME_ABORT: u8 = 0b00_0100; //stopped by a runtime error or in an unexpected way
    pub const COMPLETED: u8 = 0b00_1000; //the requested test duration was completed
    pub const QUIT_JOB_REQUESTED: u8 = 0b01_0000; //console closing or termination: quit all jobs at once
    pub const WRITES_NOT_APPLIED: u8 = 0b10_0000; //test writes never reached memory: results invalid
}

/// What Ctrl+C does
#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum HandlerMode {
    /// exit at once
    Immediate = 0,
    /// only report the request, polled with close_requested()
    Graceful = 1,
    /// report the first request, exit at once on the second one
    GracefulThenImmediate = 2,
}

static APP_STATUS: AtomicU8 = AtomicU8::new(app_status::SIGNATURE);

pub fn raise_status_bit(bit: u8) {
    APP_STATUS.fetch_or(bit, SeqCst);
}

pub fn fetch_status() -> u8 {
    APP_STATUS.load(SeqCst) & !app_status::SIGNATURE_MASK
}

pub fn check_any_bits_set(value: u8, bits: u8) -> bool {
    (value & bits) != 0
}

/// Returns true once per Ctrl+C, or always after a request to quit all jobs
pub fn close_requested() -> bool {
    if check_any_bits_set(fetch_status(), app_status::QUIT_JOB_REQUESTED) {
        return true;
    }
    INTERRUPT_REQUESTED.swap(false, SeqCst)
}

/// Exits with the status bits as the exit code (the worker process protocol)
pub fn immediate_exit(raise_abort_flag: bool) -> ! {
    if raise_abort_flag {
        raise_status_bit(app_status::RUNTIME_ABORT)
    }
    exit_process(APP_STATUS.load(SeqCst) as u32);
}

fn report_interrupt_request(quit_job: bool) {
    if quit_job {
        raise_status_bit(app_status::QUIT_JOB_REQUESTED);
    }
    let count = INTERRUPT_COUNT.fetch_add(1, SeqCst).saturating_add(1);
    let mode = HANDLER_MODE.load(SeqCst);
    let exit_now = mode == HandlerMode::Immediate as u8
        || (mode == HandlerMode::GracefulThenImmediate as u8 && (count >= 2 || quit_job));
    if exit_now {
        //don't raise abort - interrupt request is an expected way to stop
        immediate_exit(false)
    }
    INTERRUPT_REQUESTED.swap(true, SeqCst);
}

// This function can't gaurantee that graceful handler would not be run since it maybe already running.
// But it would try to avoid graceful handlers
pub fn prefer_immediate_to_graceful() {
    HANDLER_MODE.store(HandlerMode::Immediate as u8, SeqCst);
}

pub fn setup_handler(mode: HandlerMode) {
    HANDLER_MODE.store(mode as u8, SeqCst);
    setup_handler_impl();
}

#[cfg(windows)]
pub fn setup_handler_impl() {
    use windows_sys::Win32::System::Console;
    unsafe extern "system" fn os_handler(ctrltype: u32) -> windows_sys::core::BOOL {
        let quit_job = ctrltype != Console::CTRL_C_EVENT && ctrltype != Console::CTRL_BREAK_EVENT;
        report_interrupt_request(quit_job);
        if quit_job {
            //don't return for events returning from which causes immediate termination
            std::thread::sleep(std::time::Duration::from_secs(25));
            immediate_exit(true);
        }
        1
    }
    unsafe {
        Console::SetConsoleCtrlHandler(Some(os_handler), 1);
    }
}

#[cfg(windows)]
pub fn exit_process(code: u32) -> ! {
    unsafe { windows_sys::Win32::System::Threading::ExitProcess(code) }
}

#[cfg(unix)]
pub fn setup_handler_impl() {
    use nix::sys::signal;
    extern "C" fn os_handler(
        sig: nix::libc::c_int,
        _: *mut nix::libc::siginfo_t,
        _: *mut core::ffi::c_void,
    ) {
        let quit_job = sig != nix::libc::SIGINT;
        report_interrupt_request(quit_job);
    }
    unsafe {
        let sig_action = signal::SigAction::new(
            signal::SigHandler::SigAction(os_handler),
            signal::SaFlags::empty(),
            signal::SigSet::empty(),
        );
        let _ = signal::sigaction(signal::SIGINT, &sig_action);
        let _ = signal::sigaction(signal::SIGTERM, &sig_action);
        let _ = signal::sigaction(signal::SIGHUP, &sig_action);
        let _ = signal::sigaction(signal::SIGQUIT, &sig_action);
    }
}

#[cfg(unix)]
pub fn exit_process(code: u32) -> ! {
    unsafe { nix::libc::_exit(code as i32) }
}
