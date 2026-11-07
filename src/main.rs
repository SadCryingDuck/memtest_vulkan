//! memtest_vulkan: tests GPU video memory with Vulkan compute shaders.
//!
//! The console process lists the devices and lets the user pick one, then runs the test in a
//! worker process (the same executable started with --worker). If the driver fails while
//! allocating, the worker exits and the console retries with less memory.

mod cli;
mod close;
mod devices;
mod erupt_vendored_utils_loading;
mod input;
mod iobuf;
mod output;
mod ram;
mod shader;
mod tester;
mod vk_util;

pub use erupt::{CustomEntryLoader, LoaderError}; // republish for vendored part

use devices::{LoadedDevices, NamedComputeDevice};
use shader::{ALLOCATION_TRY_STEP, GB, MIN_WANTED_ALLOCATION};
use std::{
    ffi::OsString,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicBool, AtomicI64, AtomicIsize, Ordering::SeqCst},
    time,
};

/// Exit codes of the console process, documented in --help
mod exit_code {
    pub const PASSED: u32 = 0;
    pub const ERRORS_FOUND: u32 = 1;
    pub const WRITES_NOT_APPLIED: u32 = 2;
    pub const NOT_RUN: u32 = 3;
    pub const INCOMPLETE: u32 = 4;
}

#[derive(Default)]
pub struct ProcessEnv {
    argv0: Option<OsString>,
    device_label: AtomicIsize, //0 = autoselect, >0 index as displayed&sorted by memtest_vulkan, <0 -index as reported by device enumeration
    max_test_bytes: AtomicI64,
    verbose_requested: AtomicBool,
    /// This is the console process: shows the devices and runs the test in a worker process
    pub interactive: bool,
    pub options: cli::Options,
    /// Absolute path of the log file, if logging
    pub log_path: Option<PathBuf>,
}

impl ProcessEnv {
    fn effective_index_in_loaded_devices(&self) -> usize {
        // converts internal represntation to always-nonnegative index
        // 0 or nagative value means that selection is done at the moment of filling loaded_devices
        // so report selected index if positive and 0 otherwise
        (self.device_label.load(SeqCst) - 1).try_into().unwrap_or(0)
    }

    pub fn requested_index_in_device_iteration(&self) -> Option<usize> {
        // convert negative value to an index, zero or positive values convert to None
        (-1 - self.device_label.load(SeqCst)).try_into().ok()
    }

    pub fn verbose(&self) -> bool {
        self.verbose_requested.load(SeqCst)
    }

    fn make_verbose(&self) {
        self.verbose_requested.store(true, SeqCst)
    }

    pub fn max_test_bytes(&self) -> i64 {
        self.max_test_bytes.load(SeqCst)
    }

    fn mem_budget_estimated(&self) -> bool {
        self.max_test_bytes() > 0
    }

    fn set_mem_budget_limit(&self, new_limit: i64) {
        if self.mem_budget_estimated() {
            self.max_test_bytes.fetch_min(new_limit, SeqCst);
        } else {
            self.max_test_bytes.store(new_limit, SeqCst);
        }
    }

    fn wait_for_key(&self) -> bool {
        self.interactive && !self.options.batch
    }
}

fn prompt_for_label(verbose: bool) -> Option<isize> {
    let mut device_test_index = Some(0isize);
    let prompt_start = time::Instant::now();
    let mut prompt_duration: Option<time::Duration> = Some(time::Duration::from_secs(10));

    let mut input_reader = input::Reader::default();
    let no_timer_prompt = String::from(
        "                                             Enter the number of the GPU to test:",
    );
    loop {
        let mut prompt = &no_timer_prompt;
        let formatted_prompt: String;
        if let Some(effective_duration) = prompt_duration {
            if effective_duration < prompt_start.elapsed() {
                println!();
                println!("    ...testing GPU 1");
                break;
            } else {
                let duration_left = effective_duration - prompt_start.elapsed();
                formatted_prompt = std::format!(
                    "(GPU 1 will be tested in {} seconds)   Enter the number of the GPU to test:",
                    duration_left.as_secs()
                );
                prompt = &formatted_prompt;
            }
        }
        match input_reader.input_digit_step(prompt, &time::Duration::from_millis(250)) {
            Ok(input::ReaderEvent::Edited) => prompt_duration = None,
            Ok(input::ReaderEvent::Canceled) => {
                println!();
                device_test_index = None;
                break;
            }
            Ok(input::ReaderEvent::Completed) => {
                match input_reader.current_input.len()
                    == input_reader.current_input.matches('0').count()
                {
                    true => {
                        //empty or all zeroes
                        println!();
                        println!("    ...testing GPU 1");
                    }
                    false => match input_reader.current_input.parse::<isize>() {
                        Ok(parsed_idx) => {
                            device_test_index = Some(parsed_idx);
                            println!();
                        }
                        Err(_) => {
                            input_reader.current_input.clear();
                            continue;
                        }
                    },
                }
                break;
            }
            Ok(input::ReaderEvent::Timeout) => {} //just redraw prompt
            Err(e) => {
                //if input machinery doesnt work treat it as nothing was enetred
                if verbose {
                    println!("Input machinery failure: {e}");
                }
                break;
            }
        }
    }
    drop(input_reader);
    device_test_index
}

/// Starts the worker process testing `device`. Returns its exit code, or None if it couldn't start
fn run_worker<Writer: std::io::Write>(
    device: &NamedComputeDevice,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<Option<u8>, Box<dyn std::error::Error>> {
    let Some(argv0) = &env.argv0 else {
        return Ok(None);
    };
    let options = &env.options;
    let mut command = std::process::Command::new(argv0);
    command
        .arg("--worker")
        .arg("--device-enumeration-index")
        .arg(device.index_in_device_iteration.to_string())
        .arg("--size")
        .arg(env.max_test_bytes().to_string())
        .arg("--duration")
        .arg(options.duration.to_arg());
    if let Some(window_mb) = options.window_mb {
        command.arg("--window-mb").arg(window_mb.to_string());
    }
    match &env.log_path {
        Some(log_path) => command.arg("--log").arg(log_path),
        None => command.arg("--no-log"),
    };
    if env.verbose() {
        command.arg("--verbose");
    }
    let _ = log_dupler.flush();
    let Ok(mut child) = command.spawn() else {
        return Ok(None);
    };
    if env.verbose() {
        let _ = writeln!(log_dupler, "Started worker process {}", child.id());
    }
    let wait_result = child.wait();
    let parent_close_requested = close::close_requested();
    let exit_status =
        wait_result.map_err(|e| format!("Problem waiting for the test process: {e}"))?;
    if env.verbose() {
        let _ = writeln!(
            log_dupler,
            "Worker exit status {exit_status}, Ctrl+C pressed: {parent_close_requested}"
        );
    }
    let code = exit_status
        .code()
        .ok_or("The test process was terminated without an exit code")?;
    let status = code as u8;
    if (status & close::app_status::SIGNATURE_MASK) != close::app_status::SIGNATURE {
        return Err(format!("The test process failed with exit code {code}").into());
    }
    let expected_end = parent_close_requested
        || close::check_any_bits_set(
            status,
            close::app_status::QUIT_JOB_REQUESTED | close::app_status::COMPLETED,
        )
        || status == (close::app_status::SIGNATURE | close::app_status::RUNTIME_ABORT);
    if !expected_end {
        let _ = writeln!(
            log_dupler,
            "The test process ended unexpectedly (exit code {code})"
        );
        return Ok(Some(status | close::app_status::RUNTIME_ABORT));
    }
    Ok(Some(status))
}

/// Tests the single device left in `loaded_devices`, in a worker process if possible
fn test_selected_device<Writer: std::io::Write>(
    loaded_devices: LoadedDevices,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<u8, Box<dyn std::error::Error>> {
    let [single_dev] = loaded_devices.devices.as_slice() else {
        env.make_verbose();
        return Err(
            "While asking selected device exactly 1 device need to present and already selected"
                .into(),
        );
    };
    if env.interactive {
        loop {
            match run_worker(single_dev, env, log_dupler)? {
                None => {
                    let _ = writeln!(log_dupler, "Couldn't start the test process!");
                    break;
                }
                // the worker exits like this when the driver fails during allocation
                Some(status)
                    if status
                        == (close::app_status::SIGNATURE | close::app_status::RUNTIME_ABORT) =>
                {
                    let tried_bytes = env.max_test_bytes();
                    let smaller_memory = tried_bytes - ALLOCATION_TRY_STEP;
                    if smaller_memory < MIN_WANTED_ALLOCATION {
                        let _ = writeln!(
                            log_dupler,
                            "The test failed even with {:.1}GB of memory",
                            tried_bytes as f32 / GB
                        );
                        break;
                    }
                    env.set_mem_budget_limit(smaller_memory);
                    let _ = writeln!(
                        log_dupler,
                        "Retrying with less memory: {:.1}GB",
                        smaller_memory as f32 / GB
                    );
                }
                Some(status) => {
                    return Ok(
                        status | (close::fetch_status() & close::app_status::QUIT_JOB_REQUESTED)
                    );
                }
            }
        }
        env.make_verbose();
        let _ = writeln!(
            log_dupler,
            "Falling back to testing in this process, with diagnostic output"
        );
    }
    tester::prepare_and_test_device(loaded_devices.instance(), single_dev, env, log_dupler);
}

/// Shows the devices, selects one and tests it
fn select_device_and_test<Writer: std::io::Write>(
    mut loaded_devices: LoadedDevices,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<u8, Box<dyn std::error::Error>> {
    if env.interactive {
        let _ = writeln!(log_dupler);
        for (i, desc) in loaded_devices.devices.iter().enumerate() {
            let _ = writeln!(log_dupler, "{}: {}", i + 1, desc.label);
        }
        if loaded_devices.devices.is_empty() {
            return Err("No GPU able to run Vulkan compute shaders was found. Check that the GPU driver is installed.".into());
        }
        if env.options.list {
            let _ = log_dupler.flush();
            close::exit_process(exit_code::PASSED);
        }
        if loaded_devices.devices.len() > 1 && env.options.device.is_none() {
            if env.options.batch {
                let _ = writeln!(
                    log_dupler,
                    "Testing GPU 1, use --device to choose another one"
                );
            } else if let Some(selected_label) = prompt_for_label(env.verbose()) {
                env.device_label.store(selected_label, SeqCst);
            } else {
                return Err("Test cancelled, no GPU selected".into());
            }
        }
        close::setup_handler(close::HandlerMode::Graceful); //for interactive environments setup handler only after input prompt was run
    }

    let index_from_1 = env.effective_index_in_loaded_devices() + 1;
    let available_len = loaded_devices.devices.len();
    let selected = match std::mem::take(&mut loaded_devices.devices)
        .into_iter()
        .nth(index_from_1 - 1)
    {
        None => {
            return Err(format!(
                "There is no GPU number {index_from_1}, found {available_len} GPU(s)"
            )
            .into());
        }
        Some(selected) => selected,
    };

    if env.verbose() {
        let _ = writeln!(log_dupler, "Loading memory info for selected device...");
    }
    if !env.mem_budget_estimated() || env.interactive {
        let estimate = selected.estimate_testable_memory(env, log_dupler);
        match env.options.size {
            Some(size) => {
                if estimate > 0 && size > estimate {
                    let _ = writeln!(
                        log_dupler,
                        "WARNING: --size {:.1}GB is more than the {:.1}GB of memory estimated to be free.\n\
                         Part of the test memory may be swapped, so that part wouldn't check the GPU memory.",
                        size as f32 / GB,
                        estimate as f32 / GB
                    );
                }
                env.set_mem_budget_limit(size);
            }
            None => env.set_mem_budget_limit(estimate),
        }
    }
    if !env.mem_budget_estimated() {
        return Err("Couldn't determine how much memory to test, use --size".into());
    }
    loaded_devices.devices = vec![selected];
    test_selected_device(loaded_devices, env, log_dupler)
}

fn init_running_env() -> ProcessEnv {
    let mut args = std::env::args_os();
    let argv0 = args.next();
    let options = match cli::parse(args) {
        Ok(cli::Command::Run(options)) => options,
        Ok(cli::Command::Help) => {
            println!(
                "memtest_vulkan v{} by GpuZelenograd, https://github.com/GpuZelenograd/memtest_vulkan\n",
                env!("CARGO_PKG_VERSION")
            );
            print!("{}", cli::HELP);
            close::exit_process(exit_code::PASSED);
        }
        Ok(cli::Command::Version) => {
            println!("memtest_vulkan v{}", env!("CARGO_PKG_VERSION"));
            close::exit_process(exit_code::PASSED);
        }
        Err(message) => {
            eprintln!("memtest_vulkan: {message}");
            close::exit_process(exit_code::NOT_RUN);
        }
    };
    let verbose_name = argv0
        .as_ref()
        .and_then(|argv0| {
            std::path::Path::new(argv0)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(|stem| stem.to_ascii_lowercase().contains("verbose"))
        })
        .unwrap_or_default();
    let log_path = match &options.log {
        cli::LogTarget::Default => Some(PathBuf::from("memtest_vulkan.log")),
        cli::LogTarget::File(path) => Some(path.clone()),
        cli::LogTarget::Disabled => None,
    }
    .map(|path| std::path::absolute(&path).unwrap_or(path));
    let process_env = ProcessEnv {
        verbose_requested: AtomicBool::new(
            cfg!(feature = "verbose") || options.verbose || verbose_name,
        ),
        argv0,
        interactive: !options.worker,
        log_path,
        ..Default::default()
    };
    if let Some(device) = options.device {
        process_env.device_label.store(device as isize, SeqCst);
    }
    if let Some(index) = options.device_enumeration_index {
        process_env.device_label.store(-1 - index as isize, SeqCst);
    }
    if options.worker
        && let Some(size) = options.size
    {
        process_env.set_mem_budget_limit(size);
    }
    let process_env = ProcessEnv {
        options,
        ..process_env
    };
    if process_env.interactive && !process_env.options.list {
        print!("https://github.com/GpuZelenograd/");
        let _ = std::io::stdout().flush();
        let mut color_setter = input::Reader::default();
        color_setter.set_pass_fail_accent_color(false);
        println!(
            "memtest_vulkan v{} by GpuZelenograd",
            env!("CARGO_PKG_VERSION")
        );
        drop(color_setter);
        println!("Press Ctrl+C to stop testing early. Options: memtest_vulkan --help");
    }
    process_env
}

pub fn display_this_process_result(
    maybe_err: Option<Box<dyn std::error::Error>>,
    env: &ProcessEnv,
) -> ! {
    if let Some(e) = maybe_err {
        println!("Runtime error: {e}");
        close::raise_status_bit(close::app_status::RUNTIME_ABORT);
    }
    display_testing_outcome(close::fetch_status(), env)
}

/// The verdict for the final status bits: exit code, message, and whether it's a failure
/// (for the accent color; None for no color)
fn outcome(status: u8) -> (u32, String, Option<bool>) {
    use close::{app_status::*, check_any_bits_set as has};
    if has(status, WRITES_NOT_APPLIED) {
        let mut message = "memtest_vulkan: RESULTS INVALID - test writes didn't reach the memory.\n\
             Most likely a GPU driver or memtest_vulkan problem rather than faulty VRAM - see the log."
            .to_owned();
        if has(status, RUNTIME_ERRORS) {
            message += "\nOther memory errors were found as well.";
        }
        (exit_code::WRITES_NOT_APPLIED, message, Some(true))
    } else if has(status, RUNTIME_ERRORS) {
        (
            exit_code::ERRORS_FOUND,
            "memtest_vulkan: memory/gpu ERRORS FOUND, test FAILED.".to_owned(),
            Some(true),
        )
    } else if has(status, RUNTIME_ABORT) {
        let message = if has(status, INITED_OK) {
            "memtest_vulkan: no errors found, but the test stopped because of a runtime error."
        } else {
            "memtest_vulkan: the test couldn't run because of a runtime error."
        };
        (exit_code::NOT_RUN, message.to_owned(), Some(true))
    } else if has(status, COMPLETED) {
        (
            exit_code::PASSED,
            "memtest_vulkan: no errors, test PASSED.".to_owned(),
            Some(false),
        )
    } else {
        (
            exit_code::INCOMPLETE,
            "memtest_vulkan: no errors so far, but the test was stopped early (incomplete)."
                .to_owned(),
            None,
        )
    }
}

fn display_testing_outcome(status: u8, env: &ProcessEnv) -> ! {
    if !env.interactive {
        // the worker reports the status bits to the console process
        close::immediate_exit(false);
    }
    println!();
    let mut key_reader = input::Reader::default();
    if close::check_any_bits_set(status, close::app_status::QUIT_JOB_REQUESTED) {
        //propagate closing flag to this process, so no risky-during close functions would be used
        close::raise_status_bit(close::app_status::QUIT_JOB_REQUESTED);
    }
    let (code, message, failed) = outcome(status);
    if let Some(failed) = failed {
        key_reader.set_pass_fail_accent_color(failed);
    }
    println!("{message}");
    if env.wait_for_key() {
        key_reader.wait_any_key(true);
    }
    drop(key_reader); //restore terminal state before exiting
    let _ = std::io::stdout().flush();
    close::exit_process(code)
}

fn display_result<Writer: std::io::Write>(
    result: Result<u8, Box<dyn std::error::Error>>,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> ! {
    match result {
        Ok(status) => {
            let _ = log_dupler.flush();
            display_testing_outcome(status, env)
        }
        Err(e) => {
            if env.interactive || env.verbose() {
                println!();
                //make more readable message about libvulkan
                let mut error_fmt = format!("{e}");
                let _ = writeln!(
                    log_dupler,
                    "memtest_vulkan couldn't start testing:\n{error_fmt}"
                );
                let mut source = e.source();
                while let Some(inner) = source {
                    let inner_fmt = format!("{inner}");
                    //exclude duplicate/contained errors
                    if !error_fmt.contains(&inner_fmt) {
                        let _ = writeln!(log_dupler, "    due to: {inner_fmt}");
                        error_fmt = inner_fmt;
                    }
                    source = inner.source();
                }
                let _ = log_dupler.flush();
                if env.wait_for_key() {
                    let mut key_reader = input::Reader::default();
                    key_reader.wait_any_key(true);
                    drop(key_reader); //restore terminal state before exiting
                }
            }
            if !env.interactive {
                close::immediate_exit(true)
            }
            close::exit_process(exit_code::NOT_RUN)
        }
    }
}

fn main() {
    let env = init_running_env();
    if !env.interactive {
        // first Ctrl+C finishes the current pass and prints the summary, the second one quits at once
        close::setup_handler(close::HandlerMode::GracefulThenImmediate);
    } else if env.options.batch {
        close::setup_handler(close::HandlerMode::Graceful);
    }
    const MAX_LOG_SIZE: u64 = 50 * 1024 * 1024;
    let mut log_dupler = output::LogDupler::new(
        std::io::stdout(),
        env.log_path
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned()),
        MAX_LOG_SIZE,
        if env.interactive {
            "Tester console"
        } else {
            "Tester worker"
        },
    );
    if env.interactive
        && !env.options.list
        && let Some(log_path) = &env.log_path
    {
        let _ = writeln!(log_dupler, "Log file: {}", log_path.display());
    }
    let result = devices::list_devices_ordered_labeled_from_1(&env, &mut log_dupler)
        .and_then(|loaded_devices| select_device_and_test(loaded_devices, &env, &mut log_dupler));
    display_result(result, &env, &mut log_dupler);
}
