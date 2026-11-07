//! Allocating the test memory and running the write/read test passes.

use crate::{
    ProcessEnv,
    cli::{STANDARD_DURATION, TestDuration},
    close,
    devices::NamedComputeDevice,
    display_this_process_result,
    iobuf::{ErrorTotals, IOBuf, Pattern},
    output, ram,
    shader::{
        ALLOCATION_TRY_STEP, ELEMENT_SIZE, EMULATE_ALLOCATION_LIMIT_ENV, EMULATE_FAILURE_ENV,
        EMULATE_SKIPPED_WRITE_ENV, GB, MAP_FOR_GETTING_ADDRESS_SIZE, MAX_WINDOW_MB_ENV,
        MIN_WANTED_ALLOCATION, READ_SHADER, TEST_WINDOW_1D_MAX_GROUPS, TEST_WINDOW_MAX_SIZE,
        TEST_WINDOW_SIZE_GRANULARITY, VEC_SIZE, WG_SIZE, WRITE_MARKER,
    },
    text,
    vk_util::{self, MapErrRetryWithLowerMemory, MapErrStr},
};

use ash::vk;
use core::cmp::{max, min};
use std::{
    cell::Cell,
    env,
    error::Error,
    io::{IsTerminal, Write},
    mem,
    time::{Duration, Instant},
};

type BoxResult<T> = Result<T, Box<dyn Error>>;

/// A GPU pass not finishing in this time is treated as a GPU hang
const PASS_HANG_TIMEOUT: Duration = Duration::from_secs(60);
const FENCE_WAIT_SLICE_NS: u64 = 1_000_000_000;
/// Pause of the GPU load near the end of the test: checks memory left idle while the GPU is hot
const LOAD_PAUSE: Duration = Duration::from_secs(15);
const REPORT_INTERVAL: Duration = Duration::from_secs(30);
/// How often the progress line on the console is updated
const LIVE_UPDATE_INTERVAL: Duration = Duration::from_millis(250);
/// Write passes this many times slower than writes to a single window mean the test memory
/// doesn't fit in VRAM and the driver swaps it to system RAM
const SWAP_SLOWDOWN_FACTOR: f64 = 3.0;

/// Formats a duration as m:ss or h:mm:ss
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs >= 3600 {
        format!("{}:{:02}:{:02}", secs / 3600, secs / 60 % 60, secs % 60)
    } else {
        format!("{}:{:02}", secs / 60, secs % 60)
    }
}

fn format_test_duration(duration: TestDuration) -> String {
    match duration {
        TestDuration::Limited(duration) => format_duration(duration),
        TestDuration::Endless => "endless".to_owned(),
    }
}

struct ComputePipelines {
    read: vk::Pipeline,
    write: vk::Pipeline,
    emulate_write_bugs: vk::Pipeline,
}

/// One test window: a separate allocation and buffer, so each stays within the driver's
/// maxMemoryAllocationSize and maxBufferSize. A single huge allocation exceeded them on
/// drivers limiting them to ~4GB (like Intel Arc), which silently dropped part of the writes
/// and made untouched memory look like memory errors.
struct TestWindow {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// false once dropped from the test (its memory freed); indexes of the others stay the same,
    /// since the test pattern of a window depends on its index
    active: bool,
}

/// Mode of a read pass, shown in error reports
#[derive(Clone, Copy)]
enum ReadMode {
    /// first read after writing the window
    InitialRead,
    /// window 0 isn't rewritten during the test: its data is read again and again
    NextReRead,
    /// read again after the load pause, without rewriting
    AfterPause,
}

impl ReadMode {
    fn name(self) -> &'static str {
        match self {
            Self::InitialRead => "INITIAL_READ",
            Self::NextReRead => "NEXT_RE_READ",
            Self::AfterPause => "AFTER_PAUSE",
        }
    }
}

/// A failure emulated in a GPU pass, to check how failures are reported
#[derive(Clone, Copy, Debug, PartialEq)]
enum EmulatedFailure {
    DeviceLost,
    Hang,
    Panic,
}

/// Parses the EMULATE_FAILURE_ENV value: the failure and the 1-based pass it happens in
fn parse_emulated_failure(text: &str) -> Option<(EmulatedFailure, u32)> {
    let (kind, pass) = match text.trim().split_once(':') {
        Some((kind, pass)) => (kind, pass.trim().parse().ok().filter(|&pass| pass > 0)?),
        None => (text.trim(), 1),
    };
    let failure = match kind.trim() {
        "device-lost" => EmulatedFailure::DeviceLost,
        "hang" => EmulatedFailure::Hang,
        "panic" => EmulatedFailure::Panic,
        _ => return None,
    };
    Some((failure, pass))
}

/// Largest default test window. The driver may place part of the test memory in system RAM
/// when VRAM runs short; such windows are found by their speed and dropped, so smaller
/// windows lose less of the test when that happens.
const PREFERRED_WINDOW_MAX: i64 = 1024 * 1024 * 1024;
/// Smallest window size considered when choosing the size that covers the most memory
const PREFERRED_WINDOW_MIN: i64 = 512 * 1024 * 1024;

/// Chooses the window size and count: windows of 512MB to 1GB (at most `max_window_size`),
/// in TEST_WINDOW_SIZE_GRANULARITY steps, whichever covers the most of `total`, and at least
/// 2 windows (one re-read, one rewritten). None if `total` is too small for that.
fn plan_windows(total: i64, max_window_size: i64) -> Option<(i64, i64)> {
    let round_down = |size: i64| size - size % TEST_WINDOW_SIZE_GRANULARITY;
    let largest = round_down(min(max_window_size, PREFERRED_WINDOW_MAX));
    let smallest = round_down(min(largest, PREFERRED_WINDOW_MIN));
    let mut best: Option<(i64, i64)> = None;
    let mut size = largest;
    while size >= smallest && size >= TEST_WINDOW_SIZE_GRANULARITY {
        let count = total / size;
        // larger windows first: they win a tie
        if count >= 2 && best.is_none_or(|(s, c)| size * count > s * c) {
            best = Some((size, count));
        }
        size -= TEST_WINDOW_SIZE_GRANULARITY;
    }
    // too little memory for two windows of the preferred size: two smaller ones
    best.or_else(|| {
        let size = round_down(min(total / 2, largest));
        (size >= TEST_WINDOW_SIZE_GRANULARITY).then_some((size, 2))
    })
}

fn hang_error() -> Box<dyn Error> {
    format!(
        "the GPU didn't finish a test pass within {} seconds, it seems to have hung",
        PASS_HANG_TIMEOUT.as_secs()
    )
    .into()
}

/// GPU objects used to run the test passes
struct Tester<'a> {
    device: &'a ash::Device,
    env: &'a ProcessEnv,
    queue: vk::Queue,
    cmd_buf: vk::CommandBuffer,
    desc_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipelines: ComputePipelines,
    fence: vk::Fence,
    io_buffer: vk::Buffer,
    io_memory: vk::DeviceMemory,
    mapped: *mut IOBuf,
    test_mem_index: u32,
    windows: Vec<TestWindow>,
    window_size: i64,
    /// A pass didn't finish: the GPU may still be using the memory, and waiting for it may
    /// never end
    gpu_hung: Cell<bool>,
    /// GPU passes run so far
    passes: Cell<u32>,
    emulated_failure: Option<(EmulatedFailure, u32)>,
}

impl Drop for Tester<'_> {
    fn drop(&mut self) {
        if self.gpu_hung.get() {
            // device_wait_idle has no timeout and would block forever on a hung GPU, which hides
            // the hang message; freeing memory the GPU still uses is invalid. The process is
            // about to exit, which frees everything anyway.
            return;
        }
        unsafe {
            let _ = self.device.device_wait_idle();
            self.free_windows();
            self.device.destroy_buffer(self.io_buffer, None);
            self.device.unmap_memory(self.io_memory);
            self.device.free_memory(self.io_memory, None);
            self.device.destroy_fence(self.fence, None);
        }
    }
}

impl<'a> Tester<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new<Writer: std::io::Write>(
        device: &'a ash::Device,
        env: &'a ProcessEnv,
        queue: vk::Queue,
        cmd_buf: vk::CommandBuffer,
        desc_sets: Vec<vk::DescriptorSet>,
        pipeline_layout: vk::PipelineLayout,
        pipelines: ComputePipelines,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        log_dupler: &mut output::LogDupler<Writer>,
    ) -> BoxResult<Self> {
        let io_buffer_create_info = vk::BufferCreateInfo::default()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size(mem::size_of::<IOBuf>() as vk::DeviceSize);
        let io_buffer =
            unsafe { device.create_buffer(&io_buffer_create_info, None) }.err_as_str()?;
        let io_mem_reqs = unsafe { device.get_buffer_memory_requirements(io_buffer) };
        let mut io_mem_indices = Vec::new();
        for i in 0..memory_props.memory_type_count {
            //test buffer comptibility flags expressed as bitmask
            let suitable = (io_mem_reqs.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory_props.memory_types[i as usize];
            if env.verbose() && !memory_type.property_flags.is_empty() {
                let _ = writeln!(log_dupler, "{:2} {:?} ", i, memory_type);
            }
            if suitable
                && memory_type.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            {
                io_mem_indices.push(i);
            }
        }
        // sorting by a flag value allows selection of index with the minimum count of new unknown flags
        let io_mem_index = io_mem_indices
            .into_iter()
            .min_by_key(|i| memory_props.memory_types[*i as usize].property_flags)
            .ok_or("This device lacks support for DEVICE_LOCAL+HOST_COHERENT memory type.")?;
        if env.verbose() {
            let _ = writeln!(
                log_dupler,
                "CoherentIO memory          type {} inside heap {:?}",
                io_mem_index,
                memory_props.memory_heaps
                    [memory_props.memory_types[io_mem_index as usize].heap_index as usize]
            );
        }
        let io_memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(io_mem_reqs.size)
            .memory_type_index(io_mem_index);
        let io_memory =
            unsafe { device.allocate_memory(&io_memory_allocate_info, None) }.err_as_str()?;
        let mapped: *mut IOBuf = unsafe {
            device.map_memory(io_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::default())
        }
        .err_as_str()?
        .cast::<IOBuf>();
        unsafe { device.bind_buffer_memory(io_buffer, io_memory, 0) }
            .err_as_str_context("bind_buffer_memory")?;
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(desc_sets[0])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(io_buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)])],
                &[],
            );
        }

        // memory types usable for test buffers, found with a small buffer of the same kind
        let probe_buffer =
            unsafe { device.create_buffer(&test_buffer_create_info(MIN_WANTED_ALLOCATION), None) }
                .err_as_str()?;
        let test_mem_reqs = unsafe { device.get_buffer_memory_requirements(probe_buffer) };
        unsafe { device.destroy_buffer(probe_buffer, None) };
        let test_mem_index = (0..memory_props.memory_type_count)
            .filter(|mem_index| {
                //test buffer comptibility flags expressed as bitmask
                let memory_type = memory_props.memory_types[*mem_index as usize];
                let suitable = (test_mem_reqs.memory_type_bits & (1 << mem_index)) != 0;
                suitable
                    && memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .max_by_key(|mem_index| {
                let memory_type = memory_props.memory_types[*mem_index as usize];
                let heap_size = memory_props.memory_heaps[memory_type.heap_index as usize].size;
                // Among greatest heap_size select index with HOST_VISIBLE flag if it is available,
                // but with the minimum count of other flags and minimal index
                (
                    heap_size,
                    memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                    std::cmp::Reverse(memory_type.property_flags),
                    std::cmp::Reverse(*mem_index),
                )
            })
            .ok_or("DEVICE_LOCAL test memory type not available")?;

        let fence =
            unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.err_as_str()?;
        Ok(Self {
            device,
            env,
            queue,
            cmd_buf,
            desc_sets,
            pipeline_layout,
            pipelines,
            fence,
            io_buffer,
            io_memory,
            mapped,
            test_mem_index,
            windows: Vec::new(),
            window_size: 0,
            gpu_hung: Cell::new(false),
            passes: Cell::new(0),
            emulated_failure: env::var(EMULATE_FAILURE_ENV)
                .ok()
                .and_then(|text| parse_emulated_failure(&text)),
        })
    }

    /// Count of windows still tested
    fn window_count(&self) -> i64 {
        self.windows.iter().filter(|window| window.active).count() as i64
    }

    fn is_active(&self, window_idx: usize) -> bool {
        self.windows[window_idx].active
    }

    /// Frees a window and stops testing it
    fn drop_window(&mut self, window_idx: usize) {
        let window = &mut self.windows[window_idx];
        if window.active {
            window.active = false;
            unsafe {
                self.device.destroy_buffer(window.buffer, None);
                self.device.free_memory(window.memory, None);
            }
        }
    }

    fn tested_bytes(&self) -> i64 {
        self.window_size * self.window_count()
    }

    fn free_windows(&mut self) {
        for window in self.windows.drain(..).filter(|window| window.active) {
            unsafe {
                self.device.destroy_buffer(window.buffer, None);
                self.device.free_memory(window.memory, None);
            }
        }
    }

    /// Splits up to `total` bytes into windows (see `plan_windows`) and allocates them
    fn allocate_windows(&mut self, total: i64, max_window_size: i64) -> BoxResult<()> {
        self.free_windows();
        // allow emulating running out of memory, for testing the retry with less memory
        if let Some(limit) = env::var(EMULATE_ALLOCATION_LIMIT_ENV)
            .ok()
            .and_then(|text| crate::cli::parse_size(&text).ok())
            && total > limit
        {
            Err::<(), _>(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
                .err_retry_with_lower_memory("allocate_memory (emulated limit)")?;
        }
        let Some((size, count)) = plan_windows(total, max_window_size) else {
            return Err("test memory is too small".into());
        };
        self.window_size = size;
        for _ in 0..count {
            let buffer = unsafe {
                self.device
                    .create_buffer(&test_buffer_create_info(self.window_size), None)
            }
            .err_retry_with_lower_memory("create_buffer")?;
            let buffer_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
            let allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(buffer_reqs.size)
                .memory_type_index(self.test_mem_index);
            let memory = match unsafe { self.device.allocate_memory(&allocate_info, None) }
                .err_retry_with_lower_memory("allocate_memory")
            {
                Ok(memory) => memory,
                Err(e) => {
                    unsafe { self.device.destroy_buffer(buffer, None) };
                    return Err(e);
                }
            };
            self.windows.push(TestWindow {
                buffer,
                memory,
                active: true,
            });
            unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }
                .err_retry_with_lower_memory("bind_buffer_memory")?;
        }
        Ok(())
    }

    fn io_write(&self, io: IOBuf) {
        unsafe { std::ptr::write(self.mapped, io) }
    }

    fn io_read(&self) -> IOBuf {
        unsafe { std::ptr::read(self.mapped) }
    }

    /// Runs one GPU pass over a window and waits for it.
    /// fill_marker: first fill the window with WRITE_MARKER (used before its initial write).
    /// pipeline: None records no dispatch (used by the skipped-write emulation).
    fn run_pass(
        &self,
        window_idx: usize,
        fill_marker: bool,
        pipeline: Option<vk::Pipeline>,
    ) -> BoxResult<Duration> {
        let device = self.device;
        let buffer = self.windows[window_idx].buffer;
        let test_element_count = (self.window_size / ELEMENT_SIZE) as u32;
        let start = Instant::now();
        let pass = self.passes.get() + 1;
        self.passes.set(pass);
        match self.emulated_failure {
            Some((EmulatedFailure::DeviceLost, at)) if at == pass => {
                Err::<(), _>(vk::Result::ERROR_DEVICE_LOST)
                    .err_retry_with_lower_memory("queue_submit (emulated)")?;
            }
            Some((EmulatedFailure::Hang, at)) if at == pass => {
                self.gpu_hung.set(true);
                return Err(hang_error());
            }
            Some((EmulatedFailure::Panic, at)) if at == pass => {
                panic!("emulated panic in GPU pass {pass}");
            }
            _ => {}
        }
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.desc_sets[0])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(0)
                        .range(self.window_size as u64)])],
                &[],
            );
            let cmd_buf = self.cmd_buf;
            device
                .begin_command_buffer(cmd_buf, &vk::CommandBufferBeginInfo::default())
                .err_retry_with_lower_memory("begin_command_buffer")?;
            if fill_marker {
                device.cmd_fill_buffer(cmd_buf, buffer, 0, vk::WHOLE_SIZE, WRITE_MARKER);
                // the fill must complete before the shader writes/reads the window
                device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )],
                    &[],
                    &[],
                );
            }
            if let Some(pipeline) = pipeline {
                device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &self.desc_sets,
                    &[],
                );
                device.cmd_dispatch(
                    cmd_buf,
                    (TEST_WINDOW_1D_MAX_GROUPS / WG_SIZE) as u32,
                    test_element_count / VEC_SIZE as u32 / TEST_WINDOW_1D_MAX_GROUPS as u32,
                    1,
                );
            }
            // Make shader writes to the IO buffer visible to the host:
            // waiting on the fence alone doesn't guarantee host visibility per the Vulkan spec,
            // and a stale IO buffer would be reported as a memory error.
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
            device
                .end_command_buffer(cmd_buf)
                .err_retry_with_lower_memory("end_command_buffer")?;
            let cmd_bufs = [cmd_buf];
            let submit_info = &[vk::SubmitInfo::default().command_buffers(&cmd_bufs)];
            device
                .queue_submit(self.queue, submit_info, self.fence)
                .err_retry_with_lower_memory("queue_submit")?;
            // wait in slices, so a GPU hang gets reported instead of waiting forever
            loop {
                // ash reports a wait that timed out as Err(TIMEOUT): that's not a failure,
                // just the end of one wait slice
                match device.wait_for_fences(&[self.fence], true, FENCE_WAIT_SLICE_NS) {
                    Err(vk::Result::TIMEOUT) => {}
                    waited => {
                        waited.err_retry_with_lower_memory("wait_for_fences")?;
                        break;
                    }
                }
                if start.elapsed() > PASS_HANG_TIMEOUT {
                    self.gpu_hung.set(true);
                    return Err(hang_error());
                }
            }
            device
                .reset_fences(&[self.fence])
                .err_retry_with_lower_memory("reset_fences")?;
        }
        Ok(start.elapsed())
    }

    /// Writes a test pattern to a window
    fn write_window(
        &self,
        window_idx: usize,
        pattern: Pattern,
        fill_marker: bool,
        pipeline: Option<vk::Pipeline>,
    ) -> BoxResult<Duration> {
        let mut io = IOBuf::for_initial_iteration();
        io.set_pattern(pattern);
        self.io_write(io);
        self.run_pass(window_idx, fill_marker, pipeline)
    }

    /// Checks a window against the pattern described by `io_in`, returning the findings
    fn read_window(&self, window_idx: usize, io_in: IOBuf) -> BoxResult<(IOBuf, Duration)> {
        self.io_write(io_in);
        let elapsed = self.run_pass(window_idx, false, Some(self.pipelines.read))?;
        let io_out = self.io_read();
        io_out.check_vec_first()?;
        Ok((io_out, elapsed))
    }

    /// Logs the errors found by a read pass, adding them to the totals.
    /// Returns true if there were any.
    fn report_errors<Writer: std::io::Write>(
        &self,
        log_dupler: &mut output::LogDupler<Writer>,
        totals: &mut ErrorTotals,
        io_out: &IOBuf,
        window_idx: usize,
        mode: ReadMode,
    ) -> BoxResult<bool> {
        let test_offset = self.window_size * window_idx as i64;
        let Some((error_range, total_errors)) = io_out.get_error_addresses_and_count(test_offset)
        else {
            return Ok(false);
        };
        if total_errors == 0 {
            // The shader flagged an error but counted no wrong values: the test's own results
            // (in GPU memory too) can't be trusted, so the test can't go on
            writeln!(log_dupler, "{io_out}")?;
            return Err(format!(
                "the GPU flagged an error in window {} but reported no wrong values, so the \
                 test's own results seem corrupted",
                window_idx + 1
            )
            .into());
        }
        let unwritten = totals.add(io_out);
        let unwritten_words = unwritten.as_ref().map_or(0, |evidence| evidence.words());
        if unwritten.is_some() {
            close::raise_status_bit(close::app_status::WRITES_NOT_APPLIED);
        }
        if total_errors > unwritten_words {
            // wrong values that can't be explained by missing writes
            close::raise_status_bit(close::app_status::RUNTIME_ERRORS);
        }
        let test_elems = self.window_size / ELEMENT_SIZE;
        writeln!(
            log_dupler,
            "Error found. Mode {}, total errors 0x{:X} out of 0x{:X} ({:2.8}%)\nErrors address range: {:?}  iteration:{}\n{}",
            mode.name(),
            total_errors,
            test_elems,
            total_errors as f64 / test_elems as f64 * 100.0f64,
            error_range,
            io_out.iter,
            io_out
        )?;
        if let Some(evidence) = unwritten {
            let message = format!(
                "0x{unwritten_words:X} of these wrong values {}. The test's writes never reached \
                 this memory. This is most likely a GPU driver or memtest_vulkan problem rather than \
                 faulty VRAM. Retry with --window-mb 512 or another driver version; if it persists, \
                 the GPU itself may be faulty.",
                evidence.reason()
            );
            writeln!(
                log_dupler,
                "{}",
                text::wrap("WRITES NOT APPLIED: ", &message)
            )?;
        } else {
            let mut this_pass = ErrorTotals::default();
            this_pass.add(io_out);
            if let Some(cause) = this_pass.likely_cause() {
                writeln!(log_dupler, "{}", text::wrap("Likely cause: ", &cause))?;
            }
        }
        Ok(true)
    }
}

fn test_buffer_create_info(size: i64) -> vk::BufferCreateInfo<'static> {
    vk::BufferCreateInfo::default()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        // TRANSFER_DST: windows are filled with WRITE_MARKER before their first write
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
        .size(size as u64)
}

/// Bytes moved and time spent, for speed reports
#[derive(Default, Clone, Copy)]
struct Throughput {
    written: i64,
    write_time: Duration,
    read: i64,
    read_time: Duration,
}

impl Throughput {
    fn write_gbps(&self) -> f64 {
        gbps(self.written, self.write_time)
    }
    fn read_gbps(&self) -> f64 {
        gbps(self.read, self.read_time)
    }
    fn add(&mut self, other: &Throughput) {
        self.written += other.written;
        self.write_time += other.write_time;
        self.read += other.read;
        self.read_time += other.read_time;
    }
}

fn gbps(bytes: i64, time: Duration) -> f64 {
    let secs = time.as_secs_f64();
    if secs > 0.0001 {
        bytes as f64 / GB as f64 / secs
    } else {
        0.0
    }
}

/// What the main loop found, for the summary
struct RunReport {
    elapsed: Duration,
    iterations: u32,
    total: Throughput,
    totals: ErrorTotals,
    last_error_at: Option<Duration>,
    swap_warning: Option<String>,
    /// windows dropped because the driver placed them in system RAM
    skipped_windows: Vec<usize>,
    /// tested at least the standard 5 minutes, or the shorter duration asked for
    completed: bool,
    stopped_by_user: bool,
    stopped_on_error: bool,
}

/// Sets up the device and runs the test on it; never returns
pub fn prepare_and_test_device<Writer: std::io::Write>(
    instance: &ash::Instance,
    selected: &NamedComputeDevice,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> ! {
    let queue_create_info = vec![
        vk::DeviceQueueCreateInfo::default()
            .queue_family_index(selected.queue_family_index)
            .queue_priorities(&[1.0]),
    ];

    let device_create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_info);

    let device =
        unsafe { instance.create_device(selected.physical_device, &device_create_info, None) }
            .unwrap_or_display(env);
    let queue = unsafe { device.get_device_queue(selected.queue_family_index, 0) };

    let cmd_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(selected.queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cmd_pool =
        unsafe { device.create_command_pool(&cmd_pool_info, None) }.unwrap_or_display(env);

    let cmd_buf_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY);
    let cmd_bufs = unsafe { device.allocate_command_buffers(&cmd_buf_info) }.unwrap_or_display(env);

    let desc_pool_sizes = &[vk::DescriptorPoolSize::default()
        .descriptor_count(2)
        .ty(vk::DescriptorType::STORAGE_BUFFER)];
    let desc_pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(desc_pool_sizes)
        .max_sets(1);
    let desc_pool =
        unsafe { device.create_descriptor_pool(&desc_pool_info, None) }.unwrap_or_display(env);

    let desc_layout_bindings = &[
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let desc_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(desc_layout_bindings);
    let desc_layouts = [
        unsafe { device.create_descriptor_set_layout(&desc_layout_info, None) }
            .unwrap_or_display(env),
    ];

    let desc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(desc_pool)
        .set_layouts(&desc_layouts);
    let desc_sets = unsafe { device.allocate_descriptor_sets(&desc_info) }.unwrap_or_display(env);

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&desc_layouts);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .unwrap_or_display(env);

    let spv_code = Vec::from(READ_SHADER);
    let create_info = vk::ShaderModuleCreateInfo::default().code(&spv_code);
    let shader_mod =
        unsafe { device.create_shader_module(&create_info, None) }.unwrap_or_display(env);

    let pipeline_infos = [c"read", c"write", c"emulate_write_bugs"].map(|name| {
        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_mod)
            .name(name);
        vk::ComputePipelineCreateInfo::default()
            .layout(pipeline_layout)
            .stage(shader_stage)
    });

    // on failure ash returns the pipelines that were created too: only the error matters here
    let pipelines = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
    }
    .map_err(|(_, error)| error)
    .unwrap_or_display(env);
    let pipelines = ComputePipelines {
        read: pipelines[0],
        write: pipelines[1],
        emulate_write_bugs: pipelines[2],
    };

    let result = Tester::new(
        &device,
        env,
        queue,
        cmd_bufs[0],
        desc_sets,
        pipeline_layout,
        pipelines,
        &selected.memory_props,
        log_dupler,
    )
    .and_then(|mut tester| test_device(&mut tester, selected, log_dupler));
    if let Err(e) = result {
        let _ = log_dupler.flush();
        display_this_process_result(Some(e), env)
    }
    let _ = log_dupler.flush();
    display_this_process_result(None, env)
}

/// Largest window size the device limits (and the user) allow
fn max_window_size<Writer: std::io::Write>(
    selected: &NamedComputeDevice,
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> BoxResult<i64> {
    // Every window is a separate allocation and buffer, bound as one storage buffer descriptor.
    // So a window must fit in maxStorageBufferRange, maxMemoryAllocationSize and maxBufferSize.
    // Exceeding them is invalid usage, and on some drivers it silently drops writes,
    // which then shows up as false memory errors.
    let storage_range_limit = selected.physical_props.limits.max_storage_buffer_range as i64;
    let size_limits = selected.size_limits;
    // Manual cap of the window size, for diagnosing drivers that mishandle big buffers
    // (if errors disappear with smaller windows, the problem isn't the VRAM).
    let window_mb = env.options.window_mb.or_else(|| {
        env::var(MAX_WINDOW_MB_ENV)
            .ok()
            .and_then(|s| s.trim().parse::<i64>().ok())
            .filter(|&mb| mb > 0)
    });
    let window_size_override =
        window_mb.map(|mb| max(mb.saturating_mul(1024 * 1024), TEST_WINDOW_SIZE_GRANULARITY));
    if let Some(override_size) = window_size_override {
        writeln!(
            log_dupler,
            "Test window size limited to {}MB",
            (override_size - override_size % TEST_WINDOW_SIZE_GRANULARITY) / (1024 * 1024)
        )?;
    }
    let max_window_size = [
        Some(storage_range_limit),
        size_limits.max_memory_allocation_size,
        size_limits.max_buffer_size,
        window_size_override,
    ]
    .into_iter()
    .flatten()
    .map(|limit| limit - limit % TEST_WINDOW_SIZE_GRANULARITY)
    .fold(TEST_WINDOW_MAX_SIZE, min);
    if max_window_size < TEST_WINDOW_SIZE_GRANULARITY {
        return Err(format!(
            "device limits (maxStorageBufferRange 0x{storage_range_limit:X}, {size_limits:?}) are smaller than the minimal test window 0x{TEST_WINDOW_SIZE_GRANULARITY:X}"
        )
        .into());
    }
    if env.verbose() {
        let _ = writeln!(
            log_dupler,
            "maxStorageBufferRange 0x{storage_range_limit:X}, {size_limits:?}, max test window {:5.3}GB",
            max_window_size as f32 / GB
        );
    }
    Ok(max_window_size)
}

fn test_device<Writer: std::io::Write>(
    tester: &mut Tester,
    selected: &NamedComputeDevice,
    log_dupler: &mut output::LogDupler<Writer>,
) -> BoxResult<()> {
    let env = tester.env;
    let memory_props = selected.memory_props;
    let mut allocation_size = env.max_test_bytes();
    if allocation_size < MIN_WANTED_ALLOCATION {
        return Err(format!(
            "the memory to test ({:.2}GB) is smaller than the minimum of {:.2}GB",
            allocation_size as f32 / GB,
            MIN_WANTED_ALLOCATION as f32 / GB
        )
        .into());
    }
    // allow emulating writes that never reach memory, for testing the detection of it
    let skip_write_iteration = env::var(EMULATE_SKIPPED_WRITE_ENV)
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok());
    // allow write bugs emulation for testing purposes
    let emulate_write_bugs_iteration = env::var("MEMTEST_VULKAN_EMULATE_WRITE_BUG_ITERATION")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or_default();
    let max_window_size = max_window_size(selected, env, log_dupler)?;

    let mut totals = ErrorTotals::default();
    let mut last_error_at = None;
    let mut warn_on_budget_alloc_fail = true;
    //The error state before all allocation tries
    let mut last_err: Box<dyn Error> = "No heap reports memory enough for testing".into();
    loop {
        if allocation_size < MIN_WANTED_ALLOCATION {
            return Err(last_err);
        }
        if env.verbose() {
            let _ = writeln!(
                log_dupler,
                "Trying {:7.3}GB...",
                allocation_size as f32 / GB
            );
        }
        // allocate, then fill every window to verify that the allocation is really usable
        let attempt = tester
            .allocate_windows(allocation_size, max_window_size)
            .and_then(|()| {
                initial_fill(
                    tester,
                    log_dupler,
                    &mut totals,
                    &mut last_error_at,
                    skip_write_iteration == Some(0),
                )
            });
        match attempt {
            Ok(()) => break,
            Err(e) => last_err = e,
        }
        // Only running out of memory is retried with less memory. Other errors, like a lost
        // device or a hang, can come from unstable memory: retrying them could end in PASSED.
        // Memory errors found by the fill so far must be reported, so they end the test too.
        let errors_found = close::check_any_bits_set(
            close::fetch_status(),
            close::app_status::RUNTIME_ERRORS | close::app_status::WRITES_NOT_APPLIED,
        );
        if !vk_util::is_out_of_memory(last_err.as_ref()) || errors_found {
            return Err(last_err);
        }
        if env.verbose() {
            let _ = writeln!(
                log_dupler,
                "Not testing {:5.1}GB because of error: {}",
                allocation_size as f32 / GB,
                last_err
            );
        }
        if !env.interactive {
            close::exit_to_retry_with_less_memory();
        }
        allocation_size -= ALLOCATION_TRY_STEP;
        // only announce a retry if there's still a size worth trying
        if warn_on_budget_alloc_fail && allocation_size >= MIN_WANTED_ALLOCATION {
            warn_on_budget_alloc_fail = false;
            let _ = writeln!(
                log_dupler,
                "Couldn't use {:5.1}GB, trying less memory. Closing other programs using the GPU can help.",
                (allocation_size + ALLOCATION_TRY_STEP) as f32 / GB
            );
        }
    }

    let test_memory_type = memory_props.memory_types[tester.test_mem_index as usize];
    if test_memory_type
        .property_flags
        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    {
        let first_memory = tester.windows[0].memory;
        match unsafe {
            tester.device.map_memory(
                first_memory,
                0,
                MAP_FOR_GETTING_ADDRESS_SIZE,
                vk::MemoryMapFlags::default(),
            )
        }
        .err_as_str()
        {
            Err(e) => {
                if env.verbose() {
                    let _ = writeln!(log_dupler, "Failed getting virtual address due to {e}");
                }
            }
            Ok(virt_address) => {
                let addr_details =
                    ram::virt_addr_details(virt_address, selected.physical_props.device_type);
                unsafe { tester.device.unmap_memory(first_memory) }; // After getting addresses the actual mapping is not needed anymore
                if let Some(details) = addr_details {
                    writeln!(
                        log_dupler,
                        "CPU address of tested memory: virtual=x{:0pw$X} {}",
                        virt_address.addr(),
                        details,
                        pw = ram::POINTER_HEX_PRINT_WIDTH
                    )?
                }
            }
        }
    }
    if env.verbose() {
        let _ = writeln!(
            log_dupler,
            "Test memory type {:2}: {:?} {:?}",
            tester.test_mem_index,
            test_memory_type,
            memory_props.memory_heaps[test_memory_type.heap_index as usize],
        );
    }
    writeln!(log_dupler, "Testing {}", selected.label)?;
    writeln!(
        log_dupler,
        "Test plan: {:.1}GB as {} windows of {:.3}GB, duration {}{}",
        tester.tested_bytes() as f32 / GB,
        tester.window_count(),
        tester.window_size as f32 / GB,
        format_test_duration(env.options.duration),
        if env.options.duration == TestDuration::default() {
            " (standard test)"
        } else {
            ""
        }
    )?;

    if std::io::stdout().is_terminal() {
        log_dupler.enable_live_line();
    }
    let report = main_loop(
        tester,
        log_dupler,
        totals,
        last_error_at,
        skip_write_iteration,
        emulate_write_bugs_iteration,
    )?;
    if report.completed {
        close::raise_status_bit(close::app_status::COMPLETED);
    }
    write_summary(tester, selected, log_dupler, &report)?;
    Ok(())
}

/// Fills every window with the marker, writes it and checks it once
fn initial_fill<Writer: std::io::Write>(
    tester: &Tester,
    log_dupler: &mut output::LogDupler<Writer>,
    totals: &mut ErrorTotals,
    last_error_at: &mut Option<Duration>,
    skip_last_write: bool,
) -> BoxResult<()> {
    let mut initial = IOBuf::for_initial_iteration();
    initial.set_pattern(Pattern::initial());
    let window_count = tester.windows.len();
    for window_idx in 0..window_count {
        let skip_write = skip_last_write && window_idx == window_count - 1;
        tester.write_window(
            window_idx,
            Pattern::initial(),
            true,
            (!skip_write).then_some(tester.pipelines.write),
        )?;
        let (io_out, _) = tester.read_window(window_idx, initial)?;
        // memory errors found by the initial fill are real errors too
        if tester.report_errors(
            log_dupler,
            totals,
            &io_out,
            window_idx,
            ReadMode::InitialRead,
        )? {
            *last_error_at = Some(Duration::ZERO);
        }
    }
    Ok(())
}

/// Finds windows much slower to read than the fastest one: memory the driver placed in system
/// RAM instead of VRAM (checked with per-window speed, since the memory budget reported by
/// drivers can't be trusted for this). Windows that had errors are ignored, since reporting
/// errors slows the read down. Returns the slow window indexes and how many times slower they are.
fn find_slow_windows(read_times: &[Duration], had_errors: &[bool]) -> Option<(Vec<usize>, f64)> {
    let fastest = read_times
        .iter()
        .zip(had_errors)
        .filter(|&(time, &errors)| !errors && !time.is_zero())
        .map(|(time, _)| *time)
        .min()?;
    let slow: Vec<usize> = (0..read_times.len())
        .filter(|&i| {
            !had_errors[i]
                && read_times[i].as_secs_f64() > fastest.as_secs_f64() * SWAP_SLOWDOWN_FACTOR
        })
        .collect();
    let slowest = slow.iter().map(|&i| read_times[i]).max()?;
    Some((slow, slowest.as_secs_f64() / fastest.as_secs_f64()))
}

/// The progress line shown in place on the console, like
/// `  2:35 / 5:00  [##########----------]  52%   186.2GB/s  no errors`
fn progress_text(
    elapsed: Duration,
    target: Option<Duration>,
    write_gbps: f64,
    error_state: &str,
) -> String {
    const BAR_WIDTH: usize = 20;
    let speed = if write_gbps > 0.0 {
        format!("{write_gbps:6.1}GB/s")
    } else {
        " ".repeat(10)
    };
    match target {
        Some(target) => {
            let done = (elapsed.as_secs_f64() / target.as_secs_f64().max(1.0)).min(1.0);
            let filled = (done * BAR_WIDTH as f64).round() as usize;
            format!(
                "  {} / {}  [{}{}] {:3.0}%  {speed}  {error_state}",
                format_duration(elapsed),
                format_duration(target),
                "#".repeat(filled),
                "-".repeat(BAR_WIDTH - filled),
                done * 100.0
            )
        }
        None => format!(
            "  {} / endless  {speed}  {error_state}",
            format_duration(elapsed)
        ),
    }
}

/// Status of the errors found, for the progress lines
fn error_state(totals: &ErrorTotals, last_error_at: Option<Duration>) -> String {
    if totals.unwritten > 0 && totals.memory_errors == 0 {
        return "WRITES NOT APPLIED".to_owned();
    }
    match last_error_at {
        None => "no errors".to_owned(),
        Some(at) => format!(
            "ERRORS: {} (last {})",
            totals.memory_errors + totals.unwritten,
            format_duration(at)
        ),
    }
}

/// Lists window numbers (from 1) like "2, 3"
fn window_numbers(windows: &[usize]) -> String {
    windows
        .iter()
        .map(|window_idx| (window_idx + 1).to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Which of the slow windows can be dropped: never window 0 (it holds the re-read data), and
/// only while at least one other fast window stays for the write test
fn windows_to_drop(slow: &[usize], active: &[bool]) -> Vec<usize> {
    let fast_rewritten_windows = (1..active.len())
        .filter(|window_idx| active[*window_idx] && !slow.contains(window_idx))
        .count();
    if fast_rewritten_windows == 0 {
        return Vec::new();
    }
    slow.iter()
        .copied()
        .filter(|&window_idx| window_idx != 0 && active[window_idx])
        .collect()
}

/// Drops windows the driver placed in system RAM (found by `find_slow_windows`), so the rest of
/// the test checks VRAM only. Window 0 holds the re-read data and at least one other window is
/// needed for the write test, so if those are slow they're kept: then returns a warning instead.
fn handle_slow_windows<Writer: std::io::Write>(
    tester: &mut Tester,
    log_dupler: &mut output::LogDupler<Writer>,
    slow: &[usize],
    slowdown: f64,
    skipped_windows: &mut Vec<usize>,
) -> BoxResult<Option<String>> {
    let total_windows = tester.windows.len();
    let active: Vec<bool> = (0..total_windows).map(|i| tester.is_active(i)).collect();
    let droppable = windows_to_drop(slow, &active);
    let window_gb = tester.window_size as f32 / GB;
    if !droppable.is_empty() {
        for &window_idx in &droppable {
            tester.drop_window(window_idx);
        }
        skipped_windows.extend(&droppable);
        let message = format!(
            "Test window{} {} of {total_windows} ({:.1}GB) read {slowdown:.0}x slower than the \
             others: the driver placed that memory in system RAM, most likely because other \
             programs use the VRAM. Testing continues without it, on {:.1}GB. Close other \
             programs using the GPU (browsers, Discord, games, overlays) to test more of the VRAM.",
            if droppable.len() > 1 { "s" } else { "" },
            window_numbers(&droppable),
            droppable.len() as f32 * window_gb,
            tester.tested_bytes() as f32 / GB,
        );
        writeln!(log_dupler, "{}", text::wrap("", &message))?;
    }
    let kept: Vec<usize> = slow
        .iter()
        .copied()
        .filter(|window_idx| !droppable.contains(window_idx))
        .collect();
    if kept.is_empty() {
        return Ok(None);
    }
    Ok(Some(format!(
        "test window{} {} of {total_windows} ({:.1}GB) read{} {slowdown:.0}x slower than the \
         others. The driver most likely placed that memory in system RAM because other programs \
         use the VRAM, so that part of the test checks system RAM instead of VRAM. Close other \
         programs using the GPU (browsers, Discord, games, overlays) or test less memory with --size.",
        if kept.len() > 1 { "s" } else { "" },
        window_numbers(&kept),
        kept.len() as f32 * window_gb,
        if kept.len() > 1 { "" } else { "s" },
    )))
}

fn main_loop<Writer: std::io::Write>(
    tester: &mut Tester,
    log_dupler: &mut output::LogDupler<Writer>,
    mut totals: ErrorTotals,
    mut last_error_at: Option<Duration>,
    skip_write_iteration: Option<i32>,
    emulate_write_bugs_iteration: i32,
) -> BoxResult<RunReport> {
    let env = tester.env;
    let window_count = tester.windows.len();
    let duration = env.options.duration;
    let target = match duration {
        TestDuration::Limited(target) => Some(target),
        TestDuration::Endless => None,
    };
    // pause the load once, 30 seconds before the (standard) test end, then re-check all windows
    let pause_at = match target {
        Some(target) if target < Duration::from_secs(60) => None,
        _ => target
            .unwrap_or(STANDARD_DURATION)
            .checked_sub(Duration::from_secs(30)),
    };
    let mut paused = false;
    let announce_standard_result = target.is_none_or(|target| target > STANDARD_DURATION);
    let mut standard_announced = false;
    let mut swap_warning: Option<String> = None;
    let mut skipped_windows: Vec<usize> = Vec::new();

    let start = Instant::now();
    let mut next_report_at = Duration::ZERO;
    let mut period = Throughput::default();
    let mut total = Throughput::default();
    let mut buffer_in = IOBuf::for_initial_iteration();
    let mut iterations = 0u32;
    let mut stop_requested = false;
    let mut target_reached = false;
    let mut reports = 0u32;
    // per-window read time and errors in the current report period, to spot windows not in VRAM
    let mut window_read_times = vec![Duration::ZERO; window_count];
    let mut window_had_errors = vec![false; window_count];
    let stop_on_error = env.options.stop_on_error;
    let mut stopped_on_error = false;
    let mut alerted = false;
    let mut last_live_update: Option<Instant> = None;
    if last_error_at.is_some() {
        // errors found already by the initial fill
        crate::alert::alert();
        alerted = true;
        stopped_on_error = stop_on_error;
    }

    'test: for iteration in 1.. {
        if stopped_on_error {
            break;
        }
        // write all windows except window 0, which keeps the initial data to check retention
        for window_idx in 1..window_count {
            if !tester.is_active(window_idx) {
                continue;
            }
            let write_pipeline = if skip_write_iteration == Some(iteration) && window_idx == 1 {
                None // emulate a write that never reaches memory
            } else if iteration != emulate_write_bugs_iteration {
                Some(tester.pipelines.write)
            } else {
                Some(tester.pipelines.emulate_write_bugs)
            };
            period.write_time += tester.write_window(
                window_idx,
                Pattern::for_iteration(buffer_in.iter, window_idx as i64),
                false,
                write_pipeline,
            )?;
            period.written += tester.window_size;
            if close::close_requested() {
                stop_requested = true;
                break 'test;
            }
        }
        // read all windows; after the load pause, read them all once more
        let mut modes = vec![(ReadMode::NextReRead, ReadMode::InitialRead)];
        let elapsed = start.elapsed();
        if !paused && pause_at.is_some_and(|pause_at| elapsed >= pause_at) {
            paused = true;
            modes.push((ReadMode::AfterPause, ReadMode::AfterPause));
        }
        for (pass, (window0_mode, other_mode)) in modes.into_iter().enumerate() {
            if pass > 0 {
                writeln!(
                    log_dupler,
                    "Pausing the GPU load for {} seconds, then checking all memory again...",
                    LOAD_PAUSE.as_secs()
                )?;
                let pause_start = Instant::now();
                while pause_start.elapsed() < LOAD_PAUSE {
                    log_dupler.show_live_line(&format!(
                        "  Pausing the GPU load: {}s left",
                        // saturating: the pause may end between the loop check and here
                        LOAD_PAUSE.saturating_sub(pause_start.elapsed()).as_secs() + 1
                    ));
                    std::thread::sleep(Duration::from_millis(250));
                    if close::close_requested() {
                        stop_requested = true;
                        break 'test;
                    }
                }
            }
            for window_idx in 0..window_count {
                if !tester.is_active(window_idx) {
                    continue;
                }
                let io_in = if window_idx == 0 {
                    // window 0 is never rewritten: check the data of the initial fill
                    let mut io = IOBuf::for_initial_iteration();
                    io.set_pattern(Pattern::initial());
                    io
                } else {
                    let mut io = buffer_in;
                    io.set_pattern(Pattern::for_iteration(buffer_in.iter, window_idx as i64));
                    // The data this window held before this iteration's write: the initial fill
                    // or the previous iteration's write.
                    // Reading it back means this iteration's write never reached memory.
                    io.set_previous_pattern(Some(if iteration == 1 {
                        Pattern::initial()
                    } else {
                        Pattern::for_iteration(buffer_in.iter.wrapping_sub(1), window_idx as i64)
                    }));
                    io
                };
                let (io_out, read_time) = tester.read_window(window_idx, io_in)?;
                period.read_time += read_time;
                period.read += tester.window_size;
                window_read_times[window_idx] += read_time;
                let mode = if window_idx == 0 {
                    window0_mode
                } else {
                    other_mode
                };
                if tester.report_errors(log_dupler, &mut totals, &io_out, window_idx, mode)? {
                    last_error_at = Some(start.elapsed());
                    window_had_errors[window_idx] = true;
                    if !alerted {
                        crate::alert::alert();
                        alerted = true;
                    }
                    if stop_on_error {
                        stopped_on_error = true;
                        break 'test;
                    }
                }
                if close::close_requested() {
                    stop_requested = true;
                    break 'test;
                }
            }
        }
        iterations = iteration as u32;
        buffer_in.prepare_next_iter_write();

        let elapsed = start.elapsed();
        target_reached = target.is_some_and(|target| elapsed >= target);
        if elapsed >= next_report_at || target_reached {
            reports += 1;
            if reports == 2 {
                close::raise_status_bit(close::app_status::INITED_OK);
            }
            // memory placed in system RAM: some windows much slower than others
            if let Some((slow, slowdown)) =
                find_slow_windows(&window_read_times, &window_had_errors)
                && let Some(warning) =
                    handle_slow_windows(tester, log_dupler, &slow, slowdown, &mut skipped_windows)?
                && swap_warning.is_none()
            {
                writeln!(log_dupler, "{}", text::wrap("WARNING: ", &warning))?;
                swap_warning = Some(warning);
            }
            window_read_times.fill(Duration::ZERO);
            window_had_errors.fill(false);
            writeln!(
                log_dupler,
                "{:>7} / {:<7}  written:{:7.1}GB{:7.1}GB/s  checked:{:7.1}GB{:6.1}GB/s  {}",
                format_duration(elapsed),
                format_test_duration(duration),
                period.written as f32 / GB,
                period.write_gbps(),
                period.read as f32 / GB,
                period.read_gbps(),
                error_state(&totals, last_error_at)
            )?;
            total.add(&period);
            period = Throughput::default();
            next_report_at = match reports {
                1 => Duration::from_secs(1),
                2 => Duration::from_secs(6),
                _ => elapsed + REPORT_INTERVAL,
            };
        }
        if announce_standard_result && !standard_announced && elapsed >= STANDARD_DURATION {
            standard_announced = true;
            let status = close::fetch_status();
            let verdict =
                if close::check_any_bits_set(status, close::app_status::WRITES_NOT_APPLIED) {
                    "INVALID - test writes were not applied (most likely a driver/tool problem)"
                } else if close::check_any_bits_set(status, close::app_status::RUNTIME_ERRORS) {
                    "FAILED - ERRORS FOUND"
                } else {
                    "PASSED"
                };
            writeln!(
                log_dupler,
                "Standard 5-minute test {verdict}. Continuing {}; press Ctrl+C to stop.",
                match duration {
                    TestDuration::Limited(target) => format!("until {}", format_duration(target)),
                    TestDuration::Endless => "without a time limit".to_owned(),
                }
            )?;
        }
        if target_reached {
            break;
        }
        // progress shown in place on the console, a few times per second
        if last_live_update.is_none_or(|at| at.elapsed() >= LIVE_UPDATE_INTERVAL) {
            last_live_update = Some(Instant::now());
            log_dupler.show_live_line(&progress_text(
                start.elapsed(),
                target,
                period.write_gbps(),
                &error_state(&totals, last_error_at),
            ));
        }
    }
    total.add(&period);
    if stop_requested {
        writeln!(log_dupler, "Testing stopped by Ctrl+C.")?;
    }
    if stopped_on_error {
        writeln!(
            log_dupler,
            "Testing stopped at the first error (--stop-on-error)."
        )?;
        // the test ended as asked, not unexpectedly
        close::raise_status_bit(close::app_status::COMPLETED);
    }
    // stopping an endless or long test after the standard 5 minutes still gives a full result
    let elapsed = start.elapsed();
    let required = target.map_or(STANDARD_DURATION, |target| target.min(STANDARD_DURATION));
    Ok(RunReport {
        elapsed,
        iterations,
        total,
        totals,
        last_error_at,
        swap_warning,
        skipped_windows,
        completed: target_reached || elapsed >= required,
        stopped_by_user: stop_requested,
        stopped_on_error,
    })
}

/// What to do after the test, for each kind of result
fn next_steps(totals: &ErrorTotals, completed: bool) -> &'static str {
    if totals.unwritten > 0 {
        "Retry with --window-mb 512, or update or change the GPU driver, and see Troubleshooting \
         in the README. Faulty VRAM gives scattered errors, not this pattern."
    } else if totals.memory_errors > 0 {
        "If the GPU memory is overclocked or undervolted, go back to stock settings and test \
         again. If errors remain at stock settings, check the GPU temperatures and cooling. \
         Errors at stock settings and normal temperatures point to faulty VRAM."
    } else if completed {
        "Nothing to fix. When tuning an overclock, this setting passed: you can try the next \
         step, and confirm the final setting with a longer test, like --duration 1h."
    } else {
        "Run the full test (the standard 5 minutes or longer) for a reliable result."
    }
}

fn write_summary<Writer: std::io::Write>(
    tester: &Tester,
    selected: &NamedComputeDevice,
    log_dupler: &mut output::LogDupler<Writer>,
    report: &RunReport,
) -> BoxResult<()> {
    let env = tester.env;
    let totals = &report.totals;
    let rule = "-".repeat(78);
    writeln!(log_dupler, "{rule}")?;
    writeln!(log_dupler, "GPU:        {}", selected.label)?;
    writeln!(
        log_dupler,
        "Tested:     {:.1}GB as {} windows of {:.3}GB, {} passes in {}{}",
        tester.tested_bytes() as f32 / GB,
        tester.window_count(),
        tester.window_size as f32 / GB,
        report.iterations,
        format_duration(report.elapsed),
        match env.options.duration {
            TestDuration::Limited(target) if report.stopped_by_user || report.stopped_on_error => {
                format!(" of {} planned", format_duration(target))
            }
            _ => String::new(),
        }
    )?;
    writeln!(
        log_dupler,
        "Throughput: written {:.1}GB at {:.1}GB/s, checked {:.1}GB at {:.1}GB/s",
        report.total.written as f32 / GB,
        report.total.write_gbps(),
        report.total.read as f32 / GB,
        report.total.read_gbps()
    )?;
    let result = if totals.unwritten > 0 {
        let mut result = format!(
            "INVALID - {} values the test wrote never reached the memory. Most likely a GPU \
             driver or memtest_vulkan problem rather than faulty VRAM.",
            totals.unwritten
        );
        if totals.memory_errors > 0 {
            result += &format!(" Also found {} other wrong values.", totals.memory_errors);
        }
        result
    } else if totals.memory_errors > 0 {
        format!(
            "FAILED - {} wrong value{} in {} check{}{}{}",
            totals.memory_errors,
            if totals.memory_errors == 1 { "" } else { "s" },
            totals.error_passes,
            if totals.error_passes == 1 { "" } else { "s" },
            report
                .last_error_at
                .map(|at| format!(", last at {}", format_duration(at)))
                .unwrap_or_default(),
            if report.stopped_on_error {
                ", stopped at the first error"
            } else {
                ""
            }
        )
    } else if report.completed
        && report.stopped_by_user
        && env.options.duration != TestDuration::default()
    {
        "PASSED - no errors found (stopped by Ctrl+C after the standard 5 minutes)".to_owned()
    } else if report.completed {
        "PASSED - no errors found".to_owned()
    } else {
        "INCOMPLETE - stopped early, no errors found so far".to_owned()
    };
    // labels are 12 characters wide, wrapped lines continue under the text after them
    writeln!(log_dupler, "{}", text::wrap("Result:     ", &result))?;
    if let Some(cause) = totals.likely_cause()
        && totals.unwritten == 0
    {
        writeln!(log_dupler, "{}", text::wrap("Cause:      ", &cause))?;
    }
    writeln!(
        log_dupler,
        "{}",
        text::wrap("Next steps: ", next_steps(totals, report.completed))
    )?;
    if !report.skipped_windows.is_empty() {
        let skipped = format!(
            "{:.1}GB (window {}) the driver placed in system RAM, as other programs use the \
             VRAM. Close other programs using the GPU to test it too.",
            report.skipped_windows.len() as f32 * tester.window_size as f32 / GB,
            window_numbers(&report.skipped_windows)
        );
        writeln!(log_dupler, "{}", text::wrap("Skipped:    ", &skipped))?;
    }
    if let Some(warning) = &report.swap_warning {
        writeln!(log_dupler, "{}", text::wrap("Warning:    ", warning))?;
    }
    if let Some(log_path) = &env.log_path {
        writeln!(log_dupler, "Log file:   {}", log_path.display())?;
    }
    writeln!(log_dupler, "{rule}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(values: &[u64]) -> Vec<Duration> {
        values.iter().map(|&v| Duration::from_millis(v)).collect()
    }

    #[test]
    fn windows_cover_the_most_memory() {
        const MB: i64 = 1024 * 1024;
        let device_max = TEST_WINDOW_MAX_SIZE;
        // B580: 10.69GB testable. 1GB windows would cover 10GB, 640MB windows cover 10.6GB
        assert_eq!(plan_windows(10_949 * MB, device_max), Some((640 * MB, 17)));
        // a tie keeps the larger windows: 1.5GB as 2 of 768MB rather than 3 of 512MB
        assert_eq!(plan_windows(1536 * MB, device_max), Some((768 * MB, 2)));
        assert_eq!(plan_windows(8192 * MB, device_max), Some((1024 * MB, 8)));
        // less than two preferred windows: two smaller ones
        assert_eq!(plan_windows(512 * MB, device_max), Some((256 * MB, 2)));
        assert_eq!(plan_windows(200 * MB, device_max), None);
        // --window-mb 256, or a device limiting buffers to 256MB
        assert_eq!(plan_windows(10_240 * MB, 256 * MB), Some((256 * MB, 40)));
        for total in (400..=24_576).step_by(97).map(|mb| mb * MB) {
            let (size, count) = plan_windows(total, device_max).unwrap();
            assert!(count >= 2 && size * count <= total && size <= PREFERRED_WINDOW_MAX);
            assert_eq!(size % TEST_WINDOW_SIZE_GRANULARITY, 0);
        }
    }

    #[test]
    fn emulated_failures_parse() {
        use EmulatedFailure::*;
        assert_eq!(parse_emulated_failure("hang"), Some((Hang, 1)));
        assert_eq!(
            parse_emulated_failure(" device-lost:20 "),
            Some((DeviceLost, 20))
        );
        assert_eq!(parse_emulated_failure("panic:3"), Some((Panic, 3)));
        assert_eq!(parse_emulated_failure("hang:0"), None);
        assert_eq!(parse_emulated_failure("hang:x"), None);
        assert_eq!(parse_emulated_failure("crash"), None);
    }

    #[test]
    fn window_in_system_ram_is_found() {
        // measured on Intel Arc B580 with 10.5GB tested: window 3 was placed in system RAM
        let (slow, slowdown) =
            find_slow_windows(&ms(&[94, 93, 1340]), &[false, false, false]).unwrap();
        assert_eq!(slow, vec![2]);
        assert!(slowdown > 14.0, "{slowdown}");
    }

    #[test]
    fn similar_windows_are_fine() {
        assert_eq!(find_slow_windows(&ms(&[94, 110, 120]), &[false; 3]), None);
        assert_eq!(find_slow_windows(&ms(&[0, 0]), &[false; 2]), None);
    }

    #[test]
    fn windows_with_errors_are_ignored() {
        // reporting many errors makes a read slow without any swapping
        assert_eq!(
            find_slow_windows(&ms(&[94, 5000, 95]), &[false, true, false]),
            None
        );
    }

    #[test]
    fn slow_windows_are_dropped_but_window_0_kept() {
        // on Intel Arc B580 with 10.5GB tested, window 3 of 3 was placed in system RAM
        assert_eq!(windows_to_drop(&[2], &[true; 3]), vec![2]);
        assert_eq!(windows_to_drop(&[1, 3], &[true; 4]), vec![1, 3]);
        // window 0 holds the re-read data
        assert_eq!(windows_to_drop(&[0, 2], &[true; 3]), vec![2]);
        // at least one fast window must stay for the write test
        assert_eq!(windows_to_drop(&[1], &[true; 2]), Vec::<usize>::new());
        assert_eq!(
            windows_to_drop(&[2], &[true, false, true]),
            Vec::<usize>::new()
        );
        // already dropped windows aren't dropped again
        assert_eq!(
            windows_to_drop(&[1, 2], &[true, false, true, true]),
            vec![2]
        );
    }

    #[test]
    fn progress_line_shows_time_bar_and_status() {
        let line = progress_text(
            Duration::from_secs(150),
            Some(Duration::from_secs(300)),
            186.24,
            "no errors",
        );
        assert!(line.contains("2:30 / 5:00"), "{line}");
        assert!(line.contains("[##########----------]  50%"), "{line}");
        assert!(line.contains("186.2GB/s  no errors"), "{line}");
        assert!(line.len() < 80, "{line}");
        let endless = progress_text(Duration::from_secs(61), None, 0.0, "ERRORS: 3 (last 0:42)");
        assert!(endless.contains("1:01 / endless"), "{endless}");
        assert!(!endless.contains("GB/s"), "{endless}");
        let overtime = progress_text(
            Duration::from_secs(400),
            Some(Duration::from_secs(300)),
            1.0,
            "",
        );
        assert!(
            overtime.contains("[####################] 100%"),
            "{overtime}"
        );
    }

    #[test]
    fn next_steps_match_the_result() {
        let mut totals = ErrorTotals::default();
        assert!(next_steps(&totals, true).contains("Nothing to fix"));
        assert!(next_steps(&totals, false).contains("full test"));
        totals.memory_errors = 3;
        assert!(next_steps(&totals, true).contains("stock settings"));
        totals.unwritten = 100;
        assert!(next_steps(&totals, true).contains("--window-mb 512"));
    }

    #[test]
    fn formats_durations() {
        assert_eq!(format_duration(Duration::from_secs(5)), "0:05");
        assert_eq!(format_duration(Duration::from_secs(300)), "5:00");
        assert_eq!(format_duration(Duration::from_secs(3725)), "1:02:05");
    }
}
