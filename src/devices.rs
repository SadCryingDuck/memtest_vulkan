//! Finding Vulkan devices able to run the test, and their properties.

use crate::{
    ProcessEnv, output, ram,
    shader::{GB, TEST_DATA_KEEP_FREE},
    vk_util::{LAYER_KHRONOS_VALIDATION, MapErrStr, VK_LOADER_DEBUG, debug_callback, load_vulkan},
};
use ash::vk;
use core::cmp::{max, min};
use std::{env, ffi::CStr, fmt, io::Write};

#[derive(Default)]
struct DriverVersionDebug(u32);

impl fmt::Debug for DriverVersionDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let upper10bits = self.0 >> 22;
        if upper10bits > 2 {
            // NVIDIA (0x75a04180 = 470.129.06) and intel-mesa-on-linux (0x5402006 = 21.2.6) driver versioning
            return write!(f, "v{}(0x{:X})", upper10bits, self.0);
        }
        let upper18bits = self.0 >> 14;
        if upper18bits > 2 && upper18bits < 500 {
            //intel-on-windows driver versioning (0x19453c = [30.0.]101.1340)
            return write!(f, "v{}(0x{:X})", upper18bits, self.0);
        }
        if self.0 < 64 {
            //basic small number versioning like llvm
            return write!(f, "ver{}", self.0);
        }
        // don't parse AMD versioning like "0x8000E6"
        write!(f, "0x{:X}", self.0)
    }
}

/// Per-device size limits that a single test window (one allocation + one buffer) must respect.
/// None means the limit isn't reported by the driver.
#[derive(Clone, Copy, Default, Debug)]
pub struct DeviceSizeLimits {
    /// VkPhysicalDeviceMaintenance3Properties::maxMemoryAllocationSize (Vulkan 1.1)
    pub max_memory_allocation_size: Option<i64>,
    /// VkPhysicalDeviceMaintenance4Properties::maxBufferSize (Vulkan 1.3 / VK_KHR_maintenance4)
    pub max_buffer_size: Option<i64>,
}

/// What VK_EXT_memory_budget reported per memory heap (zeros if it isn't supported)
#[derive(Clone, Copy, Default)]
pub struct MemoryBudget {
    pub heap_budget: [u64; vk::MAX_MEMORY_HEAPS],
    pub heap_usage: [u64; vk::MAX_MEMORY_HEAPS],
}

pub struct NamedComputeDevice {
    /// Name, memory size and type, shown in the list after the GPU number
    pub label: String,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub physical_props: vk::PhysicalDeviceProperties,
    pub memory_props: vk::PhysicalDeviceMemoryProperties,
    pub budget_props: MemoryBudget,
    pub index_in_device_iteration: usize,
    pub size_limits: DeviceSizeLimits,
}

impl NamedComputeDevice {
    fn device_type_name(&self) -> &'static str {
        match self.physical_props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "discrete GPU",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated GPU",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual GPU",
            vk::PhysicalDeviceType::CPU => "CPU (software)",
            _ => "other device",
        }
    }

    /// How much memory can be tested: the free part of the biggest device-local heap, minus a
    /// reserve for the system and other programs. Also returns the system memory if free system
    /// RAM is what limited it (integrated GPUs use system RAM as video memory).
    pub fn testable_memory(&self) -> (i64, Option<ram::SystemMemory>) {
        // integrated GPUs use system RAM as video memory: leave enough of it for the system
        let system_memory = (self.physical_props.device_type
            != vk::PhysicalDeviceType::DISCRETE_GPU)
            .then(ram::system_memory)
            .flatten();
        let mut max_budget = 0;
        let mut limited_by_system_memory = false;
        for i in 0..self.memory_props.memory_heap_count as usize {
            if !self.memory_props.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            {
                continue;
            }
            let mut heap_free = self.memory_props.memory_heaps[i].size as i64;
            let usage = self.budget_props.heap_usage[i] as i64;
            if usage > 0 && usage < heap_free {
                heap_free -= usage;
            }
            let budget = self.budget_props.heap_budget[i] as i64;
            if budget > 0 {
                heap_free = min(heap_free, budget);
            }
            let mut limited_here = false;
            if let Some(memory) = system_memory {
                let usable = memory.usable_for_gpu() as i64;
                if usable < heap_free {
                    heap_free = usable;
                    limited_here = true;
                }
            }
            if heap_free - TEST_DATA_KEEP_FREE > max_budget {
                max_budget = heap_free - TEST_DATA_KEEP_FREE;
                limited_by_system_memory = limited_here;
            }
        }
        (
            max_budget,
            system_memory.filter(|_| limited_by_system_memory),
        )
    }

    /// `testable_memory`, logging how it was found
    pub fn estimate_testable_memory<Writer: std::io::Write>(
        &self,
        env: &ProcessEnv,
        log_dupler: &mut output::LogDupler<Writer>,
    ) -> i64 {
        if env.verbose() {
            for i in 0..self.memory_props.memory_heap_count as usize {
                let flags = self.memory_props.memory_heaps[i].flags;
                let flags = if flags.is_empty() {
                    "(empty)".to_string()
                } else {
                    format!("{flags:?}")
                };
                let _ = writeln!(
                    log_dupler,
                    "heap size {:4.1}GB budget {:4.1}GB usage {:4.1}GB flags={flags}",
                    self.memory_props.memory_heaps[i].size as f32 / GB,
                    self.budget_props.heap_budget[i] as f32 / GB,
                    self.budget_props.heap_usage[i] as f32 / GB,
                );
            }
        }
        let (max_budget, limited_by) = self.testable_memory();
        // explain the limit, unless the user chose the size anyway
        if let Some(memory) = limited_by
            && env.options.size.is_none()
        {
            let note = format!(
                "testing up to {:.1}GB of system RAM, leaving {:.1}GB for the system ({:.1}GB of \
                 {:.1}GB available now). Use --size to change it.",
                max(max_budget, 0) as f32 / GB,
                memory.reserve_for_system() as f32 / GB,
                memory.available as f32 / GB,
                memory.total as f32 / GB,
            );
            let _ = writeln!(
                log_dupler,
                "{}",
                crate::text::wrap("Integrated GPU: ", &note)
            );
        }
        max_budget
    }
}

/// The debug messenger printing validation layer messages, and its extension loader
type DebugMessenger = (ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT);

fn load_instance<Writer: std::io::Write>(
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<(ash::Entry, ash::Instance, Option<DebugMessenger>), Box<dyn std::error::Error>> {
    let override_vk_loader_debug = env::var_os(VK_LOADER_DEBUG).is_none();
    if env.verbose() && override_vk_loader_debug {
        // memtest_vulkan is single-threaded during init.
        unsafe { env::set_var(VK_LOADER_DEBUG, "error,warn") };
    }

    let mut entry = load_vulkan()?;
    if env.verbose() {
        let version = unsafe { entry.try_enumerate_instance_version() }
            .ok()
            .flatten()
            .unwrap_or(vk::API_VERSION_1_0);
        let _ = writeln!(
            log_dupler,
            "Verbose mode. Vulkan loader {}.{}.{}",
            vk::api_version_major(version),
            vk::api_version_minor(version),
            vk::api_version_patch(version)
        );
        let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
        let names: Vec<String> = layers
            .iter()
            .map(|layer| {
                layer
                    .layer_name_as_c_str()
                    .map_or("?".into(), |name| name.to_string_lossy().into_owned())
            })
            .collect();
        let _ = writeln!(log_dupler, "Layers: {}", names.join(", "));
        let extensions =
            unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
        let names: Vec<String> = extensions
            .iter()
            .map(|ext| {
                ext.extension_name_as_c_str()
                    .map_or("?".into(), |name| name.to_string_lossy().into_owned())
            })
            .collect();
        let _ = writeln!(log_dupler, "Extensions: {}", names.join(", "));
        let _ = writeln!(log_dupler);
    }

    let instance_extensions = [ash::ext::debug_utils::NAME.as_ptr()];
    let layers = [LAYER_KHRONOS_VALIDATION.as_ptr()];
    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
    let instance_create_info = vk::InstanceCreateInfo::default()
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&layers)
        .application_info(&app_info);

    match unsafe { entry.create_instance(&instance_create_info, None) }
        .err_as_str_context("instance with validation")
    {
        Ok(instance) => {
            let mut severity = vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
            if env.verbose() {
                severity |= vk::DebugUtilsMessageSeverityFlagsEXT::INFO;
                //lists all extensions, very verbose
                //severity |= vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE;
                let _ = writeln!(log_dupler, "DebugUtilsMessenger with seveity {severity:?}");
            }
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(severity)
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let messenger = unsafe { debug_utils.create_debug_utils_messenger(&create_info, None) }
                .ok()
                .map(|messenger| (debug_utils, messenger));
            return Ok((entry, instance, messenger));
        }
        Err(e) => {
            if env.verbose() {
                let _ = writeln!(log_dupler, "Not using validation layers due to {e}");
            }
        }
    }

    //fallback creation without validation and extensions
    let simple_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let simple_instance_try = unsafe { entry.create_instance(&simple_create_info, None) }
        .err_as_str_context("instance. Try specifying icd.json via VK_DRIVER_FILES env var");
    match simple_instance_try {
        Ok(instance) => Ok((entry, instance, None)),
        Err(e) => {
            if !override_vk_loader_debug {
                return Err(e);
            }
            drop(entry);
            //retry instance creation with loader debug enabled
            // memtest_vulkan is single-threaded during init.
            unsafe { env::set_var(VK_LOADER_DEBUG, "all") };
            entry = load_vulkan()?;
            let debug_instance_try = unsafe { entry.create_instance(&simple_create_info, None) }
                .map_err(|_second_error_ignired| e)?;
            Ok((entry, debug_instance_try, None))
        }
    }
}

pub struct LoadedDevices {
    // keeps the Vulkan loader library loaded; dropped after Drop destroyed the instance
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_messenger: Option<DebugMessenger>,
    pub devices: Vec<NamedComputeDevice>,
}

impl LoadedDevices {
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }
}

impl Drop for LoadedDevices {
    fn drop(&mut self) {
        unsafe {
            if let Some((debug_utils, messenger)) = self.debug_messenger.take() {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// A device found during enumeration, before numbering
struct Candidate {
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    properties: vk::PhysicalDeviceProperties,
    pci_bus: Option<(u32, u32)>,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    budget_props: MemoryBudget,
    index_in_device_iteration: usize,
    size_limits: DeviceSizeLimits,
}

/// Queries one physical device; None if it can't run compute shaders
unsafe fn query_candidate(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    index_in_device_iteration: usize,
) -> Option<Candidate> {
    let queue_family_index =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
            .into_iter()
            .position(|properties| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))?
            as u32;

    // Only chain extension structures the device supports: passing structures of
    // unsupported extensions in pNext is invalid usage per the Vulkan spec.
    let device_extensions =
        unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .unwrap_or_default();
    let has_extension = |wanted: &CStr| {
        device_extensions
            .iter()
            .any(|ext| ext.extension_name_as_c_str() == Ok(wanted))
    };
    let has_pci_bus_info = has_extension(ash::ext::pci_bus_info::NAME);
    let has_memory_budget_ext = has_extension(ash::ext::memory_budget::NAME);
    // maxBufferSize: the instance is created for Vulkan 1.1, so using the 1.3 core
    // structure is only valid when the device exposes it as VK_KHR_maintenance4
    let has_maintenance4 = has_extension(ash::khr::maintenance4::NAME);

    let mut properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let effective_version = (
        vk::api_version_major(properties.api_version),
        vk::api_version_minor(properties.api_version),
    );

    let mut pci_bus_info = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut maintenance4 = vk::PhysicalDeviceMaintenance4Properties::default();
    let mut size_limits = DeviceSizeLimits::default();

    //older vulkan implementations like broadcom on RaspberryPi lacks vk_1_1 support even if application requested it
    let is_vk_1_1 = effective_version >= (1, 1);
    let memory_props = if is_vk_1_1 {
        {
            // maxMemoryAllocationSize is Vulkan 1.1 core. The chained structures stay borrowed
            // until device_props2 goes away at the end of this block.
            let mut device_props2 =
                vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
            if has_pci_bus_info {
                device_props2 = device_props2.push_next(&mut pci_bus_info);
            }
            if has_maintenance4 {
                device_props2 = device_props2.push_next(&mut maintenance4);
            }
            unsafe {
                instance.get_physical_device_properties2(physical_device, &mut device_props2)
            };
            properties = device_props2.properties;
        }
        size_limits.max_memory_allocation_size =
            Some(maintenance3.max_memory_allocation_size as i64).filter(|&size| size > 0);
        if has_maintenance4 {
            size_limits.max_buffer_size =
                Some(maintenance4.max_buffer_size as i64).filter(|&size| size > 0);
        }
        let mut memory_props2 = vk::PhysicalDeviceMemoryProperties2::default();
        if has_memory_budget_ext {
            memory_props2 = memory_props2.push_next(&mut budget);
        }
        unsafe {
            instance.get_physical_device_memory_properties2(physical_device, &mut memory_props2)
        };
        memory_props2.memory_properties
    } else {
        unsafe { instance.get_physical_device_memory_properties(physical_device) }
    };

    Some(Candidate {
        physical_device,
        queue_family_index,
        properties,
        pci_bus: (has_pci_bus_info && is_vk_1_1)
            .then_some((pci_bus_info.pci_bus, pci_bus_info.pci_device)),
        memory_props,
        budget_props: MemoryBudget {
            heap_budget: budget.heap_budget,
            heap_usage: budget.heap_usage,
        },
        index_in_device_iteration,
        size_limits,
    })
}

/// Lists the devices able to run the test, numbered from 1 in the order they are offered:
/// discrete GPUs first, then integrated ones
pub fn list_devices_ordered_labeled_from_1<Writer: std::io::Write>(
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<LoadedDevices, Box<dyn std::error::Error>> {
    let (entry, instance, debug_messenger) = load_instance(env, log_dupler)?;
    let mut candidates: Vec<Candidate> = unsafe { instance.enumerate_physical_devices() }
        .err_as_str()?
        .into_iter()
        .enumerate()
        .filter(|(index_in_device_iteration, _)| {
            // the worker process only looks at the device it was asked to test
            [None, Some(*index_in_device_iteration)]
                .contains(&env.requested_index_in_device_iteration())
        })
        .filter_map(|(index_in_device_iteration, physical_device)| unsafe {
            query_candidate(&instance, physical_device, index_in_device_iteration)
        })
        .collect();
    // assuming that for multi-GPU scenarious the minimal PCIe bus number is used for picture output
    // and the maximal is the test target: select by default "the descrete GPU with maximal PCIe bus number".
    // Without bus info keep the enumeration order (the sort is stable).
    candidates.sort_by_key(|candidate| {
        let type_rank = match candidate.properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 2,
        };
        (
            type_rank,
            std::cmp::Reverse(candidate.pci_bus.map_or(0, |(bus, _)| bus)),
        )
    });
    let mut devices: Vec<NamedComputeDevice> = Vec::new();
    for candidate in candidates {
        let props = candidate.properties;
        let mut total_memory = 0i64;
        for heap in &candidate.memory_props.memory_heaps
            [..candidate.memory_props.memory_heap_count as usize]
        {
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                total_memory = max(total_memory, heap.size as i64);
            }
        }
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_str()
            .unwrap_or("Invalid device_name")
            .to_owned();
        let mut device = NamedComputeDevice {
            label: String::new(),
            physical_device: candidate.physical_device,
            queue_family_index: candidate.queue_family_index,
            physical_props: props,
            memory_props: candidate.memory_props,
            budget_props: candidate.budget_props,
            index_in_device_iteration: candidate.index_in_device_iteration,
            size_limits: candidate.size_limits,
        };
        let mut details = format!("DevId 0x{:04X}", props.device_id);
        if let Some((bus, slot)) = candidate.pci_bus {
            details += &format!(", Bus 0x{bus:02X}:{slot:02X}");
        }
        if env.verbose() {
            details += &format!(
                ", API {}.{}.{}, driver {:?}, -{}-",
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version),
                DriverVersionDebug(props.driver_version),
                candidate.index_in_device_iteration + 1,
            );
        }
        device.label = format!(
            "{name}  {}GB {}  ({details})",
            (total_memory as f32 / GB).ceil(),
            device.device_type_name(),
        );
        devices.push(device);
    }
    Ok(LoadedDevices {
        _entry: entry,
        instance,
        debug_messenger,
        devices,
    })
}
