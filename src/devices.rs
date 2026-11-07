//! Finding Vulkan devices able to run the test, and their properties.

use crate::{
    ProcessEnv,
    erupt_vendored_utils_loading::{EntryLoader, new_loader},
    output, ram,
    shader::{GB, TEST_DATA_KEEP_FREE},
    vk_util::{
        LAYER_KHRONOS_VALIDATION_ARRAY, MapErrStr, NO_EXTENSIONS_ARRAY, VK_LOADER_DEBUG,
        debug_callback,
    },
};
use core::cmp::{max, min};
use erupt::{
    ExtendableFrom, InstanceLoader, ObjectHandle,
    extensions::{ext_debug_utils, ext_memory_budget, ext_pci_bus_info, khr_maintenance4},
    vk,
};
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

pub struct NamedComputeDevice {
    /// Name, memory size and type, shown in the list after the GPU number
    pub label: String,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub physical_props: vk::PhysicalDeviceProperties,
    pub memory_props: vk::PhysicalDeviceMemoryProperties,
    pub budget_props: ext_memory_budget::PhysicalDeviceMemoryBudgetPropertiesEXT,
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
                let _ = writeln!(
                    log_dupler,
                    "heap size {:4.1}GB budget {:4.1}GB usage {:4.1}GB flags={:#?}",
                    self.memory_props.memory_heaps[i].size as f32 / GB,
                    self.budget_props.heap_budget[i] as f32 / GB,
                    self.budget_props.heap_usage[i] as f32 / GB,
                    self.memory_props.memory_heaps[i].flags,
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

fn load_instance<Writer: std::io::Write>(
    env: &ProcessEnv,
    log_dupler: &mut output::LogDupler<Writer>,
) -> Result<
    (
        erupt::InstanceLoader,
        EntryLoader,
        vk::DebugUtilsMessengerEXT,
    ),
    Box<dyn std::error::Error>,
> {
    let override_vk_loader_debug = env::var_os(VK_LOADER_DEBUG).is_none();
    if env.verbose() && override_vk_loader_debug {
        // memtest_vulkan is single-threaded during init.
        unsafe { env::set_var(VK_LOADER_DEBUG, "error,warn") };
    }

    let mut entry = new_loader()?;
    if env.verbose() {
        let _ = writeln!(
            log_dupler,
            "Verbose mode. Vulkan loader {}.{}.{}",
            vk::api_version_major(entry.instance_version()),
            vk::api_version_minor(entry.instance_version()),
            vk::api_version_patch(entry.instance_version())
        );
        for (idx, prop) in unsafe { entry.enumerate_instance_layer_properties(None) }
            .err_as_str()
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            if idx == 0 {
                let _ = write!(log_dupler, "Layers: ");
            } else {
                let _ = write!(log_dupler, ", ");
            }

            let _ = write!(log_dupler, "{}", unsafe {
                CStr::from_ptr(prop.layer_name.as_ptr())
                    .to_str()
                    .unwrap_or("Invalid property_name")
            });
        }
        let _ = writeln!(log_dupler);
        for (idx, ext) in unsafe { entry.enumerate_instance_extension_properties(None, None) }
            .err_as_str()
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            if idx == 0 {
                let _ = write!(log_dupler, "Extensions: ");
            } else {
                let _ = write!(log_dupler, ", ");
            }

            let _ = write!(log_dupler, "{}", unsafe {
                CStr::from_ptr(ext.extension_name.as_ptr())
                    .to_str()
                    .unwrap_or("Invalid extension_name")
            });
        }
        let _ = writeln!(log_dupler);
        let _ = writeln!(log_dupler);
    }

    let instance_extensions = vec![ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME];

    let app_info = vk::ApplicationInfoBuilder::new().api_version(vk::API_VERSION_1_1);
    let instance_create_info = vk::InstanceCreateInfoBuilder::new()
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&*LAYER_KHRONOS_VALIDATION_ARRAY)
        .application_info(&app_info);

    let mut messenger = vk::DebugUtilsMessengerEXT::null();
    match unsafe {
        InstanceLoader::new(&entry, &instance_create_info)
            .err_as_str_context("instance with validation")
    } {
        Ok(instance_with_validation) => {
            let mut severity = ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                | ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT;
            if env.verbose() {
                severity |= ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::INFO_EXT;
                //lists all extensions, very verbose
                //severity |= ext_debug_utils::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT;
                let _ = writeln!(log_dupler, "DebugUtilsMessenger with seveity {severity:?}");
            }
            let create_info = ext_debug_utils::DebugUtilsMessengerCreateInfoEXTBuilder::new()
                .message_severity(severity)
                .message_type(ext_debug_utils::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(debug_callback));
            messenger = unsafe {
                instance_with_validation.create_debug_utils_messenger_ext(&create_info, None)
            }
            .result()
            .unwrap_or_default();
            return Ok((instance_with_validation, entry, messenger));
        }
        Err(e) => {
            if env.verbose() {
                let _ = writeln!(log_dupler, "Not using validation layers due to {e}");
            }
        }
    }

    //fallback creation without validation and extensions
    let simple_instance_try = unsafe {
        InstanceLoader::new(
            &entry,
            &vk::InstanceCreateInfoBuilder::new()
                .enabled_extension_names(&*NO_EXTENSIONS_ARRAY)
                .application_info(&app_info),
        )
    }
    .err_as_str_context("instance. Try specifying icd.json via VK_DRIVER_FILES env var");
    match simple_instance_try {
        Ok(instance) => Ok((instance, entry, messenger)),
        Err(e) => {
            if !override_vk_loader_debug {
                return Err(e);
            }
            drop(entry);
            //retry instance creation with loader debug enabled
            // memtest_vulkan is single-threaded during init.
            unsafe { env::set_var(VK_LOADER_DEBUG, "all") };
            entry = new_loader()?;
            let debug_instance_try = unsafe {
                InstanceLoader::new(
                    &entry,
                    &vk::InstanceCreateInfoBuilder::new().application_info(&app_info),
                )
            }
            .map_err(|_second_error_ignired| e)?;
            Ok((debug_instance_try, entry, messenger))
        }
    }
}

pub struct LoadedDevices {
    // use Option to enforce InstanceLoader must be dropped before EntryLoader inside Drop implementation;
    // outside Drop theay are always Some
    instance: Option<erupt::InstanceLoader>,
    entry: Option<EntryLoader>,
    messenger: vk::DebugUtilsMessengerEXT,
    pub devices: Vec<NamedComputeDevice>,
}

impl LoadedDevices {
    pub fn instance(&self) -> &erupt::InstanceLoader {
        self.instance.as_ref().unwrap()
    }
}

impl Drop for LoadedDevices {
    fn drop(&mut self) {
        unsafe {
            if !self.messenger.is_null() {
                self.instance()
                    .destroy_debug_utils_messenger_ext(self.messenger, None);
            }
            self.instance().destroy_instance(None);
        }
        self.instance = None;
        self.entry = None;
    }
}

/// A device found during enumeration, before numbering
struct Candidate {
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    properties: vk::PhysicalDeviceProperties,
    pci_bus: Option<(u32, u32)>,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    budget_props: ext_memory_budget::PhysicalDeviceMemoryBudgetPropertiesEXT,
    index_in_device_iteration: usize,
    size_limits: DeviceSizeLimits,
}

/// Queries one physical device; None if it can't run compute shaders
unsafe fn query_candidate(
    instance: &InstanceLoader,
    physical_device: vk::PhysicalDevice,
    index_in_device_iteration: usize,
) -> Option<Candidate> {
    let queue_family_index =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device, None) }
            .into_iter()
            .position(|properties| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))?
            as u32;

    // Only chain extension structures the device supports: passing structures of
    // unsupported extensions in pNext is invalid usage per the Vulkan spec.
    let device_extensions =
        unsafe { instance.enumerate_device_extension_properties(physical_device, None, None) }
            .result()
            .unwrap_or_default();
    let has_extension = |name: *const std::os::raw::c_char| {
        let wanted = unsafe { CStr::from_ptr(name) };
        device_extensions
            .iter()
            .any(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) } == wanted)
    };
    let has_pci_bus_info = has_extension(ext_pci_bus_info::EXT_PCI_BUS_INFO_EXTENSION_NAME);
    let has_memory_budget_ext = has_extension(ext_memory_budget::EXT_MEMORY_BUDGET_EXTENSION_NAME);
    // maxBufferSize: the instance is created for Vulkan 1.1, so using the 1.3 core
    // structure is only valid when the device exposes it as VK_KHR_maintenance4
    let has_maintenance4 = has_extension(khr_maintenance4::KHR_MAINTENANCE_4_EXTENSION_NAME);

    let mut properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let effective_version = (
        vk::api_version_major(properties.api_version),
        vk::api_version_minor(properties.api_version),
    );

    let mut pci_structure_request =
        ext_pci_bus_info::PhysicalDevicePCIBusInfoPropertiesEXTBuilder::new();
    let mut budget_request =
        ext_memory_budget::PhysicalDeviceMemoryBudgetPropertiesEXTBuilder::new();
    let mut maintenance3_request = vk::PhysicalDeviceMaintenance3PropertiesBuilder::new();
    let mut maintenance4_request = vk::PhysicalDeviceMaintenance4PropertiesBuilder::new();
    let mut size_limits = DeviceSizeLimits::default();

    //older vulkan implementations like broadcom on RaspberryPi lacks vk_1_1 support even if application requested it
    let is_vk_1_1 = effective_version >= (1, 1);
    let memory_props = if is_vk_1_1 {
        // maxMemoryAllocationSize is Vulkan 1.1 core
        let mut device_props2 =
            vk::PhysicalDeviceProperties2Builder::new().extend_from(&mut maintenance3_request);
        if has_pci_bus_info {
            device_props2 = device_props2.extend_from(&mut pci_structure_request);
        }
        if has_maintenance4 {
            device_props2 = device_props2.extend_from(&mut maintenance4_request);
        }

        unsafe { instance.get_physical_device_properties2(physical_device, &mut device_props2) };
        properties = device_props2.properties;
        size_limits.max_memory_allocation_size =
            Some(maintenance3_request.max_memory_allocation_size as i64).filter(|&size| size > 0);
        if has_maintenance4 {
            size_limits.max_buffer_size =
                Some(maintenance4_request.max_buffer_size as i64).filter(|&size| size > 0);
        }
        let mut memory_props2 = vk::PhysicalDeviceMemoryProperties2Builder::new();
        if has_memory_budget_ext {
            memory_props2 = memory_props2.extend_from(&mut budget_request);
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
        pci_bus: (has_pci_bus_info && is_vk_1_1).then_some((
            pci_structure_request.pci_bus,
            pci_structure_request.pci_device,
        )),
        memory_props,
        budget_props: *budget_request,
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
    let (instance, entry, messenger) = load_instance(env, log_dupler)?;
    let mut candidates: Vec<Candidate> = unsafe { instance.enumerate_physical_devices(None) }
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
        instance: Some(instance),
        entry: Some(entry),
        messenger,
        devices,
    })
}
