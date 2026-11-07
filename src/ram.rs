//! System memory information.

pub const POINTER_HEX_PRINT_WIDTH: usize = size_of::<usize>() * 2;

const GIB: u64 = 1024 * 1024 * 1024;

/// Physical system memory, in bytes
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SystemMemory {
    /// usable by programs without swapping
    pub available: u64,
    pub total: u64,
}

impl SystemMemory {
    /// System RAM to leave for the OS and other programs when an integrated GPU tests it:
    /// 20% of the total, at least 2GB
    pub fn reserve_for_system(&self) -> u64 {
        (self.total / 5).max(2 * GIB)
    }

    /// How much system RAM an integrated GPU (which uses system RAM as video memory) may test
    pub fn usable_for_gpu(&self) -> u64 {
        self.available.saturating_sub(self.reserve_for_system())
    }
}

#[cfg(windows)]
pub fn system_memory() -> Option<SystemMemory> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(SystemMemory {
        available: status.ullAvailPhys,
        total: status.ullTotalPhys,
    })
}

#[cfg(target_os = "linux")]
pub fn system_memory() -> Option<SystemMemory> {
    parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn system_memory() -> Option<SystemMemory> {
    None
}

/// Parses MemAvailable and MemTotal from /proc/meminfo
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_meminfo(text: &str) -> Option<SystemMemory> {
    let field = |name: &str| -> Option<u64> {
        let line = text.lines().find(|line| line.starts_with(name))?;
        let kib: u64 = line[name.len()..]
            .trim()
            .trim_end_matches("kB")
            .trim()
            .parse()
            .ok()?;
        Some(kib * 1024)
    };
    Some(SystemMemory {
        available: field("MemAvailable:")?,
        total: field("MemTotal:")?,
    })
}

#[cfg(target_os = "linux")]
pub fn virt_addr_details(
    #[expect(unused)] virt_addr: *const core::ffi::c_void,
    device_type: erupt::vk::PhysicalDeviceType,
) -> Option<String> {
    if device_type == erupt::vk::PhysicalDeviceType::DISCRETE_GPU {
        return None;
    }
    None
}

#[cfg(not(target_os = "linux"))]
pub fn virt_addr_details(
    #[expect(unused)] virt_addr: *const core::ffi::c_void,
    #[expect(unused)] device_type: erupt::vk::PhysicalDeviceType,
) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_a_fifth_of_ram_for_the_system() {
        // the PC this was developed on: 31GB total, 14.6GB available
        let memory = SystemMemory {
            available: 146 * GIB / 10,
            total: 31 * GIB,
        };
        assert_eq!(memory.reserve_for_system(), 31 * GIB / 5);
        assert_eq!(memory.usable_for_gpu(), 146 * GIB / 10 - 31 * GIB / 5);
    }

    #[test]
    fn keeps_at_least_2gb_on_small_systems() {
        let memory = SystemMemory {
            available: 3 * GIB,
            total: 8 * GIB,
        };
        assert_eq!(memory.usable_for_gpu(), GIB);
        let nearly_full = SystemMemory {
            available: GIB,
            total: 8 * GIB,
        };
        assert_eq!(nearly_full.usable_for_gpu(), 0);
    }

    #[test]
    fn parses_proc_meminfo() {
        let text = "MemTotal:       32554076 kB\nMemFree:         1209000 kB\nMemAvailable:   15308812 kB\n";
        assert_eq!(
            parse_meminfo(text),
            Some(SystemMemory {
                available: 15308812 * 1024,
                total: 32554076 * 1024
            })
        );
        assert_eq!(parse_meminfo("MemTotal: 1 kB\n"), None);
    }
}
