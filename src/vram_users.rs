//! Which other programs use a GPU's dedicated memory (VRAM), so the user knows what to close
//! to test more of it. Windows only: read from the GPU performance counters (like Task Manager).

/// Programs using the dedicated memory of the GPU with this LUID, as (name, bytes), largest
/// first, without this process. None if it can't be found out on this system.
#[cfg(windows)]
pub fn dedicated_users(luid: [u8; 8]) -> Option<Vec<(String, u64)>> {
    let tag = luid_instance_tag(luid);
    let own_pid = std::process::id();
    let instances: Vec<(u32, u64)> = unsafe { windows::dedicated_usage_by_pid() }?
        .into_iter()
        .filter_map(|(instance, bytes)| pid_of_instance(&instance, &tag).map(|pid| (pid, bytes)))
        .collect();
    // no counters for this GPU at all: unknown rather than unused, so the caller estimates
    if instances.is_empty() {
        return None;
    }
    let usage: Vec<(u32, u64)> = instances
        .into_iter()
        .filter(|&(pid, bytes)| pid != own_pid && bytes > 0)
        .collect();
    let names = unsafe { windows::process_names() };
    Some(group_by_name(usage.into_iter().map(|(pid, bytes)| {
        let name = names
            .get(&pid)
            .map_or_else(|| "other programs".to_owned(), |exe| friendly_name(exe));
        (name, bytes)
    })))
}

#[cfg(not(windows))]
pub fn dedicated_users(_luid: [u8; 8]) -> Option<Vec<(String, u64)>> {
    None
}

// used by the Windows code only
#[cfg(any(windows, test))]
/// The part of the performance counter instance names identifying the GPU, like
/// `luid_0x00000000_0x0000cd9b` in `pid_1304_luid_0x00000000_0x0000cd9b_phys_0`
fn luid_instance_tag(luid: [u8; 8]) -> String {
    // a LUID is a u32 LowPart followed by an i32 HighPart, little-endian
    let low = u32::from_le_bytes([luid[0], luid[1], luid[2], luid[3]]);
    let high = u32::from_le_bytes([luid[4], luid[5], luid[6], luid[7]]);
    format!("luid_0x{high:08x}_0x{low:08x}")
}

// used by the Windows code only
#[cfg(any(windows, test))]
/// The process ID of a counter instance of the GPU identified by `tag`
fn pid_of_instance(instance: &str, tag: &str) -> Option<u32> {
    let instance = instance.to_ascii_lowercase();
    let rest = instance.strip_prefix("pid_")?;
    let (pid, after) = rest.split_once('_')?;
    after.starts_with(tag).then_some(())?;
    pid.parse().ok()
}

// used by the Windows code only
#[cfg(any(windows, test))]
/// A readable name for a program from its executable file name
fn friendly_name(exe: &str) -> String {
    let name = match exe.rsplit_once('.') {
        Some((stem, extension)) if extension.eq_ignore_ascii_case("exe") => stem,
        _ => exe,
    };
    match name.to_ascii_lowercase().as_str() {
        "dwm" => "Windows desktop".to_owned(),
        "explorer" => "Windows Explorer".to_owned(),
        _ => name.to_owned(),
    }
}

// used by the Windows code only
#[cfg(any(windows, test))]
/// Adds up the usage of programs with the same name (like the processes of a browser),
/// largest first
fn group_by_name(usage: impl Iterator<Item = (String, u64)>) -> Vec<(String, u64)> {
    let mut grouped: Vec<(String, u64)> = Vec::new();
    for (name, bytes) in usage {
        match grouped.iter_mut().find(|(existing, _)| *existing == name) {
            Some((_, total)) => *total += bytes,
            None => grouped.push((name, bytes)),
        }
    }
    grouped.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    grouped
}

#[cfg(windows)]
mod windows {
    use std::{collections::HashMap, ptr::null_mut};
    use windows_sys::Win32::{
        Foundation::{CloseHandle, INVALID_HANDLE_VALUE},
        System::{
            Diagnostics::ToolHelp::{
                CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW,
                TH32CS_SNAPPROCESS,
            },
            Performance::{
                PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE, PDH_MORE_DATA, PdhAddEnglishCounterW,
                PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhOpenQueryW,
            },
        },
    };

    /// Each instance of the "GPU Process Memory" counter: (instance name, dedicated bytes)
    pub unsafe fn dedicated_usage_by_pid() -> Option<Vec<(String, u64)>> {
        let mut query = null_mut();
        if unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut query) } != 0 {
            return None;
        }
        let result = unsafe { read_counter(query) };
        unsafe { PdhCloseQuery(query) };
        result
    }

    unsafe fn read_counter(
        query: windows_sys::Win32::System::Performance::PDH_HQUERY,
    ) -> Option<Vec<(String, u64)>> {
        let path: Vec<u16> = "\\GPU Process Memory(*)\\Dedicated Usage\0"
            .encode_utf16()
            .collect();
        let mut counter = null_mut();
        if unsafe { PdhAddEnglishCounterW(query, path.as_ptr(), 0, &mut counter) } != 0
            || unsafe { PdhCollectQueryData(query) } != 0
        {
            return None;
        }
        // the first call reports the buffer size; processes may start in between, so retry
        for _ in 0..3 {
            let (mut size, mut count) = (0u32, 0u32);
            let status = unsafe {
                PdhGetFormattedCounterArrayW(
                    counter,
                    PDH_FMT_LARGE,
                    &mut size,
                    &mut count,
                    null_mut(),
                )
            };
            if status != PDH_MORE_DATA {
                return (status == 0).then(Vec::new);
            }
            // the items, followed by their names; u64 elements keep the items aligned
            let mut buffer = vec![0u64; (size as usize).div_ceil(8)];
            let items = buffer.as_mut_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>();
            let status = unsafe {
                PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &mut size, &mut count, items)
            };
            if status == PDH_MORE_DATA {
                continue;
            }
            if status != 0 {
                return None;
            }
            let items = unsafe { std::slice::from_raw_parts(items, count as usize) };
            return Some(
                items
                    .iter()
                    // CStatus 0 or 1: valid data
                    .filter(|item| item.FmtValue.CStatus <= 1)
                    .map(|item| {
                        let bytes = unsafe { item.FmtValue.Anonymous.largeValue };
                        (unsafe { wide_to_string(item.szName) }, bytes.max(0) as u64)
                    })
                    .collect(),
            );
        }
        None
    }

    unsafe fn wide_to_string(text: *const u16) -> String {
        if text.is_null() {
            return String::new();
        }
        let mut len = 0;
        while unsafe { *text.add(len) } != 0 {
            len += 1;
        }
        String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(text, len) })
    }

    /// Executable file names of the running processes by process ID
    pub unsafe fn process_names() -> HashMap<u32, String> {
        let mut names = HashMap::new();
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
        if snapshot == INVALID_HANDLE_VALUE {
            return names;
        }
        let mut entry = PROCESSENTRY32W {
            dwSize: size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        let mut found = unsafe { Process32FirstW(snapshot, &mut entry) };
        while found != 0 {
            let len = entry
                .szExeFile
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(entry.szExeFile.len());
            names.insert(
                entry.th32ProcessID,
                String::from_utf16_lossy(&entry.szExeFile[..len]),
            );
            found = unsafe { Process32NextW(snapshot, &mut entry) };
        }
        unsafe { CloseHandle(snapshot) };
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_instances_match_the_gpu() {
        // LUID 0x00000000_0000CD9B, as Vulkan reports it (LowPart first, little-endian)
        let tag = luid_instance_tag([0x9B, 0xCD, 0, 0, 0, 0, 0, 0]);
        assert_eq!(tag, "luid_0x00000000_0x0000cd9b");
        assert_eq!(
            pid_of_instance("pid_1304_luid_0x00000000_0x0000CD9B_phys_0", &tag),
            Some(1304)
        );
        assert_eq!(
            pid_of_instance("pid_1304_luid_0x00000000_0x0000EB47_phys_0", &tag),
            None
        );
        assert_eq!(pid_of_instance("_Total", &tag), None);
    }

    #[test]
    fn programs_are_named_and_grouped() {
        assert_eq!(friendly_name("dwm.exe"), "Windows desktop");
        assert_eq!(friendly_name("Discord.exe"), "Discord");
        assert_eq!(friendly_name("firefox.EXE"), "firefox");
        let grouped = group_by_name(
            [
                ("firefox".to_owned(), 100),
                ("Discord".to_owned(), 150),
                ("firefox".to_owned(), 200),
            ]
            .into_iter(),
        );
        assert_eq!(
            grouped,
            vec![("firefox".to_owned(), 300), ("Discord".to_owned(), 150)]
        );
    }
}
