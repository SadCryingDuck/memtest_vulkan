//! Vulkan helpers: loading Vulkan, error conversion and validation-layer plumbing.

use crate::{ProcessEnv, display_this_process_result};

use ash::vk;
use std::{
    ffi::{CStr, c_void},
    fmt,
};

pub const VK_LOADER_DEBUG: &str = "VK_LOADER_DEBUG";
pub const LAYER_KHRONOS_VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";

/// The Vulkan loader library (vulkan-1.dll, libvulkan.so.1) couldn't be loaded
#[derive(Debug)]
pub struct LoaderLoadError(ash::LoadingError);

impl fmt::Display for LoaderLoadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Khronos 'vulkan loader' vendor-agnostic dynamic library failed to load"
        )
    }
}

impl std::error::Error for LoaderLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// Loads the system's Vulkan loader library
pub fn load_vulkan() -> Result<ash::Entry, LoaderLoadError> {
    unsafe { ash::Entry::load() }.map_err(LoaderLoadError)
}

/// Converts Vulkan errors to readable messages naming what was being created
pub trait MapErrStr {
    type ValueType;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>>;
    fn err_as_str_context(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>>;
    fn unwrap_or_display(self, env: &ProcessEnv) -> Self::ValueType
    where
        Self: Sized,
    {
        match self.err_as_str() {
            Err(e) => display_this_process_result(Some(e), env),
            Ok(v) => v,
        }
    }
}

/// Converts Vulkan errors like `MapErrStr`, marking running out of memory as `OutOfMemory`:
/// while allocating, that's the one error worth retrying with less memory
pub trait MapErrRetryWithLowerMemory {
    type ValueType;
    fn err_retry_with_lower_memory(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>>;
}

/// The driver ran out of memory. Other errors, like ERROR_DEVICE_LOST or a GPU hang, can be
/// signs of unstable memory, so they end the test instead of being retried with less memory,
/// which could end in a PASSED result for memory that failed.
#[derive(Debug)]
pub struct OutOfMemory(String);

impl fmt::Display for OutOfMemory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for OutOfMemory {}

pub fn is_out_of_memory(error: &(dyn std::error::Error + 'static)) -> bool {
    error.is::<OutOfMemory>()
}

fn describe<T>(result: vk::Result, context: Option<&str>) -> String {
    // the code name (like ERROR_INCOMPATIBLE_DRIVER) is what users search for and the
    // README's troubleshooting refers to; the description explains it
    let mut message = format!("{result:?} ({result})");
    // calls that return nothing, like allocate_memory's emulated limit, have no type to name
    let type_name = std::any::type_name::<T>();
    if type_name != "()" {
        message += " while getting ";
        message += type_name;
    }
    if let Some(context) = context {
        message += " in context ";
        message += context;
    }
    message
}

impl<T> MapErrStr for Result<T, vk::Result> {
    type ValueType = T;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        self.map_err(|result| describe::<T>(result, None).into())
    }
    fn err_as_str_context(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        self.map_err(|result| describe::<T>(result, Some(context)).into())
    }
}

impl<T> MapErrRetryWithLowerMemory for Result<T, vk::Result> {
    type ValueType = T;
    fn err_retry_with_lower_memory(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        self.map_err(|result| {
            let message = describe::<T>(result, Some(context));
            if matches!(
                result,
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY
            ) {
                Box::new(OutOfMemory(message)) as Box<dyn std::error::Error>
            } else {
                message.into()
            }
        })
    }
}

pub unsafe extern "system" fn debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let cstr_message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };

    eprintln!("{}", cstr_message.to_string_lossy());
    vk::FALSE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_names_the_code_and_type() {
        assert_eq!(
            describe::<vk::Buffer>(
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
                Some("create_buffer")
            ),
            "ERROR_OUT_OF_DEVICE_MEMORY (A device memory allocation has failed) \
             while getting ash::vk::definitions::Buffer in context create_buffer"
        );
    }

    #[test]
    fn only_running_out_of_memory_is_retried() {
        let error = |result| {
            Err::<(), _>(result)
                .err_retry_with_lower_memory("allocate_memory")
                .unwrap_err()
        };
        assert!(is_out_of_memory(
            error(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY).as_ref()
        ));
        assert!(is_out_of_memory(
            error(vk::Result::ERROR_OUT_OF_HOST_MEMORY).as_ref()
        ));
        // a lost device can come from unstable memory: never retried with less memory
        let lost = error(vk::Result::ERROR_DEVICE_LOST);
        assert!(!is_out_of_memory(lost.as_ref()));
        assert!(
            lost.to_string().starts_with("ERROR_DEVICE_LOST ("),
            "{lost}"
        );
    }

    #[test]
    fn describe_skips_the_type_for_calls_returning_nothing() {
        let lost = vk::Result::ERROR_DEVICE_LOST;
        assert_eq!(
            describe::<()>(lost, Some("wait_for_fences")),
            format!("ERROR_DEVICE_LOST ({lost}) in context wait_for_fences")
        );
        assert_eq!(
            describe::<()>(lost, None),
            format!("ERROR_DEVICE_LOST ({lost})")
        );
    }
}
