//! Vulkan helpers: loading Vulkan, error conversion and validation-layer plumbing.

use crate::{ProcessEnv, close, display_this_process_result};

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

pub trait MapErrRetryWithLowerMemory {
    type ValueType;
    fn err_retry_with_lower_memory(
        self,
        env: &ProcessEnv,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>>;
}

fn describe<T>(result: vk::Result, context: Option<&str>) -> String {
    // the code name (like ERROR_INCOMPATIBLE_DRIVER) is what users search for and the
    // README's troubleshooting refers to; the description explains it
    let mut message = format!(
        "{result:?} ({result}) while getting {}",
        std::any::type_name::<T>()
    );
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
        env: &ProcessEnv,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        self.map_err(|result| {
            let msg = describe::<T>(result, Some(context));
            if !env.interactive
                && !close::check_any_bits_set(close::fetch_status(), close::app_status::INITED_OK)
            {
                let out_of_memory = matches!(
                    result,
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY
                );
                // Running out of memory is expected while searching for the testable size.
                // Other errors (like ERROR_DEVICE_LOST after a GPU hang/driver reset) may indicate
                // instability, so always show them instead of retrying silently.
                if env.verbose() || !out_of_memory {
                    println!("Retrying with lower memory due to {}", msg);
                }
                //immediate exit in non-interactive during init to initiate try with lower memory
                close::immediate_exit(true);
            }
            msg.into()
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
