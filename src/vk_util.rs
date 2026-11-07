//! Vulkan helpers: error conversion and validation-layer plumbing.

use crate::{ProcessEnv, close, display_this_process_result};

use erupt::vk;
use std::ffi::{CStr, c_void};

pub struct CStrStaticPtr<const N: usize>([*const std::os::raw::c_char; N]);

impl<const N: usize> core::ops::Deref for CStrStaticPtr<N> {
    type Target = [*const std::os::raw::c_char; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl<const N: usize> Sync for CStrStaticPtr<N> {}

pub const VK_LOADER_DEBUG: &str = "VK_LOADER_DEBUG";
const LAYER_KHRONOS_VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";
pub static LAYER_KHRONOS_VALIDATION_ARRAY: CStrStaticPtr<1> =
    CStrStaticPtr([LAYER_KHRONOS_VALIDATION.as_ptr()]);

// erupt CreateInfoBuilder's has problem without passing enabled_extensions argument,
// https://github.com/GpuZelenograd/memtest_vulkan/issues/53
// Workaround it by explicitly passing empty array
pub static NO_EXTENSIONS_ARRAY: CStrStaticPtr<0> = CStrStaticPtr([]);

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

impl<T> MapErrStr for std::result::Result<T, erupt::LoaderError> {
    type ValueType = T;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        self.map_err(|res| {
            let msg = match res {
                erupt::LoaderError::SymbolNotAvailable => {
                    "SymbolNotAvailable in Loader".to_string()
                }
                erupt::LoaderError::VulkanError(result) => format!("{}", result),
            } + " while getting "
                + std::any::type_name::<Self::ValueType>();
            msg.into()
        })
    }
    fn err_as_str_context(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        self.map_err(|res| {
            let msg = match res {
                erupt::LoaderError::SymbolNotAvailable => {
                    "SymbolNotAvailable in Loader".to_string()
                }
                erupt::LoaderError::VulkanError(result) => format!("{}", result),
            } + " while getting "
                + std::any::type_name::<Self::ValueType>()
                + " in context "
                + context;
            msg.into()
        })
    }
}

impl<T> MapErrStr for erupt::utils::VulkanResult<T> {
    type ValueType = T;
    fn err_as_str(self) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        let result = self.result();
        result.map_err(|res| {
            let msg =
                res.to_string() + " while getting " + std::any::type_name::<Self::ValueType>();
            msg.into()
        })
    }
    fn err_as_str_context(
        self,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        let result = self.result();
        result.map_err(|res| {
            let msg = res.to_string()
                + " while getting "
                + std::any::type_name::<Self::ValueType>()
                + " in context "
                + context;
            msg.into()
        })
    }
}
impl<T> MapErrRetryWithLowerMemory for erupt::utils::VulkanResult<T> {
    type ValueType = T;
    fn err_retry_with_lower_memory(
        self,
        env: &ProcessEnv,
        context: &str,
    ) -> Result<Self::ValueType, Box<dyn std::error::Error>> {
        let result = self.result();
        result.map_err(|res| {
            let msg = res.to_string()
                + " while getting "
                + std::any::type_name::<Self::ValueType>()
                + " in context "
                + context;
            if !env.interactive
                && !close::check_any_bits_set(close::fetch_status(), close::app_status::INITED_OK)
            {
                let out_of_memory = matches!(
                    res,
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
    _message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let cstr_message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };

    eprintln!("{}", cstr_message.to_string_lossy());
    vk::FALSE
}
