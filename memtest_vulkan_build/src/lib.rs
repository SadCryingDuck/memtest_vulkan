extern crate proc_macro;
use proc_macro::TokenStream;

struct InlineSpirV(TokenStream);

fn naga_compile(src: &str) -> Result<Vec<u32>, String> {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;

    let opts = naga::back::spv::Options {
        lang_version: (1, 3), // Vulkan 1.1 must support spirv 1.3
        flags: naga::back::spv::WriterFlags::DEBUG,
        // no out-of-bounds checks: every index is in bounds by construction, and checks would
        // slow down the memory accesses being tested
        bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        // the shader loops are bounded already, extra loop counters would only add work
        force_loop_bounding: false,
        ..Default::default()
    };

    // Attempt to validate WGSL, error if invalid
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .map_err(|e| e.emit_to_string(src))?;
    let spv =
        naga::back::spv::write_vec(&module, &info, &opts, None).map_err(|e| format!("{:?}", e))?;
    Ok(spv)
}

fn gen_token_stream(feedback: Vec<u32>) -> TokenStream {
    (quote::quote! {
        {
            &[#(#feedback),*]
        }
    })
    .into()
}

impl syn::parse::Parse for InlineSpirV {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let src = input.parse::<syn::LitStr>().map(|x| x.value())?;
        let compiled = naga_compile(&src).map_err(|e| syn::parse::Error::new(input.span(), e))?;
        Ok(Self(gen_token_stream(compiled)))
    }
}

/// Compile inline shader source and embed the SPIR-V binary word sequence.
/// Returns a `&'static [u32]`.
#[proc_macro]
pub fn compiled_vk_compute_spirv(tokens: TokenStream) -> TokenStream {
    syn::parse_macro_input!(tokens as InlineSpirV).0
}
