use cargo_gpu_install::install::Install;
use cargo_gpu_install::spirv_builder::{ShaderPanicStrategy, SpirvMetadata};
use std::path::PathBuf;

pub fn main() -> anyhow::Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let crate_path: PathBuf = [manifest_dir, "..", "rust-gpu-tokenizer-shaders"]
        .iter()
        .copied()
        .collect();

    // vulkan1.1: naga's SPIR-V frontend rejects the VulkanMemoryModel
    // capability that rust-gpu emits for vulkan1.2+ targets.
    let install = Install::from_shader_crate(crate_path.clone())
        .within_build_script()
        .run()?;
    let mut builder = install.to_spirv_builder(crate_path, "spirv-unknown-vulkan1.1");
    builder.build_script.defaults = true;
    builder.shader_panic_strategy = ShaderPanicStrategy::SilentExit;
    builder.spirv_metadata = SpirvMetadata::Full;

    let result = builder.build()?;
    let artifact = result.module.unwrap_single();
    println!("cargo:rustc-env=WGPU_KERNEL_SPV_PATH={}", artifact.display());
    Ok(())
}
