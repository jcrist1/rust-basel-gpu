//! Raw Vulkan (ash) backend for the BPE tokenizer kernel.
//!
//! Mirrors the wgpu streaming harness: fixed-size sub-batches, host-visible
//! buffers, one fence-synchronized dispatch per call. Buffers are host-visible
//! for simplicity (fine on unified memory / MoltenVK; a discrete GPU would want
//! device-local + staging).

use std::ffi::{CStr, CString};
use std::mem::size_of_val;

use anyhow::anyhow;
use ash::{Device, Entry, Instance, vk};
use rust_gpu_tokenizer_shaders::{MergeRule, PACKED_WORDS, SMALL_LEN, SmallStr};

/// Binding indices, matching the kernel signature.
const BIND_PACKED: usize = 0;
const BIND_RULES: usize = 1;
const BIND_LEFT_INDEX: usize = 2;
const BIND_BYTE_TO_ID: usize = 3;
const BIND_HOT: usize = 4;
const BIND_OUT_LENS: usize = 6;

pub struct AshStream {
    device: Device,
    _instance: Instance,
    _entry: Entry,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    /// Buffers in binding order; `(buffer, memory)`.
    buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    packed: Vec<u32>,
    pub num: usize,
    pub wg: usize,
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mem_props.memory_type_count).find(|&i| {
        type_bits & (1 << i) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(flags)
    })
}

unsafe fn create_host_buffer(
    device: &Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?
    };
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_type = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| anyhow!("no host-visible coherent memory type"))?;
    let mem = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type),
            None,
        )?
    };
    unsafe { device.bind_buffer_memory(buffer, mem, 0)? };
    Ok((buffer, mem))
}

unsafe fn upload<T: Copy>(device: &Device, mem: vk::DeviceMemory, data: &[T]) -> anyhow::Result<()> {
    let bytes = std::mem::size_of_val(data) as u64;
    if bytes == 0 {
        return Ok(());
    }
    let ptr = unsafe { device.map_memory(mem, 0, bytes, vk::MemoryMapFlags::empty())? };
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut T, data.len());
        device.unmap_memory(mem);
    }
    Ok(())
}

impl AshStream {
    pub fn new(
        rules: &[MergeRule],
        hot: &[MergeRule],
        byte_to_id: &[u32],
        num_ids: usize,
        batch_chunks: usize,
    ) -> anyhow::Result<Self> {
        let wg: usize = std::env::var("BPE_WG")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);
        let num = batch_chunks.next_multiple_of(wg);

        unsafe {
            let entry = Entry::linked();

            // Instance (portability enumeration is required on macOS/MoltenVK).
            let available = entry.enumerate_instance_extension_properties(None)?;
            let has_ext = |name: &CStr| {
                available
                    .iter()
                    .any(|e| e.extension_name_as_c_str().ok() == Some(name))
            };
            let mut instance_exts: Vec<*const i8> = Vec::new();
            let mut flags = vk::InstanceCreateFlags::empty();
            if has_ext(c"VK_KHR_portability_enumeration") {
                instance_exts.push(c"VK_KHR_portability_enumeration".as_ptr());
                flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }
            let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_extension_names(&instance_exts)
                    .flags(flags),
                None,
            )?;

            let physical = instance
                .enumerate_physical_devices()?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no Vulkan physical device"))?;
            let families = instance.get_physical_device_queue_family_properties(physical);
            let queue_family = families
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| anyhow!("no compute queue family"))? as u32;

            // Portability subset must be enabled if advertised.
            let dev_available = instance.enumerate_device_extension_properties(physical)?;
            let has_dev_ext = |name: &CStr| {
                dev_available
                    .iter()
                    .any(|e| e.extension_name_as_c_str().ok() == Some(name))
            };
            let mut dev_exts: Vec<*const i8> = Vec::new();
            if has_dev_ext(c"VK_KHR_portability_subset") {
                dev_exts.push(c"VK_KHR_portability_subset".as_ptr());
            }
            let priorities = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let device = instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_info)
                    .enabled_extension_names(&dev_exts),
                None,
            )?;
            let queue = device.get_device_queue(queue_family, 0);

            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let command_buffer = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..7)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let set_layouts = [set_layout];
            let pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )?;

            let spv = ash::util::read_spv(&mut std::fs::File::open(env!(
                "WGPU_KERNEL_SPV_PATH"
            ))?)?;
            let shader_module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&spv),
                None,
            )?;
            let entry_name = CString::new(format!("bpe{wg}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_name);
            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout);
            let pipeline = device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| anyhow!("create compute pipeline: {e:?}"))?[0];

            let mem_props = instance.get_physical_device_memory_properties(physical);
            let mut buffers: Vec<(vk::Buffer, vk::DeviceMemory)> = Vec::with_capacity(7);
            // Binding order must match the kernel.
            let sizes = [
                (num * PACKED_WORDS * 4) as u64,          // packed
                size_of_val(rules) as u64,                // rules
                ((num_ids + 1) * 4) as u64,               // left_index
                size_of_val(byte_to_id) as u64,           // byte_to_id
                size_of_val(hot) as u64,                  // hot
                (num * SMALL_LEN * 4) as u64,             // out_tokens
                (num * 4) as u64,                         // out_lens
            ];
            for &size in &sizes {
                buffers.push(create_host_buffer(&device, &mem_props, size)?);
            }

            // Upload the static tables.
            let left_index = crate::build_left_index(rules, num_ids);
            upload(&device, buffers[BIND_RULES].1, rules)?;
            upload(&device, buffers[BIND_LEFT_INDEX].1, &left_index)?;
            upload(&device, buffers[BIND_BYTE_TO_ID].1, byte_to_id)?;
            upload(&device, buffers[BIND_HOT].1, hot)?;

            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(7)];
            let descriptor_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(1),
                None,
            )?;
            let descriptor_set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )?[0];

            let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|(buf, _)| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(*buf)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = (0..7)
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&buffer_infos[i]))
                })
                .collect();
            device.update_descriptor_sets(&writes, &[]);

            Ok(Self {
                device,
                _instance: instance,
                _entry: entry,
                queue,
                command_pool,
                command_buffer,
                fence,
                descriptor_pool,
                descriptor_set,
                pipeline,
                pipeline_layout,
                shader_module,
                buffers,
                packed: vec![0u32; num * PACKED_WORDS],
                num,
                wg,
            })
        }
    }

    /// Encode at most `self.num` pretokens; returns the token count.
    pub fn encode(&mut self, pretokens: &[SmallStr]) -> anyhow::Result<u64> {
        let groups = pretokens.len().div_ceil(self.wg);
        let covered = groups * self.wg;
        let used = covered * PACKED_WORDS;
        self.packed[..used].fill(0);
        for (index, chunk) in pretokens.iter().enumerate() {
            let base = index * PACKED_WORDS;
            let len = chunk.len as usize;
            for i in 0..len {
                let b = chunk.bytes[i] & 0xFF;
                self.packed[base + i / 4] |= b << ((i % 4) * 8);
            }
            self.packed[base + 5] |= (len as u32) << 16;
        }

        unsafe {
            upload(&self.device, self.buffers[BIND_PACKED].1, &self.packed[..used])?;

            self.device.reset_command_buffer(
                self.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )?;
            self.device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )?;
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            self.device
                .cmd_dispatch(self.command_buffer, groups as u32, 1, 1);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            self.device.end_command_buffer(self.command_buffer)?;

            let command_buffers = [self.command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            self.device.queue_submit(self.queue, &[submit], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.fence])?;

            let bytes = (pretokens.len() * 4) as u64;
            let ptr = self.device.map_memory(
                self.buffers[BIND_OUT_LENS].1,
                0,
                bytes,
                vk::MemoryMapFlags::empty(),
            )?;
            let lens = std::slice::from_raw_parts(ptr as *const u32, pretokens.len());
            let total: u64 = lens.iter().map(|&l| l as u64).sum();
            self.device.unmap_memory(self.buffers[BIND_OUT_LENS].1);
            Ok(total)
        }
    }
}

impl Drop for AshStream {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_shader_module(self.shader_module, None);
            for (buffer, memory) in &self.buffers {
                self.device.destroy_buffer(*buffer, None);
                self.device.free_memory(*memory, None);
            }
        }
    }
}
