//! BPE tokenizer pipeline:
//!   1. pretokenize shakespeare.txt with the tiktoken-style regex (Qwen2)
//!   2. load the Qwen2 vocab (TKV1) and reconstruct merge rules from it
//!      (token ids are merge ranks; each merged token = concat of two earlier)
//!   3. encode on the CPU (oracle) with the rule table (same code as GPU)
//!   4. validate the reconstruction against an exact byte-space reference
//!   5. encode on the GPU (wgpu): rules + byte_to_id are shared read-only
//!      buffers, pretokens in, token ids out — then benchmark.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use rayon::prelude::*;

use rust_gpu_tokenizer_shaders::{MergeRule, NUM_BASE, SmallStr, TokenSeq, bpe_encode};
use tokenizer_host::{
    bpe_encode_ref, build_hot_table, build_left_index, byte_to_id_table, count_rule_hits, load_tkv,
    pack_chunks, pre_tokenize, read_text_sample, reconstruct_rules, spawn_pretokenizer,
};

mod ash_backend;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("parquet") {
        let default_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../sample_data.parquet");
        let (path, mode) = match args.get(2).map(String::as_str) {
            // `parquet <mode>` (mode in place of the path)
            Some(m @ ("gpu" | "cpu" | "both" | "ash" | "all" | "compare")) => (default_path, m),
            Some(p) => (p, args.get(3).map(String::as_str).unwrap_or("gpu")),
            None => (default_path, "gpu"),
        };
        return tokenize_parquet(path, mode);
    }
    benchmark();
    Ok(())
}

fn benchmark() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/shakespeare.txt");
    let mut text = String::new();
    File::open(path)
        .expect("Failed to open file")
        .read_to_string(&mut text)
        .expect("Failed to read file to string");

    let t = Instant::now();
    let pre_tokenized = pre_tokenize(&text);
    let truncated = pre_tokenized.iter().filter(|c| c.offset > 0).count();
    println!(
        "pretokenize: {} chunks in {:?} ({truncated} truncated to {} chars)",
        pre_tokenized.len(),
        t.elapsed(),
        rust_gpu_tokenizer_shaders::SMALL_LEN,
    );

    let t = Instant::now();
    let vocab = load_tkv(concat!(env!("CARGO_MANIFEST_DIR"), "/qwen2.tkv"));
    let (rules, rank) = reconstruct_rules(&vocab);
    let byte_to_id = byte_to_id_table(&vocab);
    let left_index = build_left_index(&rules, vocab.len());
    println!(
        "vocab: {} tokens -> {} rules in {:?}",
        vocab.len(),
        rules.len(),
        t.elapsed()
    );

    let t = Instant::now();
    let counts = count_rule_hits(&pre_tokenized, &rules, &left_index, &byte_to_id);
    let hot_slots: usize = std::env::var("BPE_HOT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(262144);
    let hot = build_hot_table(&counts, hot_slots);
    let lookups: u64 = counts.values().map(|(c, _)| *c).sum();
    println!(
        "hot rules: {} / {} slots filled from {lookups} lookups in {:?}",
        hot.iter().filter(|r| r.left != u32::MAX).count(),
        hot.len(),
        t.elapsed()
    );

    // Validate the reconstruction on a sample: rule-table encoding (what the
    // GPU does) must agree with the exact byte-space reference.
    let mut bytes_to_id: HashMap<&[u8], u32> = HashMap::new();
    for (i, t) in vocab.iter().enumerate() {
        bytes_to_id.entry(t.as_slice()).or_insert(i as u32);
    }
    let t = Instant::now();
    let sample = pre_tokenized.len().min(5_000);
    let mut agree = 0usize;
    for chunk in &pre_tokenized[..sample] {
        let bytes: Vec<u8> = chunk.bytes[..chunk.len as usize]
            .iter()
            .map(|&c| (c & 0xFF) as u8)
            .collect();
        let oracle = bpe_encode(*chunk, &rules, &left_index, &hot, &byte_to_id);
        let reference = bpe_encode_ref(&bytes, &bytes_to_id, &rank);
        if oracle.tokens[..oracle.len as usize] == reference[..] {
            agree += 1;
        }
    }
    println!(
        "reconstruction check: {agree}/{sample} chunks agree ({:.1}%) in {:?}",
        100.0 * agree as f64 / sample as f64,
        t.elapsed()
    );

    // CPU oracle over the whole corpus.
    let t = Instant::now();
    let expected: Vec<TokenSeq> = pre_tokenized
        .iter()
        .map(|&c| bpe_encode(c, &rules, &left_index, &hot, &byte_to_id))
        .collect();
    let total_tokens: usize = expected.iter().map(|s| s.len as usize).sum();
    let max_len = expected.iter().map(|s| s.len as usize).max().unwrap_or(0);
    let over = |cap: usize| expected.iter().filter(|s| s.len as usize > cap).count();
    println!(
        "output len: max {max_len}, >8: {}, >12: {}, >16: {}, >20: {}",
        over(8),
        over(12),
        over(16),
        over(20)
    );
    println!(
        "cpu encode (serial): {:?} ({} chunks -> {} tokens, {:.2} Mchunks/s)",
        t.elapsed(),
        pre_tokenized.len(),
        total_tokens,
        pre_tokenized.len() as f64 / t.elapsed().as_secs_f64() / 1e6,
    );

    // Multithreaded CPU encode with rayon. Warm the pool up first so the
    // timing measures the work, not thread spawn.
    let _ = pre_tokenized[..1000]
        .par_iter()
        .map(|&c| bpe_encode(c, &rules, &left_index, &hot, &byte_to_id))
        .collect::<Vec<_>>();
    let t = Instant::now();
    let parallel: Vec<TokenSeq> = pre_tokenized
        .par_iter()
        .with_min_len(512)
        .map(|&c| bpe_encode(c, &rules, &left_index, &hot, &byte_to_id))
        .collect();
    let elapsed = t.elapsed();
    let mismatches = parallel
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "cpu encode (rayon, {} threads): {elapsed:?} ({} chunks, {:.2} Mchunks/s) {}",
        rayon::current_num_threads(),
        pre_tokenized.len(),
        pre_tokenized.len() as f64 / elapsed.as_secs_f64() / 1e6,
        if mismatches == 0 {
            "OK".to_string()
        } else {
            format!("{mismatches} MISMATCHES")
        }
    );

    pollster::block_on(run_gpu(
        &pre_tokenized,
        &rules,
        &hot,
        &byte_to_id,
        vocab.len(),
        &expected,
        max_len,
    ))
    .expect("gpu encode failed");
}

async fn run_gpu(
    chunks: &[SmallStr],
    rules: &[MergeRule],
    hot: &[MergeRule],
    byte_to_id: &[u32; NUM_BASE],
    num_ids: usize,
    expected: &[TokenSeq],
    max_len: usize,
) -> anyhow::Result<()> {
    use anyhow::Context as _;
    use wgpu::util::DeviceExt;

    let t = Instant::now();
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None).await?;
    // Raise the compute workgroup limits so we can test up to 1024 threads
    // per workgroup (the default caps total invocations at 256).
    let adapter_limits = adapter.limits();
    let limits = wgpu::Limits {
        max_compute_workgroup_size_x: 1024.min(adapter_limits.max_compute_workgroup_size_x),
        max_compute_workgroup_size_y: 1024.min(adapter_limits.max_compute_workgroup_size_y),
        max_compute_invocations_per_workgroup: 1024
            .min(adapter_limits.max_compute_invocations_per_workgroup),
        ..wgpu::Limits::default()
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Default::default(),
        })
        .await
        .context("request device")?;

    let module = device.create_shader_module(wgpu::include_spirv!(env!("WGPU_KERNEL_SPV_PATH")));
    eprintln!("wgpu setup in {:?}", t.elapsed());

    let t = Instant::now();
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            // chunks (in, read-only)
            bgl_entry(0, true),
            // rules (read-only, shared)
            bgl_entry(1, true),
            // left_index (read-only, shared)
            bgl_entry(2, true),
            // byte_to_id (read-only, shared)
            bgl_entry(3, true),
            // hot-rule cache (read-only, shared)
            bgl_entry(4, true),
            // out_tokens (write)
            bgl_entry(5, false),
            // out_lens (write)
            bgl_entry(6, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let wg: usize = std::env::var("BPE_WG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let entry = format!("bpe{wg}");
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&entry),
        compilation_options: Default::default(),
        cache: None,
    });
    eprintln!("pipeline in {:?}", t.elapsed());

    // Probe: truncate the rule table to see how much time is rule lookup.
    let rules: &[MergeRule] = match std::env::var("BPE_RULES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(n) => &rules[..n.min(rules.len())],
        None => rules,
    };
    let no_readback = std::env::var("BPE_NOREAD").is_ok();
    let no_upload = std::env::var("BPE_NOUPLOAD").is_ok();
    let left_index = build_left_index(rules, num_ids);
    let rules_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(rules),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let left_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&left_index),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let hot_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(hot),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b2i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(byte_to_id),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // A slot owns its own input/output/readback buffers; the rule table and
    // byte map are shared read-only. `num` is padded to the 64-thread tile,
    // `real` is the number of genuine chunks. Output is lane-major, so the
    // readback only needs the first `max_len` lanes plus the length array.
    struct Slot {
        chunks: wgpu::Buffer,
        packed: Vec<u32>,
        out_tokens: wgpu::Buffer,
        out_lens: wgpu::Buffer,
        readback: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
        num: usize,
        real: usize,
        offset: usize,
    }
    let make_slot = |chunk: &[SmallStr], offset: usize| -> Slot {
        let num = chunk.len().next_multiple_of(wg);
        let packed = pack_chunks(chunk, wg);
        let chunks_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let tokens_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (num * rust_gpu_tokenizer_shaders::SMALL_LEN * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let lens_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (num * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((max_len * num + num) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chunks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rules_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: left_index_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b2i_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: hot_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tokens_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: lens_buf.as_entire_binding(),
                },
            ],
        });
        Slot {
            chunks: chunks_buf,
            packed,
            out_tokens: tokens_buf,
            out_lens: lens_buf,
            readback,
            bind_group,
            num,
            real: chunk.len(),
            offset,
        }
    };

    // Compare one slot's readback (lane-major tokens + lengths) to the oracle.
    fn verify_block(
        data: &[u8],
        offset: usize,
        real: usize,
        num: usize,
        max_len: usize,
        expected: &[TokenSeq],
    ) -> usize {
        let lane_bytes = max_len * num * 4;
        let tokens: &[u32] = bytemuck::cast_slice(&data[..lane_bytes]);
        let lens: &[u32] = bytemuck::cast_slice(&data[lane_bytes..lane_bytes + num * 4]);
        let mut mism = 0;
        for i in 0..real {
            let e = &expected[offset + i];
            let len = lens[i] as usize;
            if len != e.len as usize {
                mism += 1;
                continue;
            }
            let mut ok = true;
            for k in 0..len {
                if tokens[k * num + i] != e.tokens[k] {
                    ok = false;
                    break;
                }
            }
            if !ok {
                mism += 1;
            }
        }
        mism
    }

    let iterations: usize = std::env::var("BPE_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // --- serial baseline: one buffer, wait for readback each iteration ---
    let serial = make_slot(chunks, 0);
    let lane_bytes = (max_len * serial.num * 4) as u64;
    let len_bytes = (serial.num * 4) as u64;
    let rb_bytes = lane_bytes + len_bytes;
    for iter in 0..iterations {
        let t = Instant::now();
        // Upload the same as the pipelined path does, so both measure the
        // full streaming cost (new text arriving each batch).
        if !no_upload {
            queue.write_buffer(&serial.chunks, 0, bytemuck::cast_slice(&serial.packed));
        }
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_bind_group(0, &serial.bind_group, &[]);
            cpass.set_pipeline(&pipeline);
            cpass.dispatch_workgroups((serial.num / wg) as u32, 1, 1);
        }
        let mismatches = if no_readback {
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::PollType::wait_indefinitely())?;
            usize::MAX
        } else {
            encoder.copy_buffer_to_buffer(&serial.out_tokens, 0, &serial.readback, 0, lane_bytes);
            encoder.copy_buffer_to_buffer(
                &serial.out_lens,
                0,
                &serial.readback,
                lane_bytes,
                len_bytes,
            );
            queue.submit(Some(encoder.finish()));
            let slice = serial.readback.slice(..rb_bytes);
            slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let m = {
                let data = slice.get_mapped_range()?;
                verify_block(&data, 0, serial.real, serial.num, max_len, expected)
            };
            serial.readback.unmap();
            m
        };
        let elapsed = t.elapsed();
        println!(
            "  serial iter {iter}: {elapsed:.3?} ({:.2} Mchunks/s) {}",
            chunks.len() as f64 / elapsed.as_secs_f64() / 1e6,
            if mismatches == 0 {
                "OK".to_string()
            } else if mismatches == usize::MAX {
                "SKIP".to_string()
            } else {
                format!("{mismatches} MISMATCHES")
            }
        );
    }
    drop(serial);

    // --- pipelined: stream chunk slices through a ring of slots so uploads,
    // dispatch and readback overlap ---
    let pipe_slots: usize = std::env::var("BPE_SLOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let per = chunks.len().div_ceil(pipe_slots);
    let slices: Vec<&[SmallStr]> = chunks.chunks(per).collect();
    let mut offset = 0usize;
    let slots: Vec<Slot> = slices
        .iter()
        .map(|s| {
            let slot = make_slot(s, offset);
            offset += s.len();
            slot
        })
        .collect();
    let slot_count = slots.len();
    eprintln!("pipelined: {} slots of ~{} chunks", slot_count, per);

    let (tx, rx) = std::sync::mpsc::channel::<usize>();
    let mut total = std::time::Duration::ZERO;
    for iter in 0..iterations {
        let t = Instant::now();
        for (si, slot) in slots.iter().enumerate() {
            // Upload (queue-ordered after this slot's previous use), dispatch,
            // and arm the readback; none of these block.
            if !no_upload {
                queue.write_buffer(&slot.chunks, 0, bytemuck::cast_slice(&slot.packed));
            }
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());
                cpass.set_bind_group(0, &slot.bind_group, &[]);
                cpass.set_pipeline(&pipeline);
                cpass.dispatch_workgroups((slot.num / wg) as u32, 1, 1);
            }
            let lb = (max_len * slot.num * 4) as u64;
            let nb = (slot.num * 4) as u64;
            encoder.copy_buffer_to_buffer(&slot.out_tokens, 0, &slot.readback, 0, lb);
            encoder.copy_buffer_to_buffer(&slot.out_lens, 0, &slot.readback, lb, nb);
            queue.submit(Some(encoder.finish()));
            let tx = tx.clone();
            slot.readback
                .slice(..lb + nb)
                .map_async(wgpu::MapMode::Read, move |_| {
                    let _ = tx.send(si);
                });
            // Drive completed map callbacks while the GPU keeps working.
            let _ = device.poll(wgpu::PollType::Poll);
        }

        // Drain this iteration's readbacks (barrier between iterations).
        let mut mismatches = 0usize;
        let mut remaining = slot_count;
        while remaining > 0 {
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .context("waiting for readback")?;
            while let Ok(si) = rx.try_recv() {
                let slot = &slots[si];
                let rb = ((max_len * slot.num + slot.num) * 4) as u64;
                let data = slot.readback.slice(..rb).get_mapped_range()?;
                mismatches +=
                    verify_block(&data, slot.offset, slot.real, slot.num, max_len, expected);
                drop(data);
                slot.readback.unmap();
                remaining -= 1;
            }
        }
        let elapsed = t.elapsed();
        total += elapsed;
        println!(
            "  pipelined iter {iter}: {elapsed:.3?} ({:.2} Mchunks/s) {}",
            chunks.len() as f64 / elapsed.as_secs_f64() / 1e6,
            if mismatches == 0 {
                "OK".to_string()
            } else {
                format!("{mismatches} MISMATCHES")
            }
        );
    }
    let avg = total / iterations as u32;
    let io_bytes = chunks.len() as f64
        * (rust_gpu_tokenizer_shaders::PACKED_WORDS * 4 + max_len * 4 + 4) as f64;
    println!(
        "pipelined: {iterations} iters, avg {avg:.3?}, {:.2} Mchunks/s, {:.2} GB/s host<->device",
        chunks.len() as f64 / avg.as_secs_f64() / 1e6,
        2.0 * io_bytes / avg.as_secs_f64() / 1e9,
    );
    Ok(())
}

fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        count: None,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Streaming parquet tokenization (performance run; tokens are discarded).
// ---------------------------------------------------------------------------

/// Chunks per GPU submission. Fixed so buffers are allocated once and the
/// kernel's `num` (= packed.len()/PACKED_WORDS) stays constant.
const STREAM_BATCH_CHUNKS: usize = 1 << 19;

/// Feed each row of the `text` column to `f`.
struct StreamGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    chunks_buf: wgpu::Buffer,
    out_tokens: wgpu::Buffer,
    out_lens: wgpu::Buffer,
    lens_readback: wgpu::Buffer,
    packed: Vec<u32>,
    num: usize,
    wg: usize,
    t_pack: std::time::Duration,
    t_sync: std::time::Duration,
}

impl StreamGpu {
    async fn new(
        rules: &[MergeRule],
        hot: &[MergeRule],
        byte_to_id: &[u32; NUM_BASE],
        num_ids: usize,
        batch_chunks: usize,
    ) -> anyhow::Result<Self> {
        use anyhow::Context as _;
        use wgpu::util::DeviceExt;

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None).await?;
        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_compute_workgroup_size_x: 1024.min(adapter_limits.max_compute_workgroup_size_x),
            max_compute_workgroup_size_y: 1024.min(adapter_limits.max_compute_workgroup_size_y),
            max_compute_invocations_per_workgroup: 1024
                .min(adapter_limits.max_compute_invocations_per_workgroup),
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Default::default(),
            })
            .await
            .context("request device")?;

        let module =
            device.create_shader_module(wgpu::include_spirv!(env!("WGPU_KERNEL_SPV_PATH")));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(0, true),
                bgl_entry(1, true),
                bgl_entry(2, true),
                bgl_entry(3, true),
                bgl_entry(4, true),
                bgl_entry(5, false),
                bgl_entry(6, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let wg: usize = std::env::var("BPE_WG")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);
        let entry = format!("bpe{wg}");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(&entry),
            compilation_options: Default::default(),
            cache: None,
        });

        let num = batch_chunks.next_multiple_of(wg);
        let packed = vec![0u32; num * rust_gpu_tokenizer_shaders::PACKED_WORDS];
        let chunks_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_tokens = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (num * rust_gpu_tokenizer_shaders::SMALL_LEN * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_lens = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (num * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let lens_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (num * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let left_index = build_left_index(rules, num_ids);
        let rules_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(rules),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let left_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&left_index),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let hot_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(hot),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b2i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(byte_to_id),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chunks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rules_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: left_index_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b2i_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: hot_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_tokens.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_lens.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            chunks_buf,
            out_tokens,
            out_lens,
            lens_readback,
            packed,
            num,
            wg,
            t_pack: std::time::Duration::ZERO,
            t_sync: std::time::Duration::ZERO,
        })
    }

    /// Encode `pretokens` (at most `self.num`) and return the token count.
    /// The token ids themselves are left on the device and discarded.
    async fn encode(&mut self, pretokens: &[SmallStr]) -> anyhow::Result<u64> {
        use anyhow::Context as _;
        debug_assert!(pretokens.len() <= self.num);

        let tp = Instant::now();
        // Cover the whole dispatched grid (rounded up to the workgroup size)
        // and zero it, so padding slots read len = 0 rather than stale data.
        let groups = pretokens.len().div_ceil(self.wg);
        let covered = groups * self.wg;
        let used = covered * rust_gpu_tokenizer_shaders::PACKED_WORDS;
        self.packed[..used].fill(0);
        for (index, chunk) in pretokens.iter().enumerate() {
            let base = index * rust_gpu_tokenizer_shaders::PACKED_WORDS;
            let len = chunk.len as usize;
            for i in 0..len {
                let b = chunk.bytes[i] & 0xFF;
                self.packed[base + i / 4] |= b << ((i % 4) * 8);
            }
            self.packed[base + 5] |= (len as u32) << 16;
        }
        // Upload only the chunks actually present; the kernel's output stride
        // still uses the fixed buffer length, which is fine.
        self.queue.write_buffer(
            &self.chunks_buf,
            0,
            bytemuck::cast_slice(&self.packed[..used]),
        );
        self.t_pack += tp.elapsed();

        let ts = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_pipeline(&self.pipeline);
            // Only dispatch the workgroups covering real chunks.
            cpass.dispatch_workgroups(groups as u32, 1, 1);
        }
        let bytes = (pretokens.len() * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.out_lens, 0, &self.lens_readback, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.lens_readback.slice(..bytes);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("waiting for readback")?;
        let total: u64 = {
            let data = slice.get_mapped_range()?;
            let lens: &[u32] = bytemuck::cast_slice(&data);
            lens[..pretokens.len()].iter().map(|&l| l as u64).sum()
        };
        self.lens_readback.unmap();
        self.t_sync += ts.elapsed();
        Ok(total)
    }
    /// Like [`Self::encode`] but returns the flattened token ids (chunk-major
    /// order) so callers can diff backends. Intended for small slices.
    async fn encode_tokens(&mut self, pretokens: &[SmallStr]) -> anyhow::Result<Vec<u32>> {
        use anyhow::Context as _;
        debug_assert!(pretokens.len() <= self.num);

        let groups = pretokens.len().div_ceil(self.wg);
        let covered = groups * self.wg;
        let used = covered * rust_gpu_tokenizer_shaders::PACKED_WORDS;
        self.packed[..used].fill(0);
        for (index, chunk) in pretokens.iter().enumerate() {
            let base = index * rust_gpu_tokenizer_shaders::PACKED_WORDS;
            let len = chunk.len as usize;
            for i in 0..len {
                let b = chunk.bytes[i] & 0xFF;
                self.packed[base + i / 4] |= b << ((i % 4) * 8);
            }
            self.packed[base + 5] |= (len as u32) << 16;
        }
        self.queue.write_buffer(
            &self.chunks_buf,
            0,
            bytemuck::cast_slice(&self.packed[..used]),
        );

        let lens_bytes = (pretokens.len() * 4) as u64;
        let tokens_bytes = (self.num * rust_gpu_tokenizer_shaders::SMALL_LEN * 4) as u64;
        let tokens_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: tokens_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_pipeline(&self.pipeline);
            cpass.dispatch_workgroups(groups as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_lens, 0, &self.lens_readback, 0, lens_bytes);
        encoder.copy_buffer_to_buffer(&self.out_tokens, 0, &tokens_readback, 0, tokens_bytes);
        self.queue.submit(Some(encoder.finish()));

        let lens_slice = self.lens_readback.slice(..lens_bytes);
        lens_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        let tokens_slice = tokens_readback.slice(..);
        tokens_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("waiting for readback")?;

        let lens: Vec<u32> = {
            let data = lens_slice.get_mapped_range()?;
            bytemuck::cast_slice(&data).to_vec()
        };
        let mut out = Vec::new();
        {
            let data = tokens_slice.get_mapped_range()?;
            let tokens: &[u32] = bytemuck::cast_slice(&data);
            for i in 0..pretokens.len() {
                let len = lens[i] as usize;
                for k in 0..len {
                    out.push(tokens[k * self.num + i]);
                }
            }
        }
        self.lens_readback.unmap();
        tokens_readback.unmap();
        Ok(out)
    }
}

/// Compare the actual token ids produced by the CPU, wgpu and ash backends on
/// the first `TOKENIZE_COMPARE` tokens (default 1000) of the file.
fn compare_tokens(path: &str) -> anyhow::Result<()> {
    let n: usize = std::env::var("TOKENIZE_COMPARE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let vocab = load_tkv(concat!(env!("CARGO_MANIFEST_DIR"), "/qwen2.tkv"));
    let (rules, _rank) = reconstruct_rules(&vocab);
    let byte_to_id = byte_to_id_table(&vocab);
    let left_index = build_left_index(&rules, vocab.len());

    let sample = read_text_sample(path, 2 << 20)?;
    let all = pre_tokenize(&sample);
    // Each pretoken yields at least one token, so n + slack chunks suffice.
    let m = (n + 512).min(all.len());
    let pre = &all[..m];

    let counts = count_rule_hits(pre, &rules, &left_index, &byte_to_id);
    let hot = build_hot_table(&counts, 262144);

    // CPU, in order.
    let mut cpu = Vec::new();
    for &c in pre {
        let s = bpe_encode(c, &rules, &left_index, &hot, &byte_to_id);
        cpu.extend_from_slice(&s.tokens[..s.len as usize]);
        if cpu.len() >= n {
            break;
        }
    }
    cpu.truncate(n);

    let tile = m.next_multiple_of(256);

    let mut gpu = pollster::block_on(StreamGpu::new(&rules, &hot, &byte_to_id, vocab.len(), tile))?;
    let mut gpu_tokens = pollster::block_on(gpu.encode_tokens(pre))?;
    gpu_tokens.truncate(n);

    let mut ash = ash_backend::AshStream::new(&rules, &hot, &byte_to_id, vocab.len(), tile)?;
    let mut ash_tokens = ash.encode_tokens(pre)?;
    ash_tokens.truncate(n);

    println!("comparing first {} tokens from {} pretokens", cpu.len(), m);
    let mut ok = compare_seq("wgpu", &cpu, &gpu_tokens);
    ok &= compare_seq("ash", &cpu, &ash_tokens);
    if ok {
        println!("all backends identical");
    }
    let k = cpu.len().min(20);
    println!("first {k} tokens: {:?}", &cpu[..k]);
    Ok(())
}

fn compare_seq(name: &str, cpu: &[u32], other: &[u32]) -> bool {
    if cpu == other {
        println!("  {name}: identical ({} tokens)", cpu.len());
        true
    } else {
        let idx = cpu
            .iter()
            .zip(other)
            .position(|(a, b)| a != b)
            .unwrap_or(cpu.len().min(other.len()));
        println!(
            "  {name}: MISMATCH at token {idx} (cpu {:?} vs {name} {:?}); lengths cpu={} {name}={}",
            cpu.get(idx),
            other.get(idx),
            cpu.len(),
            other.len()
        );
        false
    }
}

fn tokenize_parquet(path: &str, mode: &str) -> anyhow::Result<()> {
    use std::time::Duration;

    if mode == "compare" {
        return compare_tokens(path);
    }

    let do_cpu = matches!(mode, "cpu" | "both" | "all");
    let do_gpu = matches!(mode, "gpu" | "both" | "all");
    let do_ash = matches!(mode, "ash" | "all");
    anyhow::ensure!(
        do_cpu || do_gpu || do_ash,
        "unknown mode {mode:?} (expected gpu, ash, cpu, both, or all)"
    );

    let t = Instant::now();
    let vocab = load_tkv(concat!(env!("CARGO_MANIFEST_DIR"), "/qwen2.tkv"));
    let (rules, _rank) = reconstruct_rules(&vocab);
    let byte_to_id = byte_to_id_table(&vocab);
    let left_index = build_left_index(&rules, vocab.len());
    println!(
        "vocab: {} tokens -> {} rules in {:?}",
        vocab.len(),
        rules.len(),
        t.elapsed()
    );

    // Profile the hot/negative cache on the first few MB of the file.
    let t = Instant::now();
    let sample = read_text_sample(path, 8 << 20)?;
    let sample_pt = pre_tokenize(&sample);
    let counts = count_rule_hits(&sample_pt, &rules, &left_index, &byte_to_id);
    let hot_slots: usize = std::env::var("BPE_HOT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(262144);
    let hot = build_hot_table(&counts, hot_slots);
    println!(
        "hot: {} slots from {} sample chunks in {:?}",
        hot.iter().filter(|r| r.left != u32::MAX).count(),
        sample_pt.len(),
        t.elapsed()
    );
    drop(sample_pt);
    drop(sample);

    let max_rows: u64 = std::env::var("TOKENIZE_MAX_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(u64::MAX);
    let rx = spawn_pretokenizer(path.to_string(), max_rows, 128);

    pollster::block_on(async {
        let stream_chunks: usize = std::env::var("BPE_STREAM_CHUNKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(STREAM_BATCH_CHUNKS);
        let mut gpu = if do_gpu {
            Some(StreamGpu::new(&rules, &hot, &byte_to_id, vocab.len(), stream_chunks).await?)
        } else {
            None
        };
        let mut ash = if do_ash {
            Some(ash_backend::AshStream::new(
                &rules,
                &hot,
                &byte_to_id,
                vocab.len(),
                STREAM_BATCH_CHUNKS,
            )?)
        } else {
            None
        };
        let batch_num = gpu
            .as_ref()
            .map(|g| g.num)
            .or_else(|| ash.as_ref().map(|a| a.num))
            .unwrap_or(STREAM_BATCH_CHUNKS);

        let mut rows = 0u64;
        let mut bytes = 0u64;
        let mut chunks = 0u64;
        let mut tokens_cpu = 0u64;
        let mut tokens_gpu = 0u64;
        let mut tokens_ash = 0u64;
        let mut pending: Vec<SmallStr> = Vec::with_capacity(batch_num);
        let mut pending_ash: Vec<SmallStr> = Vec::with_capacity(batch_num);
        let mut t_decode = Duration::ZERO;
        let mut t_pre = Duration::ZERO;
        let mut t_cpu = Duration::ZERO;
        let mut t_gpu = Duration::ZERO;
        let mut t_ash = Duration::ZERO;
        let start = Instant::now();

        loop {
            let msg = match rx.recv() {
                Ok(m) => m,
                Err(_) => break, // producer finished
            };
            let b = msg?;
            rows += b.rows as u64;
            bytes += b.bytes;
            chunks += b.pretokens.len() as u64;
            t_decode += b.t_decode;
            t_pre += b.t_pre;

            if do_cpu {
                let tc = Instant::now();
                tokens_cpu += b
                    .pretokens
                    .par_iter()
                    .map(|&c| bpe_encode(c, &rules, &left_index, &hot, &byte_to_id).len as u64)
                    .sum::<u64>();
                t_cpu += tc.elapsed();
            }

            if let Some(gpu) = gpu.as_mut() {
                let tg = Instant::now();
                let num = gpu.num;
                // Accumulate into full sub-batches to amortize GPU round-trips.
                pending.extend_from_slice(&b.pretokens);
                while pending.len() >= num {
                    tokens_gpu += gpu.encode(&pending[..num]).await?;
                    let keep = pending.len() - num;
                    pending.copy_within(num.., 0);
                    pending.truncate(keep);
                }
                t_gpu += tg.elapsed();
            }

            if let Some(ash) = ash.as_mut() {
                let ta = Instant::now();
                let num = ash.num;
                pending_ash.extend_from_slice(&b.pretokens);
                while pending_ash.len() >= num {
                    tokens_ash += ash.encode(&pending_ash[..num])?;
                    let keep = pending_ash.len() - num;
                    pending_ash.copy_within(num.., 0);
                    pending_ash.truncate(keep);
                }
                t_ash += ta.elapsed();
            }
        }

        if let Some(ash) = ash.as_mut()
            && !pending_ash.is_empty()
        {
            let ta = Instant::now();
            tokens_ash += ash.encode(&pending_ash)?;
            t_ash += ta.elapsed();
        }

        if let Some(gpu) = gpu.as_mut()
            && !pending.is_empty()
        {
            let tg = Instant::now();
            tokens_gpu += gpu.encode(&pending).await?;
            t_gpu += tg.elapsed();
        }

        let elapsed = start.elapsed();
        let secs = elapsed.as_secs_f64();
        println!(
            "rows: {rows}, text: {:.1} MB, chunks: {chunks}",
            bytes as f64 / 1e6
        );
        let total_tokens = if do_cpu {
            tokens_cpu
        } else if do_gpu {
            tokens_gpu
        } else {
            tokens_ash
        };
        println!("tokens: {total_tokens}");
        println!("decode {t_decode:?}, pretokenize {t_pre:?}");
        if do_cpu {
            let cs = t_cpu.as_secs_f64();
            println!(
                "cpu (rayon, {} threads): {tokens_cpu} tokens in {t_cpu:?} ({:.2} Mchunks/s, {:.2} Mtokens/s)",
                rayon::current_num_threads(),
                chunks as f64 / cs / 1e6,
                tokens_cpu as f64 / cs / 1e6,
            );
        }
        if let Some(gpu) = gpu.as_ref() {
            let gs = t_gpu.as_secs_f64();
            println!(
                "gpu (wg{}): {tokens_gpu} tokens in {t_gpu:?} ({:.2} Mchunks/s, {:.2} Mtokens/s)",
                gpu.wg,
                chunks as f64 / gs / 1e6,
                tokens_gpu as f64 / gs / 1e6,
            );
            println!(
                "gpu internals: pack+upload {:?}, submit+sync {:?}",
                gpu.t_pack, gpu.t_sync
            );
        }
        if let Some(ash) = ash.as_ref() {
            let as_ = t_ash.as_secs_f64();
            println!(
                "ash (wg{}): {tokens_ash} tokens in {t_ash:?} ({:.2} Mchunks/s, {:.2} Mtokens/s)",
                ash.wg,
                chunks as f64 / as_ / 1e6,
                tokens_ash as f64 / as_ / 1e6,
            );
        }
        let mut counts = Vec::new();
        if do_cpu {
            counts.push(("cpu", tokens_cpu));
        }
        if do_gpu {
            counts.push(("gpu", tokens_gpu));
        }
        if do_ash {
            counts.push(("ash", tokens_ash));
        }
        if counts.len() > 1 {
            let all_match = counts.iter().all(|&(_, c)| c == counts[0].1);
            println!("token counts match: {all_match} ({counts:?})");
        }
        println!(
            "wall: {elapsed:?} ({:.2} Mchunks/s, {:.1} MB/s)",
            chunks as f64 / secs / 1e6,
            bytes as f64 / secs / 1e6,
        );
        Ok::<(), anyhow::Error>(())
    })
}
