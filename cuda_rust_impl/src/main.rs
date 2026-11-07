//! CUDA (cuda-oxide) BPE tokenizer backend.
//!
//! The device core is shared with the rust-gpu backend through `bpe-core`;
//! this crate only adds the `#[kernel]` entry point and the host harness
//! (vocab/rules/hot cache and parquet streaming come from `tokenizer-host`).
//!
//! Build and run on a Linux host with an NVIDIA GPU:
//!   cd cuda_rust_impl && cargo oxide run
//!   cargo oxide run -- parquet ../sample_data.parquet both

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::anyhow;
use bpe_core::{MergeRule, PACKED_WORDS, SMALL_LEN, SmallStr, bpe_encode, unpack_chunk};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread};
use rayon::prelude::*;
use tokenizer_host::{
    build_hot_table, build_left_index, byte_to_id_table, count_rule_hits, load_tkv, pre_tokenize,
    read_text_sample, reconstruct_rules, spawn_pretokenizer,
};

const VOCAB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../rust-gpu-tokenizer/qwen2.tkv"
);
const SHAKESPEARE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../rust-gpu-tokenizer/shakespeare.txt"
);
const DEFAULT_PARQUET: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../sample_data.parquet");

#[cuda_module]
mod kernels {
    use super::*;

    /// One invocation encodes one pretokenized chunk. The output is
    /// lane-major (`out_tokens[k * num + index]`) so only the used lanes are
    /// written; `out_lens[index]` is the token count.
    #[kernel]
    pub fn bpe_encode_chunks(
        packed: &[u32],
        rules: &[MergeRule],
        left_index: &[u32],
        byte_to_id: &[u32],
        hot: &[MergeRule],
        out_tokens: &mut [u32],
        out_lens: &mut [u32],
    ) {
        let index = thread::index_1d().get();
        let num = packed.len() / PACKED_WORDS;
        if index >= num {
            return;
        }
        let seq = bpe_encode(
            unpack_chunk(packed, index),
            rules,
            left_index,
            hot,
            byte_to_id,
        );
        let mut k = 0;
        while k < seq.len as usize {
            out_tokens[k * num + index] = seq.tokens[k];
            k += 1;
        }
        out_lens[index] = seq.len;
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Upload a rule table as raw `u32` words, then reinterpret the device buffer
/// as `MergeRule`. This sidesteps `DeviceBuffer::from_host`'s `DeviceCopy`
/// bound: `MergeRule` is `Pod`, so the cast is layout-preserving.
fn upload_rules(
    stream: &CudaStream,
    rules: &[MergeRule],
) -> anyhow::Result<DeviceBuffer<MergeRule>> {
    let mut words = Vec::with_capacity(rules.len() * 4);
    for rule in rules {
        words.extend_from_slice(&[rule.left, rule.right, rule.rank, rule.new_id]);
    }
    let buf = DeviceBuffer::<u32>::from_host(stream, &words)?;
    buf.cast_chunks::<MergeRule>()
        .map_err(|_| anyhow!("rule buffer is not layout-compatible with MergeRule"))
}

struct CudaGpu {
    /// Kept alive for the module and buffers below.
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    packed: DeviceBuffer<u32>,
    rules: DeviceBuffer<MergeRule>,
    left_index: DeviceBuffer<u32>,
    byte_to_id: DeviceBuffer<u32>,
    hot: DeviceBuffer<MergeRule>,
    out_tokens: DeviceBuffer<u32>,
    out_lens: DeviceBuffer<u32>,
    packed_host: Vec<u32>,
    lens_host: Vec<u32>,
    tokens_host: Vec<u32>,
    pub num: usize,
    pub wg: usize,
}

impl CudaGpu {
    fn new(
        rules: &[MergeRule],
        hot: &[MergeRule],
        byte_to_id: &[u32],
        num_ids: usize,
        batch_chunks: usize,
    ) -> anyhow::Result<Self> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let wg = env_usize("BPE_WG", 256).clamp(1, 1024);
        let num = batch_chunks.max(1).next_multiple_of(wg);

        let module = kernels::load(&ctx)?;

        let left_index_host = build_left_index(rules, num_ids);
        let rules = upload_rules(&stream, rules)?;
        let hot = upload_rules(&stream, hot)?;
        let left_index = DeviceBuffer::from_host(&stream, &left_index_host)?;
        let byte_to_id = DeviceBuffer::from_host(&stream, byte_to_id)?;

        let packed = DeviceBuffer::<u32>::zeroed(&stream, num * PACKED_WORDS)?;
        let out_tokens = DeviceBuffer::<u32>::zeroed(&stream, num * SMALL_LEN)?;
        let out_lens = DeviceBuffer::<u32>::zeroed(&stream, num)?;

        Ok(Self {
            _ctx: ctx,
            stream,
            module,
            packed,
            rules,
            left_index,
            byte_to_id,
            hot,
            out_tokens,
            out_lens,
            packed_host: vec![0; num * PACKED_WORDS],
            lens_host: vec![0; num],
            tokens_host: vec![0; num * SMALL_LEN],
            num,
            wg,
        })
    }

    /// Pack and launch `pretokens` (at most `self.num`), without waiting.
    fn dispatch(&mut self, pretokens: &[SmallStr]) -> anyhow::Result<()> {
        assert!(pretokens.len() <= self.num, "batch exceeds device buffers");
        let groups = pretokens.len().div_ceil(self.wg);
        let used = groups * self.wg;
        if groups == 0 {
            return Ok(());
        }

        let words = used * PACKED_WORDS;
        self.packed_host[..words].fill(0);
        for (index, chunk) in pretokens.iter().enumerate() {
            let base = index * PACKED_WORDS;
            let len = chunk.len as usize;
            for i in 0..len {
                let b = chunk.bytes[i] & 0xFF;
                self.packed_host[base + i / 4] |= b << ((i % 4) * 8);
            }
            self.packed_host[base + 5] |= (len as u32) << 16;
        }

        // `copy_from_host` requires an equal-length source, so upload the
        // whole (fixed-size) buffer; the unlaunched tail is never read.
        self.packed
            .copy_from_host(&self.stream, &self.packed_host)?;

        let config = LaunchConfig {
            grid_dim: (groups as u32, 1, 1),
            block_dim: (self.wg as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the grid covers exactly `used` chunks (<= self.num), the
        // rule/index tables are immutable, and both output buffers are sized
        // for `self.num` chunks.
        unsafe {
            self.module.bpe_encode_chunks(
                &self.stream,
                config,
                &self.packed,
                &self.rules,
                &self.left_index,
                &self.byte_to_id,
                &self.hot,
                &mut self.out_tokens,
                &mut self.out_lens,
            )?;
        }
        Ok(())
    }

    /// Encode at most `self.num` pretokens; returns the token count.
    fn encode(&mut self, pretokens: &[SmallStr]) -> anyhow::Result<u64> {
        self.dispatch(pretokens)?;
        if pretokens.is_empty() {
            return Ok(0);
        }
        self.out_lens
            .copy_to_host(&self.stream, &mut self.lens_host)?;
        Ok(self.lens_host[..pretokens.len()]
            .iter()
            .map(|&l| l as u64)
            .sum())
    }

    /// Like [`Self::encode`] but returns the flattened token ids (chunk-major).
    fn encode_tokens(&mut self, pretokens: &[SmallStr]) -> anyhow::Result<Vec<u32>> {
        self.dispatch(pretokens)?;
        if pretokens.is_empty() {
            return Ok(Vec::new());
        }
        self.out_lens
            .copy_to_host(&self.stream, &mut self.lens_host)?;
        self.out_tokens
            .copy_to_host(&self.stream, &mut self.tokens_host)?;
        let mut out = Vec::new();
        for i in 0..pretokens.len() {
            let len = self.lens_host[i] as usize;
            for k in 0..len {
                out.push(self.tokens_host[k * self.num + i]);
            }
        }
        Ok(out)
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("parquet") {
        let (path, mode) = match args.get(2).map(String::as_str) {
            Some(m @ ("cuda" | "gpu" | "cpu" | "both" | "all" | "compare")) => (DEFAULT_PARQUET, m),
            Some(p) => (p, args.get(3).map(String::as_str).unwrap_or("cuda")),
            None => (DEFAULT_PARQUET, "cuda"),
        };
        return tokenize_parquet(path, mode);
    }
    benchmark()
}

fn benchmark() -> anyhow::Result<()> {
    let text = std::fs::read_to_string(SHAKESPEARE)?;
    let t = Instant::now();
    let pre = pre_tokenize(&text);
    println!("pretokenize: {} chunks in {:?}", pre.len(), t.elapsed());

    let vocab = load_tkv(VOCAB);
    let (rules, _rank) = reconstruct_rules(&vocab);
    let byte_to_id = byte_to_id_table(&vocab);
    let left_index = build_left_index(&rules, vocab.len());
    println!("vocab: {} tokens -> {} rules", vocab.len(), rules.len());

    let counts = count_rule_hits(&pre, &rules, &left_index, &byte_to_id);
    let hot = build_hot_table(&counts, env_usize("BPE_HOT", 262144));

    let t = Instant::now();
    let cpu: u64 = pre
        .par_iter()
        .map(|&c| bpe_encode(c, &rules, &left_index, &hot, &byte_to_id).len as u64)
        .sum();
    let cpu_elapsed = t.elapsed();
    println!(
        "cpu (rayon, {} threads): {cpu} tokens in {cpu_elapsed:?} ({:.2} Mchunks/s)",
        rayon::current_num_threads(),
        pre.len() as f64 / cpu_elapsed.as_secs_f64() / 1e6,
    );

    if pre.is_empty() {
        return Ok(());
    }
    let mut gpu = CudaGpu::new(&rules, &hot, &byte_to_id, vocab.len(), pre.len())?;
    let t = Instant::now();
    let gpu_tokens = gpu.encode(&pre)?;
    let gpu_elapsed = t.elapsed();
    println!(
        "cuda (wg{}): {gpu_tokens} tokens in {gpu_elapsed:?} ({:.2} Mchunks/s)",
        gpu.wg,
        pre.len() as f64 / gpu_elapsed.as_secs_f64() / 1e6,
    );
    println!("token counts match: {}", cpu == gpu_tokens);
    Ok(())
}

fn compare_tokens(path: &str) -> anyhow::Result<()> {
    let n = env_usize("TOKENIZE_COMPARE", 1000);

    let vocab = load_tkv(VOCAB);
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
    let mut gpu = CudaGpu::new(&rules, &hot, &byte_to_id, vocab.len(), tile)?;
    let mut gpu_tokens = gpu.encode_tokens(pre)?;
    gpu_tokens.truncate(n);

    println!("comparing first {} tokens from {} pretokens", cpu.len(), m);
    let ok = compare_seq("cuda", &cpu, &gpu_tokens);
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
    if mode == "compare" {
        return compare_tokens(path);
    }

    let do_cpu = matches!(mode, "cpu" | "both" | "all");
    let do_cuda = matches!(mode, "cuda" | "gpu" | "both" | "all");
    anyhow::ensure!(
        do_cpu || do_cuda,
        "unknown mode {mode:?} (expected cuda, gpu, cpu, both, all, or compare)"
    );

    let t = Instant::now();
    let vocab = load_tkv(VOCAB);
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
    let hot = build_hot_table(&counts, env_usize("BPE_HOT", 262144));
    println!(
        "hot: {} slots from {} sample chunks in {:?}",
        hot.iter().filter(|r| r.left != u32::MAX).count(),
        sample_pt.len(),
        t.elapsed()
    );
    drop(sample_pt);
    drop(sample);

    let max_rows = env_u64("TOKENIZE_MAX_ROWS", u64::MAX);
    let stream_chunks = env_usize("BPE_STREAM_CHUNKS", 1 << 19);
    let rx = spawn_pretokenizer(path.to_string(), max_rows, 128);

    let mut gpu = if do_cuda {
        Some(CudaGpu::new(
            &rules,
            &hot,
            &byte_to_id,
            vocab.len(),
            stream_chunks,
        )?)
    } else {
        None
    };
    let batch_num = gpu.as_ref().map(|g| g.num).unwrap_or(stream_chunks);

    let mut rows = 0u64;
    let mut bytes = 0u64;
    let mut chunks = 0u64;
    let mut tokens_cpu = 0u64;
    let mut tokens_cuda = 0u64;
    let mut pending: Vec<SmallStr> = Vec::with_capacity(batch_num);
    let mut t_decode = Duration::ZERO;
    let mut t_pre = Duration::ZERO;
    let mut t_cpu = Duration::ZERO;
    let mut t_cuda = Duration::ZERO;
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
                tokens_cuda += gpu.encode(&pending[..num])?;
                let keep = pending.len() - num;
                pending.copy_within(num.., 0);
                pending.truncate(keep);
            }
            t_cuda += tg.elapsed();
        }
    }

    if let Some(gpu) = gpu.as_mut() {
        if !pending.is_empty() {
            let tg = Instant::now();
            tokens_cuda += gpu.encode(&pending)?;
            t_cuda += tg.elapsed();
        }
    }

    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    println!(
        "rows: {rows}, text: {:.1} MB, chunks: {chunks}",
        bytes as f64 / 1e6
    );
    let total_tokens = if do_cpu { tokens_cpu } else { tokens_cuda };
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
        let gs = t_cuda.as_secs_f64();
        println!(
            "cuda (wg{}): {tokens_cuda} tokens in {t_cuda:?} ({:.2} Mchunks/s, {:.2} Mtokens/s)",
            gpu.wg,
            chunks as f64 / gs / 1e6,
            tokens_cuda as f64 / gs / 1e6,
        );
    }
    if do_cpu && do_cuda {
        println!(
            "token counts match: {} (cpu {tokens_cpu}, cuda {tokens_cuda})",
            tokens_cpu == tokens_cuda
        );
    }
    println!(
        "wall: {elapsed:?} ({:.2} Mchunks/s, {:.1} MB/s)",
        chunks as f64 / secs / 1e6,
        bytes as f64 / secs / 1e6,
    );
    Ok(())
}
