#![no_std]
//! GPU BPE tokenizer kernels.
//!
//! The backend-agnostic core (types, merge lookup, encoder) lives in
//! `bpe-core`; this crate only adds the SPIR-V entry points. One invocation
//! encodes one pretokenized chunk. The merge table, the left-id index and the
//! (profile-guided) hot/negative cache are shared read-only buffers.

use glam::UVec3;
use spirv_std::spirv;

pub use bpe_core::{
    MergeRule, NUM_BASE, PACKED_WORDS, SMALL_LEN, SmallStr, TokenSeq, bpe_encode, find_rule,
    unpack_chunk,
};

/// BPE tokenizer kernel: each invocation encodes one pretokenized chunk.
/// Generated at several workgroup sizes so the host can pick the best one
/// (occupancy vs scheduling trade-off); the entry point is `bpe<N>`.
macro_rules! bpe_kernel {
    ($name:ident, $threads:literal) => {
        #[allow(clippy::too_many_arguments)]
        #[spirv(compute(threads($threads)))]
        pub fn $name(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] packed: &[u32],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rules: &[MergeRule],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] left_index: &[u32],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] byte_to_id: &[u32],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] hot: &[MergeRule],
            // Output is transposed (lane-major): token k of chunk i lives at
            // `k * num + i`, so only the used lanes are written and the host
            // can read back just the lanes it needs. `out_lens[i]` is the
            // token count.
            #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] out_tokens: &mut [u32],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] out_lens: &mut [u32],
        ) {
            let index = id.x as usize;
            let seq = bpe_encode(
                unpack_chunk(packed, index),
                rules,
                left_index,
                hot,
                byte_to_id,
            );
            let num = packed.len() / PACKED_WORDS;
            let mut k = 0;
            while k < seq.len as usize {
                out_tokens[k * num + index] = seq.tokens[k];
                k += 1;
            }
            out_lens[index] = seq.len;
        }
    };
}

bpe_kernel!(bpe32, 32);
bpe_kernel!(bpe64, 64);
bpe_kernel!(bpe128, 128);
bpe_kernel!(bpe256, 256);
bpe_kernel!(bpe512, 512);
bpe_kernel!(bpe1024, 1024);
