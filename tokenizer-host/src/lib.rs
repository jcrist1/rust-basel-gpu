//! Host-side BPE tokenizer plumbing shared by every backend.
//!
//! This crate owns the vocab/rule reconstruction, the profile-guided hot
//! cache, the Qwen2 pre-tokenizer and the parquet streaming producer. Device
//! kernels live in `bpe-core`; only the launch/transport layer is per-backend.

pub mod encoding;
pub mod parquet;
pub mod pre_tokenize;
pub mod vocab;

pub use parquet::{PreBatch, read_text_sample, spawn_pretokenizer, with_text_rows};
pub use vocab::{
    bpe_encode_ref, build_hot_table, build_left_index, byte_to_id_table, count_rule_hits, load_tkv,
    pack_chunks, pre_tokenize, reconstruct_rules,
};
