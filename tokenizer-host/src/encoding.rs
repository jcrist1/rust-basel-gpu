//! Pre-tokenizer split patterns.
//!
//! Only the Qwen2 pattern is wired up today. The pre-tokenizer itself
//! ([`crate::pre_tokenize`]) is generic over patterns, so other encodings can
//! be added here as needed.

// qwen2 pattern: similar to cl100k but \p{N} matches single digits (not 1-3).
pub const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";
