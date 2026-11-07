#![no_std]
//! Backend-agnostic BPE core shared by the rust-gpu and CUDA kernels.
//!
//! Everything here is plain `core` Rust: one invocation encodes one
//! pretokenized chunk by repeatedly merging the lowest-rank adjacent pair.
//! The merge table, the left-id index and the (profile-guided) hot/negative
//! cache are read-only slices supplied by the host.

/// Longest pretoken, in characters, that a kernel invocation handles.
pub const SMALL_LEN: usize = 22;

/// Number of base byte tokens.
pub const NUM_BASE: usize = 256;

/// A pretoken: `len` char lanes. For BPE only the low byte of each lane is
/// used (this corpus is ~ASCII; a production version would UTF-8-encode).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct SmallStr {
    pub bytes: [u32; SMALL_LEN],
    pub offset: u32,
    pub len: u32,
}

/// Host-side only: `str::chars` uses internals some device compilers cannot
/// handle, and kernels never build a `SmallStr` from a `&str`.
#[cfg(feature = "std")]
impl SmallStr {
    const fn new() -> Self {
        Self {
            bytes: [0; SMALL_LEN],
            offset: 0,
            len: 0,
        }
    }
}

#[cfg(feature = "std")]
impl From<&str> for SmallStr {
    /// Keeps the *tail* of overlong strings and records how many chars were
    /// skipped in `offset`.
    fn from(value: &str) -> Self {
        let skip = value.chars().count().saturating_sub(SMALL_LEN);
        let mut small = SmallStr::new();
        for (lane, c) in small.bytes.iter_mut().zip(value.chars().skip(skip)) {
            *lane = c as u32;
            small.len += 1;
        }
        small.offset = skip as u32;
        small
    }
}

/// One BPE merge rule: `left right` -> `new_id`, applied in `rank` order
/// (lowest rank first). The table is sorted by (left, right) so the kernel
/// can binary-search it.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct MergeRule {
    pub left: u32,
    pub right: u32,
    pub rank: u32,
    pub new_id: u32,
}

/// Fixed-capacity token id sequence produced by `bpe_encode` (cap = SMALL_LEN:
/// a chunk can only shrink under merging).
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct TokenSeq {
    pub tokens: [u32; SMALL_LEN],
    pub len: u32,
}

#[cfg(feature = "pod")]
unsafe impl bytemuck::Zeroable for SmallStr {}
#[cfg(feature = "pod")]
unsafe impl bytemuck::Pod for SmallStr {}
#[cfg(feature = "pod")]
unsafe impl bytemuck::Zeroable for MergeRule {}
#[cfg(feature = "pod")]
unsafe impl bytemuck::Pod for MergeRule {}
#[cfg(feature = "pod")]
unsafe impl bytemuck::Zeroable for TokenSeq {}
#[cfg(feature = "pod")]
unsafe impl bytemuck::Pod for TokenSeq {}

/// Words per chunk in the byte-packed input layout: 22 one-byte lanes plus a
/// length byte, rounded up to 24 bytes (6 u32 words).
pub const PACKED_WORDS: usize = 6;

/// Unpack a chunk from the byte-packed input: byte lane `i` is byte `i % 4`
/// of word `i / 4`; the length lives in byte 22 (bits 16..24 of word 5).
/// Equivalent to `SmallStr::from(&str)` followed by `lane & 0xFF`.
pub fn unpack_chunk(packed: &[u32], index: usize) -> SmallStr {
    let base = index * PACKED_WORDS;
    let len = (packed[base + 5] >> 16) & 0xFF;
    let mut bytes = [0u32; SMALL_LEN];
    let mut i = 0;
    while i < len as usize {
        let w = packed[base + i / 4];
        bytes[i] = (w >> ((i % 4) * 8)) & 0xFF;
        i += 1;
    }
    SmallStr {
        bytes,
        offset: 0,
        len,
    }
}

/// Two-level lookup: `left_index[left]..left_index[left + 1]` is the slice of
/// the (left, right)-sorted rule table with that left id (usually 0 or 1
/// entries), then binary search that slice by `right`. Field comparisons are
/// spelled out: tuple `<` returns `Ordering` (a repr(i8) enum) which rust-gpu
/// would lower as a u8 constant (Int8 capability).
pub fn find_rule(
    rules: &[MergeRule],
    left_index: &[u32],
    hot: &[MergeRule],
    left: u32,
    right: u32,
) -> Option<MergeRule> {
    // Direct-mapped hot-rule cache: the common pairs never touch the 1.5 MB
    // global rule table. Empty slots hold `left = u32::MAX`, which never
    // matches a real token id.
    if !hot.is_empty() {
        let h = (left.wrapping_mul(0x9E37_79B1) ^ right) as usize & (hot.len() - 1);
        let e = hot[h];
        if e.left == left && e.right == right {
            // A cached *negative* result carries rank == u32::MAX.
            return if e.rank == u32::MAX { None } else { Some(e) };
        }
    }
    let li = left as usize;
    let mut lo = left_index[li] as usize;
    let mut hi = left_index[li + 1] as usize;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let r = rules[mid];
        if r.right < right {
            lo = mid + 1;
        } else if r.right > right {
            hi = mid;
        } else {
            return Some(r);
        }
    }
    None
}

/// BPE-encode one pretokenized chunk: seed with base byte ids (via
/// `byte_to_id`, since the vocab permutes the 256 byte tokens), then
/// repeatedly merge the lowest-rank adjacent pair until no rule applies.
// The nested `if let` is intentional: cuda-oxide's MIR importer handles it
// more reliably than an `&&` let-chain.
#[allow(clippy::collapsible_if)]
pub fn bpe_encode(
    chunk: SmallStr,
    rules: &[MergeRule],
    left_index: &[u32],
    hot: &[MergeRule],
    byte_to_id: &[u32],
) -> TokenSeq {
    // Explicit construction (not `Default`): the array `Default` impl lowers
    // through `MaybeUninit`, which rust-gpu emits with u8 ops (needs Int8).
    let mut seq = TokenSeq {
        tokens: [0u32; SMALL_LEN],
        len: 0,
    };
    let mut n = chunk.len as usize;
    let mut i = 0;
    while i < n {
        seq.tokens[i] = byte_to_id[(chunk.bytes[i] & 0xFF) as usize];
        i += 1;
    }
    loop {
        // Find the lowest-rank adjacent pair.
        let mut best_rank = u32::MAX;
        let mut best_pos = 0usize;
        let mut best_id = 0u32;
        let mut j = 0;
        while j + 1 < n {
            if let Some(rule) = find_rule(rules, left_index, hot, seq.tokens[j], seq.tokens[j + 1])
            {
                if rule.rank < best_rank {
                    best_rank = rule.rank;
                    best_pos = j;
                    best_id = rule.new_id;
                }
            }
            j += 1;
        }
        if best_rank == u32::MAX {
            break;
        }
        // Merge at best_pos: replace the pair with the merged id, shift left.
        seq.tokens[best_pos] = best_id;
        let mut k = best_pos + 1;
        while k + 1 < n {
            seq.tokens[k] = seq.tokens[k + 1];
            k += 1;
        }
        n -= 1;
    }
    seq.len = n as u32;
    seq
}
