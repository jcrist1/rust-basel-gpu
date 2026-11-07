//! Vocab loading, merge-rule reconstruction and the profile-guided hot cache.
//!
//! Shared by every backend: the GPU kernels consume exactly the tables built
//! here, and the CPU oracle uses the same `bpe_encode` from `bpe-core`.

use std::collections::HashMap;
use std::sync::OnceLock;

use bpe_core::{MergeRule, NUM_BASE, PACKED_WORDS, SMALL_LEN, SmallStr, find_rule};

use crate::encoding::QWEN2_PATTERN;
use crate::pre_tokenize::{FastPath, PreTokenizer, RegexPreTokenizer, WhitespaceRules};

/// Pretokenize `text` with the Qwen2 split pattern.
pub fn pre_tokenize(text: &str) -> Vec<SmallStr> {
    // Compile the regex once; `pre_tokenize` is called per row in the
    // streaming path and `Regex::new` is far too expensive to repeat.
    static PRE_TOKENIZER: OnceLock<RegexPreTokenizer> = OnceLock::new();
    let pre_tokenizer = PRE_TOKENIZER.get_or_init(|| {
        RegexPreTokenizer::new(
            QWEN2_PATTERN,
            FastPath::Qwen2,
            WhitespaceRules::NewlineFirst,
        )
    });
    // Average pretoken is ~4-5 bytes; preallocate to avoid regrowth.
    let mut out = Vec::with_capacity(text.len() / 5 + 8);
    let mut pos = 0;
    while let Some((start, end)) = pre_tokenizer.next_match(text, pos) {
        match text.get(start..end) {
            Some(piece) => out.push(SmallStr::from(piece)),
            None => break,
        }
        pos = end;
    }
    out
}

/// Load a TKV1 vocab: 13-byte header (magic, u32 vocab_size, u32, u8),
/// u8 length table, concatenated token bytes. The byte blob ends exactly at
/// EOF.
pub fn load_tkv(path: &str) -> Vec<Vec<u8>> {
    let data = std::fs::read(path).expect("read tkv");
    assert_eq!(&data[0..4], b"TKV1", "bad magic");
    let vocab_size = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let lens = &data[13..13 + vocab_size];
    let mut off = 13 + vocab_size;
    let mut tokens = Vec::with_capacity(vocab_size);
    for &l in lens {
        tokens.push(data[off..off + l as usize].to_vec());
        off += l as usize;
    }
    assert_eq!(off, data.len(), "trailing bytes in tkv");
    tokens
}

/// Byte value -> base token id. The first 255 vocab entries are the single
/// bytes in bytes_to_unicode order; this particular vocab has no single-byte
/// 0xAD token (only the merged "\xad "), so that entry is left at 0.
pub fn byte_to_id_table(tokens: &[Vec<u8>]) -> [u32; NUM_BASE] {
    let mut table = [0u32; NUM_BASE];
    let mut seen = [false; NUM_BASE];
    for (id, t) in tokens.iter().take(NUM_BASE).enumerate() {
        if t.len() == 1 {
            table[t[0] as usize] = id as u32;
            seen[t[0] as usize] = true;
        }
    }
    let missing: Vec<usize> = (0..NUM_BASE).filter(|&b| !seen[b]).collect();
    if !missing.is_empty() {
        eprintln!(
            "warning: no single-byte token for {:?} (left at id 0)",
            missing
        );
    }
    table
}

/// Reconstruct the merge table. Token ids are *not* merge ranks in this file
/// (the vocab is not dependency-ordered), so derive ranks topologically:
/// each merged token is the concatenation of two earlier tokens, and a rule
/// may only fire after its parts exist. A token may split into two existing
/// tokens in several ways; we emit a rule for *every* valid split (same rank,
/// same new id) so the table is exactly equivalent to the byte-space
/// reference encoder. Ranks are relaxed to `max(part ranks) + 1` until stable.
pub fn reconstruct_rules(tokens: &[Vec<u8>]) -> (Vec<MergeRule>, Vec<u32>) {
    let mut bytes_to_id: HashMap<&[u8], u32> = HashMap::new();
    for (id, t) in tokens.iter().enumerate() {
        bytes_to_id.entry(t.as_slice()).or_insert(id as u32);
    }
    let n = tokens.len();
    // All valid splits per token id (empty for base tokens, duplicates, and
    // tokens that cannot be decomposed into two existing tokens).
    let mut splits: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
    let mut unsplittable = 0usize;
    for id in NUM_BASE..n {
        let tok = &tokens[id];
        if bytes_to_id[tok.as_slice()] != id as u32 {
            continue; // duplicate bytes: the first occurrence owns the rules
        }
        for i in 1..tok.len() {
            if let (Some(&l), Some(&r)) = (bytes_to_id.get(&tok[..i]), bytes_to_id.get(&tok[i..])) {
                splits[id].push((l, r));
            }
        }
        if splits[id].is_empty() {
            unsplittable += 1;
        }
    }
    if unsplittable > 0 {
        eprintln!("warning: {unsplittable} vocab tokens had no valid split");
    }

    let mut rank: Vec<u32> = (0..n as u32).collect();
    let mut converged = false;
    for _ in 0..200 {
        let mut changed = false;
        for id in NUM_BASE..n {
            for &(l, r) in &splits[id] {
                let want = rank[l as usize].max(rank[r as usize]) + 1;
                if rank[id] < want {
                    rank[id] = want;
                    changed = true;
                }
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    if !converged {
        eprintln!("warning: rank relaxation did not converge (dependency cycle)");
    }

    let mut rules: Vec<MergeRule> = (NUM_BASE..n)
        .flat_map(|id| {
            let rk = rank[id];
            splits[id].iter().map(move |&(l, r)| MergeRule {
                left: l,
                right: r,
                rank: rk,
                new_id: id as u32,
            })
        })
        .collect();
    rules.sort_by_key(|r| (r.left, r.right));
    rules.dedup_by(|a, b| a.left == b.left && a.right == b.right);
    (rules, rank)
}

/// `left_index[l]..left_index[l + 1]` bounds the rules with left id `l` in the
/// (left, right)-sorted table. Sized for every possible token id.
pub fn build_left_index(rules: &[MergeRule], num_ids: usize) -> Vec<u32> {
    let mut idx = vec![0u32; num_ids + 1];
    let mut r = 0usize;
    for (left, slot) in idx.iter_mut().enumerate().take(num_ids) {
        *slot = r as u32;
        while r < rules.len() && rules[r].left as usize == left {
            r += 1;
        }
    }
    idx[num_ids] = rules.len() as u32;
    idx
}

/// Count how often each merge rule fires over the corpus (same merge loop as
/// `bpe_encode`) so we can build a profile-guided hot-rule cache.
pub fn count_rule_hits(
    chunks: &[SmallStr],
    rules: &[MergeRule],
    left_index: &[u32],
    byte_to_id: &[u32; NUM_BASE],
) -> HashMap<(u32, u32), (u64, MergeRule)> {
    let mut counts: HashMap<(u32, u32), (u64, MergeRule)> = HashMap::new();
    let mut seq = [0u32; SMALL_LEN];
    for chunk in chunks {
        let n0 = chunk.len as usize;
        for i in 0..n0 {
            seq[i] = byte_to_id[(chunk.bytes[i] & 0xFF) as usize];
        }
        let mut n = n0;
        loop {
            let mut best_rank = u32::MAX;
            let mut best_pos = 0usize;
            let mut best_id = 0u32;
            let mut j = 0usize;
            while j + 1 < n {
                let (l, r) = (seq[j], seq[j + 1]);
                match find_rule(rules, left_index, &[], l, r) {
                    Some(rule) => {
                        counts
                            .entry((l, r))
                            .and_modify(|(c, _)| *c += 1)
                            .or_insert((1, rule));
                        if rule.rank < best_rank {
                            best_rank = rule.rank;
                            best_pos = j;
                            best_id = rule.new_id;
                        }
                    }
                    None => {
                        // Cache frequent negative results (rank = sentinel).
                        counts.entry((l, r)).or_insert((
                            1,
                            MergeRule {
                                left: l,
                                right: r,
                                rank: u32::MAX,
                                new_id: 0,
                            },
                        ));
                    }
                }
                j += 1;
            }
            if best_rank == u32::MAX {
                break;
            }
            seq[best_pos] = best_id;
            let mut k = best_pos + 1;
            while k + 1 < n {
                seq[k] = seq[k + 1];
                k += 1;
            }
            n -= 1;
        }
    }
    counts
}

/// Direct-mapped hot-rule table (power-of-two `slots`), filled by descending
/// hit count. Colliding rules are skipped; empty slots stay sentinelled.
pub fn build_hot_table(
    counts: &HashMap<(u32, u32), (u64, MergeRule)>,
    slots: usize,
) -> Vec<MergeRule> {
    let empty = MergeRule {
        left: u32::MAX,
        right: u32::MAX,
        rank: u32::MAX,
        new_id: u32::MAX,
    };
    let mut hot = vec![empty; slots];
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_by_key(|e| std::cmp::Reverse(e.1.0));
    let mut inserted = 0usize;
    for (&(left, right), &(_, rule)) in entries {
        let h = ((left.wrapping_mul(0x9E37_79B1) ^ right) as usize) & (slots - 1);
        if hot[h].left == u32::MAX {
            hot[h] = rule;
            inserted += 1;
            if inserted == slots {
                break;
            }
        }
    }
    hot
}

/// Pack chunks for a kernel: 6 u32 words per chunk holding 22 one-byte
/// lanes plus the length byte (see `bpe_core::unpack_chunk`).
pub fn pack_chunks(chunks: &[SmallStr], tile: usize) -> Vec<u32> {
    let num = chunks.len().next_multiple_of(tile);
    let mut packed = vec![0u32; num * PACKED_WORDS];
    for (index, chunk) in chunks.iter().enumerate() {
        let base = index * PACKED_WORDS;
        let len = chunk.len as usize;
        for i in 0..len {
            let b = chunk.bytes[i] & 0xFF;
            packed[base + i / 4] |= b << ((i % 4) * 8);
        }
        packed[base + 5] |= (len as u32) << 16;
    }
    packed
}

/// Exact byte-space BPE: merge the adjacent pair whose concatenation has the
/// lowest rank. Oracle for validating the reconstructed rule table.
pub fn bpe_encode_ref(bytes: &[u8], bytes_to_id: &HashMap<&[u8], u32>, rank: &[u32]) -> Vec<u32> {
    let mut parts: Vec<Vec<u8>> = bytes.iter().map(|&b| vec![b]).collect();
    loop {
        let mut best: Option<(u32, usize)> = None; // (rank, pos)
        for j in 0..parts.len().saturating_sub(1) {
            let mut cat = parts[j].clone();
            cat.extend_from_slice(&parts[j + 1]);
            if let Some(&id) = bytes_to_id.get(cat.as_slice()) {
                let r = rank[id as usize];
                if best.is_none_or(|(br, _)| r < br) {
                    best = Some((r, j));
                }
            }
        }
        let Some((_, j)) = best else { break };
        let mut merged = std::mem::take(&mut parts[j]);
        merged.extend_from_slice(&parts[j + 1]);
        parts[j] = merged;
        parts.remove(j + 1);
    }
    parts.iter().map(|p| bytes_to_id[p.as_slice()]).collect()
}
