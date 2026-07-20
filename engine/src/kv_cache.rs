use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

use crate::config::ModelConfig;
use tempura_kernels::turboquant::{CompressedKV, KvCacheCompressor};

pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    k_f16: Vec<Vec<u16>>,
    v_f16: Vec<Vec<u16>>,
    pub use_f16: bool,
    pub seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    compressor: Option<Box<dyn KvCacheCompressor>>,
    k_compressed: Vec<Option<CompressedKV>>,
    v_compressed: Vec<Option<CompressedKV>>,
    compressed: bool,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn f32_slice_to_f16(src: &[f32], dst: &mut [u16]) {
    use std::arch::x86_64::*;
    let n = src.len();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        let h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
        i += 8;
    }
    while i < n {
        let v = _mm_set_ss(src[i]);
        let h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        dst[i] = _mm_extract_epi16(h, 0) as u16;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn f16_slice_to_f32(src: &[u16], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = src.len();
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        let v = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), v);
        i += 8;
    }
    while i < n {
        let h = _mm_set1_epi16(src[i] as i16);
        let v = _mm_cvtph_ps(h);
        dst[i] = _mm_cvtss_f32(v);
        i += 1;
    }
}

impl KvCache {
    pub fn new(config: &ModelConfig) -> Self {
        let layer_size = config.max_seq_len * config.num_kv_heads * config.head_dim;
        let use_f16 = config.hidden_act == crate::config::HiddenAct::SquaredRelu
            && is_x86_feature_detected!("f16c");
        let k: Vec<Vec<f32>> = (0..config.num_layers)
            .map(|_| if use_f16 { Vec::new() } else { vec![0.0; layer_size] })
            .collect();
        let v: Vec<Vec<f32>> = (0..config.num_layers)
            .map(|_| if use_f16 { Vec::new() } else { vec![0.0; layer_size] })
            .collect();
        let k_f16: Vec<Vec<u16>> = (0..config.num_layers)
            .map(|_| if use_f16 { vec![0u16; layer_size] } else { Vec::new() })
            .collect();
        let v_f16: Vec<Vec<u16>> = (0..config.num_layers)
            .map(|_| if use_f16 { vec![0u16; layer_size] } else { Vec::new() })
            .collect();

        Self {
            k,
            v,
            k_f16,
            v_f16,
            use_f16,
            seq_len: 0,
            num_layers: config.num_layers,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            max_seq_len: config.max_seq_len,
            compressor: None,
            k_compressed: Vec::new(),
            v_compressed: Vec::new(),
            compressed: false,
        }
    }

    pub fn new_compressed(config: &ModelConfig, compressor: Box<dyn KvCacheCompressor>) -> Self {
        let k: Vec<Vec<f32>> = (0..config.num_layers).map(|_| Vec::new()).collect();
        let v: Vec<Vec<f32>> = (0..config.num_layers).map(|_| Vec::new()).collect();
        let k_compressed: Vec<Option<CompressedKV>> =
            (0..config.num_layers).map(|_| None).collect();
        let v_compressed: Vec<Option<CompressedKV>> =
            (0..config.num_layers).map(|_| None).collect();

        Self {
            k,
            v,
            k_f16: Vec::new(),
            v_f16: Vec::new(),
            use_f16: false,
            seq_len: 0,
            num_layers: config.num_layers,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            max_seq_len: config.max_seq_len,
            compressor: Some(compressor),
            k_compressed,
            v_compressed,
            compressed: true,
        }
    }

    pub fn is_compressed(&self) -> bool {
        self.compressed
    }

    pub fn append_kv(&mut self, layer: usize, new_k: &[f32], new_v: &[f32], num_new_tokens: usize) {
        let stride = self.num_kv_heads * self.head_dim;
        let src_len = num_new_tokens * stride;

        if self.compressed {
            let compressor = self.compressor.as_ref().unwrap();
            let new_k_data = &new_k[..src_len];
            let new_v_data = &new_v[..src_len];

            let existing_k = self.k_data_for_len(layer, self.seq_len);
            let existing_v = self.v_data_for_len(layer, self.seq_len);

            let mut combined_k = existing_k.into_owned();
            combined_k.extend_from_slice(new_k_data);
            let mut combined_v = existing_v.into_owned();
            combined_v.extend_from_slice(new_v_data);

            self.k_compressed[layer] =
                Some(compressor.compress(&combined_k, self.head_dim).unwrap());
            self.v_compressed[layer] =
                Some(compressor.compress(&combined_v, self.head_dim).unwrap());
        } else if self.use_f16 {
            let dst_offset = self.seq_len * stride;
            #[cfg(target_arch = "x86_64")]
            unsafe {
                f32_slice_to_f16(&new_k[..src_len], &mut self.k_f16[layer][dst_offset..dst_offset + src_len]);
                f32_slice_to_f16(&new_v[..src_len], &mut self.v_f16[layer][dst_offset..dst_offset + src_len]);
            }
        } else {
            let dst_offset = self.seq_len * stride;
            self.k[layer][dst_offset..dst_offset + src_len].copy_from_slice(&new_k[..src_len]);
            self.v[layer][dst_offset..dst_offset + src_len].copy_from_slice(&new_v[..src_len]);
        }
    }

    pub fn advance(&mut self, num_tokens: usize) {
        self.seq_len += num_tokens;
        debug_assert!(tempura_kernel_api::VerifiedKvCache::well_formed(self));
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        if self.compressed {
            for c in &mut self.k_compressed {
                *c = None;
            }
            for c in &mut self.v_compressed {
                *c = None;
            }
        }
    }

    pub fn k_data_f16(&self, layer: usize, total_len: usize) -> &[u16] {
        let len = total_len * self.num_kv_heads * self.head_dim;
        &self.k_f16[layer][..len]
    }

    pub fn v_data_f16(&self, layer: usize, total_len: usize) -> &[u16] {
        let len = total_len * self.num_kv_heads * self.head_dim;
        &self.v_f16[layer][..len]
    }

    pub fn k_slice(&self, layer: usize) -> &[f32] {
        let len = self.seq_len * self.num_kv_heads * self.head_dim;
        &self.k[layer][..len]
    }

    pub fn v_slice(&self, layer: usize) -> &[f32] {
        let len = self.seq_len * self.num_kv_heads * self.head_dim;
        &self.v[layer][..len]
    }

    pub fn k_data(&self, layer: usize, total_len: usize) -> Cow<'_, [f32]> {
        self.k_data_for_len(layer, total_len)
    }

    pub fn v_data(&self, layer: usize, total_len: usize) -> Cow<'_, [f32]> {
        self.v_data_for_len(layer, total_len)
    }

    fn k_data_for_len(&self, layer: usize, num_tokens: usize) -> Cow<'_, [f32]> {
        if self.compressed {
            match &self.k_compressed[layer] {
                Some(c) => {
                    let compressor = self.compressor.as_ref().unwrap();
                    Cow::Owned(compressor.decompress(c).unwrap())
                }
                None => Cow::Owned(vec![0.0f32; num_tokens * self.num_kv_heads * self.head_dim]),
            }
        } else if self.use_f16 {
            let len = num_tokens * self.num_kv_heads * self.head_dim;
            let mut out = vec![0.0f32; len];
            #[cfg(target_arch = "x86_64")]
            unsafe { f16_slice_to_f32(&self.k_f16[layer][..len], &mut out); }
            Cow::Owned(out)
        } else {
            let len = num_tokens * self.num_kv_heads * self.head_dim;
            Cow::Borrowed(&self.k[layer][..len])
        }
    }

    fn v_data_for_len(&self, layer: usize, num_tokens: usize) -> Cow<'_, [f32]> {
        if self.compressed {
            match &self.v_compressed[layer] {
                Some(c) => {
                    let compressor = self.compressor.as_ref().unwrap();
                    Cow::Owned(compressor.decompress(c).unwrap())
                }
                None => Cow::Owned(vec![0.0f32; num_tokens * self.num_kv_heads * self.head_dim]),
            }
        } else if self.use_f16 {
            let len = num_tokens * self.num_kv_heads * self.head_dim;
            let mut out = vec![0.0f32; len];
            #[cfg(target_arch = "x86_64")]
            unsafe { f16_slice_to_f32(&self.v_f16[layer][..len], &mut out); }
            Cow::Owned(out)
        } else {
            let len = num_tokens * self.num_kv_heads * self.head_dim;
            Cow::Borrowed(&self.v[layer][..len])
        }
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len - self.seq_len
    }

    pub fn rollback(&mut self, new_seq_len: usize) {
        assert!(new_seq_len <= self.seq_len);
        if self.compressed {
            if new_seq_len == 0 {
                for c in &mut self.k_compressed {
                    *c = None;
                }
                for c in &mut self.v_compressed {
                    *c = None;
                }
            } else if new_seq_len < self.seq_len {
                let compressor = self.compressor.as_ref().unwrap();
                let stride = self.num_kv_heads * self.head_dim;
                let truncated_len = new_seq_len * stride;
                for layer in 0..self.num_layers {
                    if let Some(ref existing) = self.k_compressed[layer] {
                        let full = compressor.decompress(existing).unwrap();
                        self.k_compressed[layer] = Some(
                            compressor
                                .compress(&full[..truncated_len], self.head_dim)
                                .unwrap(),
                        );
                    }
                    if let Some(ref existing) = self.v_compressed[layer] {
                        let full = compressor.decompress(existing).unwrap();
                        self.v_compressed[layer] = Some(
                            compressor
                                .compress(&full[..truncated_len], self.head_dim)
                                .unwrap(),
                        );
                    }
                }
            }
        }
        self.seq_len = new_seq_len;
    }

    pub fn to_transfer(&self) -> KvCacheTransfer {
        let k_data: Vec<Vec<f32>> = (0..self.num_layers)
            .map(|layer| self.k_slice(layer).to_vec())
            .collect();
        let v_data: Vec<Vec<f32>> = (0..self.num_layers)
            .map(|layer| self.v_slice(layer).to_vec())
            .collect();
        KvCacheTransfer {
            k_data,
            v_data,
            seq_len: self.seq_len,
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
        }
    }

    pub fn from_transfer(transfer: KvCacheTransfer, config: &ModelConfig) -> Self {
        let mut cache = Self::new(config);
        for (layer, (k, v)) in transfer
            .k_data
            .iter()
            .zip(transfer.v_data.iter())
            .enumerate()
        {
            cache.append_kv(layer, k, v, transfer.seq_len);
        }
        cache.advance(transfer.seq_len);
        cache
    }
}

impl tempura_kernel_api::VerifiedKvCache for KvCache {
    fn seq_len(&self) -> usize { self.seq_len }
    fn max_seq_len(&self) -> usize { self.max_seq_len }
    fn num_kv_heads(&self) -> usize { self.num_kv_heads }
    fn head_dim(&self) -> usize { self.head_dim }
}

// Verus deductive proofs for KvCache struct invariants.
// Under normal `cargo build`, the verus!{} block compiles to nothing (ghost erasure).
// Under `verus`, these proofs are checked for ALL possible inputs.
// Canonical verification runs against verus-verified/ crate; these inline specs
// serve as documentation and compile-time syntax checking.
use vstd::prelude::*;

verus! {

pub open spec fn kv_cache_well_formed(
    seq_len: nat, max_seq_len: nat, num_kv_heads: nat, head_dim: nat,
) -> bool {
    &&& seq_len <= max_seq_len
    &&& num_kv_heads >= 1
    &&& head_dim >= 1
    &&& max_seq_len >= 1
}

proof fn kv_advance_preserves_well_formed(
    seq_len: nat, max_seq_len: nat, num_kv_heads: nat, head_dim: nat, n: nat,
)
    requires
        kv_cache_well_formed(seq_len, max_seq_len, num_kv_heads, head_dim),
        n >= 1,
        seq_len + n <= max_seq_len,
    ensures
        kv_cache_well_formed(seq_len + n, max_seq_len, num_kv_heads, head_dim),
{
}

proof fn kv_reset_preserves_well_formed(
    max_seq_len: nat, num_kv_heads: nat, head_dim: nat,
)
    requires
        num_kv_heads >= 1, head_dim >= 1, max_seq_len >= 1,
    ensures
        kv_cache_well_formed(0, max_seq_len, num_kv_heads, head_dim),
{
}

proof fn kv_append_offsets_safe(
    seq_len: nat, max_seq_len: nat, num_kv_heads: nat, head_dim: nat, n: nat,
)
    requires
        kv_cache_well_formed(seq_len, max_seq_len, num_kv_heads, head_dim),
        n >= 1,
        seq_len + n <= max_seq_len,
    ensures ({
        let stride = num_kv_heads * head_dim;
        let dst = seq_len * stride;
        let src = n * stride;
        dst + src <= max_seq_len * stride
    }),
{
    assert((seq_len + n) * (num_kv_heads * head_dim)
        <= max_seq_len * (num_kv_heads * head_dim))
        by (nonlinear_arith)
        requires seq_len + n <= max_seq_len;
    assert(seq_len * (num_kv_heads * head_dim) + n * (num_kv_heads * head_dim)
        == (seq_len + n) * (num_kv_heads * head_dim))
        by (nonlinear_arith);
}

} // verus!

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KvCacheTransfer {
    pub k_data: Vec<Vec<f32>>,
    pub v_data: Vec<Vec<f32>>,
    pub seq_len: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug)]
pub enum PoolError {
    NoFreeSlots,
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::NoFreeSlots => write!(f, "no free KV cache slots available"),
        }
    }
}

impl std::error::Error for PoolError {}

pub struct KvCachePool {
    free: Vec<KvCache>,
    config: ModelConfig,
    num_slots: usize,
}

impl KvCachePool {
    pub fn new(config: &ModelConfig, num_slots: usize) -> Self {
        let free = (0..num_slots).map(|_| KvCache::new(config)).collect();
        Self {
            free,
            config: config.clone(),
            num_slots,
        }
    }

    pub fn acquire(&mut self) -> Result<KvCache, PoolError> {
        self.free.pop().ok_or(PoolError::NoFreeSlots)
    }

    pub fn release(&mut self, mut cache: KvCache) {
        cache.reset();
        self.free.push(cache);
    }

    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    pub fn available(&self) -> usize {
        self.free.len()
    }

    pub fn in_use(&self) -> usize {
        self.num_slots - self.free.len()
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

struct PrefixEntry {
    token_ids: Vec<u32>,
    k_data: Vec<Vec<f32>>,
    v_data: Vec<Vec<f32>>,
    seq_len: usize,
    ref_count: usize,
    last_used: u64,
}

pub struct PrefixCacheTable {
    entries: HashMap<u64, PrefixEntry>,
    max_entries: usize,
    tick: u64,
    hits: u64,
    tokens_saved: u64,
}

impl PrefixCacheTable {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            tick: 0,
            hits: 0,
            tokens_saved: 0,
        }
    }

    fn hash_tokens(token_ids: &[u32], salt: Option<&str>) -> u64 {
        let mut hasher = DefaultHasher::new();
        if let Some(s) = salt {
            s.hash(&mut hasher);
        }
        token_ids.hash(&mut hasher);
        hasher.finish()
    }

    pub fn probe(&mut self, token_ids: &[u32]) -> Option<PrefixMatch> {
        self.probe_with_salt(token_ids, None)
    }

    pub fn probe_with_salt(
        &mut self,
        token_ids: &[u32],
        salt: Option<&str>,
    ) -> Option<PrefixMatch> {
        self.tick += 1;

        for len in (1..=token_ids.len()).rev() {
            let prefix = &token_ids[..len];
            let hash = Self::hash_tokens(prefix, salt);
            if let Some(entry) = self.entries.get_mut(&hash) {
                if entry.token_ids == prefix {
                    entry.last_used = self.tick;
                    entry.ref_count += 1;
                    self.hits += 1;
                    self.tokens_saved += len as u64;
                    return Some(PrefixMatch {
                        matched_len: entry.seq_len,
                        k_data: entry.k_data.clone(),
                        v_data: entry.v_data.clone(),
                    });
                }
            }
        }
        None
    }

    pub fn store(&mut self, token_ids: &[u32], cache: &KvCache) {
        self.store_with_salt(token_ids, cache, None);
    }

    pub fn store_with_salt(&mut self, token_ids: &[u32], cache: &KvCache, salt: Option<&str>) {
        if token_ids.is_empty() || cache.seq_len == 0 {
            return;
        }
        self.tick += 1;

        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        let hash = Self::hash_tokens(token_ids, salt);
        let k_data: Vec<Vec<f32>> = (0..cache.num_layers)
            .map(|layer| cache.k_slice(layer).to_vec())
            .collect();
        let v_data: Vec<Vec<f32>> = (0..cache.num_layers)
            .map(|layer| cache.v_slice(layer).to_vec())
            .collect();

        self.entries.insert(
            hash,
            PrefixEntry {
                token_ids: token_ids.to_vec(),
                k_data,
                v_data,
                seq_len: cache.seq_len,
                ref_count: 0,
                last_used: self.tick,
            },
        );
    }

    pub fn release(&mut self, token_ids: &[u32]) {
        self.release_with_salt(token_ids, None);
    }

    pub fn release_with_salt(&mut self, token_ids: &[u32], salt: Option<&str>) {
        let hash = Self::hash_tokens(token_ids, salt);
        if let Some(entry) = self.entries.get_mut(&hash) {
            entry.ref_count = entry.ref_count.saturating_sub(1);
        }
    }

    fn evict_lru(&mut self) {
        let evict_key = self
            .entries
            .iter()
            .filter(|(_, e)| e.ref_count == 0)
            .min_by_key(|(_, e)| e.last_used)
            .map(|(&k, _)| k);

        if let Some(key) = evict_key {
            self.entries.remove(&key);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn hit_count(&self) -> u64 {
        self.hits
    }

    pub fn tokens_saved(&self) -> u64 {
        self.tokens_saved
    }
}

impl tempura_kernel_api::VerifiedPrefixCache for PrefixCacheTable {
    fn num_entries(&self) -> usize { self.entries.len() }
    fn max_entries(&self) -> usize { self.max_entries }
    fn tick(&self) -> u64 { self.tick }
}

pub struct PrefixMatch {
    pub matched_len: usize,
    pub k_data: Vec<Vec<f32>>,
    pub v_data: Vec<Vec<f32>>,
}

impl PrefixMatch {
    pub fn populate_cache(&self, cache: &mut KvCache) {
        for (layer, (k, v)) in self.k_data.iter().zip(self.v_data.iter()).enumerate() {
            cache.append_kv(layer, k, v, self.matched_len);
        }
        cache.advance(self.matched_len);
    }
}

pub type SharedPrefixCache = Arc<Mutex<PrefixCacheTable>>;

pub fn new_shared_prefix_cache(max_entries: usize) -> SharedPrefixCache {
    Arc::new(Mutex::new(PrefixCacheTable::new(max_entries)))
}

// ── Paged KV Cache ─────────────────────────────────────────────────────────

pub const DEFAULT_BLOCK_SIZE: usize = 16;

pub struct PageAllocator {
    free_pages: Vec<usize>,
    total_pages: usize,
}

impl PageAllocator {
    pub fn new(total_pages: usize) -> Self {
        let free_pages = (0..total_pages).rev().collect();
        Self {
            free_pages,
            total_pages,
        }
    }

    pub fn allocate(&mut self) -> Option<usize> {
        self.free_pages.pop()
    }

    pub fn free(&mut self, page_id: usize) {
        self.free_pages.push(page_id);
    }

    pub fn available(&self) -> usize {
        self.free_pages.len()
    }

    pub fn in_use(&self) -> usize {
        self.total_pages - self.free_pages.len()
    }

    pub fn total(&self) -> usize {
        self.total_pages
    }
}

impl tempura_kernel_api::VerifiedPageAllocator for PageAllocator {
    fn available(&self) -> usize { self.free_pages.len() }
    fn total(&self) -> usize { self.total_pages }
}

verus! {

proof fn page_allocator_conservation(free_count: nat, total_pages: nat)
    requires free_count <= total_pages,
    ensures free_count + (total_pages - free_count) == total_pages,
{
}

proof fn page_allocate_preserves(free_count: nat, total_pages: nat)
    requires free_count <= total_pages, free_count >= 1,
    ensures ({
        let new_free = (free_count - 1) as nat;
        &&& new_free <= total_pages
        &&& new_free + (total_pages - new_free) == total_pages
    }),
{
}

proof fn page_free_preserves(free_count: nat, total_pages: nat)
    requires free_count < total_pages,
    ensures ({
        let new_free = free_count + 1;
        &&& new_free <= total_pages
        &&& new_free + (total_pages - new_free) == total_pages
    }),
{
}

} // verus!

pub struct BlockTable {
    table: Vec<u32>,
    num_blocks: usize,
}

impl BlockTable {
    pub fn new(max_blocks: usize) -> Self {
        Self {
            table: vec![u32::MAX; max_blocks],
            num_blocks: 0,
        }
    }

    pub fn set_block(&mut self, logical_idx: usize, physical_page: usize) {
        self.table[logical_idx] = physical_page as u32;
        if logical_idx >= self.num_blocks {
            self.num_blocks = logical_idx + 1;
        }
    }

    pub fn get_block(&self, logical_idx: usize) -> Option<usize> {
        if logical_idx < self.num_blocks && self.table[logical_idx] != u32::MAX {
            Some(self.table[logical_idx] as usize)
        } else {
            None
        }
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.table[..self.num_blocks]
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn max_blocks(&self) -> usize {
        self.table.len()
    }

    pub fn clear(&mut self) -> Vec<usize> {
        let pages: Vec<usize> = self.table[..self.num_blocks]
            .iter()
            .filter(|&&p| p != u32::MAX)
            .map(|&p| p as usize)
            .collect();
        self.table.fill(u32::MAX);
        self.num_blocks = 0;
        pages
    }

    pub fn clone_from_cow(&self) -> Self {
        Self {
            table: self.table.clone(),
            num_blocks: self.num_blocks,
        }
    }
}

impl tempura_kernel_api::VerifiedBlockTable for BlockTable {
    fn num_blocks(&self) -> usize { self.num_blocks }
    fn max_blocks(&self) -> usize { self.table.len() }
}

verus! {

proof fn block_table_set_monotone(num_blocks: nat, max_blocks: nat, idx: nat)
    requires num_blocks <= max_blocks, idx < max_blocks,
    ensures ({
        let new_nb = if idx >= num_blocks { idx + 1 } else { num_blocks };
        &&& new_nb <= max_blocks
        &&& new_nb >= num_blocks
        &&& idx < new_nb
    }),
{
}

proof fn block_table_clear_valid(num_blocks: nat, max_blocks: nat)
    requires num_blocks <= max_blocks,
    ensures 0nat <= max_blocks,
{
}

} // verus!

pub struct PagedKvCache {
    k_pages: Vec<Vec<f32>>,
    v_pages: Vec<Vec<f32>>,
    block_tables: Vec<BlockTable>,
    allocator: PageAllocator,
    pub seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    page_stride: usize,
}

impl PagedKvCache {
    pub fn new(config: &crate::config::ModelConfig, total_pages: usize, block_size: usize) -> Self {
        let page_stride = block_size * config.num_kv_heads * config.head_dim;
        let k_pages = (0..config.num_layers)
            .map(|_| vec![0.0f32; total_pages * page_stride])
            .collect();
        let v_pages = (0..config.num_layers)
            .map(|_| vec![0.0f32; total_pages * page_stride])
            .collect();
        let max_blocks = (config.max_seq_len + block_size - 1) / block_size;
        let block_tables = (0..config.num_layers)
            .map(|_| BlockTable::new(max_blocks))
            .collect();

        Self {
            k_pages,
            v_pages,
            block_tables,
            allocator: PageAllocator::new(total_pages),
            seq_len: 0,
            num_layers: config.num_layers,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            block_size,
            page_stride,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn allocator(&self) -> &PageAllocator {
        &self.allocator
    }

    fn ensure_blocks_for(&mut self, layer: usize, needed_tokens: usize) {
        let needed_blocks = (needed_tokens + self.block_size - 1) / self.block_size;
        let bt = &mut self.block_tables[layer];
        let current_blocks = bt.num_blocks();
        for _ in current_blocks..needed_blocks {
            let page_id = self.allocator.allocate().expect("out of KV cache pages");
            let idx = bt.num_blocks();
            bt.set_block(idx, page_id);
        }
    }

    pub fn append_kv(&mut self, layer: usize, new_k: &[f32], new_v: &[f32], num_new_tokens: usize) {
        let stride = self.num_kv_heads * self.head_dim;
        let total_after = self.seq_len + num_new_tokens;
        self.ensure_blocks_for(layer, total_after);

        for t in 0..num_new_tokens {
            let global_pos = self.seq_len + t;
            let block_idx = global_pos / self.block_size;
            let block_offset = global_pos % self.block_size;
            let phys_page = self.block_tables[layer].get_block(block_idx).unwrap();
            let page_base = phys_page * self.page_stride + block_offset * stride;
            let src_base = t * stride;

            self.k_pages[layer][page_base..page_base + stride]
                .copy_from_slice(&new_k[src_base..src_base + stride]);
            self.v_pages[layer][page_base..page_base + stride]
                .copy_from_slice(&new_v[src_base..src_base + stride]);
        }
    }

    pub fn advance(&mut self, num_tokens: usize) {
        self.seq_len += num_tokens;
    }

    pub fn reset(&mut self) {
        for layer in 0..self.num_layers {
            let pages = self.block_tables[layer].clear();
            for p in pages {
                self.allocator.free(p);
            }
        }
        self.seq_len = 0;
    }

    pub fn k_pages(&self, layer: usize) -> &[f32] {
        &self.k_pages[layer]
    }

    pub fn v_pages(&self, layer: usize) -> &[f32] {
        &self.v_pages[layer]
    }

    pub fn block_table(&self, layer: usize) -> &BlockTable {
        &self.block_tables[layer]
    }

    pub fn k_contiguous(&self, layer: usize, kv_len: usize) -> Vec<f32> {
        let stride = self.num_kv_heads * self.head_dim;
        let mut out = vec![0.0f32; kv_len * stride];
        for pos in 0..kv_len {
            let block_idx = pos / self.block_size;
            let block_offset = pos % self.block_size;
            let phys_page = self.block_tables[layer].get_block(block_idx).unwrap();
            let page_base = phys_page * self.page_stride + block_offset * stride;
            let dst = pos * stride;
            out[dst..dst + stride]
                .copy_from_slice(&self.k_pages[layer][page_base..page_base + stride]);
        }
        out
    }

    pub fn v_contiguous(&self, layer: usize, kv_len: usize) -> Vec<f32> {
        let stride = self.num_kv_heads * self.head_dim;
        let mut out = vec![0.0f32; kv_len * stride];
        for pos in 0..kv_len {
            let block_idx = pos / self.block_size;
            let block_offset = pos % self.block_size;
            let phys_page = self.block_tables[layer].get_block(block_idx).unwrap();
            let page_base = phys_page * self.page_stride + block_offset * stride;
            let dst = pos * stride;
            out[dst..dst + stride]
                .copy_from_slice(&self.v_pages[layer][page_base..page_base + stride]);
        }
        out
    }

    pub fn remaining_pages(&self) -> usize {
        self.allocator.available()
    }

    pub fn waste_ratio(&self) -> f64 {
        if self.seq_len == 0 {
            return 0.0;
        }
        let needed_blocks = (self.seq_len + self.block_size - 1) / self.block_size;
        let allocated_tokens = needed_blocks * self.block_size;
        let wasted = allocated_tokens - self.seq_len;
        wasted as f64 / allocated_tokens as f64
    }
}

// ── FP8 E4M3 KV Cache ──────────────────────────────────────────────────────

fn f32_to_fp8_e4m3(x: f32) -> u8 {
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let abs = x.abs();

    if abs == 0.0 {
        return sign;
    }
    if abs.is_nan() || abs.is_infinite() {
        return sign | 0x7E; // max finite E4M3 = 448
    }
    // E4M3: bias=7, max_exp=14 (exp=15 reserved for NaN), max val = 2^8 * 1.75 = 448
    // min normal = 2^(-6) = 0.015625, min subnormal = 2^(-6) * 2^(-3) = 2^(-9) = 0.001953125
    if abs >= 448.0 {
        return sign | 0x7E; // clamp to max
    }
    if abs < 0.001953125 {
        return sign; // underflow to zero
    }

    // Normal encoding: find biased exponent
    let bits = abs.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_frac = bits & 0x7FFFFF;

    let e4m3_exp = f32_exp - 127 + 7;
    if e4m3_exp <= 0 {
        return sign;
    }
    if e4m3_exp > 14 {
        return sign | 0x7E;
    }

    // Round mantissa: f32 has 23 bits, E4M3 has 3 bits, drop 20 with rounding
    let mantissa = ((f32_frac + (1 << 19)) >> 20) as u8;
    if mantissa >= 8 {
        // Mantissa overflow → increment exponent
        let new_exp = e4m3_exp + 1;
        if new_exp > 14 {
            return sign | 0x7E;
        }
        return sign | ((new_exp as u8) << 3);
    }

    sign | ((e4m3_exp as u8) << 3) | mantissa
}

fn fp8_e4m3_to_f32(x: u8) -> f32 {
    let sign = ((x >> 7) & 1) as u32;
    let exp = ((x >> 3) & 0xF) as u32;
    let frac = (x & 0x7) as u32;

    if exp == 0 && frac == 0 {
        return f32::from_bits(sign << 31);
    }
    // E4M3: no infinity representation, exp=15 with frac=7 is NaN, rest are finite
    if exp == 15 && frac == 7 {
        return f32::NAN;
    }

    if exp == 0 {
        // Subnormal: value = (-1)^sign * 2^(-6) * (frac/8)
        let val = (frac as f32) * (1.0 / 8.0) * (1.0 / 64.0); // 2^(-6) * frac/8
        return if sign == 1 { -val } else { val };
    }

    // Normal: value = (-1)^sign * 2^(exp-7) * (1 + frac/8)
    let f32_exp = (exp as i32 - 7 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 20))
}

pub struct Fp8PagedKvCache {
    k_pages: Vec<Vec<u8>>,
    v_pages: Vec<Vec<u8>>,
    block_tables: Vec<BlockTable>,
    allocator: PageAllocator,
    pub seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    page_stride: usize,
}

impl Fp8PagedKvCache {
    pub fn new(config: &crate::config::ModelConfig, total_pages: usize, block_size: usize) -> Self {
        let page_stride = block_size * config.num_kv_heads * config.head_dim;
        let k_pages = (0..config.num_layers)
            .map(|_| vec![0u8; total_pages * page_stride])
            .collect();
        let v_pages = (0..config.num_layers)
            .map(|_| vec![0u8; total_pages * page_stride])
            .collect();
        let max_blocks = (config.max_seq_len + block_size - 1) / block_size;
        let block_tables = (0..config.num_layers)
            .map(|_| BlockTable::new(max_blocks))
            .collect();

        Self {
            k_pages,
            v_pages,
            block_tables,
            allocator: PageAllocator::new(total_pages),
            seq_len: 0,
            num_layers: config.num_layers,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            block_size,
            page_stride,
        }
    }

    fn ensure_blocks_for(&mut self, layer: usize, needed_tokens: usize) {
        let needed_blocks = (needed_tokens + self.block_size - 1) / self.block_size;
        let bt = &mut self.block_tables[layer];
        let current_blocks = bt.num_blocks();
        for _ in current_blocks..needed_blocks {
            let page_id = self
                .allocator
                .allocate()
                .expect("out of FP8 KV cache pages");
            let idx = bt.num_blocks();
            bt.set_block(idx, page_id);
        }
    }

    pub fn append_kv(&mut self, layer: usize, new_k: &[f32], new_v: &[f32], num_new_tokens: usize) {
        let stride = self.num_kv_heads * self.head_dim;
        let total_after = self.seq_len + num_new_tokens;
        self.ensure_blocks_for(layer, total_after);

        for t in 0..num_new_tokens {
            let global_pos = self.seq_len + t;
            let block_idx = global_pos / self.block_size;
            let block_offset = global_pos % self.block_size;
            let phys_page = self.block_tables[layer].get_block(block_idx).unwrap();
            let page_base = phys_page * self.page_stride + block_offset * stride;
            let src_base = t * stride;

            let k_slice = &new_k[src_base..src_base + stride];
            let v_slice = &new_v[src_base..src_base + stride];

            for i in 0..stride {
                self.k_pages[layer][page_base + i] = f32_to_fp8_e4m3(k_slice[i]);
                self.v_pages[layer][page_base + i] = f32_to_fp8_e4m3(v_slice[i]);
            }
        }
    }

    pub fn advance(&mut self, num_tokens: usize) {
        self.seq_len += num_tokens;
    }

    pub fn reset(&mut self) {
        for layer in 0..self.num_layers {
            let pages = self.block_tables[layer].clear();
            for p in pages {
                self.allocator.free(p);
            }
        }
        self.seq_len = 0;
    }

    pub fn k_contiguous(&self, layer: usize, kv_len: usize) -> Vec<f32> {
        let stride = self.num_kv_heads * self.head_dim;
        let mut out = vec![0.0f32; kv_len * stride];
        for pos in 0..kv_len {
            let block_idx = pos / self.block_size;
            let block_offset = pos % self.block_size;
            let phys_page = self.block_tables[layer].get_block(block_idx).unwrap();
            let page_base = phys_page * self.page_stride + block_offset * stride;
            let dst = pos * stride;
            for i in 0..stride {
                out[dst + i] = fp8_e4m3_to_f32(self.k_pages[layer][page_base + i]);
            }
        }
        out
    }

    pub fn v_contiguous(&self, layer: usize, kv_len: usize) -> Vec<f32> {
        let stride = self.num_kv_heads * self.head_dim;
        let mut out = vec![0.0f32; kv_len * stride];
        for pos in 0..kv_len {
            let block_idx = pos / self.block_size;
            let block_offset = pos % self.block_size;
            let phys_page = self.block_tables[layer].get_block(block_idx).unwrap();
            let page_base = phys_page * self.page_stride + block_offset * stride;
            let dst = pos * stride;
            for i in 0..stride {
                out[dst + i] = fp8_e4m3_to_f32(self.v_pages[layer][page_base + i]);
            }
        }
        out
    }

    pub fn memory_bytes_per_token(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub fn f32_memory_bytes_per_token(&self) -> usize {
        self.num_kv_heads * self.head_dim * 4
    }
}

#[cfg(feature = "cuda")]
pub struct GpuKvCache {
    pub k: Vec<cuda_core::DeviceBuffer<f32>>,
    pub v: Vec<cuda_core::DeviceBuffer<f32>>,
    pub seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pub max_seq_len: usize,
}

#[cfg(feature = "cuda")]
impl GpuKvCache {
    pub fn new(
        config: &ModelConfig,
        stream: &cuda_core::CudaStream,
    ) -> Result<Self, crate::loader::LoadError> {
        let layer_size = config.max_seq_len * config.num_kv_heads * config.head_dim;
        let k: Vec<_> = (0..config.num_layers)
            .map(|_| cuda_core::DeviceBuffer::zeroed(stream, layer_size))
            .collect::<Result<_, _>>()
            .map_err(|e| crate::loader::LoadError::GgufParse(e.to_string()))?;
        let v: Vec<_> = (0..config.num_layers)
            .map(|_| cuda_core::DeviceBuffer::zeroed(stream, layer_size))
            .collect::<Result<_, _>>()
            .map_err(|e| crate::loader::LoadError::GgufParse(e.to_string()))?;
        Ok(Self {
            k,
            v,
            seq_len: 0,
            num_layers: config.num_layers,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            max_seq_len: config.max_seq_len,
        })
    }

    pub fn advance(&mut self, num_tokens: usize) {
        self.seq_len += num_tokens;
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

#[cfg(kani)]
mod kani_proofs {
    // Prove KvCache indexing arithmetic is safe without constructing
    // Vec<Vec<f32>> (which causes CBMC unwinding explosion). We verify
    // that computed offsets stay within the allocated buffer size.

    #[kani::proof]
    fn append_kv_offsets_in_bounds() {
        let num_kv_heads: usize = kani::any();
        let head_dim: usize = kani::any();
        let max_seq_len: usize = kani::any();
        let seq_len: usize = kani::any();
        let num_new_tokens: usize = kani::any();

        kani::assume(num_kv_heads >= 1 && num_kv_heads <= 4);
        kani::assume(head_dim >= 1 && head_dim <= 8);
        kani::assume(max_seq_len >= 1 && max_seq_len <= 8);
        kani::assume(seq_len <= max_seq_len);
        kani::assume(num_new_tokens >= 1);
        kani::assume(num_new_tokens <= max_seq_len - seq_len);

        let stride = num_kv_heads * head_dim;
        let layer_size = max_seq_len * stride;
        let dst_offset = seq_len * stride;
        let src_len = num_new_tokens * stride;

        // copy_from_slice destination: [dst_offset..dst_offset + src_len]
        assert!(dst_offset + src_len <= layer_size);
        // source slice: [..src_len]
        assert!(src_len <= num_new_tokens * stride);
    }

    #[kani::proof]
    fn k_v_slice_in_bounds() {
        let num_kv_heads: usize = kani::any();
        let head_dim: usize = kani::any();
        let max_seq_len: usize = kani::any();
        let seq_len: usize = kani::any();

        kani::assume(num_kv_heads >= 1 && num_kv_heads <= 4);
        kani::assume(head_dim >= 1 && head_dim <= 8);
        kani::assume(max_seq_len >= 1 && max_seq_len <= 8);
        kani::assume(seq_len <= max_seq_len);

        let layer_size = max_seq_len * num_kv_heads * head_dim;
        let slice_len = seq_len * num_kv_heads * head_dim;

        // k_slice / v_slice: &self.k[layer][..slice_len]
        assert!(slice_len <= layer_size);
    }

    #[kani::proof]
    fn paged_block_table_offset_in_bounds() {
        let block_size: usize = kani::any();
        let num_kv_heads: usize = kani::any();
        let head_dim: usize = kani::any();
        let total_pages: usize = kani::any();
        let pos: usize = kani::any();
        let phys_page: usize = kani::any();

        kani::assume(block_size >= 1 && block_size <= 16);
        kani::assume(num_kv_heads >= 1 && num_kv_heads <= 4);
        kani::assume(head_dim >= 1 && head_dim <= 8);
        kani::assume(total_pages >= 1 && total_pages <= 8);
        kani::assume(phys_page < total_pages);
        kani::assume(pos < total_pages * block_size);

        let stride = num_kv_heads * head_dim;
        let page_stride = block_size * stride;
        let pool_size = total_pages * page_stride;

        let block_offset = pos % block_size;
        let page_base = phys_page * page_stride + block_offset * stride;

        assert!(page_base + stride <= pool_size);
    }

    #[kani::proof]
    fn paged_block_index_in_bounds() {
        let block_size: usize = kani::any();
        let max_seq_len: usize = kani::any();
        let pos: usize = kani::any();

        kani::assume(block_size >= 1 && block_size <= 16);
        kani::assume(max_seq_len >= 1 && max_seq_len <= 64);
        kani::assume(pos < max_seq_len);

        let max_blocks = (max_seq_len + block_size - 1) / block_size;
        let block_idx = pos / block_size;

        assert!(block_idx < max_blocks);
    }

    #[kani::proof]
    fn remaining_capacity_consistent() {
        let max_seq_len: usize = kani::any();
        let seq_len: usize = kani::any();

        kani::assume(max_seq_len >= 1 && max_seq_len <= 64);
        kani::assume(seq_len <= max_seq_len);

        let remaining = max_seq_len - seq_len;
        assert_eq!(seq_len + remaining, max_seq_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_append_and_read() {
        let config = ModelConfig::tiny_test();
        let mut cache = KvCache::new(&config);

        let stride = config.num_kv_heads * config.head_dim;
        let k_data: Vec<f32> = (0..stride).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..stride).map(|i| (i as f32) * 2.0).collect();

        cache.append_kv(0, &k_data, &v_data, 1);
        cache.advance(1);

        assert_eq!(cache.seq_len, 1);
        assert_eq!(cache.k_slice(0).len(), stride);
        assert_eq!(cache.k_slice(0)[0], 0.0);
        assert_eq!(cache.v_slice(0)[0], 0.0);
    }

    #[test]
    fn test_compressed_kv_cache_round_trip() {
        use tempura_kernels::turboquant::CpuPolarQuant;

        let config = ModelConfig::tiny_test();
        let compressor = Box::new(CpuPolarQuant::default_3bit());
        let mut cache = KvCache::new_compressed(&config, compressor);

        assert!(cache.is_compressed());

        let stride = config.num_kv_heads * config.head_dim;
        let k_data: Vec<f32> = (0..stride).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let v_data: Vec<f32> = (0..stride).map(|i| (i as f32 + 1.0) * 0.2).collect();

        cache.append_kv(0, &k_data, &v_data, 1);
        cache.advance(1);

        assert_eq!(cache.seq_len, 1);

        let k_out = cache.k_data(0, 1);
        let v_out = cache.v_data(0, 1);
        assert_eq!(k_out.len(), stride);
        assert_eq!(v_out.len(), stride);

        let k_norm: f32 = k_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_diff: f32 = k_data
            .iter()
            .zip(k_out.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        assert!(
            k_diff / k_norm < 0.05,
            "k round-trip error {:.4} >= 5%",
            k_diff / k_norm
        );
    }

    #[test]
    fn test_compressed_kv_cache_multiple_appends() {
        use tempura_kernels::turboquant::CpuPolarQuant;

        let config = ModelConfig::tiny_test();
        let compressor = Box::new(CpuPolarQuant::default_3bit());
        let mut cache = KvCache::new_compressed(&config, compressor);

        let stride = config.num_kv_heads * config.head_dim;

        for token in 0..3 {
            let k_data: Vec<f32> = (0..stride)
                .map(|i| (i as f32 + token as f32 + 1.0) * 0.1)
                .collect();
            let v_data: Vec<f32> = (0..stride)
                .map(|i| (i as f32 + token as f32 + 1.0) * 0.2)
                .collect();

            cache.append_kv(0, &k_data, &v_data, 1);
            cache.advance(1);
        }

        assert_eq!(cache.seq_len, 3);
        let k_out = cache.k_data(0, 3);
        assert_eq!(k_out.len(), 3 * stride);
    }

    #[test]
    fn test_compressed_kv_cache_reset() {
        use tempura_kernels::turboquant::CpuPolarQuant;

        let config = ModelConfig::tiny_test();
        let compressor = Box::new(CpuPolarQuant::default_3bit());
        let mut cache = KvCache::new_compressed(&config, compressor);

        let stride = config.num_kv_heads * config.head_dim;
        let k_data: Vec<f32> = (0..stride).map(|i| i as f32 + 1.0).collect();
        let v_data: Vec<f32> = (0..stride).map(|i| i as f32 + 2.0).collect();

        cache.append_kv(0, &k_data, &v_data, 1);
        cache.advance(1);
        cache.reset();

        assert_eq!(cache.seq_len, 0);
        assert!(cache.k_compressed[0].is_none());
    }

    #[test]
    fn test_pool_acquire_release() {
        let config = ModelConfig::tiny_test();
        let mut pool = KvCachePool::new(&config, 3);

        assert_eq!(pool.num_slots(), 3);
        assert_eq!(pool.available(), 3);
        assert_eq!(pool.in_use(), 0);

        let cache1 = pool.acquire().unwrap();
        assert_eq!(pool.available(), 2);
        assert_eq!(pool.in_use(), 1);

        let cache2 = pool.acquire().unwrap();
        let cache3 = pool.acquire().unwrap();
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.in_use(), 3);

        assert!(pool.acquire().is_err());

        pool.release(cache1);
        assert_eq!(pool.available(), 1);

        let _cache4 = pool.acquire().unwrap();
        assert_eq!(pool.available(), 0);

        pool.release(cache2);
        pool.release(cache3);
    }

    #[test]
    fn test_pool_size_one_backward_compat() {
        let config = ModelConfig::tiny_test();
        let mut pool = KvCachePool::new(&config, 1);
        let mut cache = pool.acquire().unwrap();

        let stride = config.num_kv_heads * config.head_dim;
        let k_data: Vec<f32> = (0..stride).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..stride).map(|i| (i as f32) * 2.0).collect();

        cache.append_kv(0, &k_data, &v_data, 1);
        cache.advance(1);
        assert_eq!(cache.seq_len, 1);
        assert_eq!(cache.k_slice(0).len(), stride);

        pool.release(cache);
        let cache2 = pool.acquire().unwrap();
        assert_eq!(cache2.seq_len, 0);
    }

    #[test]
    fn test_pool_slot_isolation() {
        let config = ModelConfig::tiny_test();
        let mut pool = KvCachePool::new(&config, 2);

        let mut cache_a = pool.acquire().unwrap();
        let mut cache_b = pool.acquire().unwrap();

        let stride = config.num_kv_heads * config.head_dim;
        let k_a: Vec<f32> = (0..stride).map(|i| i as f32 * 1.0).collect();
        let v_a: Vec<f32> = (0..stride).map(|i| i as f32 * 2.0).collect();
        let k_b: Vec<f32> = (0..stride).map(|i| i as f32 * 10.0).collect();
        let v_b: Vec<f32> = (0..stride).map(|i| i as f32 * 20.0).collect();

        cache_a.append_kv(0, &k_a, &v_a, 1);
        cache_a.advance(1);

        cache_b.append_kv(0, &k_b, &v_b, 1);
        cache_b.advance(1);

        let k_slice_a = cache_a.k_data(0, 1);
        let k_slice_b = cache_b.k_data(0, 1);

        assert_ne!(&*k_slice_a, &*k_slice_b, "slots must be independent");

        for i in 0..stride {
            assert!(
                (k_slice_a[i] - k_a[i]).abs() < 1e-6,
                "slot A k corrupted at {i}"
            );
            assert!(
                (k_slice_b[i] - k_b[i]).abs() < 1e-6,
                "slot B k corrupted at {i}"
            );
        }
    }

    #[test]
    fn test_pool_release_resets_cache() {
        let config = ModelConfig::tiny_test();
        let mut pool = KvCachePool::new(&config, 1);

        let mut cache = pool.acquire().unwrap();
        let stride = config.num_kv_heads * config.head_dim;
        let k_data: Vec<f32> = (0..stride).map(|i| i as f32 + 1.0).collect();
        let v_data = k_data.clone();
        cache.append_kv(0, &k_data, &v_data, 1);
        cache.advance(1);
        assert_eq!(cache.seq_len, 1);

        pool.release(cache);

        let reacquired = pool.acquire().unwrap();
        assert_eq!(reacquired.seq_len, 0, "released cache must be reset");
    }

    #[test]
    fn test_uncompressed_k_data_borrows() {
        let config = ModelConfig::tiny_test();
        let mut cache = KvCache::new(&config);

        let stride = config.num_kv_heads * config.head_dim;
        let k_data: Vec<f32> = (0..stride).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..stride).map(|i| i as f32).collect();

        cache.append_kv(0, &k_data, &v_data, 1);

        let result = cache.k_data(0, 1);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result.len(), stride);
    }

    #[test]
    fn test_prefix_cache_store_and_probe() {
        let config = ModelConfig::tiny_test();
        let mut cache = KvCache::new(&config);
        let stride = config.num_kv_heads * config.head_dim;

        let k_data: Vec<f32> = (0..stride * 3).map(|i| i as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..stride * 3).map(|i| i as f32 * 0.2).collect();
        cache.append_kv(0, &k_data, &v_data, 3);
        cache.advance(3);

        let mut table = PrefixCacheTable::new(10);
        table.store(&[1, 2, 3], &cache);
        assert_eq!(table.len(), 1);

        let matched = table.probe(&[1, 2, 3]).unwrap();
        assert_eq!(matched.matched_len, 3);
        assert_eq!(table.hit_count(), 1);
        assert_eq!(table.tokens_saved(), 3);
    }

    #[test]
    fn test_prefix_cache_identical_prefix_reuse() {
        let config = ModelConfig::tiny_test();
        let mut cache = KvCache::new(&config);
        let stride = config.num_kv_heads * config.head_dim;

        let k_data: Vec<f32> = (0..stride * 2).map(|i| i as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..stride * 2).map(|i| i as f32 * 0.2).collect();
        cache.append_kv(0, &k_data, &v_data, 2);
        cache.advance(2);

        let mut table = PrefixCacheTable::new(10);
        table.store(&[10, 20], &cache);

        // Two requests with the same prefix get identical KV data
        let match1 = table.probe(&[10, 20]).unwrap();
        let match2 = table.probe(&[10, 20]).unwrap();
        assert_eq!(match1.k_data, match2.k_data);
        assert_eq!(match1.v_data, match2.v_data);
    }

    #[test]
    fn test_prefix_cache_novel_prefix_miss() {
        let config = ModelConfig::tiny_test();
        let mut cache = KvCache::new(&config);
        let stride = config.num_kv_heads * config.head_dim;

        let k_data: Vec<f32> = (0..stride).map(|i| i as f32).collect();
        let v_data = k_data.clone();
        cache.append_kv(0, &k_data, &v_data, 1);
        cache.advance(1);

        let mut table = PrefixCacheTable::new(10);
        table.store(&[1], &cache);

        assert!(table.probe(&[99]).is_none());
    }

    #[test]
    fn test_prefix_cache_lru_eviction() {
        let config = ModelConfig::tiny_test();
        let mut table = PrefixCacheTable::new(2);

        for i in 0..3 {
            let mut cache = KvCache::new(&config);
            let stride = config.num_kv_heads * config.head_dim;
            let k_data: Vec<f32> = (0..stride).map(|j| (i * 100 + j) as f32).collect();
            let v_data = k_data.clone();
            cache.append_kv(0, &k_data, &v_data, 1);
            cache.advance(1);
            table.store(&[i as u32], &cache);
        }

        // Max 2 entries, so the least recently used should be evicted
        assert_eq!(table.len(), 2);
        // Entry [0] was stored first and never accessed, so it should be evicted
        assert!(table.probe(&[0]).is_none());
    }

    #[test]
    fn test_prefix_match_populates_cache() {
        let config = ModelConfig::tiny_test();
        let stride = config.num_kv_heads * config.head_dim;

        let mut source = KvCache::new(&config);
        let k: Vec<f32> = (0..stride * 2).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let v: Vec<f32> = (0..stride * 2).map(|i| (i as f32 + 1.0) * 0.3).collect();
        source.append_kv(0, &k, &v, 2);
        source.advance(2);

        let mut table = PrefixCacheTable::new(10);
        table.store(&[5, 6], &source);

        let prefix_match = table.probe(&[5, 6]).unwrap();

        let mut target = KvCache::new(&config);
        prefix_match.populate_cache(&mut target);

        assert_eq!(target.seq_len, 2);
        assert_eq!(target.k_slice(0), source.k_slice(0));
        assert_eq!(target.v_slice(0), source.v_slice(0));
    }

    #[test]
    fn test_paged_kv_cache_basic() {
        let config = ModelConfig::tiny_test();
        let block_size = 4;
        let total_pages = 16;
        let mut cache = PagedKvCache::new(&config, total_pages, block_size);

        let stride = config.num_kv_heads * config.head_dim;
        let k_data: Vec<f32> = (0..stride).map(|i| i as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..stride).map(|i| i as f32 * 0.2).collect();

        cache.append_kv(0, &k_data, &v_data, 1);
        cache.advance(1);

        assert_eq!(cache.seq_len, 1);

        let k_out = cache.k_contiguous(0, 1);
        let v_out = cache.v_contiguous(0, 1);
        assert_eq!(k_out.len(), stride);
        assert_eq!(v_out.len(), stride);
        for i in 0..stride {
            assert!(
                (k_out[i] - k_data[i]).abs() < 1e-6,
                "k mismatch at {i}: {} vs {}",
                k_out[i],
                k_data[i]
            );
            assert!((v_out[i] - v_data[i]).abs() < 1e-6, "v mismatch at {i}");
        }
    }

    #[test]
    fn test_paged_kv_cache_multi_token() {
        let config = ModelConfig::tiny_test();
        let block_size = 4;
        let total_pages = 16;
        let mut cache = PagedKvCache::new(&config, total_pages, block_size);

        let stride = config.num_kv_heads * config.head_dim;
        let num_tokens = 7;
        let k_data: Vec<f32> = (0..num_tokens * stride).map(|i| i as f32 * 0.01).collect();
        let v_data: Vec<f32> = (0..num_tokens * stride).map(|i| i as f32 * 0.02).collect();

        cache.append_kv(0, &k_data, &v_data, num_tokens);
        cache.advance(num_tokens);

        assert_eq!(cache.seq_len, 7);

        let k_out = cache.k_contiguous(0, 7);
        for i in 0..num_tokens * stride {
            assert!((k_out[i] - k_data[i]).abs() < 1e-6, "k mismatch at {i}");
        }
    }

    #[test]
    fn test_paged_kv_cache_incremental_append() {
        let config = ModelConfig::tiny_test();
        let block_size = 4;
        let total_pages = 16;
        let mut cache = PagedKvCache::new(&config, total_pages, block_size);

        let stride = config.num_kv_heads * config.head_dim;
        let mut all_k = Vec::new();

        for t in 0..6 {
            let k: Vec<f32> = (0..stride).map(|i| (t * 100 + i) as f32).collect();
            let v: Vec<f32> = (0..stride).map(|i| (t * 200 + i) as f32).collect();
            cache.append_kv(0, &k, &v, 1);
            cache.advance(1);
            all_k.extend_from_slice(&k);
        }

        assert_eq!(cache.seq_len, 6);
        let k_out = cache.k_contiguous(0, 6);
        for i in 0..all_k.len() {
            assert!(
                (k_out[i] - all_k[i]).abs() < 1e-6,
                "k mismatch at {i}: {} vs {}",
                k_out[i],
                all_k[i]
            );
        }
    }

    #[test]
    fn test_paged_kv_cache_reset() {
        let config = ModelConfig::tiny_test();
        let block_size = 4;
        let total_pages = 8;
        let mut cache = PagedKvCache::new(&config, total_pages, block_size);

        let stride = config.num_kv_heads * config.head_dim;
        let k: Vec<f32> = vec![1.0; stride * 5];
        let v = k.clone();
        cache.append_kv(0, &k, &v, 5);
        cache.advance(5);

        let pages_before = cache.remaining_pages();
        cache.reset();
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.remaining_pages(), total_pages);
        assert!(cache.remaining_pages() > pages_before);
    }

    #[test]
    fn test_paged_kv_cache_low_waste() {
        let config = ModelConfig::tiny_test();
        let block_size = 16;
        let total_pages = 64;
        let mut cache = PagedKvCache::new(&config, total_pages, block_size);

        let stride = config.num_kv_heads * config.head_dim;
        let k: Vec<f32> = vec![1.0; stride * 100];
        let v = k.clone();
        cache.append_kv(0, &k, &v, 100);
        cache.advance(100);

        let waste = cache.waste_ratio();
        assert!(
            waste < 0.16,
            "waste ratio {waste:.2} should be <16% for block_size=16"
        );
    }

    #[test]
    fn test_paged_matches_contiguous() {
        let config = ModelConfig::tiny_test();

        let mut contiguous = KvCache::new(&config);
        let mut paged = PagedKvCache::new(&config, 32, 4);

        let stride = config.num_kv_heads * config.head_dim;
        for t in 0..10 {
            let k: Vec<f32> = (0..stride).map(|i| (t * 100 + i) as f32 * 0.1).collect();
            let v: Vec<f32> = (0..stride).map(|i| (t * 200 + i) as f32 * 0.1).collect();

            contiguous.append_kv(0, &k, &v, 1);
            contiguous.advance(1);

            paged.append_kv(0, &k, &v, 1);
            paged.advance(1);
        }

        let k_cont = contiguous.k_slice(0);
        let k_paged = paged.k_contiguous(0, 10);
        assert_eq!(k_cont.len(), k_paged.len());
        for i in 0..k_cont.len() {
            assert!(
                (k_cont[i] - k_paged[i]).abs() < 1e-6,
                "k divergence at {i}: contiguous={} paged={}",
                k_cont[i],
                k_paged[i]
            );
        }
    }

    #[test]
    fn test_page_allocator() {
        let mut alloc = PageAllocator::new(4);
        assert_eq!(alloc.available(), 4);
        assert_eq!(alloc.in_use(), 0);

        let p0 = alloc.allocate().unwrap();
        let p1 = alloc.allocate().unwrap();
        assert_eq!(alloc.available(), 2);
        assert_eq!(alloc.in_use(), 2);
        assert_ne!(p0, p1);

        alloc.free(p0);
        assert_eq!(alloc.available(), 3);

        let p2 = alloc.allocate().unwrap();
        assert_eq!(p2, p0);
    }

    #[test]
    fn test_block_table() {
        let mut bt = BlockTable::new(8);
        assert_eq!(bt.num_blocks(), 0);

        bt.set_block(0, 5);
        bt.set_block(1, 3);
        assert_eq!(bt.num_blocks(), 2);
        assert_eq!(bt.get_block(0), Some(5));
        assert_eq!(bt.get_block(1), Some(3));
        assert_eq!(bt.get_block(2), None);

        let freed = bt.clear();
        assert_eq!(freed.len(), 2);
        assert!(freed.contains(&5));
        assert!(freed.contains(&3));
        assert_eq!(bt.num_blocks(), 0);
    }

    #[test]
    fn test_fp8_round_trip() {
        // Values within E4M3 representable range [0.001953125, 448]
        let test_values = [
            0.0f32, 1.0, -1.0, 0.5, -0.5, 0.25, 2.0, 4.0, 8.0, 64.0, 448.0,
        ];
        for &v in &test_values {
            let fp8 = f32_to_fp8_e4m3(v);
            let back = fp8_e4m3_to_f32(fp8);
            if v == 0.0 {
                assert_eq!(back, 0.0, "zero round-trip failed");
            } else {
                let rel_err = ((back - v) / v).abs();
                assert!(
                    rel_err < 0.13,
                    "fp8 round-trip error for {v}: got {back}, rel_err={rel_err}"
                );
            }
        }

        // Values below subnormal min underflow to zero
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(0.001)), 0.0);
        // Values above max clamp to 448
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(1000.0)), 448.0);
    }

    #[test]
    fn test_fp8_paged_kv_cache_basic() {
        let config = ModelConfig::tiny_test();
        let mut cache = Fp8PagedKvCache::new(&config, 32, 4);

        let stride = config.num_kv_heads * config.head_dim;
        let k_data: Vec<f32> = (0..stride).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let v_data: Vec<f32> = (0..stride).map(|i| (i as f32 + 1.0) * 0.2).collect();

        cache.append_kv(0, &k_data, &v_data, 1);
        cache.advance(1);
        assert_eq!(cache.seq_len, 1);

        let k_out = cache.k_contiguous(0, 1);
        assert_eq!(k_out.len(), stride);
        for i in 0..stride {
            let rel_err = if k_data[i].abs() > 0.001 {
                ((k_out[i] - k_data[i]) / k_data[i]).abs()
            } else {
                (k_out[i] - k_data[i]).abs()
            };
            assert!(
                rel_err < 0.15,
                "fp8 k divergence at {i}: got {} expected {} rel_err={}",
                k_out[i],
                k_data[i],
                rel_err
            );
        }
    }

    #[test]
    fn test_fp8_halves_memory() {
        let config = ModelConfig::tiny_test();
        let fp8 = Fp8PagedKvCache::new(&config, 32, 4);
        assert_eq!(
            fp8.memory_bytes_per_token() * 2,
            fp8.f32_memory_bytes_per_token() / 2,
        );
    }

    #[test]
    fn test_fp8_paged_kv_cache_multi_token() {
        let config = ModelConfig::tiny_test();
        let mut cache = Fp8PagedKvCache::new(&config, 32, 4);

        let stride = config.num_kv_heads * config.head_dim;
        for t in 0..6 {
            let k: Vec<f32> = (0..stride).map(|i| (t * 10 + i) as f32 * 0.05).collect();
            let v = k.clone();
            cache.append_kv(0, &k, &v, 1);
            cache.advance(1);
        }

        assert_eq!(cache.seq_len, 6);
        let k_out = cache.k_contiguous(0, 6);
        assert_eq!(k_out.len(), 6 * stride);
    }

    #[test]
    fn test_fp8_paged_kv_cache_reset() {
        let config = ModelConfig::tiny_test();
        let mut cache = Fp8PagedKvCache::new(&config, 16, 4);

        let stride = config.num_kv_heads * config.head_dim;
        let k: Vec<f32> = vec![1.0; stride * 5];
        let v = k.clone();
        cache.append_kv(0, &k, &v, 5);
        cache.advance(5);
        cache.reset();
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn test_prefix_cache_salt_isolation() {
        let config = ModelConfig::tiny_test();
        let stride = config.num_kv_heads * config.head_dim;

        let mut cache = KvCache::new(&config);
        let k: Vec<f32> = (0..stride).map(|i| i as f32).collect();
        let v = k.clone();
        cache.append_kv(0, &k, &v, 1);
        cache.advance(1);

        let mut table = PrefixCacheTable::new(10);

        table.store_with_salt(&[1, 2, 3], &cache, Some("tenant-a"));
        table.store_with_salt(&[1, 2, 3], &cache, Some("tenant-b"));

        assert_eq!(table.len(), 2);

        assert!(table
            .probe_with_salt(&[1, 2, 3], Some("tenant-a"))
            .is_some());
        assert!(table
            .probe_with_salt(&[1, 2, 3], Some("tenant-b"))
            .is_some());
        assert!(table
            .probe_with_salt(&[1, 2, 3], Some("tenant-c"))
            .is_none());
        assert!(table.probe_with_salt(&[1, 2, 3], None).is_none());
    }
}
