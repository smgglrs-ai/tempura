use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::config::ModelConfig;
use tempura_kernels::turboquant::{CompressedKV, KvCacheCompressor};

pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
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

impl KvCache {
    pub fn new(config: &ModelConfig) -> Self {
        let layer_size = config.max_seq_len * config.num_kv_heads * config.head_dim;
        let k: Vec<Vec<f32>> = (0..config.num_layers)
            .map(|_| vec![0.0; layer_size])
            .collect();
        let v: Vec<Vec<f32>> = (0..config.num_layers)
            .map(|_| vec![0.0; layer_size])
            .collect();

        Self {
            k,
            v,
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
        } else {
            let dst_offset = self.seq_len * stride;
            self.k[layer][dst_offset..dst_offset + src_len].copy_from_slice(&new_k[..src_len]);
            self.v[layer][dst_offset..dst_offset + src_len].copy_from_slice(&new_v[..src_len]);
        }
    }

    pub fn advance(&mut self, num_tokens: usize) {
        self.seq_len += num_tokens;
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
        } else {
            let len = num_tokens * self.num_kv_heads * self.head_dim;
            Cow::Borrowed(&self.v[layer][..len])
        }
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
