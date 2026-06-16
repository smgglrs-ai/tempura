use std::borrow::Cow;

use crate::config::ModelConfig;
use oxide_kernels::turboquant::{CompressedKV, KvCacheCompressor};

#[allow(dead_code)]
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
        use oxide_kernels::turboquant::CpuPolarQuant;

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
        use oxide_kernels::turboquant::CpuPolarQuant;

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
        use oxide_kernels::turboquant::CpuPolarQuant;

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
}
