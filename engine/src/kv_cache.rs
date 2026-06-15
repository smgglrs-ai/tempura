use crate::config::ModelConfig;

#[allow(dead_code)]
pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
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
        }
    }

    pub fn append_kv(
        &mut self,
        layer: usize,
        new_k: &[f32],
        new_v: &[f32],
        num_new_tokens: usize,
    ) {
        let stride = self.num_kv_heads * self.head_dim;
        let dst_offset = self.seq_len * stride;
        let src_len = num_new_tokens * stride;

        self.k[layer][dst_offset..dst_offset + src_len].copy_from_slice(&new_k[..src_len]);
        self.v[layer][dst_offset..dst_offset + src_len].copy_from_slice(&new_v[..src_len]);
    }

    pub fn advance(&mut self, num_tokens: usize) {
        self.seq_len += num_tokens;
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    pub fn k_slice(&self, layer: usize) -> &[f32] {
        let len = self.seq_len * self.num_kv_heads * self.head_dim;
        &self.k[layer][..len]
    }

    pub fn v_slice(&self, layer: usize) -> &[f32] {
        let len = self.seq_len * self.num_kv_heads * self.head_dim;
        &self.v[layer][..len]
    }

    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len - self.seq_len
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
}
