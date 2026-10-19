use crate::{AttentionConfig, I2SConfig, MatMulConfig, Q4kConfig, RMSNormConfig, RoPEConfig, TQ2Config};

pub fn attention_configs() -> Vec<AttentionConfig> {
    vec![
        AttentionConfig {
            batch: 1,
            seq_len: 4,
            kv_len: 4,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
        },
        AttentionConfig {
            batch: 1,
            seq_len: 1,
            kv_len: 16,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 16,
        },
        AttentionConfig {
            batch: 2,
            seq_len: 8,
            kv_len: 8,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 32,
        },
    ]
}

pub fn matmul_configs() -> Vec<MatMulConfig> {
    vec![
        MatMulConfig { m: 1, n: 16, k: 8 },
        MatMulConfig { m: 4, n: 32, k: 16 },
        MatMulConfig { m: 8, n: 64, k: 64 },
        MatMulConfig {
            m: 1,
            n: 1024,
            k: 512,
        },
    ]
}

pub fn rmsnorm_configs() -> Vec<RMSNormConfig> {
    vec![
        RMSNormConfig {
            tokens: 1,
            hidden_size: 16,
            eps: 1e-5,
        },
        RMSNormConfig {
            tokens: 4,
            hidden_size: 64,
            eps: 1e-6,
        },
        RMSNormConfig {
            tokens: 8,
            hidden_size: 256,
            eps: 1e-5,
        },
    ]
}

pub fn rope_configs() -> Vec<RoPEConfig> {
    vec![
        RoPEConfig {
            batch: 1,
            seq_len: 4,
            num_heads: 4,
            head_dim: 8,
            theta: 10000.0,
        },
        RoPEConfig {
            batch: 1,
            seq_len: 16,
            num_heads: 8,
            head_dim: 32,
            theta: 10000.0,
        },
        RoPEConfig {
            batch: 2,
            seq_len: 8,
            num_heads: 4,
            head_dim: 16,
            theta: 500000.0,
        },
    ]
}

pub fn q4k_configs() -> Vec<Q4kConfig> {
    vec![
        Q4kConfig { m: 1, n: 16, k: 256 },
        Q4kConfig { m: 1, n: 64, k: 512 },
        Q4kConfig { m: 4, n: 32, k: 256 },
    ]
}

pub fn tq2_configs() -> Vec<TQ2Config> {
    vec![
        TQ2Config { m: 1, n: 16, k: 256 },
        TQ2Config { m: 1, n: 64, k: 512 },
        TQ2Config { m: 4, n: 32, k: 256 },
    ]
}

pub fn i2s_configs() -> Vec<I2SConfig> {
    vec![
        I2SConfig { m: 1, n: 16, k: 256 },
        I2SConfig { m: 1, n: 64, k: 512 },
        I2SConfig { m: 4, n: 32, k: 256 },
    ]
}
