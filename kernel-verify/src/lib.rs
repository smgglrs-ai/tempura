use rand::{Rng, SeedableRng};
use serde::Serialize;
use tempura_kernel_api::{Attention, MatMul, RMSNorm, RoPE};

pub mod bitnet;
pub mod composition;
pub mod test_generators;

pub use composition::{CompositionReport, CompositionVerifier, OpShape};

#[derive(Debug, Clone, Serialize)]
pub struct VerificationReport {
    pub kernel_name: String,
    pub results: Vec<KernelTestResult>,
    pub pass: bool,
    pub tier: VerificationTier,
}

pub use tempura_kernel_api::VerificationTier;

#[derive(Debug, Clone, Serialize)]
pub struct KernelTestResult {
    pub test_name: String,
    pub pass: bool,
    pub max_abs_diff: f32,
    pub l2_distance: f32,
    pub detail: String,
}

pub struct DualPipelineRunner {
    pub epsilon: f32,
}

impl Default for DualPipelineRunner {
    fn default() -> Self {
        Self { epsilon: 1e-4 }
    }
}

impl DualPipelineRunner {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }

    pub fn compare_attention(
        &self,
        name: &str,
        impl_a: &dyn Attention,
        impl_b: &dyn Attention,
        configs: &[AttentionConfig],
    ) -> VerificationReport {
        let mut results = Vec::new();
        for cfg in configs {
            let result = compare_attention_single(impl_a, impl_b, cfg, self.epsilon);
            results.push(result);
        }
        let pass = results.iter().all(|r| r.pass);
        VerificationReport {
            kernel_name: name.into(),
            results,
            pass,
            tier: if pass {
                VerificationTier::Tested
            } else {
                VerificationTier::Compiled
            },
        }
    }

    pub fn compare_matmul(
        &self,
        name: &str,
        impl_a: &dyn MatMul,
        impl_b: &dyn MatMul,
        configs: &[MatMulConfig],
    ) -> VerificationReport {
        let mut results = Vec::new();
        for cfg in configs {
            let result = compare_matmul_single(impl_a, impl_b, cfg, self.epsilon);
            results.push(result);
        }
        let pass = results.iter().all(|r| r.pass);
        VerificationReport {
            kernel_name: name.into(),
            results,
            pass,
            tier: if pass {
                VerificationTier::Tested
            } else {
                VerificationTier::Compiled
            },
        }
    }

    pub fn compare_rmsnorm(
        &self,
        name: &str,
        impl_a: &dyn RMSNorm,
        impl_b: &dyn RMSNorm,
        configs: &[RMSNormConfig],
    ) -> VerificationReport {
        let mut results = Vec::new();
        for cfg in configs {
            let result = compare_rmsnorm_single(impl_a, impl_b, cfg, self.epsilon);
            results.push(result);
        }
        let pass = results.iter().all(|r| r.pass);
        VerificationReport {
            kernel_name: name.into(),
            results,
            pass,
            tier: if pass {
                VerificationTier::Tested
            } else {
                VerificationTier::Compiled
            },
        }
    }

    pub fn compare_rope(
        &self,
        name: &str,
        impl_a: &dyn RoPE,
        impl_b: &dyn RoPE,
        configs: &[RoPEConfig],
    ) -> VerificationReport {
        let mut results = Vec::new();
        for cfg in configs {
            let result = compare_rope_single(impl_a, impl_b, cfg, self.epsilon);
            results.push(result);
        }
        let pass = results.iter().all(|r| r.pass);
        VerificationReport {
            kernel_name: name.into(),
            results,
            pass,
            tier: if pass {
                VerificationTier::Tested
            } else {
                VerificationTier::Compiled
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub kv_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone)]
pub struct MatMulConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug, Clone)]
pub struct RMSNormConfig {
    pub tokens: usize,
    pub hidden_size: usize,
    pub eps: f32,
}

#[derive(Debug, Clone)]
pub struct RoPEConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub theta: f32,
}

fn random_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn compare_attention_single(
    a: &dyn Attention,
    b: &dyn Attention,
    cfg: &AttentionConfig,
    epsilon: f32,
) -> KernelTestResult {
    let q_len = cfg.batch * cfg.seq_len * cfg.num_heads * cfg.head_dim;
    let kv_len = cfg.batch * cfg.kv_len * cfg.num_kv_heads * cfg.head_dim;
    let out_len = cfg.batch * cfg.seq_len * cfg.num_heads * cfg.head_dim;

    let q = random_vec(q_len, 1);
    let k = random_vec(kv_len, 2);
    let v = random_vec(kv_len, 3);
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();

    let mut out_a = vec![0.0f32; out_len];
    let mut out_b = vec![0.0f32; out_len];

    let res_a = a.forward(
        &q,
        &k,
        &v,
        &mut out_a,
        cfg.batch,
        cfg.seq_len,
        cfg.kv_len,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        scale,
    );
    let res_b = b.forward(
        &q,
        &k,
        &v,
        &mut out_b,
        cfg.batch,
        cfg.seq_len,
        cfg.kv_len,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        scale,
    );

    if let (Ok(()), Ok(())) = (res_a, res_b) {
        let mad = max_abs_diff(&out_a, &out_b);
        let l2 = l2_distance(&out_a, &out_b);
        let pass = mad <= epsilon;
        KernelTestResult {
            test_name: format!(
                "attention b={} s={} kv={} h={} d={}",
                cfg.batch, cfg.seq_len, cfg.kv_len, cfg.num_heads, cfg.head_dim
            ),
            pass,
            max_abs_diff: mad,
            l2_distance: l2,
            detail: if pass {
                "within epsilon".into()
            } else {
                format!("max_abs_diff {mad:.6} exceeds epsilon {epsilon:.6}")
            },
        }
    } else {
        KernelTestResult {
            test_name: format!("attention b={} s={}", cfg.batch, cfg.seq_len),
            pass: false,
            max_abs_diff: f32::NAN,
            l2_distance: f32::NAN,
            detail: "one or both implementations returned an error".into(),
        }
    }
}

fn compare_matmul_single(
    a: &dyn MatMul,
    b: &dyn MatMul,
    cfg: &MatMulConfig,
    epsilon: f32,
) -> KernelTestResult {
    let a_data = random_vec(cfg.m * cfg.k, 10);
    let b_data = random_vec(cfg.k * cfg.n, 11);

    let mut out_a = vec![0.0f32; cfg.m * cfg.n];
    let mut out_b = vec![0.0f32; cfg.m * cfg.n];

    let res_a = a.forward(&a_data, &b_data, &mut out_a, cfg.m, cfg.n, cfg.k);
    let res_b = b.forward(&a_data, &b_data, &mut out_b, cfg.m, cfg.n, cfg.k);

    if let (Ok(()), Ok(())) = (res_a, res_b) {
        let mad = max_abs_diff(&out_a, &out_b);
        let l2 = l2_distance(&out_a, &out_b);
        let pass = mad <= epsilon;
        KernelTestResult {
            test_name: format!("matmul m={} n={} k={}", cfg.m, cfg.n, cfg.k),
            pass,
            max_abs_diff: mad,
            l2_distance: l2,
            detail: if pass {
                "within epsilon".into()
            } else {
                format!("max_abs_diff {mad:.6} exceeds epsilon {epsilon:.6}")
            },
        }
    } else {
        KernelTestResult {
            test_name: format!("matmul m={} n={} k={}", cfg.m, cfg.n, cfg.k),
            pass: false,
            max_abs_diff: f32::NAN,
            l2_distance: f32::NAN,
            detail: "one or both implementations returned an error".into(),
        }
    }
}

fn compare_rmsnorm_single(
    a: &dyn RMSNorm,
    b: &dyn RMSNorm,
    cfg: &RMSNormConfig,
    epsilon: f32,
) -> KernelTestResult {
    let x_data = random_vec(cfg.tokens * cfg.hidden_size, 20);
    let weight = random_vec(cfg.hidden_size, 21);

    let mut x_a = x_data.clone();
    let mut x_b = x_data.clone();

    let res_a = a.forward(&mut x_a, &weight, cfg.hidden_size, cfg.eps);
    let res_b = b.forward(&mut x_b, &weight, cfg.hidden_size, cfg.eps);

    if let (Ok(()), Ok(())) = (res_a, res_b) {
        let mad = max_abs_diff(&x_a, &x_b);
        let l2 = l2_distance(&x_a, &x_b);
        let pass = mad <= epsilon;
        KernelTestResult {
            test_name: format!("rmsnorm tokens={} hidden={}", cfg.tokens, cfg.hidden_size),
            pass,
            max_abs_diff: mad,
            l2_distance: l2,
            detail: if pass {
                "within epsilon".into()
            } else {
                format!("max_abs_diff {mad:.6} exceeds epsilon {epsilon:.6}")
            },
        }
    } else {
        KernelTestResult {
            test_name: format!("rmsnorm tokens={} hidden={}", cfg.tokens, cfg.hidden_size),
            pass: false,
            max_abs_diff: f32::NAN,
            l2_distance: f32::NAN,
            detail: "one or both implementations returned an error".into(),
        }
    }
}

fn compare_rope_single(
    a: &dyn RoPE,
    b: &dyn RoPE,
    cfg: &RoPEConfig,
    epsilon: f32,
) -> KernelTestResult {
    let x_data = random_vec(cfg.batch * cfg.seq_len * cfg.num_heads * cfg.head_dim, 30);
    let positions: Vec<u32> = (0..cfg.batch * cfg.seq_len).map(|i| i as u32).collect();

    let mut x_a = x_data.clone();
    let mut x_b = x_data.clone();

    let res_a = a.forward(
        &mut x_a,
        cfg.batch,
        cfg.seq_len,
        cfg.num_heads,
        cfg.head_dim,
        &positions,
        cfg.theta,
    );
    let res_b = b.forward(
        &mut x_b,
        cfg.batch,
        cfg.seq_len,
        cfg.num_heads,
        cfg.head_dim,
        &positions,
        cfg.theta,
    );

    if let (Ok(()), Ok(())) = (res_a, res_b) {
        let mad = max_abs_diff(&x_a, &x_b);
        let l2 = l2_distance(&x_a, &x_b);
        let pass = mad <= epsilon;
        KernelTestResult {
            test_name: format!(
                "rope b={} s={} h={} d={}",
                cfg.batch, cfg.seq_len, cfg.num_heads, cfg.head_dim
            ),
            pass,
            max_abs_diff: mad,
            l2_distance: l2,
            detail: if pass {
                "within epsilon".into()
            } else {
                format!("max_abs_diff {mad:.6} exceeds epsilon {epsilon:.6}")
            },
        }
    } else {
        KernelTestResult {
            test_name: format!("rope b={} s={}", cfg.batch, cfg.seq_len),
            pass: false,
            max_abs_diff: f32::NAN,
            l2_distance: f32::NAN,
            detail: "one or both implementations returned an error".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempura_kernels::attention::CpuAttention;
    use tempura_kernels::matmul::CpuMatMul;
    use tempura_kernels::rmsnorm::CpuRMSNorm;
    use tempura_kernels::rope::CpuRoPE;

    #[test]
    fn test_identical_attention_zero_divergence() {
        let runner = DualPipelineRunner::new(1e-6);
        let cpu = CpuAttention;
        let configs = test_generators::attention_configs();
        let report = runner.compare_attention("CpuAttention-self", &cpu, &cpu, &configs);
        assert!(report.pass);
        assert_eq!(report.tier, VerificationTier::Tested);
        for r in &report.results {
            assert_eq!(r.max_abs_diff, 0.0);
        }
    }

    #[test]
    fn test_identical_matmul_zero_divergence() {
        let runner = DualPipelineRunner::new(1e-6);
        let cpu = CpuMatMul;
        let configs = test_generators::matmul_configs();
        let report = runner.compare_matmul("CpuMatMul-self", &cpu, &cpu, &configs);
        assert!(report.pass);
        for r in &report.results {
            assert_eq!(r.max_abs_diff, 0.0);
        }
    }

    #[test]
    fn test_identical_rmsnorm_zero_divergence() {
        let runner = DualPipelineRunner::new(1e-6);
        let cpu = CpuRMSNorm;
        let configs = test_generators::rmsnorm_configs();
        let report = runner.compare_rmsnorm("CpuRMSNorm-self", &cpu, &cpu, &configs);
        assert!(report.pass);
    }

    #[test]
    fn test_identical_rope_zero_divergence() {
        let runner = DualPipelineRunner::new(1e-6);
        let cpu = CpuRoPE;
        let configs = test_generators::rope_configs();
        let report = runner.compare_rope("CpuRoPE-self", &cpu, &cpu, &configs);
        assert!(report.pass);
    }

    #[test]
    fn test_report_serializes_to_json() {
        let runner = DualPipelineRunner::default();
        let cpu = CpuAttention;
        let configs = vec![test_generators::attention_configs()[0].clone()];
        let report = runner.compare_attention("test", &cpu, &cpu, &configs);
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("\"pass\": true"));
        assert!(json.contains("\"tier\": \"tested\""));
    }
}
