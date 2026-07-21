use crate::Result;
pub use crate::{CompressedKV, KvCacheCompressor};

#[cfg(feature = "cpu")]
pub struct CpuPolarQuant {
    pub bits: u8,
    pub group_size: usize,
}

#[cfg(feature = "cpu")]
impl CpuPolarQuant {
    pub fn new(bits: u8, group_size: usize) -> Result<Self> {
        if bits == 0 || bits > 8 {
            return Err(crate::KernelError::InvalidConfig(
                "bits must be 1..=8".into(),
            ));
        }
        if group_size == 0 {
            return Err(crate::KernelError::InvalidConfig(
                "group_size must be > 0".into(),
            ));
        }
        Ok(Self { bits, group_size })
    }

    pub fn default_3bit() -> Self {
        Self {
            bits: 3,
            group_size: 32,
        }
    }

    pub fn default_2bit() -> Self {
        Self {
            bits: 2,
            group_size: 32,
        }
    }
}

#[cfg(feature = "cpu")]
impl KvCacheCompressor for CpuPolarQuant {
    fn compress(&self, data: &[f32], head_dim: usize) -> Result<CompressedKV> {
        if head_dim == 0 {
            return Err(crate::KernelError::InvalidConfig(
                "head_dim must be > 0".into(),
            ));
        }
        if !data.len().is_multiple_of(head_dim) {
            return Err(crate::KernelError::DimensionMismatch {
                expected: 0,
                got: data.len() % head_dim,
            });
        }

        let num_vectors = data.len() / head_dim;
        let groups_per_vec = head_dim.div_ceil(self.group_size);
        let total_groups = num_vectors * groups_per_vec;
        let levels = (1u32 << self.bits) - 1;

        let mut mins = Vec::with_capacity(total_groups);
        let mut maxs = Vec::with_capacity(total_groups);
        let mut quantized = Vec::with_capacity(data.len());

        for vi in 0..num_vectors {
            let vec_start = vi * head_dim;

            for gi in 0..groups_per_vec {
                let g_start = vec_start + gi * self.group_size;
                let g_end = (g_start + self.group_size).min(vec_start + head_dim);
                let group = &data[g_start..g_end];

                let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                mins.push(min_val);
                maxs.push(max_val);

                let range = max_val - min_val;
                for &val in group {
                    if range < 1e-30 {
                        quantized.push(0);
                    } else {
                        let normalized = (val - min_val) / range;
                        let q = (normalized * levels as f32).round().min(levels as f32) as u8;
                        quantized.push(q);
                    }
                }
            }
        }

        Ok(CompressedKV {
            mins,
            maxs,
            quantized,
            num_vectors,
            head_dim,
            group_size: self.group_size,
            bits: self.bits,
        })
    }

    fn compress_append(&self, existing: &mut CompressedKV, new_data: &[f32]) -> Result<()> {
        let head_dim = existing.head_dim;
        if head_dim == 0 {
            return Err(crate::KernelError::InvalidConfig(
                "head_dim must be > 0".into(),
            ));
        }
        if !new_data.len().is_multiple_of(head_dim) {
            return Err(crate::KernelError::DimensionMismatch {
                expected: 0,
                got: new_data.len() % head_dim,
            });
        }

        let num_new_vectors = new_data.len() / head_dim;
        let groups_per_vec = head_dim.div_ceil(self.group_size);
        let levels = (1u32 << self.bits) - 1;

        for vi in 0..num_new_vectors {
            let vec_start = vi * head_dim;

            for gi in 0..groups_per_vec {
                let g_start = vec_start + gi * self.group_size;
                let g_end = (g_start + self.group_size).min(vec_start + head_dim);
                let group = &new_data[g_start..g_end];

                let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                existing.mins.push(min_val);
                existing.maxs.push(max_val);

                let range = max_val - min_val;
                for &val in group {
                    if range < 1e-30 {
                        existing.quantized.push(0);
                    } else {
                        let normalized = (val - min_val) / range;
                        let q = (normalized * levels as f32).round().min(levels as f32) as u8;
                        existing.quantized.push(q);
                    }
                }
            }
        }

        existing.num_vectors += num_new_vectors;
        Ok(())
    }

    fn decompress(&self, compressed: &CompressedKV) -> Result<Vec<f32>> {
        let head_dim = compressed.head_dim;
        let group_size = compressed.group_size;
        let groups_per_vec = compressed.groups_per_vector();
        let levels = (1u32 << compressed.bits) - 1;
        let mut output = Vec::with_capacity(compressed.num_vectors * head_dim);

        let mut q_idx = 0;
        for vi in 0..compressed.num_vectors {
            for gi in 0..groups_per_vec {
                let group_idx = vi * groups_per_vec + gi;
                let min_val = compressed.mins[group_idx];
                let max_val = compressed.maxs[group_idx];
                let range = max_val - min_val;

                let g_start = gi * group_size;
                let g_len = group_size.min(head_dim - g_start);

                for _ in 0..g_len {
                    let q = compressed.quantized[q_idx];
                    q_idx += 1;
                    if range < 1e-30 {
                        output.push(min_val);
                    } else {
                        let val = min_val + (q as f32 / levels as f32) * range;
                        output.push(val);
                    }
                }
            }
        }

        Ok(output)
    }
}

#[cfg(feature = "cuda")]
pub struct CudaPolarQuant {
    #[allow(dead_code)]
    gpu: crate::gpu::GpuContext,
    bits: u8,
    group_size: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaPolarQuant {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaPolarQuant {}

#[cfg(feature = "cuda")]
impl CudaPolarQuant {
    pub fn new(bits: u8, group_size: usize) -> Result<Self> {
        if bits == 0 || bits > 8 {
            return Err(crate::KernelError::InvalidConfig(
                "bits must be 1..=8".into(),
            ));
        }
        Ok(Self {
            gpu: crate::gpu::GpuContext::new(0)?,
            bits,
            group_size,
        })
    }
}

#[cfg(feature = "cuda")]
impl KvCacheCompressor for CudaPolarQuant {
    fn compress(&self, _data: &[f32], _head_dim: usize) -> Result<CompressedKV> {
        todo!("CudaPolarQuant compress: GPU kernel not yet implemented")
    }

    fn decompress(&self, _compressed: &CompressedKV) -> Result<Vec<f32>> {
        todo!("CudaPolarQuant decompress: GPU kernel not yet implemented")
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn compress_indices_in_bounds() {
        let head_dim: usize = kani::any();
        let num_vectors: usize = kani::any();
        let group_size: usize = kani::any();

        kani::assume(head_dim >= 1 && head_dim <= 8);
        kani::assume(num_vectors >= 1 && num_vectors <= 4);
        kani::assume(group_size >= 1 && group_size <= 8);

        let data_len = num_vectors * head_dim;
        let groups_per_vec = (head_dim + group_size - 1) / group_size;
        let total_groups = num_vectors * groups_per_vec;

        let vi: usize = kani::any();
        let gi: usize = kani::any();
        kani::assume(vi < num_vectors);
        kani::assume(gi < groups_per_vec);

        let vec_start = vi * head_dim;
        let g_start = vec_start + gi * group_size;
        let g_end_candidate = g_start + group_size;
        let g_end = if g_end_candidate < vec_start + head_dim {
            g_end_candidate
        } else {
            vec_start + head_dim
        };

        assert!(g_start < data_len);
        assert!(g_end <= data_len);

        let group_idx = vi * groups_per_vec + gi;
        assert!(group_idx < total_groups);
    }

    #[kani::proof]
    fn decompress_indices_in_bounds() {
        let head_dim: usize = kani::any();
        let num_vectors: usize = kani::any();
        let group_size: usize = kani::any();

        kani::assume(head_dim >= 1 && head_dim <= 8);
        kani::assume(num_vectors >= 1 && num_vectors <= 4);
        kani::assume(group_size >= 1 && group_size <= 8);

        let groups_per_vec = (head_dim + group_size - 1) / group_size;
        let total_groups = num_vectors * groups_per_vec;
        let total_elements = num_vectors * head_dim;

        let vi: usize = kani::any();
        let gi: usize = kani::any();
        kani::assume(vi < num_vectors);
        kani::assume(gi < groups_per_vec);

        let group_idx = vi * groups_per_vec + gi;
        assert!(group_idx < total_groups);

        let g_start = gi * group_size;
        let g_len_candidate = group_size;
        let remaining = head_dim - g_start;
        let g_len = if g_len_candidate < remaining {
            g_len_candidate
        } else {
            remaining
        };
        assert!(g_len >= 1);
        assert!(g_len <= group_size);

        let output_pos = vi * head_dim + g_start;
        assert!(output_pos + g_len <= total_elements);
    }

    #[kani::proof]
    fn compress_append_indices_in_bounds() {
        let head_dim: usize = kani::any();
        let num_existing: usize = kani::any();
        let num_new: usize = kani::any();
        let group_size: usize = kani::any();

        kani::assume(head_dim >= 1 && head_dim <= 8);
        kani::assume(num_existing >= 1 && num_existing <= 4);
        kani::assume(num_new >= 1 && num_new <= 4);
        kani::assume(group_size >= 1 && group_size <= 8);

        let groups_per_vec = (head_dim + group_size - 1) / group_size;
        let existing_groups = num_existing * groups_per_vec;
        let existing_elements = num_existing * head_dim;
        let new_data_len = num_new * head_dim;
        let total_groups_after = (num_existing + num_new) * groups_per_vec;
        let total_elements_after = (num_existing + num_new) * head_dim;

        let vi: usize = kani::any();
        let gi: usize = kani::any();
        kani::assume(vi < num_new);
        kani::assume(gi < groups_per_vec);

        let vec_start = vi * head_dim;
        let g_start = vec_start + gi * group_size;
        let g_end_candidate = g_start + group_size;
        let g_end = if g_end_candidate < vec_start + head_dim {
            g_end_candidate
        } else {
            vec_start + head_dim
        };

        assert!(g_start < new_data_len);
        assert!(g_end <= new_data_len);

        let new_group_idx = existing_groups + vi * groups_per_vec + gi;
        assert!(new_group_idx < total_groups_after);

        let new_element_offset = existing_elements + g_start;
        let g_len = g_end - g_start;
        assert!(new_element_offset + g_len <= total_elements_after);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn relative_l2_error(original: &[f32], reconstructed: &[f32]) -> f32 {
        let norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            return 0.0;
        }
        let diff: f32 = original
            .iter()
            .zip(reconstructed)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        diff / norm
    }

    #[test]
    fn test_round_trip_simple() {
        let pq = CpuPolarQuant::default_3bit();
        let head_dim = 4;
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let compressed = pq.compress(&data, head_dim).unwrap();
        let decompressed = pq.decompress(&compressed).unwrap();

        assert_eq!(decompressed.len(), data.len());
        let err = relative_l2_error(&data, &decompressed);
        assert!(err < 0.05, "relative error {err} >= 5%");
    }

    #[test]
    fn test_round_trip_random_vectors_mean_error() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let pq = CpuPolarQuant::default_3bit();
        let head_dim = 16;
        let num_vectors = 100;

        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..num_vectors * head_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let compressed = pq.compress(&data, head_dim).unwrap();
        let decompressed = pq.decompress(&compressed).unwrap();

        let mut total_err = 0.0;
        for vi in 0..num_vectors {
            let start = vi * head_dim;
            total_err += relative_l2_error(
                &data[start..start + head_dim],
                &decompressed[start..start + head_dim],
            );
        }
        let mean_err = total_err / num_vectors as f32;
        assert!(
            mean_err < 0.15,
            "mean relative error {mean_err} >= 15% across {num_vectors} vectors"
        );
    }

    #[test]
    fn test_constant_vector() {
        let pq = CpuPolarQuant::default_3bit();
        let head_dim = 8;
        let data = vec![3.14; head_dim];

        let compressed = pq.compress(&data, head_dim).unwrap();
        let decompressed = pq.decompress(&compressed).unwrap();

        for (a, b) in data.iter().zip(&decompressed) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_zero_vector() {
        let pq = CpuPolarQuant::default_3bit();
        let head_dim = 4;
        let data = vec![0.0; head_dim];

        let compressed = pq.compress(&data, head_dim).unwrap();
        let decompressed = pq.decompress(&compressed).unwrap();

        for x in &decompressed {
            assert_relative_eq!(*x, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let pq = CpuPolarQuant::default_3bit();
        let result = pq.compress(&[1.0, 2.0, 3.0], 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_vectors() {
        let pq = CpuPolarQuant::default_3bit();
        let head_dim = 4;
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // vector 0: smooth ramp
            -1.0, 0.5, 2.0, -0.5, // vector 1: wide range within groups
            10.0, 20.0, 30.0, 40.0, // vector 2: large values, smooth
        ];

        let compressed = pq.compress(&data, head_dim).unwrap();
        assert_eq!(compressed.num_vectors, 3);

        let decompressed = pq.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), data.len());

        let mut total_err = 0.0;
        for vi in 0..3 {
            let start = vi * head_dim;
            total_err += relative_l2_error(
                &data[start..start + head_dim],
                &decompressed[start..start + head_dim],
            );
        }
        let mean_err = total_err / 3.0;
        assert!(mean_err < 0.10, "mean error {mean_err} >= 10%");
    }

    #[test]
    fn test_higher_bits_lower_error() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let head_dim = 16;
        let mut rng = StdRng::seed_from_u64(99);
        let data: Vec<f32> = (0..head_dim).map(|_| rng.random_range(-1.0..1.0)).collect();

        let pq3 = CpuPolarQuant::new(3, 4).unwrap();
        let pq6 = CpuPolarQuant::new(6, 4).unwrap();

        let d3 = pq3
            .decompress(&pq3.compress(&data, head_dim).unwrap())
            .unwrap();
        let d6 = pq6
            .decompress(&pq6.compress(&data, head_dim).unwrap())
            .unwrap();

        let err3 = relative_l2_error(&data, &d3);
        let err6 = relative_l2_error(&data, &d6);

        assert!(
            err6 < err3,
            "6-bit error ({err6}) should be less than 3-bit ({err3})"
        );
    }

    #[test]
    fn test_invalid_config() {
        assert!(CpuPolarQuant::new(0, 4).is_err());
        assert!(CpuPolarQuant::new(9, 4).is_err());
        assert!(CpuPolarQuant::new(3, 0).is_err());
        assert!(CpuPolarQuant::new(1, 1).is_ok());
        assert!(CpuPolarQuant::new(8, 8).is_ok());
    }

    #[test]
    fn test_group_size_one_exact() {
        let pq = CpuPolarQuant::new(8, 1).unwrap();
        let head_dim = 4;
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let compressed = pq.compress(&data, head_dim).unwrap();
        let decompressed = pq.decompress(&compressed).unwrap();

        for (a, b) in data.iter().zip(&decompressed) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_head_dim_not_divisible_by_group_size() {
        let pq = CpuPolarQuant::new(4, 3).unwrap();
        let head_dim = 7;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let compressed = pq.compress(&data, head_dim).unwrap();
        assert_eq!(compressed.groups_per_vector(), 3); // ceil(7/3)

        let decompressed = pq.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), head_dim);

        let err = relative_l2_error(&data, &decompressed);
        assert!(err < 0.05, "error {err} >= 5%");
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_round_trip_error_bounded(
                values in prop::collection::vec(-5.0f32..5.0f32, 8..32),
            ) {
                let head_dim = values.len();
                let compressor = CpuPolarQuant::default_3bit();
                let compressed = compressor.compress(&values, head_dim).unwrap();
                let decompressed = compressor.decompress(&compressed).unwrap();

                let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-6 {
                    return Ok(());
                }
                let error: f32 = values.iter().zip(decompressed.iter())
                    .map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                let relative = error / norm;
                prop_assert!(relative < 0.25, "round-trip error {:.4} >= 25%", relative);
            }

            #[test]
            fn prop_compress_append_equivalent(
                num_first in 1usize..=4,
                num_second in 1usize..=4,
                head_dim in prop::sample::select(vec![4usize, 8, 16, 32, 64]),
            ) {
                use rand::rngs::StdRng;
                use rand::{Rng, SeedableRng};

                let mut rng = StdRng::seed_from_u64(
                    (num_first * 1000 + num_second * 100 + head_dim) as u64,
                );
                let data1: Vec<f32> = (0..num_first * head_dim)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect();
                let data2: Vec<f32> = (0..num_second * head_dim)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect();

                let compressor = CpuPolarQuant::default_3bit();

                let mut combined = data1.clone();
                combined.extend_from_slice(&data2);
                let full_compressed = compressor.compress(&combined, head_dim).unwrap();
                let full_decompressed = compressor.decompress(&full_compressed).unwrap();

                let mut incremental = compressor.compress(&data1, head_dim).unwrap();
                compressor.compress_append(&mut incremental, &data2).unwrap();
                let inc_decompressed = compressor.decompress(&incremental).unwrap();

                prop_assert_eq!(full_decompressed.len(), inc_decompressed.len());
                for (i, (a, b)) in full_decompressed.iter().zip(&inc_decompressed).enumerate() {
                    prop_assert!(
                        (a - b).abs() < 1e-6,
                        "mismatch at {}: full={} inc={}", i, a, b,
                    );
                }
            }
        }
    }

    #[test]
    fn test_realistic_head_dim_64() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let pq = CpuPolarQuant::default_3bit();
        let head_dim = 64;
        let num_vectors = 100;

        let mut rng = StdRng::seed_from_u64(123);
        let data: Vec<f32> = (0..num_vectors * head_dim)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();

        let compressed = pq.compress(&data, head_dim).unwrap();
        let decompressed = pq.decompress(&compressed).unwrap();

        let mut total_err = 0.0;
        for vi in 0..num_vectors {
            let start = vi * head_dim;
            total_err += relative_l2_error(
                &data[start..start + head_dim],
                &decompressed[start..start + head_dim],
            );
        }
        let mean_err = total_err / num_vectors as f32;
        assert!(
            mean_err < 0.15,
            "mean error {mean_err} >= 15% across {num_vectors} vectors"
        );
    }
}
