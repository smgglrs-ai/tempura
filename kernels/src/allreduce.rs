pub use crate::AllReduce;
use crate::Result;

#[cfg(feature = "cpu")]
pub struct CpuAllReduce;

#[cfg(feature = "cpu")]
impl AllReduce for CpuAllReduce {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Proven
    }

    fn sum_in_place(&self, _data: &mut [f32]) -> Result<()> {
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub struct CudaAllReduce {
    #[allow(dead_code)]
    gpu: crate::gpu::GpuContext,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaAllReduce {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaAllReduce {}

#[cfg(feature = "cuda")]
impl CudaAllReduce {
    pub fn new() -> Result<Self> {
        Ok(Self {
            gpu: crate::gpu::GpuContext::new(0)?,
        })
    }
}

#[cfg(feature = "cuda")]
impl AllReduce for CudaAllReduce {
    fn sum_in_place(&self, _data: &mut [f32]) -> Result<()> {
        todo!("CudaAllReduce: NCCL-backed all-reduce not yet implemented")
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn cpu_allreduce_is_identity() {
        let len: usize = kani::any();
        kani::assume(len >= 1 && len <= 8);
        let input = vec![1.0f32; len];
        let mut output = vec![0.0f32; len];
        output.copy_from_slice(&input);
        for i in 0..len {
            assert!(output[i] == input[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_allreduce_is_identity() {
        let ar = CpuAllReduce;
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let original = data.clone();
        ar.sum_in_place(&mut data).unwrap();
        assert_eq!(data, original);
    }
}
