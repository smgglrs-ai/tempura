pub mod attention_kernel;
pub mod matmul_kernel;
pub mod rmsnorm_kernel;
pub mod rope_kernel;

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

use crate::KernelError;

pub struct GpuContext {
    ctx: Arc<CudaContext>,
}

impl GpuContext {
    pub fn new(device_ordinal: usize) -> Result<Self, KernelError> {
        let ctx = CudaContext::new(device_ordinal)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self { ctx })
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> Arc<CudaStream> {
        self.ctx.default_stream()
    }
}

pub fn to_device(stream: &CudaStream, data: &[f32]) -> Result<DeviceBuffer<f32>, KernelError> {
    DeviceBuffer::from_host(stream, data).map_err(|e| KernelError::InvalidConfig(e.to_string()))
}

pub fn to_host(stream: &CudaStream, buf: &DeviceBuffer<f32>) -> Result<Vec<f32>, KernelError> {
    buf.to_host_vec(stream)
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))
}

pub fn to_device_u32(stream: &CudaStream, data: &[u32]) -> Result<DeviceBuffer<u32>, KernelError> {
    DeviceBuffer::from_host(stream, data).map_err(|e| KernelError::InvalidConfig(e.to_string()))
}

pub fn to_device_mut(stream: &CudaStream, data: &[f32]) -> Result<DeviceBuffer<f32>, KernelError> {
    DeviceBuffer::from_host(stream, data).map_err(|e| KernelError::InvalidConfig(e.to_string()))
}

pub fn copy_back(
    stream: &CudaStream,
    buf: &DeviceBuffer<f32>,
    dst: &mut [f32],
) -> Result<(), KernelError> {
    let host = to_host(stream, buf)?;
    dst.copy_from_slice(&host);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_round_trip() {
        let gpu =
            GpuContext::new(0).expect("Failed to create GPU context — is an NVIDIA GPU available?");
        let stream = gpu.stream();

        let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
        let dev_buf = to_device(&stream, &data).expect("Failed to copy to device");
        let result = to_host(&stream, &dev_buf).expect("Failed to copy from device");

        assert_eq!(data.len(), result.len());
        for (i, (a, b)) in data.iter().zip(result.iter()).enumerate() {
            assert_eq!(a, b, "Mismatch at index {i}: expected {a}, got {b}");
        }
    }

    #[test]
    fn test_gpu_copy_back() {
        let gpu = GpuContext::new(0).expect("Failed to create GPU context");
        let stream = gpu.stream();

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let dev_buf = to_device(&stream, &data).unwrap();
        let mut dst = vec![0.0f32; 4];
        copy_back(&stream, &dev_buf, &mut dst).unwrap();

        assert_eq!(dst, data);
    }
}
