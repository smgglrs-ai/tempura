pub mod attention_kernel;
pub mod elementwise_kernel;
pub mod flash_attention_kernel;
pub mod flash_decode_kernel;
pub mod matmul_kernel;
pub mod matmul_mma_kernel;
pub mod mma_sync;
pub mod quantized_gemm_kernel;
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

    pub fn from_context(ctx: Arc<CudaContext>) -> Self {
        Self { ctx }
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> Arc<CudaStream> {
        self.ctx.default_stream()
    }
}

pub fn bind_context(ctx: &CudaContext) -> Result<(), KernelError> {
    ctx.bind_to_thread()
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))
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

pub fn alloc_device(stream: &CudaStream, len: usize) -> Result<DeviceBuffer<f32>, KernelError> {
    DeviceBuffer::zeroed(stream, len).map_err(|e| KernelError::InvalidConfig(e.to_string()))
}

pub fn to_device_u16(stream: &CudaStream, data: &[u16]) -> Result<DeviceBuffer<u16>, KernelError> {
    DeviceBuffer::from_host(stream, data).map_err(|e| KernelError::InvalidConfig(e.to_string()))
}

pub fn alloc_device_u16(stream: &CudaStream, len: usize) -> Result<DeviceBuffer<u16>, KernelError> {
    DeviceBuffer::zeroed(stream, len).map_err(|e| KernelError::InvalidConfig(e.to_string()))
}

pub fn to_device_u8(stream: &CudaStream, data: &[u8]) -> Result<DeviceBuffer<u8>, KernelError> {
    DeviceBuffer::from_host(stream, data).map_err(|e| KernelError::InvalidConfig(e.to_string()))
}

pub fn alloc_device_u32(stream: &CudaStream, len: usize) -> Result<DeviceBuffer<u32>, KernelError> {
    DeviceBuffer::zeroed(stream, len).map_err(|e| KernelError::InvalidConfig(e.to_string()))
}

pub fn vector_add_device(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), KernelError> {
    use cuda_core::LaunchConfig;
    bind_context(ctx)?;
    let module = elementwise_kernel::kernels::load(ctx)
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
    module
        .vector_add(
            stream,
            LaunchConfig::for_num_elems(a.len() as u32),
            a,
            b,
            output,
        )
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
    Ok(())
}

pub fn silu_mul_device(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), KernelError> {
    use cuda_core::LaunchConfig;
    bind_context(ctx)?;
    let module = elementwise_kernel::kernels::load(ctx)
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
    module
        .silu_mul(
            stream,
            LaunchConfig::for_num_elems(gate.len() as u32),
            gate,
            up,
            output,
        )
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
    Ok(())
}

pub fn slice_columns_device(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    src: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
    rows: usize,
    total_cols: usize,
    col_offset: usize,
    slice_cols: usize,
) -> Result<(), KernelError> {
    use cuda_core::LaunchConfig;
    bind_context(ctx)?;
    let module = elementwise_kernel::kernels::load(ctx)
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
    let total_elems = rows * slice_cols;
    module
        .slice_columns(
            stream,
            LaunchConfig::for_num_elems(total_elems as u32),
            src,
            dst,
            rows as u32,
            total_cols as u32,
            col_offset as u32,
            slice_cols as u32,
        )
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
    Ok(())
}

pub fn copy_region_device(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    src: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
    src_offset: usize,
    dst_offset: usize,
    count: usize,
) -> Result<(), KernelError> {
    bind_context(ctx)?;
    let byte_count = count * std::mem::size_of::<f32>();
    let src_ptr = src.cu_deviceptr() + (src_offset * std::mem::size_of::<f32>()) as u64;
    let dst_ptr = dst.cu_deviceptr() + (dst_offset * std::mem::size_of::<f32>()) as u64;
    unsafe {
        cuda_core::memory::memcpy_dtod_async(dst_ptr, src_ptr, byte_count, stream.cu_stream())
    }
    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
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
