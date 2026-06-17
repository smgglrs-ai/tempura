use crate::config::ModelConfig;
use cuda_core::{CudaStream, DeviceBuffer};

pub struct GpuScratchpad {
    pub hidden: DeviceBuffer<f32>,
    pub hidden_swap: DeviceBuffer<f32>,
    pub normed: DeviceBuffer<f32>,
    pub qkv: DeviceBuffer<f32>,
    pub q: DeviceBuffer<f32>,
    pub k: DeviceBuffer<f32>,
    pub v: DeviceBuffer<f32>,
    pub q_rotated: DeviceBuffer<f32>,
    pub k_rotated: DeviceBuffer<f32>,
    pub attn_out: DeviceBuffer<f32>,
    pub o_proj: DeviceBuffer<f32>,
    pub normed2: DeviceBuffer<f32>,
    pub gate_up: DeviceBuffer<f32>,
    pub gate: DeviceBuffer<f32>,
    pub up: DeviceBuffer<f32>,
    pub silu_out: DeviceBuffer<f32>,
    pub down: DeviceBuffer<f32>,
    pub final_normed: DeviceBuffer<f32>,
    pub logits: DeviceBuffer<f32>,
}

impl GpuScratchpad {
    pub fn new(
        config: &ModelConfig,
        stream: &CudaStream,
    ) -> Result<Self, crate::weights::WeightsError> {
        let s = config.max_seq_len;
        let h = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        let alloc = |len: usize| -> Result<DeviceBuffer<f32>, crate::weights::WeightsError> {
            DeviceBuffer::zeroed(stream, len).map_err(|e| {
                crate::weights::WeightsError::GpuUpload(format!("scratchpad alloc: {e}"))
            })
        };

        Ok(Self {
            hidden: alloc(s * h)?,
            hidden_swap: alloc(s * h)?,
            normed: alloc(s * h)?,
            qkv: alloc(s * (q_dim + 2 * kv_dim))?,
            q: alloc(s * q_dim)?,
            k: alloc(s * kv_dim)?,
            v: alloc(s * kv_dim)?,
            q_rotated: alloc(s * q_dim)?,
            k_rotated: alloc(s * kv_dim)?,
            attn_out: alloc(s * q_dim)?,
            o_proj: alloc(s * h)?,
            normed2: alloc(s * h)?,
            gate_up: alloc(s * 2 * inter)?,
            gate: alloc(s * inter)?,
            up: alloc(s * inter)?,
            silu_out: alloc(s * inter)?,
            down: alloc(s * h)?,
            final_normed: alloc(s * h)?,
            logits: alloc(s * vocab)?,
        })
    }

    pub fn upload_hidden(&mut self, stream: &CudaStream, data: &[f32]) {
        let src = cuda_core::DeviceBuffer::from_host(stream, data).expect("upload hidden");
        let byte_count = data.len() * std::mem::size_of::<f32>();
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                self.hidden.cu_deviceptr(),
                src.cu_deviceptr(),
                byte_count,
                stream.cu_stream(),
            )
        }
        .expect("copy hidden to scratchpad");
    }
}
