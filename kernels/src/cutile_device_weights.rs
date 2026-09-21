use std::collections::HashMap;
use std::sync::Arc;

use cuda_core::{Device, Stream};
use cutile::api;
use cutile::prelude::DeviceOp;
use cutile::tensor::{Reshape, Tensor};

use crate::{KernelError, Result};

/// Pre-uploaded weight tensors resident on the GPU.
///
/// Eliminates per-forward-pass host-to-device transfers for static model weights.
/// Upload once at model load time; reuse across all inference calls.
pub struct CutileDeviceWeights {
    tensors: HashMap<String, Arc<Tensor<f32>>>,
    device: Arc<Device>,
    stream: Arc<Stream>,
}

unsafe impl Send for CutileDeviceWeights {}
unsafe impl Sync for CutileDeviceWeights {}

impl CutileDeviceWeights {
    /// Initialize a weight cache for the given CUDA device ordinal.
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = Device::new(ordinal).map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let stream = device
            .new_stream()
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self {
            tensors: HashMap::new(),
            device,
            stream,
        })
    }

    /// Upload a f32 slice to the device under `name` with shape [rows, cols].
    ///
    /// Replaces any previously cached tensor with the same name.
    pub fn upload(&mut self, name: &str, data: &[f32], rows: usize, cols: usize) -> Result<()> {
        let vec = Arc::new(data.to_vec());
        let tensor: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&vec)
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[rows, cols])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();
        self.tensors.insert(name.to_string(), tensor);
        Ok(())
    }

    /// Retrieve a device tensor by name.
    pub fn get(&self, name: &str) -> Option<&Arc<Tensor<f32>>> {
        self.tensors.get(name)
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
