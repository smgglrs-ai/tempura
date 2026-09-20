use crate::config::ModelConfig;

pub struct DeltaNetState {
    states: Vec<Vec<f32>>,
    conv_states: Vec<Vec<f32>>,
    num_deltanet_layers: usize,
    num_heads: usize,
    head_dim: usize,
    conv_len: usize,
}

impl DeltaNetState {
    pub fn new(config: &ModelConfig) -> Self {
        let num_heads = config.deltanet_num_heads.unwrap_or(config.num_kv_heads);
        let head_dim = config.deltanet_head_dim.unwrap_or(config.head_dim);
        let num_deltanet_layers = config
            .layer_kinds
            .as_ref()
            .map(|kinds| {
                kinds
                    .iter()
                    .filter(|k| **k == crate::config::LayerKind::DeltaNet)
                    .count()
            })
            .unwrap_or(0);

        let state_size = num_heads * head_dim * head_dim;
        let states = (0..num_deltanet_layers)
            .map(|_| vec![0.0f32; state_size])
            .collect();
        let conv_states = (0..num_deltanet_layers)
            .map(|_| Vec::new())
            .collect();

        Self {
            states,
            conv_states,
            num_deltanet_layers,
            num_heads,
            head_dim,
            conv_len: 0,
        }
    }

    pub fn state_mut(&mut self, deltanet_layer_idx: usize) -> &mut [f32] {
        debug_assert!(tempura_kernel_api::VerifiedDeltaNetState::well_formed(self));
        &mut self.states[deltanet_layer_idx]
    }

    pub fn state(&self, deltanet_layer_idx: usize) -> &[f32] {
        &self.states[deltanet_layer_idx]
    }

    pub fn conv_state(&self, deltanet_layer_idx: usize) -> &[f32] {
        &self.conv_states[deltanet_layer_idx]
    }

    pub fn update_conv_state(&mut self, deltanet_layer_idx: usize, conv_dim: usize, kernel_size: usize, new_tokens: &[f32], seq_len: usize) {
        let history = kernel_size - 1;
        let needed = history * conv_dim;
        let buf = &mut self.conv_states[deltanet_layer_idx];
        if buf.len() != needed {
            buf.resize(needed, 0.0);
            self.conv_len = history;
        }
        if seq_len >= history {
            let start = (seq_len - history) * conv_dim;
            buf.copy_from_slice(&new_tokens[start..start + needed]);
        } else {
            let keep = (history - seq_len) * conv_dim;
            buf.copy_within(needed - keep..needed, 0);
            buf[keep..].copy_from_slice(&new_tokens[..seq_len * conv_dim]);
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.states {
            s.fill(0.0);
        }
        for c in &mut self.conv_states {
            c.fill(0.0);
        }
    }

    pub fn num_deltanet_layers(&self) -> usize {
        self.num_deltanet_layers
    }

    pub fn to_transfer(&self) -> DeltaNetStateTransfer {
        DeltaNetStateTransfer {
            states: self.states.clone(),
            num_heads: self.num_heads,
            head_dim: self.head_dim,
        }
    }

    pub fn from_transfer(transfer: DeltaNetStateTransfer) -> Self {
        let num_deltanet_layers = transfer.states.len();
        let conv_states = (0..num_deltanet_layers).map(|_| Vec::new()).collect();
        Self {
            states: transfer.states,
            conv_states,
            num_deltanet_layers,
            num_heads: transfer.num_heads,
            head_dim: transfer.head_dim,
            conv_len: 0,
        }
    }
}

impl tempura_kernel_api::VerifiedDeltaNetState for DeltaNetState {
    fn num_layers(&self) -> usize {
        self.num_deltanet_layers
    }
    fn num_heads(&self) -> usize {
        self.num_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn state_size(&self) -> usize {
        self.num_heads * self.head_dim * self.head_dim
    }
    fn well_formed(&self) -> bool {
        self.states.len() == self.num_deltanet_layers
            && self.num_heads >= 1
            && self.head_dim >= 1
            && self.states.iter().all(|s| s.len() == self.num_heads * self.head_dim * self.head_dim)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaNetStateTransfer {
    pub states: Vec<Vec<f32>>,
    pub num_heads: usize,
    pub head_dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LayerKind;

    fn hybrid_test_config() -> ModelConfig {
        let mut cfg = ModelConfig::tiny_test();
        cfg.layer_kinds = Some(vec![
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::Attention,
        ]);
        cfg.num_layers = 4;
        cfg
    }

    #[test]
    fn test_deltanet_state_new() {
        let cfg = hybrid_test_config();
        let state = DeltaNetState::new(&cfg);
        assert_eq!(state.num_deltanet_layers(), 3);
        assert!(tempura_kernel_api::VerifiedDeltaNetState::well_formed(&state));
    }

    #[test]
    fn test_deltanet_state_reset() {
        let cfg = hybrid_test_config();
        let mut state = DeltaNetState::new(&cfg);
        state.state_mut(0)[0] = 42.0;
        state.reset();
        assert_eq!(state.state(0)[0], 0.0);
    }

    #[test]
    fn test_deltanet_state_transfer_roundtrip() {
        let cfg = hybrid_test_config();
        let mut state = DeltaNetState::new(&cfg);
        state.state_mut(1)[0] = 3.14;

        let transfer = state.to_transfer();
        let restored = DeltaNetState::from_transfer(transfer);
        assert_eq!(restored.state(1)[0], 3.14);
        assert_eq!(restored.num_deltanet_layers(), 3);
        assert!(tempura_kernel_api::VerifiedDeltaNetState::well_formed(&restored));
    }
}
