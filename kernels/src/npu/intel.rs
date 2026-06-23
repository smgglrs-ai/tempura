use tempura_kernel_api::graph::{
    DecodeInput, DecodeOutput, FusedTransformerLayer, LayerInput, LayerOutput, LayerWeights,
    PrefillInput, PrefillOutput,
};
use tempura_kernel_api::{Result, VerificationTier};

pub struct IntelNpuLayer;

impl FusedTransformerLayer for IntelNpuLayer {
    fn forward(&self, _input: &LayerInput, _weights: &LayerWeights) -> Result<LayerOutput> {
        todo!("Intel NPU layer requires OpenVINO Runtime / NPU plugin")
    }

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

pub struct IntelNpuPrefill;

impl tempura_kernel_api::graph::CompiledPrefill for IntelNpuPrefill {
    fn prefill(&self, _input: &PrefillInput) -> Result<PrefillOutput> {
        todo!("Intel NPU prefill requires OpenVINO Runtime / NPU plugin")
    }

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

pub struct IntelNpuDecode;

impl tempura_kernel_api::graph::CompiledDecodeStep for IntelNpuDecode {
    fn decode(&self, _input: &DecodeInput) -> Result<DecodeOutput> {
        todo!("Intel NPU decode requires OpenVINO Runtime / NPU plugin")
    }

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}
