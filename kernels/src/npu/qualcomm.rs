use tempura_kernel_api::graph::{
    DecodeInput, DecodeOutput, FusedTransformerLayer, LayerInput, LayerOutput, LayerWeights,
    PrefillInput, PrefillOutput,
};
use tempura_kernel_api::{Result, VerificationTier};

pub struct QualcommHexagonLayer;

impl FusedTransformerLayer for QualcommHexagonLayer {
    fn forward(&self, _input: &LayerInput, _weights: &LayerWeights) -> Result<LayerOutput> {
        todo!("Qualcomm Hexagon layer requires QNN SDK")
    }

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

pub struct QualcommHexagonPrefill;

impl tempura_kernel_api::graph::CompiledPrefill for QualcommHexagonPrefill {
    fn prefill(&self, _input: &PrefillInput) -> Result<PrefillOutput> {
        todo!("Qualcomm Hexagon prefill requires QNN SDK")
    }

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

pub struct QualcommHexagonDecode;

impl tempura_kernel_api::graph::CompiledDecodeStep for QualcommHexagonDecode {
    fn decode(&self, _input: &DecodeInput) -> Result<DecodeOutput> {
        todo!("Qualcomm Hexagon decode requires QNN SDK")
    }

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}
