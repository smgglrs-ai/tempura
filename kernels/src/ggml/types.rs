use super::blocks;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4, 5 deprecated (Q4_2, Q4_3)
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    // 31-33 removed
    TQ1_0 = 34,
    TQ2_0 = 35,
    // 36-38 removed
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
}

pub const GGML_TYPE_COUNT: usize = 42;

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            34 => Some(Self::TQ1_0),
            35 => Some(Self::TQ2_0),
            39 => Some(Self::MXFP4),
            40 => Some(Self::NVFP4),
            41 => Some(Self::Q1_0),
            _ => None,
        }
    }

    pub fn is_quantized(self) -> bool {
        !matches!(
            self,
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 | Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::BF16 => 1,
            Self::F64 => 1,
            Self::I8 => 1,
            Self::I16 => 1,
            Self::I32 => 1,
            Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::IQ4NL => 32,
            Self::MXFP4 => 32,
            Self::NVFP4 => 64,
            Self::Q1_0 => 128,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => 256,
            Self::IQ3XXS | Self::IQ3S => 256,
            Self::IQ1S | Self::IQ1M => 256,
            Self::IQ4XS => 256,
            Self::TQ1_0 | Self::TQ2_0 => 256,
        }
    }

    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => blocks::BLOCK_Q4_0_SIZE,
            Self::Q4_1 => blocks::BLOCK_Q4_1_SIZE,
            Self::Q5_0 => blocks::BLOCK_Q5_0_SIZE,
            Self::Q5_1 => blocks::BLOCK_Q5_1_SIZE,
            Self::Q8_0 => blocks::BLOCK_Q8_0_SIZE,
            Self::Q8_1 => blocks::BLOCK_Q8_1_SIZE,
            Self::Q2K => blocks::BLOCK_Q2K_SIZE,
            Self::Q3K => blocks::BLOCK_Q3K_SIZE,
            Self::Q4K => blocks::BLOCK_Q4K_SIZE,
            Self::Q5K => blocks::BLOCK_Q5K_SIZE,
            Self::Q6K => blocks::BLOCK_Q6K_SIZE,
            Self::Q8K => blocks::BLOCK_Q8K_SIZE,
            Self::IQ2XXS => blocks::BLOCK_IQ2XXS_SIZE,
            Self::IQ2XS => blocks::BLOCK_IQ2XS_SIZE,
            Self::IQ2S => blocks::BLOCK_IQ2S_SIZE,
            Self::IQ3XXS => blocks::BLOCK_IQ3XXS_SIZE,
            Self::IQ3S => blocks::BLOCK_IQ3S_SIZE,
            Self::IQ1S => blocks::BLOCK_IQ1S_SIZE,
            Self::IQ1M => blocks::BLOCK_IQ1M_SIZE,
            Self::IQ4NL => blocks::BLOCK_IQ4NL_SIZE,
            Self::IQ4XS => blocks::BLOCK_IQ4XS_SIZE,
            Self::TQ1_0 => blocks::BLOCK_TQ1_0_SIZE,
            Self::TQ2_0 => blocks::BLOCK_TQ2_0_SIZE,
            Self::Q1_0 => blocks::BLOCK_Q1_0_SIZE,
            Self::MXFP4 => blocks::BLOCK_MXFP4_SIZE,
            Self::NVFP4 => blocks::BLOCK_NVFP4_SIZE,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2K => "q2_K",
            Self::Q3K => "q3_K",
            Self::Q4K => "q4_K",
            Self::Q5K => "q5_K",
            Self::Q6K => "q6_K",
            Self::Q8K => "q8_K",
            Self::IQ2XXS => "iq2_xxs",
            Self::IQ2XS => "iq2_xs",
            Self::IQ2S => "iq2_s",
            Self::IQ3XXS => "iq3_xxs",
            Self::IQ3S => "iq3_s",
            Self::IQ1S => "iq1_s",
            Self::IQ1M => "iq1_m",
            Self::IQ4NL => "iq4_nl",
            Self::IQ4XS => "iq4_xs",
            Self::TQ1_0 => "tq1_0",
            Self::TQ2_0 => "tq2_0",
            Self::Q1_0 => "q1_0",
            Self::MXFP4 => "mxfp4",
            Self::NVFP4 => "nvfp4",
        }
    }

    pub fn vec_dot_type(self) -> GgmlType {
        match self {
            Self::F32 => Self::F32,
            Self::F16 => Self::F16,
            Self::BF16 => Self::BF16,
            Self::Q4_0 | Self::Q5_0 | Self::Q8_0 | Self::Q1_0 | Self::MXFP4 | Self::NVFP4 | Self::IQ4NL => Self::Q8_0,
            Self::Q4_1 | Self::Q5_1 => Self::Q8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => Self::Q8K,
            Self::TQ1_0 | Self::TQ2_0 => Self::Q8K,
            Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => Self::Q8K,
            Self::IQ3XXS | Self::IQ3S => Self::Q8K,
            Self::IQ1S | Self::IQ1M => Self::Q8K,
            Self::IQ4XS => Self::Q8K,
            _ => Self::F32,
        }
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_u32_roundtrip() {
        for id in 0..GGML_TYPE_COUNT as u32 {
            if let Some(t) = GgmlType::from_u32(id) {
                assert_eq!(t as u32, id);
            }
        }
    }

    #[test]
    fn q4k_matches_ggml() {
        assert_eq!(GgmlType::Q4K as u32, 12);
        assert_eq!(GgmlType::Q4K.block_size(), 256);
        assert_eq!(GgmlType::Q4K.type_size(), 144);
        assert_eq!(GgmlType::Q4K.vec_dot_type(), GgmlType::Q8K);
    }

    #[test]
    fn q6k_matches_ggml() {
        assert_eq!(GgmlType::Q6K as u32, 14);
        assert_eq!(GgmlType::Q6K.block_size(), 256);
        assert_eq!(GgmlType::Q6K.type_size(), 210);
        assert_eq!(GgmlType::Q6K.vec_dot_type(), GgmlType::Q8K);
    }

    #[test]
    fn q4_0_matches_ggml() {
        assert_eq!(GgmlType::Q4_0 as u32, 2);
        assert_eq!(GgmlType::Q4_0.block_size(), 32);
        assert_eq!(GgmlType::Q4_0.type_size(), 18);
        assert_eq!(GgmlType::Q4_0.vec_dot_type(), GgmlType::Q8_0);
    }

    #[test]
    fn q8_k_matches_ggml() {
        assert_eq!(GgmlType::Q8K as u32, 15);
        assert_eq!(GgmlType::Q8K.block_size(), 256);
        assert_eq!(GgmlType::Q8K.type_size(), 292);
    }

    #[test]
    fn deprecated_ids_return_none() {
        assert!(GgmlType::from_u32(4).is_none());
        assert!(GgmlType::from_u32(5).is_none());
        assert!(GgmlType::from_u32(31).is_none());
        assert!(GgmlType::from_u32(32).is_none());
        assert!(GgmlType::from_u32(33).is_none());
        assert!(GgmlType::from_u32(36).is_none());
        assert!(GgmlType::from_u32(37).is_none());
        assert!(GgmlType::from_u32(38).is_none());
    }
}
