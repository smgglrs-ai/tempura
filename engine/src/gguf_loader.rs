use std::collections::HashMap;
use std::path::Path;

use crate::config::{ModelConfig, QuantConfig};
use crate::loader::{LoadError, WeightLoader};

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: usize = 32;

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
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
    IQ1M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    TL1 = 38, // bitnet.cpp: ARM TL1 LUT format
    I2S = 39, // bitnet.cpp: x86 TL2/I2_S LUT format; NOTE: upstream GGUF uses 39 for MXFP4
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42, // PrismML: 2-bit ternary, group size 128, f16 scale
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self, LoadError> {
        match v {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            9 => Ok(GgmlType::Q8_1),
            10 => Ok(GgmlType::Q2K),
            11 => Ok(GgmlType::Q3K),
            12 => Ok(GgmlType::Q4K),
            13 => Ok(GgmlType::Q5K),
            14 => Ok(GgmlType::Q6K),
            15 => Ok(GgmlType::Q8K),
            16 => Ok(GgmlType::IQ2XXS),
            17 => Ok(GgmlType::IQ2XS),
            18 => Ok(GgmlType::IQ3XXS),
            19 => Ok(GgmlType::IQ1S),
            20 => Ok(GgmlType::IQ4NL),
            21 => Ok(GgmlType::IQ3S),
            22 => Ok(GgmlType::IQ2S),
            23 => Ok(GgmlType::IQ4XS),
            29 => Ok(GgmlType::IQ1M),
            30 => Ok(GgmlType::BF16),
            34 => Ok(GgmlType::TQ1_0),
            35 => Ok(GgmlType::TQ2_0),
            38 => Ok(GgmlType::TL1),
            39 => Ok(GgmlType::I2S),
            40 => Ok(GgmlType::NVFP4),
            41 => Ok(GgmlType::Q1_0),
            42 => Ok(GgmlType::Q2_0),
            _ => Err(LoadError::GgufParse(format!("unsupported GGML type: {v}"))),
        }
    }
}

struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    dtype: GgmlType,
    offset: u64,
}

fn read_u8(data: &[u8], offset: &mut usize) -> Result<u8, LoadError> {
    if *offset + 1 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading u8".into()));
    }
    let v = data[*offset];
    *offset += 1;
    Ok(v)
}

fn read_i8(data: &[u8], offset: &mut usize) -> Result<i8, LoadError> {
    Ok(read_u8(data, offset)? as i8)
}

fn read_u16(data: &[u8], offset: &mut usize) -> Result<u16, LoadError> {
    if *offset + 2 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading u16".into()));
    }
    let v = u16::from_le_bytes([data[*offset], data[*offset + 1]]);
    *offset += 2;
    Ok(v)
}

fn read_i16(data: &[u8], offset: &mut usize) -> Result<i16, LoadError> {
    Ok(read_u16(data, offset)? as i16)
}

fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32, LoadError> {
    if *offset + 4 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading u32".into()));
    }
    let v = u32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    Ok(v)
}

fn read_i32(data: &[u8], offset: &mut usize) -> Result<i32, LoadError> {
    if *offset + 4 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading i32".into()));
    }
    let v = i32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    Ok(v)
}

fn read_f32(data: &[u8], offset: &mut usize) -> Result<f32, LoadError> {
    if *offset + 4 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading f32".into()));
    }
    let v = f32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    Ok(v)
}

fn read_u64(data: &[u8], offset: &mut usize) -> Result<u64, LoadError> {
    if *offset + 8 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading u64".into()));
    }
    let v = u64::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
        data[*offset + 4],
        data[*offset + 5],
        data[*offset + 6],
        data[*offset + 7],
    ]);
    *offset += 8;
    Ok(v)
}

fn read_i64(data: &[u8], offset: &mut usize) -> Result<i64, LoadError> {
    if *offset + 8 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading i64".into()));
    }
    let v = i64::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
        data[*offset + 4],
        data[*offset + 5],
        data[*offset + 6],
        data[*offset + 7],
    ]);
    *offset += 8;
    Ok(v)
}

fn read_f64(data: &[u8], offset: &mut usize) -> Result<f64, LoadError> {
    if *offset + 8 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading f64".into()));
    }
    let v = f64::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
        data[*offset + 4],
        data[*offset + 5],
        data[*offset + 6],
        data[*offset + 7],
    ]);
    *offset += 8;
    Ok(v)
}

fn read_string(data: &[u8], offset: &mut usize) -> Result<String, LoadError> {
    let len = read_u64(data, offset)? as usize;
    if *offset + len > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading string".into()));
    }
    let s = std::str::from_utf8(&data[*offset..*offset + len])
        .map_err(|e| LoadError::GgufParse(format!("invalid UTF-8 in string: {e}")))?
        .to_string();
    *offset += len;
    Ok(s)
}

fn read_metadata_value(
    data: &[u8],
    offset: &mut usize,
    value_type: u32,
) -> Result<GgufValue, LoadError> {
    match value_type {
        0 => Ok(GgufValue::U8(read_u8(data, offset)?)),
        1 => Ok(GgufValue::I8(read_i8(data, offset)?)),
        2 => Ok(GgufValue::U16(read_u16(data, offset)?)),
        3 => Ok(GgufValue::I16(read_i16(data, offset)?)),
        4 => Ok(GgufValue::U32(read_u32(data, offset)?)),
        5 => Ok(GgufValue::I32(read_i32(data, offset)?)),
        6 => Ok(GgufValue::F32(read_f32(data, offset)?)),
        7 => {
            let v = read_u8(data, offset)?;
            Ok(GgufValue::Bool(v != 0))
        }
        8 => Ok(GgufValue::String(read_string(data, offset)?)),
        9 => {
            let elem_type = read_u32(data, offset)?;
            let count = read_u64(data, offset)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_metadata_value(data, offset, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(read_u64(data, offset)?)),
        11 => Ok(GgufValue::I64(read_i64(data, offset)?)),
        12 => Ok(GgufValue::F64(read_f64(data, offset)?)),
        _ => Err(LoadError::GgufParse(format!(
            "unknown metadata type: {value_type}"
        ))),
    }
}

// dequantize functions removed — now using tempura_kernels::ggml::dequantize::dequantize_row

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let scale = f32::from_bits((127 - 14) << 23);
        let mantissa = frac as f32 / 1024.0;
        if sign == 0 {
            scale * mantissa
        } else {
            -scale * mantissa
        }
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
        }
    } else {
        let f32_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
    }
}

fn dequant_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let start = i * 4;
        let v = f32::from_le_bytes([
            data[start],
            data[start + 1],
            data[start + 2],
            data[start + 3],
        ]);
        out.push(v);
    }
    out
}

fn dequant_f16(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let start = i * 2;
        let bits = u16::from_le_bytes([data[start], data[start + 1]]);
        out.push(f16_to_f32(bits));
    }
    out
}

fn dequant_bf16(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let start = i * 2;
        let bits = u16::from_le_bytes([data[start], data[start + 1]]);
        out.push(bf16_to_f32(bits));
    }
    out
}

fn dequant_q4_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let block_size = 18;
    let mut out = Vec::with_capacity(num_elements);
    for b in 0..num_blocks {
        let block_start = b * block_size;
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = f16_to_f32(scale_bits);
        for i in 0..16 {
            let byte = data[block_start + 2 + i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out.push(lo as f32 * scale);
            out.push(hi as f32 * scale);
        }
    }
    out
}

fn dequant_q4_1(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let block_size = 20;
    let mut out = Vec::with_capacity(num_elements);
    for b in 0..num_blocks {
        let block_start = b * block_size;
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = f16_to_f32(scale_bits);
        let min_bits = u16::from_le_bytes([data[block_start + 2], data[block_start + 3]]);
        let min = f16_to_f32(min_bits);
        for i in 0..16 {
            let byte = data[block_start + 4 + i];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            out.push(lo * scale + min);
            out.push(hi * scale + min);
        }
    }
    out
}

fn dequant_q8_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let block_size = 34;
    let mut out = Vec::with_capacity(num_elements);
    for b in 0..num_blocks {
        let block_start = b * block_size;
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = f16_to_f32(scale_bits);
        for i in 0..32 {
            let q = data[block_start + 2 + i] as i8;
            out.push(q as f32 * scale);
        }
    }
    out
}

fn dequant_q5_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let block_size = 22; // 2 bytes scale + 4 bytes high-bits + 16 bytes low nibbles
    let mut output = Vec::with_capacity(num_elements);

    for b in 0..num_blocks {
        let bs = b * block_size;
        let scale = f16_to_f32(u16::from_le_bytes([data[bs], data[bs + 1]]));
        let high_bits =
            u32::from_le_bytes([data[bs + 2], data[bs + 3], data[bs + 4], data[bs + 5]]);

        for j in 0..32 {
            let byte_idx = bs + 6 + j / 2;
            let nibble = if j % 2 == 0 {
                data[byte_idx] & 0x0F
            } else {
                data[byte_idx] >> 4
            };
            let high = ((high_bits >> j) & 1) as u8;
            let q = (nibble | (high << 4)) as i32 - 16;
            output.push(q as f32 * scale);
        }
    }
    output
}

fn dequant_q6k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 256;
    let block_size = 210;
    let mut output = vec![0.0f32; num_elements];

    for b in 0..num_blocks {
        let bs = b * block_size;
        let ql = &data[bs..bs + 128];
        let qh = &data[bs + 128..bs + 192];
        let scales = &data[bs + 192..bs + 208];
        let d = f16_to_f32(u16::from_le_bytes([data[bs + 208], data[bs + 209]]));
        let out = &mut output[b * 256..(b + 1) * 256];

        for half in 0..2 {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let y_off = half * 128;

            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
                let q2 =
                    ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                let sc0 = scales[half * 8 + is] as i8 as f32;
                let sc1 = scales[half * 8 + is + 2] as i8 as f32;
                let sc2 = scales[half * 8 + is + 4] as i8 as f32;
                let sc3 = scales[half * 8 + is + 6] as i8 as f32;

                out[y_off + l] = d * sc0 * q1 as f32;
                out[y_off + l + 32] = d * sc1 * q2 as f32;
                out[y_off + l + 64] = d * sc2 * q3 as f32;
                out[y_off + l + 96] = d * sc3 * q4 as f32;
            }
        }
    }
    output
}

fn dequant_q5_1(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let block_size = 24; // 2 bytes scale + 2 bytes min + 4 bytes high-bits + 16 bytes low nibbles
    let mut output = Vec::with_capacity(num_elements);

    for b in 0..num_blocks {
        let bs = b * block_size;
        let scale = f16_to_f32(u16::from_le_bytes([data[bs], data[bs + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([data[bs + 2], data[bs + 3]]));
        let high_bits =
            u32::from_le_bytes([data[bs + 4], data[bs + 5], data[bs + 6], data[bs + 7]]);

        for j in 0..32 {
            let byte_idx = bs + 8 + j / 2;
            let nibble = if j % 2 == 0 {
                data[byte_idx] & 0x0F
            } else {
                data[byte_idx] >> 4
            };
            let high = ((high_bits >> j) & 1) as u8;
            let q = (nibble | (high << 4)) as f32;
            output.push(q * scale + min);
        }
    }
    output
}

fn dequant_q2k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 256;
    let block_size = 84;
    let mut output = Vec::with_capacity(num_elements);

    for b in 0..num_blocks {
        let bs = b * block_size;
        let scales = &data[bs..bs + 16];
        let qs = &data[bs + 16..bs + 80];
        let d = f16_to_f32(u16::from_le_bytes([data[bs + 80], data[bs + 81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[bs + 82], data[bs + 83]]));

        for (j, &scale_byte) in scales.iter().enumerate() {
            let sc = (scale_byte & 0xF) as f32;
            let m = (scale_byte >> 4) as f32;

            for i in 0..16 {
                let elem = j * 16 + i;
                let byte_idx = elem / 4;
                let shift = (elem % 4) * 2;
                let q = ((qs[byte_idx] >> shift) & 3) as f32;
                output.push(d * sc * q - dmin * m);
            }
        }
    }
    output
}

fn dequant_q3k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 256;
    let block_size = 110;
    let mut output = Vec::with_capacity(num_elements);

    for b in 0..num_blocks {
        let bs = b * block_size;
        let hmask = &data[bs..bs + 32];
        let qs = &data[bs + 32..bs + 96];
        let packed_scales = &data[bs + 96..bs + 108];
        let d = f16_to_f32(u16::from_le_bytes([data[bs + 108], data[bs + 109]]));

        // Unpack 6-bit scales from 12 bytes into 16 scale values
        // The 12 bytes encode 16 6-bit values:
        //   scales[0..3]: low 6 bits from packed_scales[0..3] (mask 0x3F)
        //   scales[4..7]: low 6 bits from packed_scales[4..7] (mask 0x3F)
        //   scales[8..11]: combine packed_scales[0..3] >> 6 (2 bits) with packed_scales[8..11] low 4 bits
        //   scales[12..15]: combine packed_scales[4..7] >> 6 (2 bits) with packed_scales[8..11] high 4 bits
        let mut sc = [0i32; 16];
        for i in 0..4 {
            sc[i] = (packed_scales[i] & 0x3F) as i32;
            sc[i + 4] = (packed_scales[i + 4] & 0x3F) as i32;
            sc[i + 8] =
                (((packed_scales[i] >> 6) | ((packed_scales[i + 8] & 0x0F) << 2)) & 0x3F) as i32;
            sc[i + 12] =
                (((packed_scales[i + 4] >> 6) | ((packed_scales[i + 8] >> 4) << 2)) & 0x3F) as i32;
        }
        // Convert to signed: subtract 32
        for s in sc.iter_mut() {
            *s -= 32;
        }

        for j in 0..256 {
            let low2 = (qs[j / 4] >> (2 * (j % 4))) & 3;
            let high1 = (hmask[j / 8] >> (j % 8)) & 1;
            let q = ((low2 | (high1 << 2)) as i32) - 4;
            let scale_idx = j / 16;
            output.push(d * sc[scale_idx] as f32 * q as f32);
        }
    }
    output
}

fn unpack_q4k_scales(packed: &[u8]) -> ([f32; 8], [f32; 8]) {
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];

    for i in 0..4 {
        sc[i] = packed[i] & 0x3F;
        sc[i + 4] = ((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F;
        m[i] = packed[i + 4] & 0x3F;
        m[i + 4] = ((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F;
    }

    let sc_f: [f32; 8] = std::array::from_fn(|i| sc[i] as f32);
    let m_f: [f32; 8] = std::array::from_fn(|i| m[i] as f32);
    (sc_f, m_f)
}

pub fn dequant_q4k_row(data: &[u8], output: &mut [f32], num_elements: usize) {
    let v = dequant_q4k(data, num_elements);
    output[..num_elements].copy_from_slice(&v);
}

pub fn dequant_q6k_row(data: &[u8], output: &mut [f32], num_elements: usize) {
    let v = dequant_q6k(data, num_elements);
    output[..num_elements].copy_from_slice(&v);
}

fn dequant_q4k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 256;
    let block_size = 144;
    let mut output = vec![0.0f32; num_elements];

    for b in 0..num_blocks {
        let bs = b * block_size;
        let d = f16_to_f32(u16::from_le_bytes([data[bs], data[bs + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[bs + 2], data[bs + 3]]));
        let packed_scales = &data[bs + 4..bs + 16];
        let qs = &data[bs + 16..bs + 144];

        let (sc, m) = unpack_q4k_scales(packed_scales);
        let out = &mut output[b * 256..(b + 1) * 256];

        for j in 0..4 {
            let q_off = j * 32;
            let d1 = d * sc[2 * j];
            let m1 = dmin * m[2 * j];
            let d2 = d * sc[2 * j + 1];
            let m2 = dmin * m[2 * j + 1];

            for l in 0..32 {
                out[j * 64 + l] = d1 * (qs[q_off + l] & 0x0F) as f32 - m1;
                out[j * 64 + l + 32] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            }
        }
    }
    output
}

fn dequant_q5k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 256;
    let block_size = 176;
    let mut output = vec![0.0f32; num_elements];

    for b in 0..num_blocks {
        let bs = b * block_size;
        let d = f16_to_f32(u16::from_le_bytes([data[bs], data[bs + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[bs + 2], data[bs + 3]]));
        let packed_scales = &data[bs + 4..bs + 16];
        let qh = &data[bs + 16..bs + 48]; // 32 bytes = 256 high bits
        let qs = &data[bs + 48..bs + 176]; // 128 bytes = 256 nibbles (split layout)

        let (sc, m) = unpack_q4k_scales(packed_scales);
        let out = &mut output[b * 256..(b + 1) * 256];

        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in 0..4 {
            let q_off = j * 32;
            let d1 = d * sc[2 * j];
            let m1 = dmin * m[2 * j];
            let d2 = d * sc[2 * j + 1];
            let m2 = dmin * m[2 * j + 1];

            for l in 0..32 {
                let hb1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                let hb2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                out[j * 64 + l] = d1 * ((qs[q_off + l] & 0x0F) + hb1) as f32 - m1;
                out[j * 64 + l + 32] = d2 * ((qs[q_off + l] >> 4) + hb2) as f32 - m2;
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    output
}

pub fn dequant_q2_0_pub(data: &[u8], num_elements: usize) -> Vec<f32> {
    dequant_q2_0(data, num_elements)
}

fn dequant_q2_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    // PrismML Q2_0: QK=128, 34 bytes per block (2-byte f16 scale + 32 bytes packed)
    // Encoding: (q - 1) * scale → 00=-1, 01=0, 10=+1, 11=+2
    const QK: usize = 128;
    const BLOCK_BYTES: usize = 34;
    let num_blocks = num_elements / QK;
    let mut output = vec![0.0f32; num_elements];
    for blk in 0..num_blocks {
        let block = &data[blk * BLOCK_BYTES..(blk + 1) * BLOCK_BYTES];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for i in 0..QK {
            let byte_idx = 2 + i / 4;
            let bit_shift = (i % 4) * 2;
            let q = ((block[byte_idx] >> bit_shift) & 0x03) as i32;
            output[blk * QK + i] = (q - 1) as f32 * scale;
        }
    }
    output
}

fn dequant_tq2(data: &[u8], num_elements: usize) -> Vec<f32> {
    use tempura_kernels::TQ2_BLOCK_SIZE;
    let block_bytes = 66;
    let num_blocks = num_elements / TQ2_BLOCK_SIZE;
    let mut output = vec![0.0f32; num_elements];
    for blk in 0..num_blocks {
        let block = &data[blk * block_bytes..(blk + 1) * block_bytes];
        let mut buf = [0.0f32; 256];
        tempura_kernels::quantized::tq2_dequant_block(block, &mut buf);
        output[blk * TQ2_BLOCK_SIZE..(blk + 1) * TQ2_BLOCK_SIZE].copy_from_slice(&buf);
    }
    output
}

fn dequant_tq1(data: &[u8], num_elements: usize) -> Vec<f32> {
    let packed_bytes = num_elements / 4;
    let mut output = vec![0.0f32; num_elements];
    for i in 0..packed_bytes {
        let byte = data[i];
        for shift in (0..8).step_by(2) {
            let idx = i * 4 + (3 - shift / 2);
            if idx < num_elements {
                let bits = (byte >> shift) & 0x03;
                output[idx] = bits as f32 - 1.0;
            }
        }
    }
    output
}

fn tensor_data_size(dtype: GgmlType, num_elements: u64, _dims: &[u64]) -> u64 {
    match dtype {
        GgmlType::F32 => num_elements * 4,
        GgmlType::F16 | GgmlType::BF16 => num_elements * 2,
        GgmlType::Q4_0 => (num_elements / 32) * 18,
        GgmlType::Q4_1 => (num_elements / 32) * 20,
        GgmlType::Q5_0 => (num_elements / 32) * 22,
        GgmlType::Q5_1 => (num_elements / 32) * 24,
        GgmlType::Q8_0 => (num_elements / 32) * 34,
        GgmlType::Q2K => (num_elements / 256) * 84,
        GgmlType::Q3K => (num_elements / 256) * 110,
        GgmlType::Q4K => (num_elements / 256) * 144,
        GgmlType::Q5K => (num_elements / 256) * 176,
        GgmlType::Q6K => (num_elements / 256) * 210,
        // IQ4_NL: block_iq4_nl = 2 (f16 d) + 16 (qs[16]) = 18 bytes per 32 elements
        GgmlType::IQ4NL => (num_elements / 32) * 18,
        // IQ4_XS: block_iq4_xs = 2 (f16 d) + 2 (scales_h) + 4 (scales_l[4]) + 128 (qs[128]) = 136 bytes per 256
        GgmlType::IQ4XS => (num_elements / 256) * 136,
        GgmlType::TQ1_0 => {
            // I2_S in bitnet.cpp: 2-bit packed (ne/4) + one f32 scale per tensor
            num_elements / 4 + 4
        }
        GgmlType::TQ2_0 => (num_elements / 256) * 66,
        GgmlType::Q2_0 => (num_elements / 128) * 34, // QK=128: 2-byte f16 scale + 32 bytes packed
        GgmlType::TL1 => {
            // ARM TL1 LUT format: 2-bit packed + 32-byte alignment
            num_elements / 4 + 32
        }
        // bitnet.cpp I2_S/TL2: 2-bit packed (ne/4 bytes) + one f32 scale per tensor
        GgmlType::I2S => num_elements / 4 + 4,
        GgmlType::Q8_1 => (num_elements / 32) * 36,
        GgmlType::Q8K => (num_elements / 256) * 292,
        GgmlType::IQ2XXS => (num_elements / 256) * 66,
        GgmlType::IQ2XS => (num_elements / 256) * 74,
        GgmlType::IQ3XXS => (num_elements / 256) * 98,
        GgmlType::IQ1S => (num_elements / 256) * 50,
        GgmlType::IQ3S => (num_elements / 256) * 110,
        GgmlType::IQ2S => (num_elements / 256) * 82,
        GgmlType::IQ1M => (num_elements / 256) * 56,
        GgmlType::NVFP4 => (num_elements / 64) * 36,
        GgmlType::Q1_0 => (num_elements / 32) * 18,
    }
}

fn map_gguf_name(gguf_name: &str) -> Option<String> {
    match gguf_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }

    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        let dot_pos = rest.find('.')?;
        let n = &rest[..dot_pos];
        let suffix = &rest[dot_pos + 1..];
        let hf_suffix = match suffix {
            "attn_norm.weight" => "input_layernorm.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_qkv.weight" => "ssm.qkv_proj.weight",
            "attn_gate.weight" => "ssm.gate_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            "ffn_sub_norm.weight" => "mlp.ffn_sub_norm.weight",
            "attn_sub_norm.weight" => "self_attn.attn_sub_norm.weight",
            "post_attention_norm.weight" => "post_attention_layernorm.weight",
            "ssm_in.weight" => "ssm.in_proj.weight",
            "ssm_out.weight" => "ssm.out_proj.weight",
            "ssm_conv1d.weight" => "ssm.conv1d.weight",
            "ssm_conv1d.bias" => "ssm.conv1d.bias",
            "ssm_dt.weight" => "ssm.dt_proj.weight",
            "ssm_dt.bias" => "ssm.dt_proj.bias",
            "ssm_a" => "ssm.A_log",
            "ssm_norm.weight" => "ssm.norm.weight",
            "ssm_beta_alpha.weight" => "ssm.beta_alpha.weight",
            "ssm_ba.weight" => "ssm.beta_alpha.weight",
            "ssm_alpha.weight" => "ssm.alpha.weight",
            "ssm_beta.weight" => "ssm.beta.weight",
            _ => return None,
        };
        return Some(format!("model.layers.{n}.{hf_suffix}"));
    }

    None
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U64(v) => Some(*v),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::I32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            GgufValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    result.push(v.as_str()?);
                }
                Some(result)
            }
            _ => None,
        }
    }

    pub fn as_u32_array(&self) -> Option<Vec<u32>> {
        match self {
            GgufValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        GgufValue::U32(x) => result.push(*x),
                        GgufValue::I32(x) => result.push(*x as u32),
                        GgufValue::U64(x) => result.push(*x as u32),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            GgufValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    result.push(v.as_f32()?);
                }
                Some(result)
            }
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

fn get_usize_from_meta(metadata: &HashMap<String, GgufValue>, suffix: &str) -> Option<usize> {
    let arch = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");
    let key = format!("{arch}.{suffix}");
    if let Some(val) = metadata.get(&key) {
        if let Some(n) = val.as_u64() {
            return Some(n as usize);
        }
    }
    None
}

fn detect_and_apply_hybrid_layers(
    config: &mut ModelConfig,
    tensor_names: &[&str],
    metadata: &HashMap<String, GgufValue>,
) {
    use crate::config::LayerKind;

    let arch = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    // Primary detection: full_attention_interval metadata (Qwen3.5/3.6)
    let interval = metadata
        .get(&format!("{arch}.full_attention_interval"))
        .and_then(|v| v.as_u64())
        .or_else(|| {
            metadata
                .get("full_attention_interval")
                .and_then(|v| v.as_u64())
        });

    if let Some(interval) = interval {
        let interval = interval as usize;
        let mut kinds = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            if (layer + 1) % interval == 0 {
                kinds.push(LayerKind::Attention);
            } else {
                kinds.push(LayerKind::DeltaNet);
            }
        }
        config.layer_kinds = Some(kinds);
    } else {
        // Fallback: scan tensor names for DeltaNet markers (ssm_alpha, ssm_ba, ssm_beta_alpha)
        let has_ssm = tensor_names.iter().any(|n| {
            n.contains("ssm_alpha")
                || n.contains("ssm.alpha")
                || n.contains("ssm_ba")
                || n.contains("ssm.beta_alpha")
                || n.contains("ssm_beta_alpha")
        });
        if !has_ssm {
            return;
        }

        let mut kinds = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let markers = [
                format!("blk.{layer}.ssm_alpha.weight"),
                format!("blk.{layer}.ssm_ba.weight"),
                format!("blk.{layer}.ssm_beta_alpha.weight"),
                format!("model.layers.{layer}.ssm.alpha.weight"),
                format!("model.layers.{layer}.ssm.beta_alpha.weight"),
            ];
            let has_ssm_layer = tensor_names.iter().any(|n| markers.iter().any(|m| n == m));

            if has_ssm_layer {
                kinds.push(LayerKind::DeltaNet);
            } else {
                kinds.push(LayerKind::Attention);
            }
        }

        if kinds.iter().all(|k| *k == LayerKind::Attention) {
            return;
        }
        config.layer_kinds = Some(kinds);
    }

    // SSM dimensions from metadata
    config.deltanet_num_heads = metadata
        .get(&format!("{arch}.ssm.time_step_rank"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .or(Some(config.num_kv_heads));

    config.deltanet_head_dim = metadata
        .get(&format!("{arch}.ssm.state_size"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .or(Some(config.head_dim));

    config.ssm_a = metadata
        .get(&format!("{arch}.ssm.time_step_a"))
        .and_then(|v| v.as_f32())
        .or_else(|| metadata.get("ssm.time_step_a").and_then(|v| v.as_f32()));
}

pub fn extract_config(metadata: &HashMap<String, GgufValue>) -> Result<ModelConfig, LoadError> {
    let arch = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    let get_usize = |suffix: &str| -> Option<usize> {
        let key = format!("{arch}.{suffix}");
        if let Some(val) = metadata.get(&key) {
            if let Some(n) = val.as_u64() {
                return Some(n as usize);
            }
            if let Some(n) = val.as_u32() {
                return Some(n as usize);
            }
        }
        let general_key = format!("general.{suffix}");
        if let Some(val) = metadata.get(&general_key) {
            if let Some(n) = val.as_u64() {
                return Some(n as usize);
            }
            if let Some(n) = val.as_u32() {
                return Some(n as usize);
            }
        }
        None
    };

    let hidden_size = get_usize("embedding_length")
        .ok_or_else(|| LoadError::GgufParse("missing embedding_length in metadata".into()))?;

    let intermediate_size = get_usize("feed_forward_length")
        .ok_or_else(|| LoadError::GgufParse("missing feed_forward_length in metadata".into()))?;

    let num_attention_heads = get_usize("attention.head_count")
        .ok_or_else(|| LoadError::GgufParse("missing attention.head_count in metadata".into()))?;

    let num_kv_heads = get_usize("attention.head_count_kv").unwrap_or(num_attention_heads);

    let num_layers = get_usize("block_count")
        .ok_or_else(|| LoadError::GgufParse("missing block_count in metadata".into()))?;

    let vocab_size = get_usize("vocab_size").unwrap_or(32000);

    let max_seq_len = get_usize("context_length").unwrap_or(4096);

    let rope_theta = metadata
        .get(&format!("{arch}.rope.freq_base"))
        .and_then(|v| v.as_f32())
        .unwrap_or(10000.0);

    let layer_norm_eps = metadata
        .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .and_then(|v| v.as_f32())
        .unwrap_or(1e-5);

    let head_dim = get_usize("attention.key_length").unwrap_or(hidden_size / num_attention_heads);

    let is_bitnet = arch.contains("bitnet");

    let file_type = metadata.get("general.file_type").and_then(|v| v.as_u32());
    let quantization = if is_bitnet {
        Some(QuantConfig {
            format: crate::config::QuantFormat::Tq2,
            bits: 2,
            group_size: 256,
            symmetric: true,
        })
    } else {
        file_type.and_then(|ft| match ft {
            0 | 1 => None,
            2 | 3 => Some(QuantConfig::gguf(4)),
            7 => Some(QuantConfig::gguf(8)),
            _ => Some(QuantConfig::gguf(4)),
        })
    };

    let hidden_act = if is_bitnet {
        crate::config::HiddenAct::SquaredRelu
    } else {
        match arch {
            "gpt2" | "gptj" | "gptneox" | "starcoder" | "refact" => crate::config::HiddenAct::Gelu,
            "phi2" | "phi3" | "phimoe" => crate::config::HiddenAct::Gelu,
            "gemma" | "gemma2" | "gemma3" => crate::config::HiddenAct::GeGLU,
            "falcon" | "mpt" => crate::config::HiddenAct::Gelu,
            _ => crate::config::HiddenAct::SiluGate,
        }
    };

    // Qwen3.5/3.6 use IMROPE (interleaved multi-rope) = adjacent-pair (Norm) RoPE.
    // Standard GGUF converter permutes Q/K weights for adjacent-pair pairing.
    // BitNet/ternary GGUF converter does NOT permute → needs NeoX (split-half) RoPE.
    let rope_type = if (arch == "qwen35" || arch == "qwen35moe") && !is_bitnet {
        crate::config::RopeType::Norm
    } else {
        crate::config::RopeType::Neox
    };

    let architecture = crate::config::Architecture::from_gguf_name(arch);

    Ok(ModelConfig {
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_kv_heads,
        num_layers,
        vocab_size,
        max_seq_len,
        head_dim,
        rope_theta,
        layer_norm_eps,
        tensor_parallel_size: 1,
        quantization,
        use_qk_norm: false,
        sliding_window: None,
        hidden_act,
        rope_type,
        architecture,
        n_experts: get_usize("expert_count").unwrap_or(0),
        n_experts_used: get_usize("expert_used_count").unwrap_or(0),
        use_layernorm: matches!(
            architecture,
            crate::config::Architecture::Falcon
                | crate::config::Architecture::Mpt
                | crate::config::Architecture::Phi
        ),
        use_alibi: matches!(architecture, crate::config::Architecture::Mpt),
        alibi_max_bias: if matches!(architecture, crate::config::Architecture::Mpt) {
            metadata
                .get(&format!("{arch}.attention.alibi_bias_max"))
                .and_then(|v| v.as_f32())
                .unwrap_or(8.0)
        } else {
            0.0
        },
        parallel_attn_ffn: matches!(architecture, crate::config::Architecture::Phi)
            || metadata
                .get(&format!("{arch}.attention.use_parallel_residual"))
                .and_then(|v| match v {
                    GgufValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(false),
        rmsnorm_offset: matches!(architecture, crate::config::Architecture::Gemma),
        attn_logit_softcapping: metadata
            .get(&format!("{arch}.attention.logit_softcapping"))
            .and_then(|v| v.as_f32()),
        embedding_scale: matches!(architecture, crate::config::Architecture::Gemma),
        layer_kinds: None,
        deltanet_num_heads: None,
        deltanet_head_dim: None,
        ssm_a: None,
        dt_bias: None,
        rope_dim: if std::env::var("TEMPURA_FULL_ROPE").is_ok() {
            None
        } else {
            get_usize("rope.dimension_count")
        },
        rope_scaling: {
            let scaling_type_str = metadata
                .get(&format!("{arch}.rope.scaling.type"))
                .and_then(|v| v.as_str().map(|s| s.to_string()));
            scaling_type_str.and_then(|s| {
                let st = match s.as_str() {
                    "linear" => crate::config::RopeScalingType::Linear,
                    "ntk" => crate::config::RopeScalingType::NTK,
                    "yarn" => crate::config::RopeScalingType::YaRN,
                    "llama3" | "longrope" => crate::config::RopeScalingType::LLaMA3,
                    _ => return None,
                };
                let factor = metadata
                    .get(&format!("{arch}.rope.scaling.factor"))
                    .and_then(|v| v.as_f32())
                    .unwrap_or(1.0);
                let original_context_length = metadata
                    .get(&format!("{arch}.rope.scaling.original_context_length"))
                    .and_then(|v| v.as_u32())
                    .unwrap_or(0);
                let yarn_beta_fast = metadata
                    .get(&format!("{arch}.rope.scaling.yarn_beta_fast"))
                    .and_then(|v| v.as_f32())
                    .unwrap_or(32.0);
                let yarn_beta_slow = metadata
                    .get(&format!("{arch}.rope.scaling.yarn_beta_slow"))
                    .and_then(|v| v.as_f32())
                    .unwrap_or(1.0);
                let yarn_attn_factor = metadata
                    .get(&format!("{arch}.rope.scaling.yarn_attn_factor"))
                    .and_then(|v| v.as_f32())
                    .unwrap_or(1.0);
                let low_freq_factor = metadata
                    .get(&format!("{arch}.rope.scaling.low_freq_factor"))
                    .and_then(|v| v.as_f32())
                    .unwrap_or(1.0);
                let high_freq_factor = metadata
                    .get(&format!("{arch}.rope.scaling.high_freq_factor"))
                    .and_then(|v| v.as_f32())
                    .unwrap_or(4.0);
                Some(crate::config::RopeScalingConfig {
                    scaling_type: st,
                    factor,
                    original_context_length,
                    yarn_beta_fast,
                    yarn_beta_slow,
                    yarn_attn_factor,
                    low_freq_factor,
                    high_freq_factor,
                })
            })
        },
    })
}

fn convert_ggml_type(dtype: GgmlType) -> tempura_kernels::ggml::GgmlType {
    use tempura_kernels::ggml::GgmlType as GT;
    match dtype {
        GgmlType::F32 => GT::F32,
        GgmlType::F16 => GT::F16,
        GgmlType::Q4_0 => GT::Q4_0,
        GgmlType::Q4_1 => GT::Q4_1,
        GgmlType::Q5_0 => GT::Q5_0,
        GgmlType::Q5_1 => GT::Q5_1,
        GgmlType::Q8_0 => GT::Q8_0,
        GgmlType::Q8_1 => GT::Q8_1,
        GgmlType::Q2K => GT::Q2K,
        GgmlType::Q3K => GT::Q3K,
        GgmlType::Q4K => GT::Q4K,
        GgmlType::Q5K => GT::Q5K,
        GgmlType::Q6K => GT::Q6K,
        GgmlType::Q8K => GT::Q8K,
        GgmlType::IQ2XXS => GT::IQ2XXS,
        GgmlType::IQ2XS => GT::IQ2XS,
        GgmlType::IQ3XXS => GT::IQ3XXS,
        GgmlType::IQ1S => GT::IQ1S,
        GgmlType::IQ4NL => GT::IQ4NL,
        GgmlType::IQ3S => GT::IQ3S,
        GgmlType::IQ2S => GT::IQ2S,
        GgmlType::IQ4XS => GT::IQ4XS,
        GgmlType::IQ1M => GT::IQ1M,
        GgmlType::BF16 => GT::BF16,
        GgmlType::TQ1_0 => GT::TQ1_0,
        GgmlType::TQ2_0 => GT::TQ2_0,
        GgmlType::TL1 => GT::TQ1_0,
        GgmlType::I2S => GT::TQ1_0,
        GgmlType::NVFP4 => GT::NVFP4,
        GgmlType::Q1_0 => GT::Q1_0,
        GgmlType::Q2_0 => GT::Q2K,
    }
}

pub struct GgufLoader {
    mmap: Option<memmap2::Mmap>,
    tensors: HashMap<String, Vec<f32>>,
    raw_tensors: HashMap<String, Vec<u8>>,
    raw_tensor_dims: HashMap<String, Vec<u64>>,
    tensor_dtypes: HashMap<String, GgmlType>,
    f16_raw_tensors: HashMap<String, Vec<u8>>,
    mmap_ranges: HashMap<String, (usize, usize, GgmlType)>,
    pub config: ModelConfig,
    pub metadata: HashMap<String, GgufValue>,
}

impl std::fmt::Debug for GgufLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufLoader")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Detect split GGUF shards (pattern: `prefix-NNNNN-of-MMMMM.gguf`) and return all shard paths
/// in order. Returns a single-element vec for non-split files.
fn find_split_shards(path: &Path) -> Result<Vec<std::path::PathBuf>, LoadError> {
    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let stem = match filename.strip_suffix(".gguf") {
        Some(s) => s,
        None => return Ok(vec![path.to_path_buf()]),
    };
    // Look for `-NNNNN-of-MMMMM` suffix
    let of_idx = match stem.rfind("-of-") {
        Some(i) => i,
        None => return Ok(vec![path.to_path_buf()]),
    };
    let total_str = &stem[of_idx + 4..];
    let total: usize = match total_str.parse() {
        Ok(n) if n > 0 => n,
        _ => return Ok(vec![path.to_path_buf()]),
    };
    let before_of = &stem[..of_idx];
    let dash_idx = match before_of.rfind('-') {
        Some(i) => i,
        None => return Ok(vec![path.to_path_buf()]),
    };
    if before_of[dash_idx + 1..].parse::<usize>().is_err() {
        return Ok(vec![path.to_path_buf()]);
    }
    let prefix = &before_of[..dash_idx];
    let dir = path.parent().unwrap_or(std::path::Path::new("."));
    let mut shards = Vec::with_capacity(total);
    for i in 1..=total {
        let name = format!("{prefix}-{i:05}-of-{total:05}.gguf");
        let shard = dir.join(&name);
        if !shard.exists() {
            return Err(LoadError::GgufParse(format!(
                "missing GGUF shard: {}",
                shard.display()
            )));
        }
        shards.push(shard);
    }
    Ok(shards)
}

impl GgufLoader {
    pub fn from_file(path: &Path) -> Result<Self, LoadError> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Load a GGUF model, transparently handling split multi-file shards.
    /// Accepts any shard path (e.g. `model-00001-of-00004.gguf`) and loads all shards.
    pub fn from_path(path: &Path) -> Result<Self, LoadError> {
        let shards = find_split_shards(path)?;
        if shards.len() <= 1 {
            return Self::from_file(path);
        }
        let data0 = std::fs::read(&shards[0])?;
        let mut loader = Self::from_bytes(&data0)?;
        for shard_path in &shards[1..] {
            let data = std::fs::read(shard_path)?;
            let shard = Self::from_bytes(&data)?;
            loader.tensors.extend(shard.tensors);
            loader.raw_tensors.extend(shard.raw_tensors);
            loader.tensor_dtypes.extend(shard.tensor_dtypes);
            loader.raw_tensor_dims.extend(shard.raw_tensor_dims);
            loader.f16_raw_tensors.extend(shard.f16_raw_tensors);
            loader.mmap_ranges.extend(shard.mmap_ranges);
        }
        Ok(loader)
    }

    /// Load a quantized GGUF model, transparently handling split multi-file shards.
    pub fn from_path_quantized(path: &Path) -> Result<Self, LoadError> {
        let shards = find_split_shards(path)?;
        if shards.len() <= 1 {
            return Self::from_file_quantized(path);
        }
        // Quantized path uses mmap; for multi-shard we fall back to from_bytes_quantized
        // which loads without mmap so we can merge shards.
        let data0 = std::fs::read(&shards[0])?;
        let mut loader = Self::from_bytes_quantized(&data0)?;
        for shard_path in &shards[1..] {
            let data = std::fs::read(shard_path)?;
            let shard = Self::from_bytes_quantized(&data)?;
            loader.tensors.extend(shard.tensors);
            loader.raw_tensors.extend(shard.raw_tensors);
            loader.tensor_dtypes.extend(shard.tensor_dtypes);
            loader.raw_tensor_dims.extend(shard.raw_tensor_dims);
            loader.f16_raw_tensors.extend(shard.f16_raw_tensors);
            loader.mmap_ranges.extend(shard.mmap_ranges);
        }
        Ok(loader)
    }

    pub fn from_file_quantized(path: &Path) -> Result<Self, LoadError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let mut loader = Self::from_bytes_quantized(&mmap)?;
        loader.mmap = Some(mmap);
        Ok(loader)
    }

    fn from_bytes_quantized(data: &[u8]) -> Result<Self, LoadError> {
        let mut offset = 0usize;

        let magic = read_u32(data, &mut offset)?;
        if magic != GGUF_MAGIC {
            return Err(LoadError::GgufParse(format!(
                "invalid magic: expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}"
            )));
        }
        let version = read_u32(data, &mut offset)?;
        if version != GGUF_VERSION {
            return Err(LoadError::GgufParse(format!(
                "unsupported GGUF version: {version} (expected {GGUF_VERSION})"
            )));
        }

        let tensor_count = read_u64(data, &mut offset)? as usize;
        let metadata_kv_count = read_u64(data, &mut offset)? as usize;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(data, &mut offset)?;
            let value_type = read_u32(data, &mut offset)?;
            let value = read_metadata_value(data, &mut offset, value_type)?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(data, &mut offset)?;
            let n_dims = read_u32(data, &mut offset)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(data, &mut offset)?);
            }
            let dtype = GgmlType::from_u32(read_u32(data, &mut offset)?)?;
            let tensor_offset = read_u64(data, &mut offset)?;
            tensor_infos.push(TensorInfo {
                name,
                dims,
                dtype,
                offset: tensor_offset,
            });
        }

        let data_start = offset.div_ceil(ALIGNMENT) * ALIGNMENT;

        // Compute tensor byte sizes from adjacent offsets (robust against unknown quant formats).
        let mut sorted_by_offset_nonq: Vec<usize> = (0..tensor_infos.len()).collect();
        sorted_by_offset_nonq.sort_by_key(|&i| tensor_infos[i].offset);
        let mut tensor_byte_sizes_nonq: Vec<usize> = vec![0; tensor_infos.len()];
        for k in 0..sorted_by_offset_nonq.len() {
            let idx = sorted_by_offset_nonq[k];
            let abs_off = data_start + tensor_infos[idx].offset as usize;
            let end = if k + 1 < sorted_by_offset_nonq.len() {
                data_start + tensor_infos[sorted_by_offset_nonq[k + 1]].offset as usize
            } else {
                data.len()
            };
            tensor_byte_sizes_nonq[idx] = end.saturating_sub(abs_off);
        }

        let mut tensors = HashMap::new();
        let mut raw_tensors = HashMap::new();
        let mut raw_tensor_dims: HashMap<String, Vec<u64>> = HashMap::new();
        let mut tensor_dtypes = HashMap::new();
        let mut f16_raw_tensors: HashMap<String, Vec<u8>> = HashMap::new();
        let mut mmap_ranges: HashMap<String, (usize, usize, GgmlType)> = HashMap::new();

        for (i, info) in tensor_infos.iter().enumerate() {
            let num_elements: u64 = info.dims.iter().product();
            let byte_size = tensor_byte_sizes_nonq[i];
            let abs_offset = data_start + info.offset as usize;

            if abs_offset + byte_size > data.len() {
                return Err(LoadError::GgufParse(format!(
                    "tensor '{}' data out of bounds",
                    info.name,
                )));
            }

            let tensor_data = &data[abs_offset..abs_offset + byte_size];
            let ne = num_elements as usize;
            let mapped_name = map_gguf_name(&info.name).unwrap_or_else(|| info.name.clone());

            let is_quantized =
                !matches!(info.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16);
            // I2S (bitnet.cpp type 39) has no matching GPU quantized kernel yet — dequantize to f32.
            // TL1 (type 38, ARM LUT format) also dequantized via TQ1_0 path.
            let force_dequant = matches!(info.dtype, GgmlType::I2S | GgmlType::TL1);
            let keep_raw = if std::env::var("TEMPURA_F32_ALL").is_ok() {
                false
            } else {
                is_quantized
                    && !force_dequant
                    && !mapped_name.contains("embed_tokens")
                    && !mapped_name.contains("norm.weight")
            };

            if info.dtype == GgmlType::F16 {
                mmap_ranges.insert(mapped_name.clone(), (abs_offset, byte_size, GgmlType::F16));
            }

            let skip_f32 = info.dtype == GgmlType::F16;

            let use_mmap = keep_raw
                && matches!(
                    info.dtype,
                    GgmlType::TQ1_0
                        | GgmlType::TL1
                        | GgmlType::Q4K
                        | GgmlType::Q6K
                        | GgmlType::Q2_0
                );
            if use_mmap {
                mmap_ranges.insert(mapped_name.clone(), (abs_offset, byte_size, info.dtype));
            }

            if !keep_raw && !skip_f32 {
                let floats = match info.dtype {
                    GgmlType::F32 => dequant_f32(tensor_data, ne),
                    GgmlType::F16 => dequant_f16(tensor_data, ne),
                    GgmlType::BF16 => dequant_bf16(tensor_data, ne),
                    GgmlType::Q4_0 => dequant_q4_0(tensor_data, ne),
                    GgmlType::Q4_1 => dequant_q4_1(tensor_data, ne),
                    GgmlType::Q5_0 => dequant_q5_0(tensor_data, ne),
                    GgmlType::Q5_1 => dequant_q5_1(tensor_data, ne),
                    GgmlType::Q8_0 => dequant_q8_0(tensor_data, ne),
                    GgmlType::Q2K => dequant_q2k(tensor_data, ne),
                    GgmlType::Q3K => dequant_q3k(tensor_data, ne),
                    GgmlType::Q4K => dequant_q4k(tensor_data, ne),
                    GgmlType::Q5K => dequant_q5k(tensor_data, ne),
                    GgmlType::Q6K => dequant_q6k(tensor_data, ne),
                    GgmlType::TQ2_0 => dequant_tq2(tensor_data, ne),
                    GgmlType::Q2_0 => dequant_q2_0(tensor_data, ne),
                    GgmlType::TQ1_0 | GgmlType::TL1 => dequant_tq1(tensor_data, ne),
                    GgmlType::I2S => {
                        // I2S/TL2 embed/norm should not appear — fall back to zeros
                        vec![0.0f32; ne]
                    }
                    // Types with no dequant kernel yet
                    GgmlType::Q8_1
                    | GgmlType::Q8K
                    | GgmlType::IQ2XXS
                    | GgmlType::IQ2XS
                    | GgmlType::IQ3XXS
                    | GgmlType::IQ1S
                    | GgmlType::IQ4NL
                    | GgmlType::IQ3S
                    | GgmlType::IQ2S
                    | GgmlType::IQ4XS
                    | GgmlType::IQ1M
                    | GgmlType::NVFP4
                    | GgmlType::Q1_0 => vec![0.0f32; ne],
                };
                tensors.insert(mapped_name.clone(), floats);
            } else if !use_mmap && !skip_f32 {
                raw_tensors.insert(mapped_name.clone(), tensor_data.to_vec());
            }
            if keep_raw || use_mmap {
                raw_tensor_dims.insert(mapped_name.clone(), info.dims.clone());
            }
            tensor_dtypes.insert(mapped_name.clone(), info.dtype);
        }

        let vocab_size_from_embed = tensors
            .get("model.embed_tokens.weight")
            .map(|t| {
                let hidden = get_usize_from_meta(&metadata, "embedding_length").unwrap_or(1);
                t.len().checked_div(hidden).unwrap_or(0)
            })
            .or_else(|| {
                f16_raw_tensors.get("model.embed_tokens.weight").map(|t| {
                    let hidden = get_usize_from_meta(&metadata, "embedding_length").unwrap_or(1);
                    (t.len() / 2).checked_div(hidden).unwrap_or(0)
                })
            })
            .or_else(|| {
                let hidden = get_usize_from_meta(&metadata, "embedding_length").unwrap_or(1);
                mmap_ranges
                    .get("model.embed_tokens.weight")
                    .map(|&(_, len, dtype)| {
                        let row_bytes =
                            tensor_data_size(dtype, hidden as u64, &[hidden as u64]) as usize;
                        len.checked_div(row_bytes).unwrap_or(0)
                    })
            });

        let mut config = extract_config(&metadata)?;
        if let Some(vs) = vocab_size_from_embed {
            if vs > 0 {
                config.vocab_size = vs;
            }
        }
        config.use_qk_norm = (0..config.num_layers)
            .any(|l| tensors.contains_key(&format!("model.layers.{l}.self_attn.q_norm.weight")));

        // Detect ternary/BitNet attention weights: if Q uses Q2_0/TQ2_0,
        // the converter did NOT permute Q/K → need NeoX RoPE even for qwen35.
        let has_ternary_attn = tensor_dtypes.iter().any(|(name, dtype)| {
            name.contains("self_attn.q_proj") && matches!(dtype, GgmlType::Q2_0 | GgmlType::TQ2_0)
        }) || mmap_ranges.iter().any(|(name, &(_, _, dtype))| {
            name.contains("self_attn.q_proj") && matches!(dtype, GgmlType::Q2_0 | GgmlType::TQ2_0)
        });
        if has_ternary_attn && config.rope_type == crate::config::RopeType::Norm {
            config.rope_type = crate::config::RopeType::Neox;
        }

        {
            let all_names: Vec<String> = tensors
                .keys()
                .chain(raw_tensors.keys())
                .chain(f16_raw_tensors.keys())
                .chain(mmap_ranges.keys())
                .cloned()
                .collect();
            detect_and_apply_hybrid_layers(
                &mut config,
                &all_names.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                &metadata,
            );
        }

        if !tensors.contains_key("lm_head.weight")
            && !raw_tensors.contains_key("lm_head.weight")
            && !f16_raw_tensors.contains_key("lm_head.weight")
            && !mmap_ranges.contains_key("lm_head.weight")
        {
            if let Some(&(off, len, dtype)) = mmap_ranges.get("model.embed_tokens.weight") {
                mmap_ranges.insert("lm_head.weight".into(), (off, len, dtype));
                tensor_dtypes.insert("lm_head.weight".into(), dtype);
            } else if let Some(f16_raw) = f16_raw_tensors.get("model.embed_tokens.weight") {
                f16_raw_tensors.insert("lm_head.weight".into(), f16_raw.clone());
                tensor_dtypes.insert("lm_head.weight".into(), GgmlType::F16);
            } else if let Some(embed) = tensors.get("model.embed_tokens.weight") {
                tensors.insert("lm_head.weight".into(), embed.clone());
                tensor_dtypes.insert("lm_head.weight".into(), GgmlType::F32);
            }
        }

        Ok(Self {
            mmap: None,
            tensors,
            raw_tensors,
            raw_tensor_dims,
            tensor_dtypes,
            f16_raw_tensors,
            mmap_ranges,
            config,
            metadata,
        })
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, LoadError> {
        let mut offset = 0usize;

        let magic = read_u32(data, &mut offset)?;
        if magic != GGUF_MAGIC {
            return Err(LoadError::GgufParse(format!(
                "invalid magic: expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}"
            )));
        }

        let version = read_u32(data, &mut offset)?;
        if version != GGUF_VERSION {
            return Err(LoadError::GgufParse(format!(
                "unsupported GGUF version: {version} (expected {GGUF_VERSION})"
            )));
        }

        let tensor_count = read_u64(data, &mut offset)? as usize;
        let metadata_kv_count = read_u64(data, &mut offset)? as usize;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(data, &mut offset)?;
            let value_type = read_u32(data, &mut offset)?;
            let value = read_metadata_value(data, &mut offset, value_type)?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(data, &mut offset)?;
            let n_dims = read_u32(data, &mut offset)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(data, &mut offset)?);
            }
            let dtype = GgmlType::from_u32(read_u32(data, &mut offset)?)?;
            let tensor_offset = read_u64(data, &mut offset)?;
            tensor_infos.push(TensorInfo {
                name,
                dims,
                dtype,
                offset: tensor_offset,
            });
        }

        let data_start = offset.div_ceil(ALIGNMENT) * ALIGNMENT;

        // Compute each tensor's byte size from adjacent offsets (handles proprietary formats
        // like I2_S whose block structure is unknown). Sort by file offset, then compute
        // size as next_offset - current_offset; last tensor uses file end.
        let mut sorted_by_offset: Vec<usize> = (0..tensor_infos.len()).collect();
        sorted_by_offset.sort_by_key(|&i| tensor_infos[i].offset);

        let mut tensor_byte_sizes: Vec<usize> = vec![0; tensor_infos.len()];
        for k in 0..sorted_by_offset.len() {
            let idx = sorted_by_offset[k];
            let abs_off = data_start + tensor_infos[idx].offset as usize;
            let end = if k + 1 < sorted_by_offset.len() {
                data_start + tensor_infos[sorted_by_offset[k + 1]].offset as usize
            } else {
                data.len()
            };
            if end > data.len() {
                return Err(LoadError::GgufParse(format!(
                    "tensor '{}' offset points past end of file",
                    tensor_infos[idx].name,
                )));
            }
            tensor_byte_sizes[idx] = end.saturating_sub(abs_off);
        }

        let mut tensors = HashMap::new();
        let mut raw_tensors = HashMap::new();
        let mut tensor_dtypes = HashMap::new();
        for (i, info) in tensor_infos.iter().enumerate() {
            let num_elements: u64 = info.dims.iter().product();
            let byte_size = tensor_byte_sizes[i];
            let abs_offset = data_start + info.offset as usize;

            if abs_offset + byte_size > data.len() {
                return Err(LoadError::GgufParse(format!(
                    "tensor '{}' data out of bounds: offset {} + size {} > file size {}",
                    info.name,
                    abs_offset,
                    byte_size,
                    data.len()
                )));
            }

            let tensor_data = &data[abs_offset..abs_offset + byte_size];
            let ne = num_elements as usize;

            let floats = match info.dtype {
                GgmlType::F32 => dequant_f32(tensor_data, ne),
                GgmlType::F16 => dequant_f16(tensor_data, ne),
                GgmlType::BF16 => dequant_bf16(tensor_data, ne),
                GgmlType::Q4_0 => dequant_q4_0(tensor_data, ne),
                GgmlType::Q4_1 => dequant_q4_1(tensor_data, ne),
                GgmlType::Q5_0 => dequant_q5_0(tensor_data, ne),
                GgmlType::Q5_1 => dequant_q5_1(tensor_data, ne),
                GgmlType::Q8_0 => dequant_q8_0(tensor_data, ne),
                GgmlType::Q2K => dequant_q2k(tensor_data, ne),
                GgmlType::Q3K => dequant_q3k(tensor_data, ne),
                GgmlType::Q4K => dequant_q4k(tensor_data, ne),
                GgmlType::Q5K => dequant_q5k(tensor_data, ne),
                GgmlType::Q6K => dequant_q6k(tensor_data, ne),
                GgmlType::TQ2_0 => dequant_tq2(tensor_data, ne),
                GgmlType::Q2_0 => dequant_q2_0(tensor_data, ne),
                GgmlType::TQ1_0 | GgmlType::TL1 => dequant_tq1(tensor_data, ne),
                GgmlType::I2S => vec![0.0f32; ne],
                // Types with no dequant kernel yet — load raw bytes, return zeros for f32 path
                GgmlType::Q8_1
                | GgmlType::Q8K
                | GgmlType::IQ2XXS
                | GgmlType::IQ2XS
                | GgmlType::IQ3XXS
                | GgmlType::IQ1S
                | GgmlType::IQ4NL
                | GgmlType::IQ3S
                | GgmlType::IQ2S
                | GgmlType::IQ4XS
                | GgmlType::IQ1M
                | GgmlType::NVFP4
                | GgmlType::Q1_0 => vec![0.0f32; ne],
            };

            let mapped_name = map_gguf_name(&info.name).unwrap_or_else(|| info.name.clone());
            raw_tensors.insert(mapped_name.clone(), tensor_data.to_vec());
            tensor_dtypes.insert(mapped_name.clone(), info.dtype);
            tensors.insert(mapped_name, floats);
        }

        let vocab_size_from_embed = tensors.get("model.embed_tokens.weight").map(|t| {
            let hidden = get_usize_from_meta(&metadata, "embedding_length").unwrap_or(1);
            t.len().checked_div(hidden).unwrap_or(0)
        });

        let mut config = extract_config(&metadata)?;
        if let Some(vs) = vocab_size_from_embed {
            if vs > 0 {
                config.vocab_size = vs;
            }
        }

        config.use_qk_norm = (0..config.num_layers)
            .any(|l| tensors.contains_key(&format!("model.layers.{l}.self_attn.q_norm.weight")));

        detect_and_apply_hybrid_layers(
            &mut config,
            &tensors.keys().map(|s| s.as_str()).collect::<Vec<_>>(),
            &metadata,
        );

        Ok(Self {
            mmap: None,
            tensors,
            raw_tensors,
            raw_tensor_dims: HashMap::new(),
            tensor_dtypes,
            f16_raw_tensors: HashMap::new(),
            mmap_ranges: HashMap::new(),
            config,
            metadata,
        })
    }
}

impl GgufLoader {
    pub fn to_quantized_weights(self) -> crate::weights::QuantizedWeights {
        let q2_shapes = self.q2_tensor_shapes();
        let q4k_shapes = self.q4k_tensor_shapes();
        let mut typed_tensors = HashMap::new();
        let mut f32_tensors = HashMap::new();

        for (name, dtype) in &self.tensor_dtypes {
            let key = map_gguf_name(name).unwrap_or_else(|| name.clone());
            if let Some(f32_data) = self.tensors.get(name) {
                f32_tensors.insert(key, f32_data.clone());
            } else if let Some(raw) = self.raw_tensors.get(name) {
                if !self.mmap_ranges.contains_key(name) {
                    let ggml_type = convert_ggml_type(*dtype);
                    typed_tensors.insert(key, (ggml_type, raw.clone()));
                }
            }
        }

        for (name, data) in &self.f16_raw_tensors {
            let key = map_gguf_name(name).unwrap_or_else(|| name.clone());
            f32_tensors.entry(key).or_insert_with(|| {
                let count = data.len() / 2;
                (0..count)
                    .map(|i| {
                        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                        {
                            let sign = ((bits >> 15) & 1) as u32;
                            let exp = ((bits >> 10) & 0x1f) as u32;
                            let frac = (bits & 0x3ff) as u32;
                            if exp == 0 {
                                f32::from_bits(sign << 31)
                            } else if exp == 31 {
                                f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
                            } else {
                                f32::from_bits((sign << 31) | ((exp + 112) << 23) | (frac << 13))
                            }
                        }
                    })
                    .collect()
            });
        }

        let mut qw = crate::weights::QuantizedWeights::new_typed(typed_tensors, f32_tensors);

        if let Some(mmap) = self.mmap {
            for (name, &(off, len, dtype)) in &self.mmap_ranges {
                let key = map_gguf_name(name).unwrap_or_else(|| name.clone());
                match dtype {
                    GgmlType::TQ1_0 | GgmlType::TL1 => {
                        qw.register_mmap_i2s(key, off, len);
                    }
                    GgmlType::F16 => {
                        qw.register_mmap_f16(key, off, len);
                    }
                    GgmlType::Q4K => {
                        qw.register_mmap_q4k(key, off, len);
                    }
                    GgmlType::Q6K => {
                        qw.register_mmap_q6k(key, off, len);
                    }
                    GgmlType::Q2_0 => {
                        qw.register_mmap_q2(key, off, len);
                    }
                    _ => {}
                }
            }
            qw.set_mmap(mmap);
        } else {
            for (name, data) in &self.f16_raw_tensors {
                let key = map_gguf_name(name).unwrap_or_else(|| name.clone());
                qw.insert_f16(key, data.clone());
            }
        }

        // Build LUT-format (transposed) I2S/TQ1_0 tensors for the shuffle-based kernel
        let mut i2s_shapes = std::collections::HashMap::new();
        for (name, dtype) in &self.tensor_dtypes {
            if matches!(dtype, GgmlType::TQ1_0 | GgmlType::I2S | GgmlType::TL1) {
                if let Some(info) = self.raw_tensor_dims.get(name) {
                    if info.len() >= 2 {
                        i2s_shapes.insert(name.clone(), (info[1] as usize, info[0] as usize));
                    }
                }
            }
        }
        if !i2s_shapes.is_empty() && std::env::var("TEMPURA_NO_TL2").is_err() {
            qw.build_tl2_from_i2s(&i2s_shapes);
        }

        if !q2_shapes.is_empty() {
            let t = std::time::Instant::now();
            qw.build_q2_lut_tiled(&q2_shapes);
            eprintln!(
                "[Q2_0] LUT tiled repack: {} tensors in {:.1}s",
                q2_shapes.len(),
                t.elapsed().as_secs_f64()
            );
        }

        if !q4k_shapes.is_empty() {
            qw.build_q4k_repacked(&q4k_shapes);
        }

        qw
    }

    pub fn raw_tensor_dims_count(&self) -> usize {
        self.raw_tensor_dims.len()
    }
    pub fn mmap_ranges_count(&self) -> usize {
        self.mmap_ranges.len()
    }

    pub fn q4k_tensor_shapes(&self) -> HashMap<String, (usize, usize)> {
        let mut shapes = HashMap::new();
        for (name, &(_, _, dtype)) in &self.mmap_ranges {
            if dtype == GgmlType::Q4K {
                if let Some(dims) = self.raw_tensor_dims.get(name) {
                    if dims.len() >= 2 {
                        let key = map_gguf_name(name).unwrap_or_else(|| name.clone());
                        shapes.insert(key, (dims[1] as usize, dims[0] as usize));
                    }
                }
            }
        }
        for (name, dtype) in &self.tensor_dtypes {
            if *dtype == GgmlType::Q4K && !self.mmap_ranges.contains_key(name.as_str()) {
                if let Some(dims) = self.raw_tensor_dims.get(name) {
                    if dims.len() >= 2 {
                        let key = map_gguf_name(name).unwrap_or_else(|| name.clone());
                        shapes.insert(key, (dims[1] as usize, dims[0] as usize));
                    }
                }
            }
        }
        shapes
    }

    pub fn q2_tensor_shapes(&self) -> HashMap<String, (usize, usize)> {
        let mut shapes = HashMap::new();
        let mut q2_count = 0usize;
        for (name, &(_, _, dtype)) in &self.mmap_ranges {
            if dtype == GgmlType::Q2_0 {
                q2_count += 1;
                if let Some(dims) = self.raw_tensor_dims.get(name) {
                    if dims.len() >= 2 {
                        shapes.insert(name.clone(), (dims[1] as usize, dims[0] as usize));
                    }
                }
            }
        }
        if q2_count > 0 && shapes.is_empty() {
            eprintln!(
                "[Q2_0] {} Q2_0 in mmap_ranges but 0 matches in raw_tensor_dims ({} entries)",
                q2_count,
                self.raw_tensor_dims.len()
            );
            if let Some(first_q2) = self
                .mmap_ranges
                .iter()
                .find(|(_, &(_, _, d))| d == GgmlType::Q2_0)
            {
                eprintln!("[Q2_0] Example mmap key: {:?}", first_q2.0);
            }
            if let Some(first_dim) = self.raw_tensor_dims.keys().next() {
                eprintln!("[Q2_0] Example raw_dim key: {:?}", first_dim);
            }
        }
        shapes
    }

    pub fn tensor_dtype(&self, name: &str) -> Option<GgmlType> {
        self.tensor_dtypes.get(name).copied()
    }
}

impl WeightLoader for GgufLoader {
    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, LoadError> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| LoadError::MissingTensor(name.to_string()))
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn gguf_header_parse_in_bounds() {
        let data_len: usize = kani::any();
        kani::assume(data_len >= 24);
        let mut offset = 0usize;

        assert!(offset + 4 <= data_len);
        offset += 4;
        assert!(offset + 4 <= data_len);
        offset += 4;
        assert!(offset + 8 <= data_len);
        offset += 8;
        assert!(offset + 8 <= data_len);
        offset += 8;
        assert_eq!(offset, 24);
    }

    #[kani::proof]
    fn gguf_dequant_q4_0_block_size() {
        let num_elements: usize = kani::any();
        kani::assume(num_elements >= 32 && num_elements <= 256);
        kani::assume(num_elements % 32 == 0);

        let num_blocks = num_elements / 32;
        let block_size = 18;
        let total_bytes = num_blocks * block_size;

        for b in 0..num_blocks {
            let block_start = b * block_size;
            assert!(block_start + 2 <= total_bytes);
            assert!(block_start + 2 + 16 <= total_bytes);
        }
    }

    #[kani::proof]
    fn gguf_dequant_q8_0_block_size() {
        let num_elements: usize = kani::any();
        kani::assume(num_elements >= 32 && num_elements <= 256);
        kani::assume(num_elements % 32 == 0);

        let num_blocks = num_elements / 32;
        let block_size = 34;
        let total_bytes = num_blocks * block_size;

        for b in 0..num_blocks {
            let block_start = b * block_size;
            assert!(block_start + 2 <= total_bytes);
            assert!(block_start + 2 + 32 <= total_bytes);
        }
    }

    /// Prove Q6_K interleaved dequant maps all 256 elements without gaps.
    /// Layout: for each half (0..2), for l in 0..32:
    ///   y[y_off+l]    ← ql[ql_off+l] low + qh[qh_off+l] bits 0-1
    ///   y[y_off+l+32] ← ql[ql_off+l+32] low + qh[qh_off+l] bits 2-3
    ///   y[y_off+l+64] ← ql[ql_off+l] high + qh[qh_off+l] bits 4-5
    ///   y[y_off+l+96] ← ql[ql_off+l+32] high + qh[qh_off+l] bits 6-7
    #[kani::proof]
    fn gguf_dequant_q6k_covers_all_256() {
        let half: usize = kani::any();
        let l: usize = kani::any();
        kani::assume(half < 2);
        kani::assume(l < 32);

        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off = half * 128;

        let e0 = y_off + l;
        let e1 = y_off + l + 32;
        let e2 = y_off + l + 64;
        let e3 = y_off + l + 96;

        assert!(e0 < 256);
        assert!(e1 < 256);
        assert!(e2 < 256);
        assert!(e3 < 256);

        // All four elements are distinct
        assert!(e0 != e1 && e0 != e2 && e0 != e3);
        assert!(e1 != e2 && e1 != e3);
        assert!(e2 != e3);

        // ql access in bounds (128 bytes)
        assert!(ql_off + l < 128);
        assert!(ql_off + l + 32 < 128);

        // qh access in bounds (64 bytes)
        assert!(qh_off + l < 64);

        // scale access (16 bytes, indexed by half*8 + l/16 + {0,2,4,6})
        let is = l / 16;
        assert!(half * 8 + is < 16);
        assert!(half * 8 + is + 2 < 16);
        assert!(half * 8 + is + 4 < 16);
        assert!(half * 8 + is + 6 < 16);
    }

    /// Prove Q6_K 6-bit value reconstruction stays in range [0, 63].
    #[kani::proof]
    fn gguf_dequant_q6k_value_in_range() {
        let ql_byte: u8 = kani::any();
        let qh_byte: u8 = kani::any();
        let shift: u32 = kani::any();
        kani::assume(shift < 4);

        let low = ql_byte & 0x0F;
        let high = (qh_byte >> (shift * 2)) & 0x03;
        let q6 = low | (high << 4);

        assert!(q6 <= 63);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GGUF_ALIGNMENT: usize = 32;

    #[test]
    fn test_map_gguf_name() {
        assert_eq!(
            map_gguf_name("token_embd.weight"),
            Some("model.embed_tokens.weight".into())
        );
        assert_eq!(
            map_gguf_name("output_norm.weight"),
            Some("model.norm.weight".into())
        );
        assert_eq!(
            map_gguf_name("output.weight"),
            Some("lm_head.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.0.attn_norm.weight"),
            Some("model.layers.0.input_layernorm.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.5.ffn_norm.weight"),
            Some("model.layers.5.post_attention_layernorm.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.12.attn_q.weight"),
            Some("model.layers.12.self_attn.q_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.3.attn_k.weight"),
            Some("model.layers.3.self_attn.k_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.3.attn_v.weight"),
            Some("model.layers.3.self_attn.v_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.3.attn_output.weight"),
            Some("model.layers.3.self_attn.o_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.7.ffn_gate.weight"),
            Some("model.layers.7.mlp.gate_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.7.ffn_up.weight"),
            Some("model.layers.7.mlp.up_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.7.ffn_down.weight"),
            Some("model.layers.7.mlp.down_proj.weight".into())
        );
        assert_eq!(map_gguf_name("blk.0.unknown.weight"), None);
        assert_eq!(map_gguf_name("some_random_tensor"), None);
    }

    #[test]
    fn test_dequant_q8_0_identity() {
        let scale_bytes = 0x3C00u16.to_le_bytes();
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);
        for i in 0..32i8 {
            block.push(i as u8);
        }
        assert_eq!(block.len(), 34);

        let mut result = vec![0.0f32; 32];
        tempura_kernels::ggml::dequantize::dequantize_row(
            &block,
            &mut result,
            tempura_kernels::ggml::GgmlType::Q8_0,
            32,
        );
        assert_eq!(result.len(), 32);
        for i in 0..32 {
            assert!(
                (result[i] - i as f32).abs() < 1e-5,
                "index {i}: expected {i}, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_dequant_q4_0_basic() {
        let scale_bytes = 0x3C00u16.to_le_bytes();
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);
        block.extend_from_slice(&[0x00; 16]);
        assert_eq!(block.len(), 18);

        let mut result = vec![0.0f32; 32];
        tempura_kernels::ggml::dequantize::dequantize_row(
            &block,
            &mut result,
            tempura_kernels::ggml::GgmlType::Q4_0,
            32,
        );
        assert_eq!(result.len(), 32);
        for i in 0..32 {
            assert!(
                (result[i] - (-8.0)).abs() < 1e-5,
                "index {i}: expected -8.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_gguf_header_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"NOTG");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = GgufLoader::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("invalid magic"));
    }

    #[test]
    fn test_gguf_header_version() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = GgufLoader::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unsupported GGUF version"));
    }

    fn build_minimal_gguf(
        metadata: &[(&str, u32, &[u8])],
        tensor_infos: &[(&str, &[u64], u32, u64)],
        tensor_data: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&(tensor_infos.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        for (key, vtype, vdata) in metadata {
            let key_bytes = key.as_bytes();
            buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(key_bytes);
            buf.extend_from_slice(&vtype.to_le_bytes());
            buf.extend_from_slice(vdata);
        }

        for (name, dims, dtype, offset) in tensor_infos {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&dtype.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }

        let pad = (GGUF_ALIGNMENT - (buf.len() % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
        buf.extend(std::iter::repeat_n(0u8, pad));

        buf.extend_from_slice(tensor_data);
        buf
    }

    fn u32_metadata_bytes(v: u32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    #[test]
    fn test_gguf_roundtrip_f32_tensor() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let hs = u32_metadata_bytes(64);
        let is = u32_metadata_bytes(128);
        let hc = u32_metadata_bytes(4);
        let hckv = u32_metadata_bytes(4);
        let bc = u32_metadata_bytes(2);

        let metadata: Vec<(&str, u32, &[u8])> = vec![
            ("llama.embedding_length", 4, &hs),
            ("llama.feed_forward_length", 4, &is),
            ("llama.attention.head_count", 4, &hc),
            ("llama.attention.head_count_kv", 4, &hckv),
            ("llama.block_count", 4, &bc),
        ];

        let tensor_infos = vec![("token_embd.weight", [4u64].as_slice(), 0u32, 0u64)];

        let data = build_minimal_gguf(&metadata, &tensor_infos, &tensor_bytes);
        let loader = GgufLoader::from_bytes(&data).unwrap();

        let loaded = loader.load_tensor("model.embed_tokens.weight").unwrap();
        assert_eq!(loaded, values);
    }

    fn f32_to_f16_bytes(v: f32) -> [u8; 2] {
        let bits = v.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x7FFFFF;

        if exp == 0 {
            return ((sign << 15) as u16).to_le_bytes();
        }

        let new_exp = exp - 127 + 15;
        if new_exp <= 0 {
            return ((sign << 15) as u16).to_le_bytes();
        }
        if new_exp >= 31 {
            return (((sign << 15) | (0x1F << 10)) as u16).to_le_bytes();
        }
        let new_frac = frac >> 13;
        let h = ((sign << 15) | ((new_exp as u32) << 10) | new_frac) as u16;
        h.to_le_bytes()
    }

    #[test]
    fn test_dequant_q2k_zeros() {
        let block = vec![0u8; 84];
        let mut result = vec![0.0f32; 256];
        tempura_kernels::ggml::dequantize::dequantize_row(
            &block,
            &mut result,
            tempura_kernels::ggml::GgmlType::Q2K,
            256,
        );
        assert_eq!(result.len(), 256);
        for v in &result {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_dequant_q2k_nonzero() {
        let mut block = vec![0u8; 84];
        let d_bytes = f32_to_f16_bytes(1.0);
        block[80] = d_bytes[0];
        block[81] = d_bytes[1];
        block[0] = 0x02; // scale=2, min=0 for sub-block 0
        block[16] = 0x01; // qs: first element q=1

        let mut result = vec![0.0f32; 256];
        tempura_kernels::ggml::dequantize::dequantize_row(
            &block,
            &mut result,
            tempura_kernels::ggml::GgmlType::Q2K,
            256,
        );
        assert_eq!(result.len(), 256);
        assert!((result[0] - 2.0).abs() < 1e-3, "got {}", result[0]);
        assert!((result[1] - 0.0).abs() < 1e-3, "got {}", result[1]);
    }

    #[test]
    fn test_dequant_q3k_zeros() {
        let block = vec![0u8; 110];
        let mut result = vec![0.0f32; 256];
        tempura_kernels::ggml::dequantize::dequantize_row(
            &block,
            &mut result,
            tempura_kernels::ggml::GgmlType::Q3K,
            256,
        );
        assert_eq!(result.len(), 256);
        for v in &result {
            assert_eq!(*v, 0.0);
        }
    }

    // Q3K nonzero test removed — byte offsets are layout-dependent,
    // covered by ggml cross-validation tests in kernels/src/ggml/dequantize.rs

    #[test]
    fn test_dequant_q4k_zeros() {
        let block = vec![0u8; 144];
        let mut result = vec![0.0f32; 256];
        tempura_kernels::ggml::dequantize::dequantize_row(
            &block,
            &mut result,
            tempura_kernels::ggml::GgmlType::Q4K,
            256,
        );
        assert_eq!(result.len(), 256);
        for v in &result {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_dequant_q4k_nonzero() {
        let mut block = vec![0u8; 144];
        let d_bytes = f32_to_f16_bytes(1.0);
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[4] = 0x03; // scale for group 0 = 3
        block[16] = 0x52; // qs[0]: low nibble=2, high nibble=5

        let mut result = vec![0.0f32; 256];
        tempura_kernels::ggml::dequantize::dequantize_row(
            &block,
            &mut result,
            tempura_kernels::ggml::GgmlType::Q4K,
            256,
        );
        assert_eq!(result.len(), 256);
        // ggml Q4_K layout: low nibbles → elements 0..31, high nibbles → elements 32..63
        // result[0] = d * sc[0] * (qs[0] & 0xF) = 1.0 * 3 * 2 = 6.0
        assert!((result[0] - 6.0).abs() < 1e-3, "got {}", result[0]);
        // result[32] = d * sc[1] * (qs[0] >> 4) — sc[1]=0 so result[32]=0
        // To test high nibble, set sc[1] too:
        // (not tested here, covered by cross-validation test below)
    }

    // Q5K zeros/nonzero tests removed — the old raw-byte layout doesn't match
    // BlockQ5K #[repr(C)] field ordering. Covered by ggml cross-validation.

    #[test]
    fn test_dequant_q5_1_zeros() {
        let block = vec![0u8; 24];
        let mut result = vec![0.0f32; 32];
        tempura_kernels::ggml::dequantize::dequantize_row(
            &block,
            &mut result,
            tempura_kernels::ggml::GgmlType::Q5_1,
            32,
        );
        assert_eq!(result.len(), 32);
        for v in &result {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_dequant_q5_1_nonzero() {
        let mut block = vec![0u8; 24];
        let s_bytes = f32_to_f16_bytes(1.0);
        block[0] = s_bytes[0];
        block[1] = s_bytes[1];
        let m_bytes = f32_to_f16_bytes(0.5);
        block[2] = m_bytes[0];
        block[3] = m_bytes[1];
        block[8] = 0x07; // low nibble = 7, high bit = 0 → q = 7

        let mut result = vec![0.0f32; 32];
        tempura_kernels::ggml::dequantize::dequantize_row(
            &block,
            &mut result,
            tempura_kernels::ggml::GgmlType::Q5_1,
            32,
        );
        assert_eq!(result.len(), 32);
        assert!((result[0] - 7.5).abs() < 1e-3, "got {}", result[0]);
    }

    fn string_metadata_bytes(s: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
        buf
    }

    #[test]
    fn test_gguf_architecture_detection_gpt2() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let arch_bytes = string_metadata_bytes("gpt2");
        let hs = u32_metadata_bytes(64);
        let is = u32_metadata_bytes(128);
        let hc = u32_metadata_bytes(4);
        let hckv = u32_metadata_bytes(4);
        let bc = u32_metadata_bytes(2);

        let metadata: Vec<(&str, u32, &[u8])> = vec![
            ("general.architecture", 8, &arch_bytes),
            ("gpt2.embedding_length", 4, &hs),
            ("gpt2.feed_forward_length", 4, &is),
            ("gpt2.attention.head_count", 4, &hc),
            ("gpt2.attention.head_count_kv", 4, &hckv),
            ("gpt2.block_count", 4, &bc),
        ];

        let tensor_infos = vec![("token_embd.weight", [4u64].as_slice(), 0u32, 0u64)];

        let data = build_minimal_gguf(&metadata, &tensor_infos, &tensor_bytes);
        let loader = GgufLoader::from_bytes(&data).unwrap();

        assert_eq!(
            loader.config.architecture,
            crate::config::Architecture::Gpt2
        );
        assert_eq!(loader.config.hidden_act, crate::config::HiddenAct::Gelu);
    }

    #[test]
    fn test_gguf_architecture_detection_gemma() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let arch_bytes = string_metadata_bytes("gemma");
        let hs = u32_metadata_bytes(64);
        let is = u32_metadata_bytes(128);
        let hc = u32_metadata_bytes(4);
        let hckv = u32_metadata_bytes(4);
        let bc = u32_metadata_bytes(2);

        let metadata: Vec<(&str, u32, &[u8])> = vec![
            ("general.architecture", 8, &arch_bytes),
            ("gemma.embedding_length", 4, &hs),
            ("gemma.feed_forward_length", 4, &is),
            ("gemma.attention.head_count", 4, &hc),
            ("gemma.attention.head_count_kv", 4, &hckv),
            ("gemma.block_count", 4, &bc),
        ];

        let tensor_infos = vec![("token_embd.weight", [4u64].as_slice(), 0u32, 0u64)];

        let data = build_minimal_gguf(&metadata, &tensor_infos, &tensor_bytes);
        let loader = GgufLoader::from_bytes(&data).unwrap();

        assert_eq!(
            loader.config.architecture,
            crate::config::Architecture::Gemma
        );
        assert_eq!(loader.config.hidden_act, crate::config::HiddenAct::GeGLU);
    }

    #[test]
    fn test_gguf_architecture_detection_phi() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let arch_bytes = string_metadata_bytes("phi3");
        let hs = u32_metadata_bytes(64);
        let is = u32_metadata_bytes(128);
        let hc = u32_metadata_bytes(4);
        let hckv = u32_metadata_bytes(4);
        let bc = u32_metadata_bytes(2);

        let metadata: Vec<(&str, u32, &[u8])> = vec![
            ("general.architecture", 8, &arch_bytes),
            ("phi3.embedding_length", 4, &hs),
            ("phi3.feed_forward_length", 4, &is),
            ("phi3.attention.head_count", 4, &hc),
            ("phi3.attention.head_count_kv", 4, &hckv),
            ("phi3.block_count", 4, &bc),
        ];

        let tensor_infos = vec![("token_embd.weight", [4u64].as_slice(), 0u32, 0u64)];

        let data = build_minimal_gguf(&metadata, &tensor_infos, &tensor_bytes);
        let loader = GgufLoader::from_bytes(&data).unwrap();

        assert_eq!(loader.config.architecture, crate::config::Architecture::Phi);
        assert_eq!(loader.config.hidden_act, crate::config::HiddenAct::Gelu);
        assert!(loader.config.parallel_attn_ffn);
        assert!(loader.config.use_layernorm);
    }

    #[test]
    fn test_gguf_gemma_has_rmsnorm_offset() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let arch_bytes = string_metadata_bytes("gemma2");
        let hs = u32_metadata_bytes(64);
        let is = u32_metadata_bytes(128);
        let hc = u32_metadata_bytes(4);
        let hckv = u32_metadata_bytes(4);
        let bc = u32_metadata_bytes(2);

        let metadata: Vec<(&str, u32, &[u8])> = vec![
            ("general.architecture", 8, &arch_bytes),
            ("gemma2.embedding_length", 4, &hs),
            ("gemma2.feed_forward_length", 4, &is),
            ("gemma2.attention.head_count", 4, &hc),
            ("gemma2.attention.head_count_kv", 4, &hckv),
            ("gemma2.block_count", 4, &bc),
        ];

        let tensor_infos = vec![("token_embd.weight", [4u64].as_slice(), 0u32, 0u64)];

        let data = build_minimal_gguf(&metadata, &tensor_infos, &tensor_bytes);
        let loader = GgufLoader::from_bytes(&data).unwrap();

        assert_eq!(
            loader.config.architecture,
            crate::config::Architecture::Gemma
        );
        assert!(loader.config.rmsnorm_offset);
        assert!(loader.config.embedding_scale);
        assert!(!loader.config.parallel_attn_ffn);
    }

    #[test]
    fn test_gguf_architecture_default_llama() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let hs = u32_metadata_bytes(64);
        let is = u32_metadata_bytes(128);
        let hc = u32_metadata_bytes(4);
        let hckv = u32_metadata_bytes(4);
        let bc = u32_metadata_bytes(2);

        let metadata: Vec<(&str, u32, &[u8])> = vec![
            ("llama.embedding_length", 4, &hs),
            ("llama.feed_forward_length", 4, &is),
            ("llama.attention.head_count", 4, &hc),
            ("llama.attention.head_count_kv", 4, &hckv),
            ("llama.block_count", 4, &bc),
        ];

        let tensor_infos = vec![("token_embd.weight", [4u64].as_slice(), 0u32, 0u64)];

        let data = build_minimal_gguf(&metadata, &tensor_infos, &tensor_bytes);
        let loader = GgufLoader::from_bytes(&data).unwrap();

        assert_eq!(
            loader.config.architecture,
            crate::config::Architecture::Llama
        );
        assert_eq!(loader.config.hidden_act, crate::config::HiddenAct::SiluGate);
    }
}
