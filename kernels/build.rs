fn main() {
    #[cfg(feature = "openmp")]
    {
        cc::Build::new()
            .file("src/openmp_gemv.c")
            .flag("-fopenmp")
            .flag("-O3")
            .flag("-march=native")
            .compile("openmp_gemv");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rerun-if-changed=src/openmp_gemv.c");
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Generate f16→f32 lookup table (65536 entries × 4 bytes = 256KB).
    // Matches llama.cpp's ggml_table_f32_f16 — allows single indexed load [table+idx*4].
    let table_path = std::path::Path::new(&out_dir).join("f16_table.bin");
    if !table_path.exists() {
        let mut table = Vec::with_capacity(65536 * 4);
        for i in 0u32..=65535 {
            let f16_bits = i as u16;
            let f32_val = f16_bits_to_f32(f16_bits);
            table.extend_from_slice(&f32_val.to_le_bytes());
        }
        std::fs::write(&table_path, &table).unwrap();
    }

    // GELU lookup table: gelu(f32_from_f16(i)) stored as f16 bits (65536 × 2 = 128KB).
    // Matches llama.cpp's ggml_table_gelu_f16 — f32→f16 index + one load replaces tanh/exp.
    let gelu_path = std::path::Path::new(&out_dir).join("gelu_table.bin");
    if !gelu_path.exists() {
        let mut table = Vec::with_capacity(65536 * 2);
        for i in 0u32..=65535 {
            let x = f16_bits_to_f32(i as u16);
            let gelu_val = gelu_f32_build(x);
            let gelu_f16 = f32_to_f16_bits(gelu_val);
            table.extend_from_slice(&gelu_f16.to_le_bytes());
        }
        std::fs::write(&gelu_path, &table).unwrap();
    }

    // GELU-quick lookup table: x * sigmoid(1.702*x) stored as f16 bits (128KB).
    let gelu_quick_path = std::path::Path::new(&out_dir).join("gelu_quick_table.bin");
    if !gelu_quick_path.exists() {
        let mut table = Vec::with_capacity(65536 * 2);
        for i in 0u32..=65535 {
            let x = f16_bits_to_f32(i as u16);
            let val = gelu_quick_f32_build(x);
            let f16 = f32_to_f16_bits(val);
            table.extend_from_slice(&f16.to_le_bytes());
        }
        std::fs::write(&gelu_quick_path, &table).unwrap();
    }

    println!("cargo:rerun-if-changed=build.rs");
}

fn gelu_f32_build(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const GELU_COEF_A: f32 = 0.044715;
    if x.is_nan() {
        return x;
    }
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

fn gelu_quick_f32_build(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    x * (1.0 / (1.0 + (-1.702_f32 * x).exp()))
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        return (sign << 15) | (0x1f << 10) | ((mant >> 13) as u16 & 0x3ff);
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign << 15) | (0x1f << 10);
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign << 15;
        }
        let m = (mant | 0x800000) >> (1 - new_exp + 13);
        return (sign << 15) | (m as u16 & 0x3ff);
    }
    let truncated = (mant >> 13) as u16 & 0x3ff;
    let remainder = mant & 0x1fff;
    let round_up = remainder > 0x1000 || (remainder == 0x1000 && (truncated & 1) != 0);
    let mut h = (sign << 15) | ((new_exp as u16) << 10) | truncated;
    if round_up {
        h += 1;
    }
    h
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 { return f32::from_bits(sign << 31); }
        let mut e = 0u32; let mut m = mant;
        while (m & 0x400) == 0 { m <<= 1; e += 1; }
        return f32::from_bits((sign << 31) | ((127 - 15 + 1 - e) << 23) | ((m & 0x3ff) << 13));
    }
    if exp == 31 { return f32::from_bits((sign << 31) | (0xff << 23) | (mant << 13)); }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
}
