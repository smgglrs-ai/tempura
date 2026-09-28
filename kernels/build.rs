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

    // Generate f16→f32 lookup table (65536 entries × 4 bytes = 256KB).
    // Matches llama.cpp's ggml_table_f32_f16 — allows single indexed load [table+idx*4].
    let out_dir = std::env::var("OUT_DIR").unwrap();
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
    println!("cargo:rerun-if-changed=build.rs");
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
