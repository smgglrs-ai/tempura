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

    // Generate f16->f32 lookup table (65536 entries x 4 bytes = 256KB).
    // Matches llama.cpp's ggml_table_f32_f16 -- allows single indexed load [table+idx*4].
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

    // GELU lookup table: gelu(f32_from_f16(i)) stored as f16 bits (65536 x 2 = 128KB).
    // Matches llama.cpp's ggml_table_gelu_f16 -- f32->f16 index + one load replaces tanh/exp.
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

    // E8M0 half lookup table: ggml_table_f32_e8m0_half (256 entries x 4 bytes = 1KB).
    // Matches llama.cpp's ggml_table_f32_e8m0_half = GGML_E8M0_TO_FP32_HALF(i).
    // Value = 2^(i-128) = ggml_e8m0_to_fp32(i) / 2.
    let e8m0_half_path = std::path::Path::new(&out_dir).join("e8m0_half_table.bin");
    if !e8m0_half_path.exists() {
        let mut table = Vec::with_capacity(256 * 4);
        for i in 0u8..=255 {
            let val = e8m0_to_f32_half_build(i);
            table.extend_from_slice(&val.to_le_bytes());
        }
        std::fs::write(&e8m0_half_path, &table).unwrap();
    }

    // UE4M3 lookup table: ggml_table_f32_ue4m3 (256 entries x 4 bytes = 1KB).
    // Matches llama.cpp's ggml_ue4m3_to_fp32: unsigned 4-exp 3-mantissa, bias=7,
    // returns raw * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 x E2M1_float).
    let ue4m3_path = std::path::Path::new(&out_dir).join("ue4m3_table.bin");
    if !ue4m3_path.exists() {
        let mut table = Vec::with_capacity(256 * 4);
        for i in 0u8..=255 {
            let val = ue4m3_to_f32_build(i);
            table.extend_from_slice(&val.to_le_bytes());
        }
        std::fs::write(&ue4m3_path, &table).unwrap();
    }

    println!("cargo:rerun-if-changed=build.rs");

    // -- oxide-decode-attn: copy pre-built PTX to OUT_DIR ----------------------
    // The PTX is hand-authored and lives in tools/oxide-decode-attn/.
    // It contains oxide_decode_attn, oxide_decode_attn_kpar, and oxide_decode_attn_batch.
    {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tools/oxide-decode-attn/oxide_decode_attn.ptx");
        let dst = std::path::Path::new(&out_dir).join("oxide_decode_attn.ptx");
        std::fs::copy(&src, &dst)
            .unwrap_or_else(|e| panic!("build.rs: failed to copy oxide_decode_attn.ptx: {e}"));
        println!("cargo:rerun-if-changed=../tools/oxide-decode-attn/oxide_decode_attn.ptx");
    }

    // -- oxide-gemv: compile Q2_0 GEMV kernel with rustc-codegen-cuda ---------
    // When the `oxide-gemv` feature is requested and `cargo-oxide` is in PATH,
    // we build `tools/oxide-gemv` with `cargo oxide build`, then extract the
    // embedded PTX from the compiled binary and write it to OUT_DIR.
    //
    // Failure is non-fatal: the `OxideQ2Gemv::try_load` constructor returns
    // `None` when the PTX is absent, and the cuTile kernel is used as fallback.
    if std::env::var("CARGO_FEATURE_OXIDE_GEMV").is_ok() {
        build_oxide_q2_gemv(&out_dir);
        build_oxide_q4k_gemv(&out_dir);
        build_oxide_q6k_gemv(&out_dir);
    }
}

fn build_oxide_q2_gemv(out_dir: &str) {
    use std::process::Command;
    use std::path::Path;

    // Path to the oxide-gemv standalone crate (relative to workspace root).
    let oxide_crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()  // kernels/ -> workspace root
        .unwrap()
        .join("tools/oxide-gemv");

    // PTX destination.
    let ptx_path = Path::new(out_dir).join("oxide_q2_gemv.ptx");

    // Trigger rebuild when the oxide kernel source changes.
    println!(
        "cargo:rerun-if-changed={}",
        oxide_crate_dir.join("src/main.rs").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        oxide_crate_dir.join("Cargo.toml").display()
    );

    // Check if the oxide binary already exists from a previous `cargo oxide build`.
    // If it does, just extract the PTX. If not, run `cargo oxide build` first.
    let oxide_binary = oxide_crate_dir.join("target/release/oxide-gemv");

    if !oxide_binary.exists() {
        // Run `cargo oxide build` to compile the kernel.
        eprintln!("build.rs: running `cargo oxide build` for oxide_q2_gemv kernel...");
        let status = Command::new("cargo")
            .arg("oxide")
            .arg("build")
            .arg("--manifest-path")
            .arg(oxide_crate_dir.join("Cargo.toml"))
            .status();

        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!(
                    "build.rs: `cargo oxide build` failed (exit {:?}); \
                     oxide Q2_0 GEMV kernel disabled",
                    s.code()
                );
                return;
            }
            Err(e) => {
                eprintln!(
                    "build.rs: failed to invoke `cargo oxide build` ({e}); \
                     oxide Q2_0 GEMV kernel disabled"
                );
                return;
            }
        }
    }

    if !oxide_binary.exists() {
        eprintln!("build.rs: oxide binary not found after build; oxide kernel disabled");
        return;
    }

    // Extract PTX from the compiled binary using Python.
    // The PTX is embedded in the `.oxart` ELF section as tag+content.
    let binary = &oxide_binary;

    let extract_script = r#"
import sys, struct

with open(sys.argv[1], 'rb') as f:
    data = f.read()

tag = b'oxide_gemv.ptx\x00'
idx = data.find(tag)
if idx == -1:
    print('PTX tag not found in binary', file=sys.stderr)
    sys.exit(1)

ptx_start = idx + len(tag)
ptx_end = ptx_start
while ptx_end < len(data) and data[ptx_end] != 0:
    ptx_end += 1

sys.stdout.buffer.write(data[ptx_start:ptx_end])
"#;

    let output = Command::new("python3")
        .arg("-c")
        .arg(extract_script)
        .arg(binary.to_str().unwrap())
        .output();

    match output {
        Ok(out) if out.status.success() && !out.stdout.is_empty() => {
            std::fs::write(&ptx_path, &out.stdout).unwrap();
            eprintln!(
                "build.rs: oxide PTX extracted -> {} ({} bytes)",
                ptx_path.display(),
                out.stdout.len()
            );
        }
        Ok(out) => {
            eprintln!(
                "build.rs: PTX extraction failed; stderr: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Err(e) => {
            eprintln!("build.rs: python3 not available ({e}); oxide kernel disabled");
        }
    }
}

fn build_oxide_q4k_gemv(out_dir: &str) {
    use std::process::Command;
    use std::path::Path;

    let oxide_crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tools/oxide-gemv-q4k");

    let ptx_path = Path::new(out_dir).join("oxide_q4k_gemv.ptx");

    println!(
        "cargo:rerun-if-changed={}",
        oxide_crate_dir.join("src/main.rs").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        oxide_crate_dir.join("Cargo.toml").display()
    );

    let oxide_binary = oxide_crate_dir.join("target/release/oxide-gemv-q4k");

    if !oxide_binary.exists() {
        eprintln!("build.rs: running `cargo oxide build` for oxide_q4k_gemv kernel...");
        let status = Command::new("cargo")
            .arg("oxide")
            .arg("build")
            .arg("--manifest-path")
            .arg(oxide_crate_dir.join("Cargo.toml"))
            .status();

        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!(
                    "build.rs: `cargo oxide build` for Q4K failed (exit {:?}); \
                     oxide Q4_K GEMV kernel disabled",
                    s.code()
                );
                return;
            }
            Err(e) => {
                eprintln!(
                    "build.rs: failed to invoke `cargo oxide build` for Q4K ({e}); \
                     oxide Q4_K GEMV kernel disabled"
                );
                return;
            }
        }
    }

    if !oxide_binary.exists() {
        eprintln!("build.rs: oxide-gemv-q4k binary not found after build; Q4K kernel disabled");
        return;
    }

    let extract_script = r#"
import sys

with open(sys.argv[1], 'rb') as f:
    data = f.read()

tag = b'oxide_gemv_q4k.ptx\x00'
idx = data.find(tag)
if idx == -1:
    print('PTX tag not found in binary', file=sys.stderr)
    sys.exit(1)

ptx_start = idx + len(tag)
ptx_end = ptx_start
while ptx_end < len(data) and data[ptx_end] != 0:
    ptx_end += 1

sys.stdout.buffer.write(data[ptx_start:ptx_end])
"#;

    let output = Command::new("python3")
        .arg("-c")
        .arg(extract_script)
        .arg(oxide_binary.to_str().unwrap())
        .output();

    match output {
        Ok(out) if out.status.success() && !out.stdout.is_empty() => {
            std::fs::write(&ptx_path, &out.stdout).unwrap();
            eprintln!(
                "build.rs: oxide Q4K PTX extracted -> {} ({} bytes)",
                ptx_path.display(),
                out.stdout.len()
            );
        }
        Ok(out) => {
            eprintln!(
                "build.rs: Q4K PTX extraction failed; stderr: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Err(e) => {
            eprintln!("build.rs: python3 not available ({e}); Q4K oxide kernel disabled");
        }
    }
}

fn build_oxide_q6k_gemv(out_dir: &str) {
    use std::process::Command;
    use std::path::Path;

    let oxide_crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tools/oxide-gemv-q6k");

    let ptx_path = Path::new(out_dir).join("oxide_q6k_gemv.ptx");

    println!(
        "cargo:rerun-if-changed={}",
        oxide_crate_dir.join("src/main.rs").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        oxide_crate_dir.join("Cargo.toml").display()
    );

    let oxide_binary = oxide_crate_dir.join("target/release/oxide-gemv-q6k");

    if !oxide_binary.exists() {
        eprintln!("build.rs: running `cargo oxide build` for oxide_q6k_gemv kernel...");
        let status = Command::new("cargo")
            .arg("oxide")
            .arg("build")
            .arg("--manifest-path")
            .arg(oxide_crate_dir.join("Cargo.toml"))
            .status();

        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!(
                    "build.rs: `cargo oxide build` for Q6K failed (exit {:?}); \
                     oxide Q6_K GEMV kernel disabled",
                    s.code()
                );
                return;
            }
            Err(e) => {
                eprintln!(
                    "build.rs: failed to invoke `cargo oxide build` for Q6K ({e}); \
                     oxide Q6_K GEMV kernel disabled"
                );
                return;
            }
        }
    }

    if !oxide_binary.exists() {
        eprintln!("build.rs: oxide-gemv-q6k binary not found after build; Q6K kernel disabled");
        return;
    }

    let extract_script = r#"
import sys

with open(sys.argv[1], 'rb') as f:
    data = f.read()

tag = b'oxide_gemv_q6k.ptx\x00'
idx = data.find(tag)
if idx == -1:
    print('PTX tag not found in binary', file=sys.stderr)
    sys.exit(1)

ptx_start = idx + len(tag)
ptx_end = ptx_start
while ptx_end < len(data) and data[ptx_end] != 0:
    ptx_end += 1

sys.stdout.buffer.write(data[ptx_start:ptx_end])
"#;

    let output = Command::new("python3")
        .arg("-c")
        .arg(extract_script)
        .arg(oxide_binary.to_str().unwrap())
        .output();

    match output {
        Ok(out) if out.status.success() && !out.stdout.is_empty() => {
            std::fs::write(&ptx_path, &out.stdout).unwrap();
            eprintln!(
                "build.rs: oxide Q6K PTX extracted -> {} ({} bytes)",
                ptx_path.display(),
                out.stdout.len()
            );
        }
        Ok(out) => {
            eprintln!(
                "build.rs: Q6K PTX extraction failed; stderr: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Err(e) => {
            eprintln!("build.rs: python3 not available ({e}); Q6K oxide kernel disabled");
        }
    }
}

fn e8m0_to_f32_half_build(x: u8) -> f32 {
    // Matches ggml_e8m0_to_fp32_half: 2^(x-128).
    // x < 2: denormal bit patterns; x >= 2: normalized with exponent (x-1).
    let bits: u32 = if x < 2 {
        0x00200000u32 << x
    } else {
        (x as u32 - 1) << 23
    };
    f32::from_bits(bits)
}

fn ue4m3_to_f32_build(x: u8) -> f32 {
    // Matches ggml_ue4m3_to_fp32: UE4M3 unsigned (4 exp bits, bias=7; 3 mantissa bits).
    // Returns raw * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 x E2M1_float).
    if x == 0 || x == 0x7F {
        return 0.0f32;
    }
    let exp = (x >> 3) & 0xF;
    let man = x & 0x7;
    let raw: f32 = if exp == 0 {
        // subnormal: man * 2^(-9)
        (man as f32) * (1.0f32 / 512.0f32)
    } else {
        // normalized: (1 + man/8) * 2^(exp-7)
        (1.0f32 + (man as f32) / 8.0f32) * (2.0f32.powi(exp as i32 - 7))
    };
    raw * 0.5f32
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
