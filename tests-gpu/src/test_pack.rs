use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, ptx_asm, thread};

#[cuda_module]
mod pack_kernels {
    use super::*;

    /// Minimal MMA test: A = all 1.0 (packed as f16), B = all 1.0
    /// Each thread contributes the same fragments. Result should be:
    /// D[groupID, tid*2]   = 16.0 (sum of 16 products of 1*1)
    /// D[groupID+8, tid*2] = 16.0 (same — both halves should be equal for uniform data)
    ///
    /// Then test with A top-half=1.0, bottom-half=2.0:
    /// D[groupID, ...]   = 16.0 (top rows use a[0],a[1] = 1.0)
    /// D[groupID+8, ...] = 32.0 (bottom rows use a[2],a[3] = 2.0)
    #[kernel]
    pub fn test_mma_raw(mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= 32 { return; }

        // f16(1.0) = 0x3C00, packed pair = 0x3C003C00
        let one_pair: u32 = 0x3C003C00;
        // f16(2.0) = 0x4000, packed pair = 0x40004000
        let two_pair: u32 = 0x40004000;

        // Test 1: uniform A=1, B=1 → expect all 16.0
        let mut d0: f32;
        let mut d1: f32;
        let mut d2: f32;
        let mut d3: f32;
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
                out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
                in("r") one_pair, in("r") one_pair, in("r") one_pair, in("r") one_pair,
                in("r") one_pair, in("r") one_pair,
                in("f") 0.0f32, in("f") 0.0f32, in("f") 0.0f32, in("f") 0.0f32,
            );
        }

        // Test 2: A top=1, bottom=2, B=1
        // Embed ALL fragment loads + MMA in a single asm block using .reg variables.
        // ptxas respects .reg declarations — no deduplication across declared registers.
        let mut e0: f32;
        let mut e1: f32;
        let mut e2: f32;
        let mut e3: f32;
        unsafe {
            ptx_asm!(
                "{{ \
                   .reg .b32 %%ra<4>, %%rb<2>; \
                   mov.b32 %%ra0, %4; \
                   mov.b32 %%ra1, %4; \
                   mov.b32 %%ra2, %5; \
                   mov.b32 %%ra3, %5; \
                   mov.b32 %%rb0, %4; \
                   mov.b32 %%rb1, %4; \
                   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                   {%0,%1,%2,%3}, {%%ra0,%%ra1,%%ra2,%%ra3}, {%%rb0,%%rb1}, {%6,%7,%8,%9}; \
                 }}",
                out("=f") e0, out("=f") e1, out("=f") e2, out("=f") e3,
                in("r") one_pair, in("r") two_pair,
                in("f") 0.0f32, in("f") 0.0f32, in("f") 0.0f32, in("f") 0.0f32,
            );
        }

        // Only lane 0 reports
        if i == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = d0;
                *out.get_unchecked_mut(1) = d1;
                *out.get_unchecked_mut(2) = d2;
                *out.get_unchecked_mut(3) = d3;
                *out.get_unchecked_mut(4) = e0;
                *out.get_unchecked_mut(5) = e1;
                *out.get_unchecked_mut(6) = e2;
                *out.get_unchecked_mut(7) = e3;
            }
        }
    }

    #[kernel]
    pub fn test_four_outputs(vals: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }
        let a = vals[0];
        let b = vals[1];
        let c = vals[2];
        let d = vals[3];
        let mut r0: f32;
        let mut r1: f32;
        let mut r2: f32;
        let mut r3: f32;
        unsafe {
            ptx_asm!(
                "add.f32 %0, %4, 1.0; add.f32 %1, %5, 2.0; add.f32 %2, %6, 3.0; add.f32 %3, %7, 4.0;",
                out("=f") r0, out("=f") r1, out("=f") r2, out("=f") r3,
                in("f") a, in("f") b, in("f") c, in("f") d,
            );
        }
        unsafe {
            *out.get_unchecked_mut(0) = r0;
            *out.get_unchecked_mut(1) = r1;
            *out.get_unchecked_mut(2) = r2;
            *out.get_unchecked_mut(3) = r3;
        }
    }

    #[kernel]
    pub fn test_pack_f16x2(a_in: &[f32], b_in: &[f32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= a_in.len() {
            return;
        }
        let a = a_in[i];
        let b = b_in[i];
        let packed: u32;
        unsafe {
            ptx_asm!(
                "cvt.rn.f16x2.f32 %0, %2, %1;",
                out("=r") packed,
                in("f") a,
                in("f") b,
            );
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = packed;
        }
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = -1i32;
        let mut f = frac;
        while (f & 0x400) == 0 {
            f <<= 1;
            e -= 1;
        }
        let f32_exp = (127 + e) as u32;
        let f32_frac = (f & 0x3ff) << 13;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_frac);
    }
    if exp == 31 {
        if frac == 0 {
            return f32::from_bits((sign << 31) | 0x7f800000);
        }
        return f32::NAN;
    }
    let f32_exp = (exp + 112) << 23;
    let f32_frac = (frac as u32) << 13;
    f32::from_bits((sign << 31) | f32_exp | f32_frac)
}

pub fn run_mma_raw_test() {
    println!("=== Raw MMA diagnostic (no SMEM, hardcoded fragments) ===");
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, 8).unwrap();

    let module = pack_kernels::load(&ctx).unwrap();
    module
        .test_mma_raw(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut out_dev,
        )
        .unwrap();

    let out = out_dev.to_host_vec(&stream).unwrap();

    println!("  Test 1 (A=1,B=1): d=[{:.1}, {:.1}, {:.1}, {:.1}]  expect [16, 16, 16, 16]  {}",
        out[0], out[1], out[2], out[3],
        if out[0..4] == [16.0, 16.0, 16.0, 16.0] { "OK" } else { "BUG" });

    println!("  Test 2 (A top=1 bot=2, B=1): d=[{:.1}, {:.1}, {:.1}, {:.1}]  expect [16, 16, 32, 32]  {}",
        out[4], out[5], out[6], out[7],
        if out[4..8] == [16.0, 16.0, 32.0, 32.0] { "OK" } else { "BUG" });
}

pub fn run_four_output_test() {
    println!("=== Four-output ptx_asm diagnostic ===");
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let in_vals = vec![10.0f32, 20.0, 30.0, 40.0];
    let in_dev = DeviceBuffer::from_host(&stream, &in_vals).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, 4).unwrap();

    let module = pack_kernels::load(&ctx).unwrap();
    module
        .test_four_outputs(
            &stream,
            LaunchConfig::for_num_elems(1),
            &in_dev,
            &mut out_dev,
        )
        .unwrap();

    let out = out_dev.to_host_vec(&stream).unwrap();
    println!("  Input:  {:?}", &in_vals);
    println!("  Output: {:?}", &out);
    println!(
        "  Expected: [11, 22, 33, 44] (each +1, +2, +3, +4)  {}",
        if out == [11.0, 22.0, 33.0, 44.0] { "OK" } else { "BUG!" }
    );
}

pub fn run_pack_test() {
    println!("=== Pack f16x2 diagnostic ===");
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let a_vals = vec![1.0f32, 2.0, 0.5, 0.1];
    let b_vals = vec![3.0f32, 4.0, 1.5, 0.2];

    let a_dev = DeviceBuffer::from_host(&stream, &a_vals).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_vals).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();

    let module = pack_kernels::load(&ctx).unwrap();
    module
        .test_pack_f16x2(
            &stream,
            LaunchConfig::for_num_elems(4),
            &a_dev,
            &b_dev,
            &mut out_dev,
        )
        .unwrap();

    let out = out_dev.to_host_vec(&stream).unwrap();

    for i in 0..4 {
        let lo_bits = (out[i] & 0xFFFF) as u16;
        let hi_bits = (out[i] >> 16) as u16;
        let lo_f32 = f16_bits_to_f32(lo_bits);
        let hi_f32 = f16_bits_to_f32(hi_bits);
        let status = if (lo_f32 - a_vals[i]).abs() < 0.01 && (hi_f32 - b_vals[i]).abs() < 0.01 {
            "OK: lo=a, hi=b"
        } else if (lo_f32 - b_vals[i]).abs() < 0.01 && (hi_f32 - a_vals[i]).abs() < 0.01 {
            "SWAPPED: lo=b, hi=a"
        } else {
            "UNEXPECTED"
        };
        println!(
            "  pack(a={:.1}, b={:.1}) → lo={:.4}, hi={:.4}  [{status}]",
            a_vals[i], b_vals[i], lo_f32, hi_f32
        );
    }
}
