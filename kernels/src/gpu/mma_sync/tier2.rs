/// Tier 2: Non-block-scaled narrow precision (sm_120+)
///
/// `kind::f8f6f4` MMA instructions. 2× throughput vs bf16 for FP8,
/// 4× for FP4.
///
/// m16n8k32 for FP8/FP6, m16n8k64 for FP4.
/// Per-thread fragments: D[4×f32], A[4×u32], B[2×u32], C[4×f32].
use cuda_device::ptx_asm;

// --- FP8 (E4M3) variants ---

/// mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
///
/// A: 4 × u32, each holding 4 packed E4M3 values (k=32).
/// B: 2 × u32, each holding 4 packed E4M3 values.
#[inline(always)]
pub fn m16n8k32_f32_e4m3_e4m3(d: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2], c: &[f32; 4]) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
        );
    }
}

/// mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e4m3.f32
///
/// Mixed FP8: A uses E5M2 (wider range), B uses E4M3 (higher precision).
#[inline(always)]
pub fn m16n8k32_f32_e5m2_e4m3(d: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2], c: &[f32; 4]) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e4m3.f32 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
        );
    }
}

/// mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e5m2.f32
#[inline(always)]
pub fn m16n8k32_f32_e4m3_e5m2(d: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2], c: &[f32; 4]) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e5m2.f32 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
        );
    }
}

/// mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e5m2.f32
#[inline(always)]
pub fn m16n8k32_f32_e5m2_e5m2(d: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2], c: &[f32; 4]) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e5m2.f32 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
        );
    }
}

// --- FP6 variants ---

/// mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e3m2.f32
#[inline(always)]
pub fn m16n8k32_f32_e3m2_e3m2(d: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2], c: &[f32; 4]) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e3m2.f32 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
        );
    }
}

/// mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e2m3.f32
#[inline(always)]
pub fn m16n8k32_f32_e2m3_e2m3(d: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2], c: &[f32; 4]) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e2m3.f32 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
        );
    }
}

// --- FP4 (E2M1) non-block-scaled ---

/// mma.sync.aligned.kind::f8f6f4.m16n8k64.row.col.f32.e2m1.e2m1.f32
///
/// FP4 × FP4: k=64, 4× throughput vs bf16.
/// A: 4 × u32, each holding 8 packed E2M1 values.
/// B: 2 × u32, each holding 8 packed E2M1 values.
#[inline(always)]
pub fn m16n8k64_f32_e2m1_e2m1(d: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2], c: &[f32; 4]) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::f8f6f4.m16n8k64.row.col.f32.e2m1.e2m1.f32 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
        );
    }
}
