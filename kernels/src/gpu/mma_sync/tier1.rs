/// Tier 1: Standard MMA (sm_80+)
///
/// m16n8k16 for f16/bf16, m16n8k8 for tf32.
/// Per-thread fragments: D[4×f32], A[4×u32], B[2×u32], C[4×f32].
/// (tf32 uses A[2×u32], B[1×u32] with k=8.)

use cuda_device::ptx_asm;

/// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
///
/// A fragments: 4 × u32, each holding 2 packed f16 values.
/// B fragments: 2 × u32, each holding 2 packed f16 values.
#[inline(always)]
pub fn m16n8k16_f32_f16(d: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2], c: &[f32; 4]) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
        );
    }
}

/// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
#[inline(always)]
pub fn m16n8k16_f32_bf16(d: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2], c: &[f32; 4]) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
        );
    }
}

/// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
///
/// TF32: 19-bit (1 sign + 8 exponent + 10 mantissa).
/// A fragments: 2 × u32 (k=8 instead of k=16).
/// B fragments: 1 × u32.
#[inline(always)]
pub fn m16n8k8_f32_tf32(d: &mut [f32; 4], a: &[u32; 2], b: &[u32; 1], c: &[f32; 4]) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \
             {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1],
            in("r") b[0],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
        );
    }
}
