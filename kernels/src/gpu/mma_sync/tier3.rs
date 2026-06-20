/// Tier 3: Block-scaled MX formats (sm_120a / sm_121a)
///
/// `kind::mxf8f6f4` MMA instructions with hardware block scaling.
/// Uses E8M0 scale factors with block size 32.
///
/// Extra operands vs Tier 2: scale factors SFA (for A) and SFB (for B),
/// each a u32 holding packed u8 scale values.
///
/// Compilation requires: `-gencode arch=compute_120a,code=sm_120a`
/// (or sm_121a for GB10/GX10).
use cuda_device::ptx_asm;

/// Block-scaled FP8 × FP8.
///
/// mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
///
/// SFA/SFB: u32 holding packed E8M0 scale factors (one per block of 32 elements).
#[inline(always)]
pub fn m16n8k32_f32_e4m3_e4m3_ue8m0(
    d: &mut [f32; 4],
    a: &[u32; 4],
    b: &[u32; 2],
    c: &[f32; 4],
    sfa: u32,
    sfb: u32,
) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {%15};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
            in("r") sfa,
            in("r") sfb,
        );
    }
}

/// Block-scaled FP8 E5M2 × E4M3.
///
/// mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e5m2.e4m3.f32.ue8m0
#[inline(always)]
pub fn m16n8k32_f32_e5m2_e4m3_ue8m0(
    d: &mut [f32; 4],
    a: &[u32; 4],
    b: &[u32; 2],
    c: &[f32; 4],
    sfa: u32,
    sfb: u32,
) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e5m2.e4m3.f32.ue8m0 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {%15};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
            in("r") sfa,
            in("r") sfb,
        );
    }
}

/// Block-scaled FP6 E3M2 × E3M2.
///
/// mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e3m2.e3m2.f32.ue8m0
#[inline(always)]
pub fn m16n8k32_f32_e3m2_e3m2_ue8m0(
    d: &mut [f32; 4],
    a: &[u32; 4],
    b: &[u32; 2],
    c: &[f32; 4],
    sfa: u32,
    sfb: u32,
) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e3m2.e3m2.f32.ue8m0 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {%15};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
            in("r") sfa,
            in("r") sfb,
        );
    }
}

/// Block-scaled MXFP4 E2M1 × E2M1 (OCP Microscaling spec).
///
/// mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
///
/// Uses scale_vec::2X because atom_K=64 with block_size=32 means 2 scale values per tile.
#[inline(always)]
pub fn m16n8k64_f32_e2m1_e2m1_mxf4_ue8m0(
    d: &mut [f32; 4],
    a: &[u32; 4],
    b: &[u32; 2],
    c: &[f32; 4],
    sfa: u32,
    sfb: u32,
) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 \
             {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {%15};",
            out("=f") d[0], out("=f") d[1], out("=f") d[2], out("=f") d[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
            in("r") sfa,
            in("r") sfb,
        );
    }
}
