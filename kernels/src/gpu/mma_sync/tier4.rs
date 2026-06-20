/// Tier 4: NVFP4 block-scaled (sm_120a / sm_121a)
///
/// `kind::mxf4nvf4` — NVIDIA's proprietary FP4 format with finer-grained
/// scaling than MXFP4. Uses E4M3 scale factors with block size 16
/// (vs MXFP4's E8M0 with block size 32).
///
/// Compilation requires: `-gencode arch=compute_120a,code=sm_120a`
use cuda_device::ptx_asm;

/// NVFP4 with E4M3 scales, block size 16.
///
/// mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
///
/// scale_vec::4X because atom_K=64 with block_size=16 = 4 scale values per tile.
/// SFA/SFB: u32 holding 4 packed E4M3 scale factors.
#[inline(always)]
pub fn m16n8k64_f32_e2m1_e2m1_nvf4_ue4m3(
    d: &mut [f32; 4],
    a: &[u32; 4],
    b: &[u32; 2],
    c: &[f32; 4],
    sfa: u32,
    sfb: u32,
) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 \
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

/// NVFP4 with E8M0 scales, block size 32 (coarser scaling, OCP-compatible).
///
/// mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
#[inline(always)]
pub fn m16n8k64_f32_e2m1_e2m1_nvf4_ue8m0(
    d: &mut [f32; 4],
    a: &[u32; 4],
    b: &[u32; 2],
    c: &[f32; 4],
    sfa: u32,
    sfb: u32,
) {
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 \
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
