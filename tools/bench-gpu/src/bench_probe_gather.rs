// probe-gather -- isolate which part of the paged gather kernel fails to
// JIT ("Failed to compile type", bench-decode-ops 2026-11-11).  Six
// minimal variants of the GatherScatterView usage:
//   A. f32 rank-2 [BC, D] gather, iota sparse indices
//   B. f32 rank-2 [BC, D] gather, computed indices (divi/remi/mul/add)
//   C. i32 rank-1 [BC] gather (block-table lookup)
//   D. f32 rank-2 gather with the Some(4) latency hint
//   E. f32 rank-1 gather
//   F. i32 rank-2 gather
//
// RESULT (2026-11-11, RTX 5090, cuda-oxide a0a4dc78a): A B D F OK (and
// bit-correct), C E FAIL "Failed to compile type" -- RANK-1 gather views
// do not JIT; rank-2 works for both element types.  Computed indices and
// the latency hint are fine.  Workaround used by cutile_paged_gather.rs:
// block table as rank-2 [-1, 1].
//
// Run: scripts/gpu-remote.sh bench probe-gather

use std::sync::Arc;
use cuda_core::Stream;
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};

const BC: i32 = 32;
const D: i32 = 128;

#[cutile::module]
pub mod var_a {
    use cutile::core::*;
    use cutile::tileir::*;

    #[cutile::entry()]
    pub unsafe fn gather_iota<const BC: i32, const D: i32>(
        out: &mut Tensor<f32, { [BC, D] }>,
        src: &Tensor<f32, { [-1, D] }>,
    ) {
        let idx: Tile<i32, { [BC] }> = iota(shape![BC]);
        let view: GatherScatterView<f32, { [BC, D] }, 0> =
            make_gather_scatter_view(src, shape![BC, D], padding::None);
        let (tile, _): (Tile<f32, { [BC, D] }>, Token) = load_view_raw(
            &view,
            (idx, 0i32),
            None,
            ordering::Weak,
            scope::Device,
            tma::Disabled,
            None,
            [false, false],
            shape![BC, D],
        );
        out.store(tile);
    }
}

#[cutile::module]
pub mod var_b {
    use cutile::core::*;
    use cutile::tileir::*;

    #[cutile::entry()]
    pub unsafe fn gather_computed<const BC: i32, const D: i32>(
        out: &mut Tensor<f32, { [BC, D] }>,
        src: &Tensor<f32, { [-1, D] }>,
        n_kv_heads: i32,
        block_size: i32,
    ) {
        let toks: Tile<i32, { [BC] }> = iota(shape![BC]);
        let bs: Tile<i32, { [BC] }> = broadcast_scalar(block_size, shape![BC]);
        let bidx: Tile<i32, { [BC] }> = divi(toks, bs, rounding::Zero);
        let off: Tile<i32, { [BC] }> = remi(toks, bs);
        let row: Tile<i32, { [BC] }> = bidx * bs + off;
        let pool_row: Tile<i32, { [BC] }> =
            row * broadcast_scalar(n_kv_heads, shape![BC]);
        let view: GatherScatterView<f32, { [BC, D] }, 0> =
            make_gather_scatter_view(src, shape![BC, D], padding::None);
        let (tile, _): (Tile<f32, { [BC, D] }>, Token) = load_view_raw(
            &view,
            (pool_row, 0i32),
            None,
            ordering::Weak,
            scope::Device,
            tma::Disabled,
            None,
            [false, false],
            shape![BC, D],
        );
        out.store(tile);
    }
}

#[cutile::module]
pub mod var_c {
    use cutile::core::*;
    use cutile::tileir::*;

    #[cutile::entry()]
    pub unsafe fn gather_i32_table<const BC: i32>(
        out: &mut Tensor<i32, { [BC] }>,
        table: &Tensor<i32, { [-1] }>,
        block_size: i32,
    ) {
        let toks: Tile<i32, { [BC] }> = iota(shape![BC]);
        let bs: Tile<i32, { [BC] }> = broadcast_scalar(block_size, shape![BC]);
        let bidx: Tile<i32, { [BC] }> = divi(toks, bs, rounding::Zero);
        let view: GatherScatterView<i32, { [BC] }, 0> =
            make_gather_scatter_view(table, shape![BC], padding::None);
        let (page, _): (Tile<i32, { [BC] }>, Token) = load_view_raw(
            &view,
            (bidx,),
            None,
            ordering::Weak,
            scope::Device,
            tma::Disabled,
            None,
            [false],
            shape![BC],
        );
        out.store(page);
    }
}

#[cutile::module]
pub mod var_d {
    use cutile::core::*;
    use cutile::tileir::*;

    #[cutile::entry()]
    pub unsafe fn gather_lat4<const BC: i32, const D: i32>(
        out: &mut Tensor<f32, { [BC, D] }>,
        src: &Tensor<f32, { [-1, D] }>,
    ) {
        let idx: Tile<i32, { [BC] }> = iota(shape![BC]);
        let view: GatherScatterView<f32, { [BC, D] }, 0> =
            make_gather_scatter_view(src, shape![BC, D], padding::None);
        let (tile, _): (Tile<f32, { [BC, D] }>, Token) = load_view_raw(
            &view,
            (idx, 0i32),
            None,
            ordering::Weak,
            scope::Device,
            tma::Disabled,
            Some(4),
            [false, false],
            shape![BC, D],
        );
        out.store(tile);
    }
}

#[cutile::module]
pub mod var_e {
    use cutile::core::*;
    use cutile::tileir::*;

    /// f32 RANK-1 gather (isolate: is C's failure the i32 type or rank-1?).
    #[cutile::entry()]
    pub unsafe fn gather_f32_rank1<const BC: i32>(
        out: &mut Tensor<f32, { [BC] }>,
        table: &Tensor<f32, { [-1] }>,
        block_size: i32,
    ) {
        let toks: Tile<i32, { [BC] }> = iota(shape![BC]);
        let bs: Tile<i32, { [BC] }> = broadcast_scalar(block_size, shape![BC]);
        let bidx: Tile<i32, { [BC] }> = divi(toks, bs, rounding::Zero);
        let view: GatherScatterView<f32, { [BC] }, 0> =
            make_gather_scatter_view(table, shape![BC], padding::None);
        let (page, _): (Tile<f32, { [BC] }>, Token) = load_view_raw(
            &view,
            (bidx,),
            None,
            ordering::Weak,
            scope::Device,
            tma::Disabled,
            None,
            [false],
            shape![BC],
        );
        out.store(page);
    }
}

#[cutile::module]
pub mod var_f {
    use cutile::core::*;
    use cutile::tileir::*;

    /// i32 RANK-2 gather (isolate: is C's failure the i32 type or rank-1?).
    #[cutile::entry()]
    pub unsafe fn gather_i32_rank2<const BC: i32, const D: i32>(
        out: &mut Tensor<i32, { [BC, D] }>,
        table: &Tensor<i32, { [-1, D] }>,
        block_size: i32,
    ) {
        let toks: Tile<i32, { [BC] }> = iota(shape![BC]);
        let bs: Tile<i32, { [BC] }> = broadcast_scalar(block_size, shape![BC]);
        let bidx: Tile<i32, { [BC] }> = divi(toks, bs, rounding::Zero);
        let view: GatherScatterView<i32, { [BC, D] }, 0> =
            make_gather_scatter_view(table, shape![BC, D], padding::None);
        let (page, _): (Tile<i32, { [BC, D] }>, Token) = load_view_raw(
            &view,
            (bidx, 0i32),
            None,
            ordering::Weak,
            scope::Device,
            tma::Disabled,
            None,
            [false, false],
            shape![BC, D],
        );
        out.store(page);
    }
}

fn main() {
    let device = Arc::new(cuda_core::Device::new(0).expect("device"));
    let ctx = device.primary_context().expect("ctx");
    let stream: Arc<Stream> = ctx.new_stream().expect("stream");
    let src: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(
        (0..(BC as usize) * (D as usize)).map(|i| i as f32).collect::<Vec<_>>(),
    ))
    .sync_on(&stream)
    .expect("upload")
    .reshape(&[BC as usize, D as usize])
    .expect("reshape")
    .into();

    let run = |label: &str, f: &mut dyn FnMut() -> Result<(), String>| {
        match f() {
            Ok(()) => println!("{label}: OK"),
            Err(e) => println!("{label}: FAIL {e}"),
        }
    };

    run("A f32 rank-2 iota", &mut || {
        let out = api::zeros::<f32>(&[BC as usize, D as usize])
            .sync_on(&stream)
            .expect("out");
        let (p, ..) = unsafe {
            var_a::gather_iota(out.partition([BC as usize, D as usize]), src.clone())
                .generics(vec![BC.to_string(), D.to_string()])
                .async_on(&stream)
                .map_err(|e| e.to_string())?
        };
        let v = Arc::new(p.unpartition())
            .to_host_vec()
            .sync_on(&stream)
            .expect("dl");
        let expect: Vec<f32> = (0..(BC as usize) * (D as usize)).map(|i| i as f32).collect();
        if v != expect {
            return Err("wrong data".into());
        }
        Ok(())
    });

    run("B f32 computed idx", &mut || {
        let out = api::zeros::<f32>(&[BC as usize, D as usize])
            .sync_on(&stream)
            .expect("out");
        let (p, ..) = unsafe {
            var_b::gather_computed(out.partition([BC as usize, D as usize]), src.clone(), 1, 16)
                .generics(vec![BC.to_string(), D.to_string()])
                .async_on(&stream)
                .map_err(|e| e.to_string())?
        };
        let v = Arc::new(p.unpartition())
            .to_host_vec()
            .sync_on(&stream)
            .expect("dl");
        let expect: Vec<f32> = (0..(BC as usize) * (D as usize)).map(|i| i as f32).collect();
        if v != expect {
            return Err("wrong data".into());
        }
        Ok(())
    });

    run("C i32 table gather", &mut || {
        let table: Arc<Tensor<i32>> = api::copy_host_vec_to_device(&Arc::new(
            (0..BC as usize).map(|i| i as i32 * 3).collect::<Vec<_>>(),
        ))
        .sync_on(&stream)
        .expect("table upload")
        .into();
        let out = api::zeros::<i32>(&[BC as usize])
            .sync_on(&stream)
            .expect("out");
        let (p, ..) = unsafe {
            var_c::gather_i32_table(out.partition([BC as usize]), table.clone(), 16)
                .generics(vec![BC.to_string()])
                .async_on(&stream)
                .map_err(|e| e.to_string())?
        };
        let v = Arc::new(p.unpartition())
            .to_host_vec()
            .sync_on(&stream)
            .expect("dl");
        let expect: Vec<i32> =
            (0..(BC as usize / 16)).map(|i| (i * 3) as i32).collect::<Vec<_>>();
        // 32 iota toks / bs=16 -> bidx = 0,0,...,1,1,...; table[bidx]*3
        let mut want = vec![0i32; BC as usize];
        for i in 0..BC as usize {
            want[i] = (i / 16) as i32 * 3;
        }
        let _ = expect;
        if v != want {
            return Err(format!("wrong data {v:?}"));
        }
        Ok(())
    });

    run("D f32 iota + lat4", &mut || {
        let out = api::zeros::<f32>(&[BC as usize, D as usize])
            .sync_on(&stream)
            .expect("out");
        let (p, ..) = unsafe {
            var_d::gather_lat4(out.partition([BC as usize, D as usize]), src.clone())
                .generics(vec![BC.to_string(), D.to_string()])
                .async_on(&stream)
                .map_err(|e| e.to_string())?
        };
        let v = Arc::new(p.unpartition())
            .to_host_vec()
            .sync_on(&stream)
            .expect("dl");
        let expect: Vec<f32> = (0..(BC as usize) * (D as usize)).map(|i| i as f32).collect();
        if v != expect {
            return Err("wrong data".into());
        }
        Ok(())
    });

    run("E f32 rank-1 gather", &mut || {
        let table: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(
            (0..BC as usize).map(|i| (i * 3) as f32).collect::<Vec<_>>(),
        ))
        .sync_on(&stream)
        .expect("table upload")
        .into();
        let out = api::zeros::<f32>(&[BC as usize])
            .sync_on(&stream)
            .expect("out");
        let (p, ..) = unsafe {
            var_e::gather_f32_rank1(out.partition([BC as usize]), table.clone(), 16)
                .generics(vec![BC.to_string()])
                .async_on(&stream)
                .map_err(|e| e.to_string())?
        };
        let v = Arc::new(p.unpartition())
            .to_host_vec()
            .sync_on(&stream)
            .expect("dl");
        let mut want = vec![0f32; BC as usize];
        for i in 0..BC as usize {
            want[i] = (i / 16) as i32 as f32 * 3.0;
        }
        if v != want {
            return Err(format!("wrong data {v:?}"));
        }
        Ok(())
    });

    run("F i32 rank-2 gather", &mut || {
        let table: Arc<Tensor<i32>> = api::copy_host_vec_to_device(&Arc::new(
            (0..(BC as usize) * (D as usize)).map(|i| i as i32).collect::<Vec<_>>(),
        ))
        .sync_on(&stream)
        .expect("table upload")
        .reshape(&[BC as usize, D as usize])
        .expect("reshape")
        .into();
        let out = api::zeros::<i32>(&[BC as usize, D as usize])
            .sync_on(&stream)
            .expect("out");
        let (p, ..) = unsafe {
            var_f::gather_i32_rank2(
                out.partition([BC as usize, D as usize]),
                table.clone(),
                16,
            )
            .generics(vec![BC.to_string(), D.to_string()])
            .async_on(&stream)
            .map_err(|e| e.to_string())?
        };
        let v = Arc::new(p.unpartition())
            .to_host_vec()
            .sync_on(&stream)
            .expect("dl");
        let mut want = vec![0i32; BC as usize * D as usize];
        for i in 0..BC as usize {
            for d in 0..D as usize {
                want[i * D as usize + d] = (i / 16) as i32 * D + d as i32;
            }
        }
        if v != want {
            return Err("wrong data".into());
        }
        Ok(())
    });
}
