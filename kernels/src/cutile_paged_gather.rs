//! Step-0 probe for the paged attention kernel (plan p20-paged-attention):
//! validates the cutile tile-level gather primitive (`GatherScatterView` +
//! `load_view_raw`) that a block-table-indirected flash attention KV tile
//! loop needs, BEFORE the kernel itself is built.  No tempura kernel used
//! the raw Tile IR gather surface before this.
//!
//! The paged index math under test (per KV tile column i, block_size bs):
//!   token    = tile_start + i
//!   bidx     = token / bs                    (divi)
//!   page     = block_table[bidx]             (i32 tile gather)
//!   row      = page * bs + token % bs        (remi)
//!   pool_row = row * n_kv_heads + kv_head    (head de-interleave)
//! and the [BC, D] K tile is gathered with `pool_row` as the sparse row
//! index, D contiguous elements per row -- paging AND de-interleaving in
//! one primitive, replacing cutile_gather_heads + a contiguous partition
//! load.  All computed indices stay in bounds by construction: the pool is
//! padded to num_tiles * BC tokens and the block table covers the padded
//! token range (ragged tail rows are loaded but zeroed by the caller-side
//! mask, matching the score-side -inf masking convention of the flash
//! kernels).
//!
//! Validated by the "paged gather view probe" section of bench-decode-ops
//! (correctness vs a CPU reference on RANDOM data with a SHUFFLED page
//! mapping -- identity tables and uniform data cannot catch index-mapping
//! bugs -- plus a throughput comparison against the dense
//! `load_pipelined` path the gather would replace).
//!
//! DSL constraint found 2026-11-11 (bench-probe-gather): RANK-1
//! GatherScatterViews fail to JIT ("Failed to compile type") for both i32
//! and f32; rank-2 views compile for both.  Hence the block table is
//! passed as rank-2 [-1, 1] and gathered with a [BC, 1] tile.  Computed
//! sparse indices (divi/remi/mul/add chains) and the Some(4) latency hint
//! both work.

#[cutile::module]
pub mod kernels {
    use cutile::core::*;
    use cutile::tileir::*;

    /// Correctness probe: one program per (kv_head, tile); gathers the
    /// [BC, D] tile through the block table and stores it dense.
    /// Rows with token >= kv_len are zeroed (ragged tail).
    #[cutile::entry()]
    pub unsafe fn paged_gather_tile<const BC: i32, const D: i32>(
        out: &mut Tensor<f32, { [BC, D] }>,
        kv_pool: &Tensor<f32, { [-1, D] }>,
        block_table: &Tensor<i32, { [-1, 1] }>,
        n_kv_heads: i32,
        num_tiles: i32,
        kv_len: i32,
        block_size: i32,
    ) {
        let blk = program_id(0);
        let kv_h = blk / num_tiles;
        let j = blk % num_tiles;

        let toks: Tile<i32, { [BC] }> =
            iota(shape![BC]) + broadcast_scalar(j * BC, shape![BC]);
        let bs: Tile<i32, { [BC] }> = broadcast_scalar(block_size, shape![BC]);
        let bidx: Tile<i32, { [BC] }> = divi(toks, bs, rounding::Zero);
        let off: Tile<i32, { [BC] }> = remi(toks, bs);

        // NOTE: the block table is rank-2 [-1, 1] because RANK-1 gather
        // views fail to JIT ("Failed to compile type", isolated by
        // bench-probe-gather 2026-11-11: rank-2 gathers compile for both
        // i32 and f32, rank-1 fails for both).  i32 elements are fine.
        let bt_view: GatherScatterView<i32, { [BC, 1] }, 0> =
            make_gather_scatter_view(block_table, shape![BC, 1], padding::None);
        let (page2, _): (Tile<i32, { [BC, 1] }>, Token) = load_view_raw(
            &bt_view,
            (bidx, 0i32),
            None,
            ordering::Weak,
            scope::Device,
            tma::Disabled,
            None,
            [false, false],
            shape![BC, 1],
        );
        let page: Tile<i32, { [BC] }> = page2.reshape(shape![BC]);

        let row: Tile<i32, { [BC] }> = page * bs + off;
        let pool_row: Tile<i32, { [BC] }> = row * broadcast_scalar(n_kv_heads, shape![BC])
            + broadcast_scalar(kv_h, shape![BC]);

        let kv_view: GatherScatterView<f32, { [BC, D] }, 0> =
            make_gather_scatter_view(kv_pool, shape![BC, D], padding::None);
        let (tile, _): (Tile<f32, { [BC, D] }>, Token) = load_view_raw(
            &kv_view,
            (pool_row, 0i32),
            None,
            ordering::Weak,
            scope::Device,
            tma::Disabled,
            None,
            [false, false],
            shape![BC, D],
        );

        let valid: Tile<bool, { [BC, D] }> = lt_tile(
            toks.reshape(shape![BC, 1]).broadcast(shape![BC, D]),
            broadcast_scalar(kv_len, shape![BC, D]),
        );
        let zero: Tile<f32, { [BC, D] }> = constant(0.0f32, shape![BC, D]);
        out.store(select(valid, tile, zero));
    }

    /// Perf probe (paged gather): grid = n_kv_heads * progs_per_head, each
    /// program walks its tile range with the full paged index chain and
    /// accumulates reduce_sum over the tile axis into a [D] checksum
    /// (prevents dead-code elimination).  No latency hint.
    #[cutile::entry()]
    pub unsafe fn paged_gather_bench<const BC: i32, const D: i32>(
        out: &mut Tensor<f32, { [D] }>,
        kv_pool: &Tensor<f32, { [-1, D] }>,
        block_table: &Tensor<i32, { [-1, 1] }>,
        n_kv_heads: i32,
        num_tiles: i32,
        tiles_per_prog: i32,
        block_size: i32,
    ) {
        let pid = program_id(0);
        let progs_per_head = num_tiles / tiles_per_prog;
        let kv_h = pid / progs_per_head;
        let start = (pid % progs_per_head) * tiles_per_prog;

        let bs: Tile<i32, { [BC] }> = broadcast_scalar(block_size, shape![BC]);
        let kvh_mul: Tile<i32, { [BC] }> = broadcast_scalar(n_kv_heads, shape![BC]);
        let kvh_add: Tile<i32, { [BC] }> = broadcast_scalar(kv_h, shape![BC]);
        let bt_view: GatherScatterView<i32, { [BC, 1] }, 0> =
            make_gather_scatter_view(block_table, shape![BC, 1], padding::None);
        let kv_view: GatherScatterView<f32, { [BC, D] }, 0> =
            make_gather_scatter_view(kv_pool, shape![BC, D], padding::None);

        let mut acc: Tile<f32, { [D] }> = constant(0.0f32, shape![D]);
        for j in start..start + tiles_per_prog {
            let toks: Tile<i32, { [BC] }> =
                iota(shape![BC]) + broadcast_scalar(j * BC, shape![BC]);
            let bidx: Tile<i32, { [BC] }> = divi(toks, bs, rounding::Zero);
            let off: Tile<i32, { [BC] }> = remi(toks, bs);
            let (page2, _): (Tile<i32, { [BC, 1] }>, Token) = load_view_raw(
                &bt_view,
                (bidx, 0i32),
                None,
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [false, false],
                shape![BC, 1],
            );
            let page: Tile<i32, { [BC] }> = page2.reshape(shape![BC]);
            let pool_row: Tile<i32, { [BC] }> = (page * bs + off) * kvh_mul + kvh_add;
            let (tile, _): (Tile<f32, { [BC, D] }>, Token) = load_view_raw(
                &kv_view,
                (pool_row, 0i32),
                None,
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [false, false],
                shape![BC, D],
            );
            acc = acc + reduce_sum(tile, 0i32);
        }
        out.store(acc);
    }

    /// Perf probe (paged gather, latency hint 4): identical to
    /// [`paged_gather_bench`] but the KV gather carries the same latency
    /// hint `load_pipelined::<4>` uses, to measure whether the software
    /// pipeliner can overlap gather loads.
    #[cutile::entry()]
    pub unsafe fn paged_gather_bench_lat4<const BC: i32, const D: i32>(
        out: &mut Tensor<f32, { [D] }>,
        kv_pool: &Tensor<f32, { [-1, D] }>,
        block_table: &Tensor<i32, { [-1, 1] }>,
        n_kv_heads: i32,
        num_tiles: i32,
        tiles_per_prog: i32,
        block_size: i32,
    ) {
        let pid = program_id(0);
        let progs_per_head = num_tiles / tiles_per_prog;
        let kv_h = pid / progs_per_head;
        let start = (pid % progs_per_head) * tiles_per_prog;

        let bs: Tile<i32, { [BC] }> = broadcast_scalar(block_size, shape![BC]);
        let kvh_mul: Tile<i32, { [BC] }> = broadcast_scalar(n_kv_heads, shape![BC]);
        let kvh_add: Tile<i32, { [BC] }> = broadcast_scalar(kv_h, shape![BC]);
        let bt_view: GatherScatterView<i32, { [BC, 1] }, 0> =
            make_gather_scatter_view(block_table, shape![BC, 1], padding::None);
        let kv_view: GatherScatterView<f32, { [BC, D] }, 0> =
            make_gather_scatter_view(kv_pool, shape![BC, D], padding::None);

        let mut acc: Tile<f32, { [D] }> = constant(0.0f32, shape![D]);
        for j in start..start + tiles_per_prog {
            let toks: Tile<i32, { [BC] }> =
                iota(shape![BC]) + broadcast_scalar(j * BC, shape![BC]);
            let bidx: Tile<i32, { [BC] }> = divi(toks, bs, rounding::Zero);
            let off: Tile<i32, { [BC] }> = remi(toks, bs);
            let (page2, _): (Tile<i32, { [BC, 1] }>, Token) = load_view_raw(
                &bt_view,
                (bidx, 0i32),
                None,
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [false, false],
                shape![BC, 1],
            );
            let page: Tile<i32, { [BC] }> = page2.reshape(shape![BC]);
            let pool_row: Tile<i32, { [BC] }> = (page * bs + off) * kvh_mul + kvh_add;
            let (tile, _): (Tile<f32, { [BC, D] }>, Token) = load_view_raw(
                &kv_view,
                (pool_row, 0i32),
                None,
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                Some(4),
                [false, false],
                shape![BC, D],
            );
            acc = acc + reduce_sum(tile, 0i32);
        }
        out.store(acc);
    }

    /// Perf probe (dense baseline): identical loop structure over a
    /// head-major [rows, D] tensor using the partition
    /// `load_pipelined::<4>` the flash kernels use today -- the access
    /// pattern the paged gather would replace.
    #[cutile::entry()]
    pub unsafe fn dense_load_bench<const BC: i32, const D: i32>(
        out: &mut Tensor<f32, { [D] }>,
        src: &Tensor<f32, { [-1, D] }>,
        num_tiles: i32,
        tiles_per_prog: i32,
    ) {
        let pid = program_id(0);
        let progs_per_head = num_tiles / tiles_per_prog;
        let kv_h = pid / progs_per_head;
        let start = (pid % progs_per_head) * tiles_per_prog;

        let part = src.partition(shape![BC, D]);
        let base: i32 = kv_h * num_tiles;
        let mut acc: Tile<f32, { [D] }> = constant(0.0f32, shape![D]);
        for j in start..start + tiles_per_prog {
            let t: Tile<f32, { [BC, D] }> = part.load_pipelined::<4>([base + j, 0i32]);
            acc = acc + reduce_sum(t, 0i32);
        }
        out.store(acc);
    }
}
