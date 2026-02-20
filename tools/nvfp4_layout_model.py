#!/usr/bin/env python3
"""Layout construction helpers for nvfp4_group_gemm tools (wagmi_v6 defaults)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Any


@dataclass(frozen=True)
class KernelConfig:
    mma_tiler_mnk: tuple[int, int, int] = (128, 128, 256)
    mma_inst_shape_k: int = 64
    sf_vec_size: int = 16
    num_ab_stage: int = 1
    threads_per_cta: int = 128


def default_kernel_config() -> KernelConfig:
    return KernelConfig()


def benchmark_problem_groups() -> list[list[tuple[int, int, int, int]]]:
    """Benchmark distribution from context dump (B200 nvfp4_group_gemm)."""
    return [
        [(80, 4096, 7168, 1), (176, 4096, 7168, 1), (128, 4096, 7168, 1), (72, 4096, 7168, 1), (64, 4096, 7168, 1), (248, 4096, 7168, 1), (96, 4096, 7168, 1), (160, 4096, 7168, 1)],
        [(40, 7168, 2048, 1), (76, 7168, 2048, 1), (168, 7168, 2048, 1), (72, 7168, 2048, 1), (164, 7168, 2048, 1), (148, 7168, 2048, 1), (196, 7168, 2048, 1), (160, 7168, 2048, 1)],
        [(192, 3072, 4096, 1), (320, 3072, 4096, 1)],
        [(128, 4096, 1536, 1), (384, 4096, 1536, 1)],
    ]


def benchmark_problem_sizes_flat() -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for group in benchmark_problem_groups():
        out.extend(group)
    return out


def build_cta_mn_list(
    problem_sizes: list[tuple[int, int, int, int]],
    *,
    tile_m: int = 128,
    tile_n: int = 128,
) -> list[tuple[int, int]]:
    return [
        ((m + tile_m - 1) // tile_m, (n + tile_n - 1) // tile_n)
        for (m, n, _k, _l) in problem_sizes
    ]


def build_cta_prefix_from_mn(cta_mn_list: list[tuple[int, int]]) -> list[int]:
    prefix = [0]
    for cta_m, cta_n in cta_mn_list:
        prefix.append(prefix[-1] + cta_m * cta_n)
    return prefix


def parse_shape_spec(spec: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected m,n,k,l shape; got: {spec!r}")
    return (parts[0], parts[1], parts[2], parts[3])


def parse_mma_tiler(spec: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected m,n,k tiler; got: {spec!r}")
    return (parts[0], parts[1], parts[2])


def round_tmem_alloc_cols(required_cols: int) -> int:
    """Match wagmi_v6 allocator rounding policy."""
    valid_cols = (32, 64, 128, 256, 512)
    need = max(1, int(required_cols))
    for cols in valid_cols:
        if need <= cols:
            return cols
    return 512


def aligned_round_up(x: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be > 0")
    return ((int(x) + alignment - 1) // alignment) * alignment


def _import_cutlass_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.nvgpu import tcgen05
        import cutlass.utils.blackwell_helpers as sm100_utils
        import cutlass.utils.blockscaled_layout as blockscaled_utils
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "CUTLASS Python DSL is required for this tool. "
            "Run in an environment with cutlass.cute + blockscaled utilities installed."
        ) from exc

    return cutlass, cute, tcgen05, sm100_utils, blockscaled_utils


def build_layout_bundle(
    shape_mnkl: tuple[int, int, int, int],
    *,
    config: KernelConfig | None = None,
) -> dict[str, Any]:
    """Build gmem/smem/tmem/epilogue layout objects matching wagmi_v6."""
    cfg = config or default_kernel_config()
    m, n, k, l = shape_mnkl
    cutlass, cute, tcgen05, sm100_utils, blockscaled_utils = _import_cutlass_stack()

    # Core gmem layouts
    a_gmem_layout = cute.make_layout((m, k, l), stride=(k, 1, m * k))
    b_gmem_layout = cute.make_layout((n, k, l), stride=(k, 1, n * k))

    atom_shape = ((32, 4), (cfg.sf_vec_size, 4))
    atom_stride = ((16, 4), (0, 1))
    atom_layout = cute.make_layout(atom_shape, stride=atom_stride)

    sfa_gmem_layout_explicit = cute.tile_to_shape(
        atom_layout,
        a_gmem_layout.shape,
        (2, 1, 3),
    )
    sfb_gmem_layout_explicit = cute.tile_to_shape(
        atom_layout,
        b_gmem_layout.shape,
        (2, 1, 3),
    )

    sfa_gmem_layout_helper = blockscaled_utils.tile_atom_to_shape_SF(
        a_gmem_layout.shape,
        cfg.sf_vec_size,
    )
    sfb_gmem_layout_helper = blockscaled_utils.tile_atom_to_shape_SF(
        b_gmem_layout.shape,
        cfg.sf_vec_size,
    )

    # MMA and staged smem layouts
    mma_op = tcgen05.MmaMXF4NVF4Op(
        cutlass.Float8E4M3FN,
        (cfg.mma_tiler_mnk[0], cfg.mma_tiler_mnk[1], cfg.mma_inst_shape_k),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)

    a_smem_layout_staged = sm100_utils.make_smem_layout_a(
        tiled_mma,
        cfg.mma_tiler_mnk,
        cutlass.Float4E2M1FN,
        cfg.num_ab_stage,
    )
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(
        tiled_mma,
        cfg.mma_tiler_mnk,
        cutlass.Float4E2M1FN,
        cfg.num_ab_stage,
    )
    sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma,
        cfg.mma_tiler_mnk,
        cfg.sf_vec_size,
        cfg.num_ab_stage,
    )
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma,
        cfg.mma_tiler_mnk,
        cfg.sf_vec_size,
        cfg.num_ab_stage,
    )

    # TMEM layouts/footprint
    acc_shape = tiled_mma.partition_shape_C(cfg.mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        cfg.mma_tiler_mnk,
        cfg.sf_vec_size,
        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        cfg.mma_tiler_mnk,
        cfg.sf_vec_size,
        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFA_fake = cute.make_tensor(
        cute.make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        tCtSFA_layout,
    )
    tCtSFB_fake = cute.make_tensor(
        cute.make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        tCtSFB_layout,
    )

    acc_cols = int(tcgen05.find_tmem_tensor_col_offset(tCtAcc_fake))
    sfa_cols = int(tcgen05.find_tmem_tensor_col_offset(tCtSFA_fake))
    sfb_cols = int(tcgen05.find_tmem_tensor_col_offset(tCtSFB_fake))

    # Epilogue store layouts
    mC_layout = cute.make_layout((m, n, l), stride=(n, 1, m * n))
    mC = cute.make_tensor(
        cute.make_ptr(cutlass.Float16, 0, cute.AddressSpace.gmem, assumed_align=16),
        mC_layout,
    )
    gC = cute.local_tile(
        mC,
        cute.slice_(cfg.mma_tiler_mnk, (None, None, 0)),
        (0, 0, 0),
    )

    simt_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float16,
        num_bits_per_copy=16,
    )
    epilogue_thread_layout = cute.make_layout(
        (1, cfg.threads_per_cta),
        stride=(cfg.threads_per_cta, 1),
    )
    epilogue_value_layout = cute.make_layout((1, 1))
    tiled_copy_r2g = cute.make_tiled_copy_tv(
        simt_atom,
        epilogue_thread_layout,
        epilogue_value_layout,
    )

    c_identity = cute.make_identity_tensor(gC.shape)
    thr_copy_r2g = tiled_copy_r2g.get_slice(0)
    epilogue_partition_layout = thr_copy_r2g.partition_D(c_identity).layout

    return {
        "shape_mnkl": shape_mnkl,
        "kernel_config": asdict(cfg),
        "tiled_mma": tiled_mma,
        "a_gmem_layout": a_gmem_layout,
        "b_gmem_layout": b_gmem_layout,
        "sfa_gmem_layout_explicit": sfa_gmem_layout_explicit,
        "sfb_gmem_layout_explicit": sfb_gmem_layout_explicit,
        "sfa_gmem_layout_helper": sfa_gmem_layout_helper,
        "sfb_gmem_layout_helper": sfb_gmem_layout_helper,
        "a_smem_layout_staged": a_smem_layout_staged,
        "b_smem_layout_staged": b_smem_layout_staged,
        "sfa_smem_layout_staged": sfa_smem_layout_staged,
        "sfb_smem_layout_staged": sfb_smem_layout_staged,
        "tCtAcc_layout": tCtAcc_fake.layout,
        "tCtSFA_layout": tCtSFA_layout,
        "tCtSFB_layout": tCtSFB_layout,
        "acc_cols": acc_cols,
        "sfa_cols": sfa_cols,
        "sfb_cols": sfb_cols,
        "total_cols": acc_cols + sfa_cols + sfb_cols,
        "alloc_cols": round_tmem_alloc_cols(acc_cols + sfa_cols + sfb_cols),
        "gC_tile_layout": gC.layout,
        "epilogue_thread_layout": epilogue_thread_layout,
        "epilogue_value_layout": epilogue_value_layout,
        "epilogue_partition_layout": epilogue_partition_layout,
    }


def tmem_order_search(
    acc_cols: int,
    sfa_cols: int,
    sfb_cols: int,
    *,
    segment_alignment: int = 1,
) -> list[dict[str, Any]]:
    """Exploratory order/alignment search for TMEM segment packing."""
    segs = {
        "acc": int(acc_cols),
        "sfa": int(sfa_cols),
        "sfb": int(sfb_cols),
    }
    names = list(segs.keys())
    out: list[dict[str, Any]] = []

    for order in permutations(names):
        cursor = 0
        placement: list[dict[str, Any]] = []
        for name in order:
            aligned = aligned_round_up(cursor, segment_alignment)
            placement.append(
                {
                    "segment": name,
                    "start_col": aligned,
                    "end_col_exclusive": aligned + segs[name],
                    "cols": segs[name],
                }
            )
            cursor = aligned + segs[name]

        used = cursor
        alloc = round_tmem_alloc_cols(used)
        out.append(
            {
                "order": order,
                "segment_alignment": segment_alignment,
                "used_cols": used,
                "alloc_cols": alloc,
                "slack_cols": max(0, alloc - used),
                "placement": placement,
            }
        )

    out.sort(key=lambda r: (r["slack_cols"], r["used_cols"], r["order"]))
    return out
