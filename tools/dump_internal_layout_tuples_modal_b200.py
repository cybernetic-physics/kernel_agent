#!/usr/bin/env python3
"""Run internal layout tuple dump on deployed Modal B200 and print/write JSON.

Deploy once:
    uv run --with modal modal deploy tools/dump_internal_layout_tuples_modal_b200.py
Then run repeatedly without `modal run`:
    uv run --with modal python tools/dump_internal_layout_tuples_modal_b200.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import modal

APP_NAME = "nvfp4-layout-tuples-b200"
DEFAULT_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
CACHE_VOLUME_NAME = "modal-tools-cache-v1"
CACHE_MOUNT_PATH = "/cache"

app = modal.App(APP_NAME)
image = modal.Image.from_registry(DEFAULT_IMAGE, add_python="3.11").pip_install(
    "nvidia-cutlass-dsl==4.4.0"
)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default="80,4096,7168,1")
    p.add_argument("--mma-tiler", default="128,128,256")
    p.add_argument("--mma-inst-shape-k", type=int, default=64)
    p.add_argument("--sf-vec-size", type=int, default=16)
    p.add_argument("--num-ab-stage", type=int, default=1)
    p.add_argument("--threads-per-cta", type=int, default=128)
    p.add_argument(
        "--try-cpasync-partition",
        action="store_true",
        default=_env_flag("NVFP4_TRY_CPASYNC_PARTITION", False),
        help="Best-effort call to cpasync.tma_partition (may emit CUTLASS type errors).",
    )
    p.add_argument(
        "--probe-modes",
        action="store_true",
        default=_env_flag("NVFP4_PROBE_MODES", False),
        help="Run all context probe modes instead of only the stable with_module_ip mode.",
    )
    p.add_argument("--gpu", default="B200")
    p.add_argument(
        "--app-name",
        default=APP_NAME,
        help=f"Deployed Modal app name (default: {APP_NAME}).",
    )
    p.add_argument(
        "--function-name",
        default="_dump_remote",
        help="Deployed Modal function name (default: _dump_remote).",
    )
    p.add_argument("--json-out", type=Path)
    # Modal CLI can inject its own argv fragments into local entrypoints.
    args, _unknown = p.parse_known_args()
    return args


def _child_code(context_mode: str) -> str:
    return f"""
import json
import os

cfg = json.loads(os.environ["NVFP4_LAYOUT_CFG"])
context_mode = {context_mode!r}

def _to_jsonable(x):
    if isinstance(x, (str, bool, int, float)) or x is None:
        return x
    if isinstance(x, (tuple, list)):
        return [_to_jsonable(v) for v in x]
    try:
        return int(x)
    except Exception:
        return str(x)

def _layout_record(layout):
    out = {{"repr": str(layout)}}
    if hasattr(layout, "shape"):
        out["shape"] = _to_jsonable(layout.shape)
    if hasattr(layout, "stride"):
        out["stride"] = _to_jsonable(layout.stride)
    return out

def _tensor_record(tensor):
    out = {{"repr": str(tensor)}}
    if hasattr(tensor, "layout"):
        out["layout"] = _layout_record(tensor.layout)
    return out

def _extract():
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blackwell_helpers as sm100_utils
    import cutlass.utils.blockscaled_layout as blockscaled_utils
    from cutlass.cute.nvgpu import tcgen05

    mma_tiler_mnk = tuple(cfg["mma_tiler_mnk"])
    mma_inst_shape_k = int(cfg["mma_inst_shape_k"])
    sf_vec_size = int(cfg["sf_vec_size"])
    num_ab_stage = int(cfg["num_ab_stage"])

    sf_dtype = cutlass.Float8E4M3FN
    ab_dtype = cutlass.Float4E2M1FN

    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)

    a_smem_layout_staged = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, ab_dtype, num_ab_stage
    )
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, ab_dtype, num_ab_stage
    )
    sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage
    )
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage
    )
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
    )
    return {{
        "a_smem_layout_staged": {{
            "outer": _layout_record(a_smem_layout_staged.outer),
            "inner": _layout_record(a_smem_layout_staged.inner),
        }},
        "b_smem_layout_staged": {{
            "outer": _layout_record(b_smem_layout_staged.outer),
            "inner": _layout_record(b_smem_layout_staged.inner),
        }},
        "sfa_smem_layout_staged": _layout_record(sfa_smem_layout_staged),
        "sfb_smem_layout_staged": _layout_record(sfb_smem_layout_staged),
        "tCtSFA_layout": _layout_record(tCtSFA_layout),
        "tCtSFB_layout": _layout_record(tCtSFB_layout),
    }}

def _extract_with_explicit_loc():
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blackwell_helpers as sm100_utils
    import cutlass.utils.blockscaled_layout as blockscaled_utils
    from cutlass.cute.nvgpu import tcgen05
    from cutlass._mlir import ir

    mma_tiler_mnk = tuple(cfg["mma_tiler_mnk"])
    mma_inst_shape_k = int(cfg["mma_inst_shape_k"])
    sf_vec_size = int(cfg["sf_vec_size"])
    num_ab_stage = int(cfg["num_ab_stage"])

    sf_dtype = cutlass.Float8E4M3FN
    ab_dtype = cutlass.Float4E2M1FN

    with ir.Context() as ctx:
        with ir.Location.unknown(ctx) as loc:
            # Avoid decorator location synthesis path by supplying loc everywhere possible.
            mma_op = tcgen05.MmaMXF4NVF4Op(
                sf_dtype,
                (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
            )
            tiled_mma = cute.make_tiled_mma(mma_op, loc=loc)

            a_smem_layout_staged = sm100_utils.make_smem_layout_a(
                tiled_mma, mma_tiler_mnk, ab_dtype, num_ab_stage, loc=loc
            )
            b_smem_layout_staged = sm100_utils.make_smem_layout_b(
                tiled_mma, mma_tiler_mnk, ab_dtype, num_ab_stage, loc=loc
            )
            sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage, loc=loc
            )
            sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage, loc=loc
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                mma_tiler_mnk,
                sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0), loc=loc),
                loc=loc,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                mma_tiler_mnk,
                sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0), loc=loc),
                loc=loc,
            )

            return {{
                "a_smem_layout_staged": {{
                    "outer": _layout_record(a_smem_layout_staged.outer),
                    "inner": _layout_record(a_smem_layout_staged.inner),
                }},
                "b_smem_layout_staged": {{
                    "outer": _layout_record(b_smem_layout_staged.outer),
                    "inner": _layout_record(b_smem_layout_staged.inner),
                }},
                "sfa_smem_layout_staged": _layout_record(sfa_smem_layout_staged),
                "sfb_smem_layout_staged": _layout_record(sfb_smem_layout_staged),
                "tCtSFA_layout": _layout_record(tCtSFA_layout),
                "tCtSFB_layout": _layout_record(tCtSFB_layout),
            }}

def _extract_with_module_ip():
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blackwell_helpers as sm100_utils
    import cutlass.utils.blockscaled_layout as blockscaled_utils
    from cutlass.cute.nvgpu import cpasync, tcgen05
    from cutlass._mlir import ir

    mma_tiler_mnk = tuple(cfg["mma_tiler_mnk"])
    m, n, k, l = tuple(cfg["shape_mnkl"])
    m_work = max(int(m), int(mma_tiler_mnk[0]))
    n_work = max(int(n), int(mma_tiler_mnk[1]))
    mma_inst_shape_k = int(cfg["mma_inst_shape_k"])
    sf_vec_size = int(cfg["sf_vec_size"])
    num_ab_stage = int(cfg["num_ab_stage"])

    sf_dtype = cutlass.Float8E4M3FN
    ab_dtype = cutlass.Float4E2M1FN

    with ir.Context():
        with ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                loc = ir.Location.current
                mma_op = tcgen05.MmaMXF4NVF4Op(
                    sf_dtype,
                    (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
                    tcgen05.CtaGroup.ONE,
                    tcgen05.OperandSource.SMEM,
                )
                tiled_mma = cute.make_tiled_mma(mma_op, loc=loc)

                a_smem_layout_staged = sm100_utils.make_smem_layout_a(
                    tiled_mma, mma_tiler_mnk, ab_dtype, num_ab_stage, loc=loc
                )
                b_smem_layout_staged = sm100_utils.make_smem_layout_b(
                    tiled_mma, mma_tiler_mnk, ab_dtype, num_ab_stage, loc=loc
                )
                sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                    tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage, loc=loc
                )
                sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                    tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage, loc=loc
                )
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    mma_tiler_mnk,
                    sf_vec_size,
                    cute.slice_(
                        sfa_smem_layout_staged, (None, None, None, 0), loc=loc
                    ),
                    loc=loc,
                )
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    mma_tiler_mnk,
                    sf_vec_size,
                    cute.slice_(
                        sfb_smem_layout_staged, (None, None, None, 0), loc=loc
                    ),
                    loc=loc,
                )

                # Build global tensors and thread partitions (tCg*)
                mA_mkl = cute.make_tensor(
                    cute.make_ptr(
                        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
                    ),
                    cute.make_layout(
                        (m_work, k, l), stride=(k, 1, m_work * k), loc=loc
                    ),
                    loc=loc,
                )
                mB_nkl = cute.make_tensor(
                    cute.make_ptr(
                        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
                    ),
                    cute.make_layout(
                        (n_work, k, l), stride=(k, 1, n_work * k), loc=loc
                    ),
                    loc=loc,
                )
                mC_mnl = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float16, 0, cute.AddressSpace.gmem, assumed_align=16
                    ),
                    cute.make_layout(
                        (m_work, n_work, l),
                        stride=(n_work, 1, m_work * n_work),
                        loc=loc,
                    ),
                    loc=loc,
                )

                mSFA_mkl = cute.make_tensor(
                    cute.make_ptr(
                        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
                    ),
                    blockscaled_utils.tile_atom_to_shape_SF(
                        mA_mkl.layout.shape, sf_vec_size, loc=loc
                    ),
                    loc=loc,
                )
                mSFB_nkl = cute.make_tensor(
                    cute.make_ptr(
                        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
                    ),
                    blockscaled_utils.tile_atom_to_shape_SF(
                        mB_nkl.layout.shape, sf_vec_size, loc=loc
                    ),
                    loc=loc,
                )

                gA_mkl = cute.local_tile(
                    mA_mkl,
                    cute.slice_(mma_tiler_mnk, (None, 0, None), loc=loc),
                    (None, None, None),
                    loc=loc,
                )
                gB_nkl = cute.local_tile(
                    mB_nkl,
                    cute.slice_(mma_tiler_mnk, (0, None, None), loc=loc),
                    (None, None, None),
                    loc=loc,
                )
                gC_mnl = cute.local_tile(
                    mC_mnl,
                    cute.slice_(mma_tiler_mnk, (None, None, 0), loc=loc),
                    (0, 0, 0),
                    loc=loc,
                )
                gSFA_mkl = cute.local_tile(
                    mSFA_mkl,
                    cute.slice_(mma_tiler_mnk, (None, 0, None), loc=loc),
                    (None, None, None),
                    loc=loc,
                )
                gSFB_nkl = cute.local_tile(
                    mSFB_nkl,
                    cute.slice_(mma_tiler_mnk, (0, None, None), loc=loc),
                    (None, None, None),
                    loc=loc,
                )

                thr_mma = tiled_mma.get_slice(0)
                tCgA = thr_mma.partition_A(gA_mkl, loc=loc)
                tCgB = thr_mma.partition_B(gB_nkl, loc=loc)
                tCgC = thr_mma.partition_C(gC_mnl, loc=loc)
                tCgSFA = thr_mma.partition_A(gSFA_mkl, loc=loc)
                tCgSFB = thr_mma.partition_B(gSFB_nkl, loc=loc)

                # Build smem tensors + TMA atoms and run tma_partition
                sA = cute.make_tensor(
                    cute.make_ptr(
                        ab_dtype, 0, cute.AddressSpace.smem, assumed_align=16
                    ),
                    a_smem_layout_staged,
                    loc=loc,
                )
                sB = cute.make_tensor(
                    cute.make_ptr(
                        ab_dtype, 0, cute.AddressSpace.smem, assumed_align=16
                    ),
                    b_smem_layout_staged,
                    loc=loc,
                )
                sSFA = cute.make_tensor(
                    cute.make_ptr(
                        sf_dtype, 0, cute.AddressSpace.smem, assumed_align=16
                    ),
                    sfa_smem_layout_staged,
                    loc=loc,
                )
                sSFB = cute.make_tensor(
                    cute.make_ptr(
                        sf_dtype, 0, cute.AddressSpace.smem, assumed_align=16
                    ),
                    sfb_smem_layout_staged,
                    loc=loc,
                )

                cluster_layout_vmnk = cute.tiled_divide(
                    cute.make_layout((1, 1, 1), loc=loc),
                    (tiled_mma.thr_id.shape,),
                    loc=loc,
                )

                tma_atom_a, _ = cute.nvgpu.make_tiled_tma_atom_A(
                    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                    mA_mkl,
                    cute.slice_(a_smem_layout_staged, (None, None, None, 0), loc=loc),
                    mma_tiler_mnk,
                    tiled_mma,
                    cluster_layout_vmnk.shape,
                    loc=loc,
                )
                tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
                    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                    mB_nkl,
                    cute.slice_(b_smem_layout_staged, (None, None, None, 0), loc=loc),
                    mma_tiler_mnk,
                    tiled_mma,
                    cluster_layout_vmnk.shape,
                    loc=loc,
                )
                tma_atom_sfa, _ = cute.nvgpu.make_tiled_tma_atom_A(
                    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                    mSFA_mkl,
                    cute.slice_(
                        sfa_smem_layout_staged, (None, None, None, 0), loc=loc
                    ),
                    mma_tiler_mnk,
                    tiled_mma,
                    cluster_layout_vmnk.shape,
                    internal_type=cutlass.Int16,
                    loc=loc,
                )
                tma_atom_sfb, _ = cute.nvgpu.make_tiled_tma_atom_B(
                    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                    mSFB_nkl,
                    cute.slice_(
                        sfb_smem_layout_staged, (None, None, None, 0), loc=loc
                    ),
                    mma_tiler_mnk,
                    tiled_mma,
                    cluster_layout_vmnk.shape,
                    internal_type=cutlass.Int16,
                    loc=loc,
                )

                grouped_sA = cute.group_modes(sA, 0, 3, loc=loc)
                grouped_tCgA = cute.group_modes(tCgA, 0, 3, loc=loc)
                grouped_sB = cute.group_modes(sB, 0, 3, loc=loc)
                grouped_tCgB = cute.group_modes(tCgB, 0, 3, loc=loc)
                grouped_sSFA = cute.group_modes(sSFA, 0, 3, loc=loc)
                grouped_tCgSFA = cute.group_modes(tCgSFA, 0, 3, loc=loc)
                grouped_sSFB = cute.group_modes(sSFB, 0, 3, loc=loc)
                grouped_tCgSFB = cute.group_modes(tCgSFB, 0, 3, loc=loc)

                cpasync_partitions = {{
                    "status": "not_attempted",
                    "error": None,
                    "inputs": {{
                        "tma_atom_a": str(tma_atom_a),
                        "tma_atom_b": str(tma_atom_b),
                        "tma_atom_sfa": str(tma_atom_sfa),
                        "tma_atom_sfb": str(tma_atom_sfb),
                        "grouped_sA": _tensor_record(grouped_sA),
                        "grouped_tCgA": _tensor_record(grouped_tCgA),
                        "grouped_sB": _tensor_record(grouped_sB),
                        "grouped_tCgB": _tensor_record(grouped_tCgB),
                        "grouped_sSFA": _tensor_record(grouped_sSFA),
                        "grouped_tCgSFA": _tensor_record(grouped_tCgSFA),
                        "grouped_sSFB": _tensor_record(grouped_sSFB),
                        "grouped_tCgSFB": _tensor_record(grouped_tCgSFB),
                    }},
                }}
                if bool(cfg.get("try_cpasync_partition", False)):
                    cpasync_partitions["status"] = "attempted"
                    try:
                        tAsA, tAgA = cpasync.tma_partition(
                            tma_atom_a,
                            0,
                            cute.make_layout(1, loc=loc),
                            grouped_sA,
                            grouped_tCgA,
                            loc=loc,
                        )
                        tBsB, tBgB = cpasync.tma_partition(
                            tma_atom_b,
                            0,
                            cute.make_layout(1, loc=loc),
                            grouped_sB,
                            grouped_tCgB,
                            loc=loc,
                        )
                        tAsSFA, tAgSFA = cpasync.tma_partition(
                            tma_atom_sfa,
                            0,
                            cute.make_layout(1, loc=loc),
                            grouped_sSFA,
                            grouped_tCgSFA,
                            loc=loc,
                        )
                        tBsSFB, tBgSFB = cpasync.tma_partition(
                            tma_atom_sfb,
                            0,
                            cute.make_layout(1, loc=loc),
                            grouped_sSFB,
                            grouped_tCgSFB,
                            loc=loc,
                        )
                        tAsSFA_fz = cute.filter_zeros(tAsSFA, loc=loc)
                        tAgSFA_fz = cute.filter_zeros(tAgSFA, loc=loc)
                        tBsSFB_fz = cute.filter_zeros(tBsSFB, loc=loc)
                        tBgSFB_fz = cute.filter_zeros(tBgSFB, loc=loc)
                        cpasync_partitions["status"] = "ok"
                        cpasync_partitions["outputs"] = {{
                            "A": {{
                                "tAs": _layout_record(tAsA.layout),
                                "tAg": _layout_record(tAgA.layout),
                            }},
                            "B": {{
                                "tBs": _layout_record(tBsB.layout),
                                "tBg": _layout_record(tBgB.layout),
                            }},
                            "SFA": {{
                                "tAs": _layout_record(tAsSFA.layout),
                                "tAg": _layout_record(tAgSFA.layout),
                                "tAs_filter_zeros": _layout_record(tAsSFA_fz.layout),
                                "tAg_filter_zeros": _layout_record(tAgSFA_fz.layout),
                            }},
                            "SFB": {{
                                "tBs": _layout_record(tBsSFB.layout),
                                "tBg": _layout_record(tBgSFB.layout),
                                "tBs_filter_zeros": _layout_record(tBsSFB_fz.layout),
                                "tBg_filter_zeros": _layout_record(tBgSFB_fz.layout),
                            }},
                        }}
                    except Exception as e:
                        cpasync_partitions["status"] = "failed"
                        cpasync_partitions["error"] = f"{{type(e).__name__}}: {{e}}"

                # S2T copy partitions (tcgen05)
                tCtSFA = cute.make_tensor(
                    cute.make_ptr(
                        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
                    ),
                    tCtSFA_layout,
                    loc=loc,
                )
                tCtSFB = cute.make_tensor(
                    cute.make_ptr(
                        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
                    ),
                    tCtSFB_layout,
                    loc=loc,
                )
                copy_atom_s2t = cute.make_copy_atom(
                    tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype, loc=loc
                )
                tCsSFA_compact = cute.filter_zeros(sSFA, loc=loc)
                tCtSFA_compact = cute.filter_zeros(tCtSFA, loc=loc)
                tCsSFB_compact = cute.filter_zeros(sSFB, loc=loc)
                tCtSFB_compact = cute.filter_zeros(tCtSFB, loc=loc)

                tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(
                    copy_atom_s2t, tCtSFA_compact, loc=loc
                )
                thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
                tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(
                    tCsSFA_compact, loc=loc
                )
                tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                    tiled_copy_s2t_sfa, tCsSFA_compact_s2t_, loc=loc
                )
                tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(
                    tCtSFA_compact, loc=loc
                )

                tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(
                    copy_atom_s2t, tCtSFB_compact, loc=loc
                )
                thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
                tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(
                    tCsSFB_compact, loc=loc
                )
                tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                    tiled_copy_s2t_sfb, tCsSFB_compact_s2t_, loc=loc
                )
                tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(
                    tCtSFB_compact, loc=loc
                )

                return {{
                    "a_smem_layout_staged": {{
                        "outer": _layout_record(a_smem_layout_staged.outer),
                        "inner": _layout_record(a_smem_layout_staged.inner),
                    }},
                    "b_smem_layout_staged": {{
                        "outer": _layout_record(b_smem_layout_staged.outer),
                        "inner": _layout_record(b_smem_layout_staged.inner),
                    }},
                    "sfa_smem_layout_staged": _layout_record(sfa_smem_layout_staged),
                    "sfb_smem_layout_staged": _layout_record(sfb_smem_layout_staged),
                    "tCtSFA_layout": _layout_record(tCtSFA_layout),
                    "tCtSFB_layout": _layout_record(tCtSFB_layout),
                    "thr_mma_partitions": {{
                        "tCgA": _layout_record(tCgA.layout),
                        "tCgB": _layout_record(tCgB.layout),
                        "tCgC": _layout_record(tCgC.layout),
                        "tCgSFA": _layout_record(tCgSFA.layout),
                        "tCgSFB": _layout_record(tCgSFB.layout),
                    }},
                    "cpasync_tma_partitions": cpasync_partitions,
                    "tcgen05_s2t_partitions": {{
                        "SFA": {{
                            "tCs_compact": _layout_record(tCsSFA_compact.layout),
                            "tCt_compact": _layout_record(tCtSFA_compact.layout),
                            "tCs_partition_S": _layout_record(tCsSFA_compact_s2t_.layout),
                            "tCs_desc_tensor": _layout_record(tCsSFA_compact_s2t.layout),
                            "tCt_partition_D": _layout_record(tCtSFA_compact_s2t.layout),
                        }},
                        "SFB": {{
                            "tCs_compact": _layout_record(tCsSFB_compact.layout),
                            "tCt_compact": _layout_record(tCtSFB_compact.layout),
                            "tCs_partition_S": _layout_record(tCsSFB_compact_s2t_.layout),
                            "tCs_desc_tensor": _layout_record(tCsSFB_compact_s2t.layout),
                            "tCt_partition_D": _layout_record(tCtSFB_compact_s2t.layout),
                        }},
                    }},
                }}

if context_mode == "with_mlir_context":
    from cutlass._mlir import ir
    with ir.Context() as ctx:
        with ir.Location.unknown(ctx):
            print(json.dumps(_extract()))
elif context_mode == "with_explicit_loc":
    print(json.dumps(_extract_with_explicit_loc()))
elif context_mode == "with_module_ip":
    print(json.dumps(_extract_with_module_ip()))
else:
    print(json.dumps(_extract()))
"""


@app.function(
    image=image,
    gpu="B200",
    timeout=900,
    volumes={CACHE_MOUNT_PATH: cache_volume},
)
def _dump_remote(cfg: dict[str, object]) -> dict[str, object]:
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["NVFP4_LAYOUT_CFG"] = json.dumps(cfg)
    # Work around allocator aborts seen with CUTLASS DSL in some container setups.
    env.setdefault("GLIBC_TUNABLES", "glibc.malloc.tcache_count=0")
    env.setdefault("XDG_CACHE_HOME", f"{CACHE_MOUNT_PATH}/xdg")
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{CACHE_MOUNT_PATH}/torchinductor")
    env.setdefault("TRITON_CACHE_DIR", f"{CACHE_MOUNT_PATH}/triton")
    env.setdefault("CUDA_CACHE_PATH", f"{CACHE_MOUNT_PATH}/cuda")

    attempts: list[dict[str, object]] = []
    if bool(cfg.get("probe_modes", False)):
        modes = ["plain", "with_mlir_context", "with_explicit_loc", "with_module_ip"]
    else:
        modes = ["with_module_ip"]

    for mode in modes:
        proc = subprocess.run(
            [sys.executable, "-c", _child_code(mode)],
            text=True,
            capture_output=True,
            env=env,
            timeout=600,
        )
        rec = {
            "mode": mode,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
        attempts.append(rec)
        if proc.returncode == 0:
            try:
                payload = json.loads(proc.stdout)
                return {"ok": True, "mode": mode, "payload": payload, "attempts": attempts}
            except json.JSONDecodeError:
                pass

    if not bool(cfg.get("probe_modes", False)):
        # Fallback diagnostics only when the stable mode fails.
        for mode in ("plain", "with_mlir_context", "with_explicit_loc"):
            proc = subprocess.run(
                [sys.executable, "-c", _child_code(mode)],
                text=True,
                capture_output=True,
                env=env,
                timeout=600,
            )
            attempts.append(
                {
                    "mode": mode,
                    "returncode": proc.returncode,
                    "stdout_tail": proc.stdout[-2000:],
                    "stderr_tail": proc.stderr[-2000:],
                }
            )

    return {"ok": False, "attempts": attempts}


def main() -> None:
    args = _parse_args()
    if args.gpu != "B200":
        raise SystemExit("dump_internal_layout_tuples_modal_b200.py is B200-only.")

    mma_tiler = [int(x.strip()) for x in args.mma_tiler.split(",")]
    shape = [int(x.strip()) for x in args.shape.split(",")]

    if len(mma_tiler) != 3:
        raise SystemExit(f"--mma-tiler expects 3 ints, got: {args.mma_tiler!r}")
    if len(shape) != 4:
        raise SystemExit(f"--shape expects 4 ints, got: {args.shape!r}")

    cfg = {
        "shape_mnkl": shape,
        "mma_tiler_mnk": mma_tiler,
        "mma_inst_shape_k": args.mma_inst_shape_k,
        "sf_vec_size": args.sf_vec_size,
        "num_ab_stage": args.num_ab_stage,
        "threads_per_cta": args.threads_per_cta,
        "try_cpasync_partition": bool(args.try_cpasync_partition),
        "probe_modes": bool(args.probe_modes),
    }

    remote_fn = modal.Function.from_name(args.app_name, args.function_name)
    result = remote_fn.remote(cfg)

    text = json.dumps(result, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"[INFO] Wrote: {args.json_out}")

    print(text)


if __name__ == "__main__":
    main()
