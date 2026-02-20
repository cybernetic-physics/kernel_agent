#!/usr/bin/env python3
"""Generate layout snapshot manifest for nvfp4_group_gemm (wagmi_v6 defaults)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.nvfp4_layout_model import (
    KernelConfig,
    build_layout_bundle,
    default_kernel_config,
    parse_mma_tiler,
    parse_shape_spec,
)
from tools.tract_layout_utils import layout_digest


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shape",
        default="80,4096,7168,1",
        help="Snapshot shape m,n,k,l.",
    )
    p.add_argument(
        "--kernel-file",
        default="kernels/nvfp4_group_gemm/wagmi_v6.py",
        help="Reference kernel file path for metadata.",
    )
    p.add_argument("--mma-tiler", default="128,128,256")
    p.add_argument("--mma-inst-shape-k", type=int, default=64)
    p.add_argument("--sf-vec-size", type=int, default=16)
    p.add_argument("--num-ab-stage", type=int, default=1)
    p.add_argument("--threads-per-cta", type=int, default=128)
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--md-out", type=Path)
    return p.parse_args()


def _digest_or_repr(layout: Any) -> dict[str, Any]:
    try:
        return {"digest": layout_digest(layout), "repr": str(layout)}
    except Exception as exc:
        return {
            "digest": None,
            "repr": str(layout),
            "error": f"digest_failed: {type(exc).__name__}: {exc}",
        }


def _markdown_report(manifest: dict[str, Any]) -> str:
    lines = []
    lines.append("# Layout Snapshot")
    lines.append("")
    lines.append(f"- Generated: `{manifest['generated_at_utc']}`")
    lines.append(f"- Kernel file: `{manifest['kernel_file']}`")
    shape = manifest["shape_mnkl"]
    lines.append(
        f"- Shape: `m={shape['m']} n={shape['n']} k={shape['k']} l={shape['l']}`"
    )
    lines.append("")
    lines.append("| Layout | Hash | Rank | Size | Cosize | Tractable |")
    lines.append("|---|---|---:|---:|---:|:---:|")

    for name, record in manifest["layouts"].items():
        dig = record.get("digest")
        if not dig:
            lines.append(f"| `{name}` | `n/a` | - | - | - | - |")
            continue
        lines.append(
            f"| `{name}` | `{dig['hash'][:12]}` | {dig['flat_rank']} | "
            f"{dig['flat_size']} | {dig['flat_cosize']} | {dig['tractable']} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    shape = parse_shape_spec(args.shape)

    cfg = default_kernel_config()
    if (
        args.mma_tiler != "128,128,256"
        or args.mma_inst_shape_k != cfg.mma_inst_shape_k
        or args.sf_vec_size != cfg.sf_vec_size
        or args.num_ab_stage != cfg.num_ab_stage
        or args.threads_per_cta != cfg.threads_per_cta
    ):
        cfg = KernelConfig(
            mma_tiler_mnk=parse_mma_tiler(args.mma_tiler),
            mma_inst_shape_k=args.mma_inst_shape_k,
            sf_vec_size=args.sf_vec_size,
            num_ab_stage=args.num_ab_stage,
            threads_per_cta=args.threads_per_cta,
        )

    try:
        bundle = build_layout_bundle(shape, config=cfg)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    layouts = {
        "a_smem_layout_staged.outer": _digest_or_repr(bundle["a_smem_layout_staged"].outer),
        "b_smem_layout_staged.outer": _digest_or_repr(bundle["b_smem_layout_staged"].outer),
        "sfa_smem_layout_staged": _digest_or_repr(bundle["sfa_smem_layout_staged"]),
        "sfb_smem_layout_staged": _digest_or_repr(bundle["sfb_smem_layout_staged"]),
        "sfa_gmem_layout.helper": _digest_or_repr(bundle["sfa_gmem_layout_helper"]),
        "sfb_gmem_layout.helper": _digest_or_repr(bundle["sfb_gmem_layout_helper"]),
        "sfa_gmem_layout.explicit": _digest_or_repr(bundle["sfa_gmem_layout_explicit"]),
        "sfb_gmem_layout.explicit": _digest_or_repr(bundle["sfb_gmem_layout_explicit"]),
        "acc_tmem_layout": _digest_or_repr(bundle["tCtAcc_layout"]),
        "sfa_tmem_layout": _digest_or_repr(bundle["tCtSFA_layout"]),
        "sfb_tmem_layout": _digest_or_repr(bundle["tCtSFB_layout"]),
        "c_tile_layout": _digest_or_repr(bundle["gC_tile_layout"]),
        "epilogue_thread_layout": _digest_or_repr(bundle["epilogue_thread_layout"]),
        "epilogue_value_layout": _digest_or_repr(bundle["epilogue_value_layout"]),
        "epilogue_partition_layout": _digest_or_repr(bundle["epilogue_partition_layout"]),
    }

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "kernel_file": args.kernel_file,
        "shape_mnkl": {"m": shape[0], "n": shape[1], "k": shape[2], "l": shape[3]},
        "kernel_config": bundle["kernel_config"],
        "notes": {
            "a_smem_layout_staged.inner": str(bundle["a_smem_layout_staged"].inner),
            "b_smem_layout_staged.inner": str(bundle["b_smem_layout_staged"].inner),
            "acc_cols": bundle["acc_cols"],
            "sfa_cols": bundle["sfa_cols"],
            "sfb_cols": bundle["sfb_cols"],
            "total_cols": bundle["total_cols"],
            "alloc_cols": bundle["alloc_cols"],
        },
        "layouts": layouts,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote snapshot JSON: {args.json_out}")

    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_markdown_report(manifest), encoding="utf-8")
        print(f"[INFO] Wrote snapshot markdown: {args.md_out}")


if __name__ == "__main__":
    main()
