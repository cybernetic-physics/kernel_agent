#!/usr/bin/env python3
"""Dump concrete shape/stride tuples for internal nvfp4_group_gemm layouts."""

from __future__ import annotations

import argparse
import json
import sys
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shape",
        default="80,4096,7168,1",
        help="Shape m,n,k,l used to instantiate the layout bundle.",
    )
    p.add_argument("--mma-tiler", default="128,128,256")
    p.add_argument("--mma-inst-shape-k", type=int, default=64)
    p.add_argument("--sf-vec-size", type=int, default=16)
    p.add_argument("--num-ab-stage", type=int, default=1)
    p.add_argument("--threads-per-cta", type=int, default=128)
    p.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write full JSON output.",
    )
    return p.parse_args()


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, bool, int, float)) or x is None:
        return x
    if isinstance(x, (tuple, list)):
        return [_to_jsonable(v) for v in x]
    try:
        return int(x)
    except Exception:
        return str(x)


def _layout_record(layout: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"repr": str(layout)}
    if hasattr(layout, "shape"):
        out["shape"] = _to_jsonable(layout.shape)
    if hasattr(layout, "stride"):
        out["stride"] = _to_jsonable(layout.stride)
    return out


def _build_config(args: argparse.Namespace) -> KernelConfig:
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
    return cfg


def main() -> None:
    args = _parse_args()
    shape = parse_shape_spec(args.shape)
    cfg = _build_config(args)

    try:
        bundle = build_layout_bundle(shape, config=cfg)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    report = {
        "shape_mnkl": {"m": shape[0], "n": shape[1], "k": shape[2], "l": shape[3]},
        "kernel_config": bundle["kernel_config"],
        "a_smem_layout_staged": {
            "outer": _layout_record(bundle["a_smem_layout_staged"].outer),
            "inner": _layout_record(bundle["a_smem_layout_staged"].inner),
        },
        "b_smem_layout_staged": {
            "outer": _layout_record(bundle["b_smem_layout_staged"].outer),
            "inner": _layout_record(bundle["b_smem_layout_staged"].inner),
        },
        "sfa_smem_layout_staged": _layout_record(bundle["sfa_smem_layout_staged"]),
        "sfb_smem_layout_staged": _layout_record(bundle["sfb_smem_layout_staged"]),
        "tCtSFA_layout": _layout_record(bundle["tCtSFA_layout"]),
        "tCtSFB_layout": _layout_record(bundle["tCtSFB_layout"]),
    }

    text = json.dumps(report, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"[INFO] Wrote: {args.json_out}")

    print(text)


if __name__ == "__main__":
    main()
