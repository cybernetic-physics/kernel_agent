#!/usr/bin/env python3
"""Exploratory TMEM segment packing search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.nvfp4_layout_model import (
    KernelConfig,
    build_layout_bundle,
    parse_mma_tiler,
    parse_shape_spec,
    tmem_order_search,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shape",
        help="Compute acc/sfa/sfb cols from shape m,n,k,l via CUTLASS.",
    )
    p.add_argument("--acc-cols", type=int)
    p.add_argument("--sfa-cols", type=int)
    p.add_argument("--sfb-cols", type=int)
    p.add_argument("--segment-alignment", type=int, default=1)
    p.add_argument("--top-k", type=int, default=6)

    p.add_argument("--mma-tiler", default="128,128,256")
    p.add_argument("--mma-inst-shape-k", type=int, default=64)
    p.add_argument("--sf-vec-size", type=int, default=16)
    p.add_argument("--num-ab-stage", type=int, default=1)
    p.add_argument("--threads-per-cta", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.shape:
        cfg = KernelConfig(
            mma_tiler_mnk=parse_mma_tiler(args.mma_tiler),
            mma_inst_shape_k=args.mma_inst_shape_k,
            sf_vec_size=args.sf_vec_size,
            num_ab_stage=args.num_ab_stage,
            threads_per_cta=args.threads_per_cta,
        )
        try:
            bundle = build_layout_bundle(parse_shape_spec(args.shape), config=cfg)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        acc_cols = int(bundle["acc_cols"])
        sfa_cols = int(bundle["sfa_cols"])
        sfb_cols = int(bundle["sfb_cols"])
    else:
        if args.acc_cols is None or args.sfa_cols is None or args.sfb_cols is None:
            raise SystemExit(
                "Provide either --shape or all of --acc-cols --sfa-cols --sfb-cols"
            )
        acc_cols = args.acc_cols
        sfa_cols = args.sfa_cols
        sfb_cols = args.sfb_cols

    rows = tmem_order_search(
        acc_cols,
        sfa_cols,
        sfb_cols,
        segment_alignment=args.segment_alignment,
    )

    print(
        f"Input columns: acc={acc_cols} sfa={sfa_cols} sfb={sfb_cols} "
        f"(alignment={args.segment_alignment})"
    )
    print("rank | order | used | alloc | slack")
    for idx, row in enumerate(rows[: max(1, args.top_k)], start=1):
        print(
            f"{idx:>4} | {row['order']} | {row['used_cols']:>4} | "
            f"{row['alloc_cols']:>5} | {row['slack_cols']:>5}"
        )


if __name__ == "__main__":
    main()
