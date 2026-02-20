#!/usr/bin/env python3
"""TMEM footprint report for nvfp4_group_gemm (wagmi_v6-style layouts)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.nvfp4_layout_model import (
    KernelConfig,
    benchmark_problem_sizes_flat,
    build_layout_bundle,
    parse_mma_tiler,
    parse_shape_spec,
    round_tmem_alloc_cols,
    aligned_round_up,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Shape m,n,k,l (repeatable).",
    )
    p.add_argument(
        "--no-benchmarks",
        action="store_true",
        help="Do not include benchmark distribution shapes from context_dump.",
    )
    p.add_argument("--mma-tiler", default="128,128,256")
    p.add_argument("--mma-inst-shape-k", type=int, default=64)
    p.add_argument("--sf-vec-size", type=int, default=16)
    p.add_argument("--num-ab-stage", type=int, default=1)
    p.add_argument("--threads-per-cta", type=int, default=128)
    p.add_argument(
        "--alignment",
        type=int,
        default=16,
        help="Alignment used for generic round-up recommendation.",
    )
    p.add_argument("--csv-out", type=Path)
    p.add_argument("--json-out", type=Path)
    return p.parse_args()


def _default_shapes(include_benchmarks: bool) -> list[tuple[int, int, int, int]]:
    shapes = [(64, 64, 256, 1), (128, 128, 512, 1)]
    if include_benchmarks:
        shapes.extend(benchmark_problem_sizes_flat())
    # Keep stable order while deduplicating.
    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []
    for shape in shapes:
        if shape not in seen:
            seen.add(shape)
            out.append(shape)
    return out


def main() -> None:
    args = _parse_args()

    config = KernelConfig(
        mma_tiler_mnk=parse_mma_tiler(args.mma_tiler),
        mma_inst_shape_k=args.mma_inst_shape_k,
        sf_vec_size=args.sf_vec_size,
        num_ab_stage=args.num_ab_stage,
        threads_per_cta=args.threads_per_cta,
    )

    shapes = _default_shapes(include_benchmarks=not args.no_benchmarks)
    if args.shape:
        shapes.extend(parse_shape_spec(s) for s in args.shape)

    rows: list[dict[str, int | str]] = []
    for shape in shapes:
        try:
            bundle = build_layout_bundle(shape, config=config)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

        acc_cols = int(bundle["acc_cols"])
        sfa_cols = int(bundle["sfa_cols"])
        sfb_cols = int(bundle["sfb_cols"])
        total_cols = int(bundle["total_cols"])
        alloc_pow2 = int(round_tmem_alloc_cols(total_cols))
        alloc_aligned = int(aligned_round_up(total_cols, args.alignment))

        rows.append(
            {
                "shape": f"{shape[0]},{shape[1]},{shape[2]},{shape[3]}",
                "acc_cols": acc_cols,
                "sfa_cols": sfa_cols,
                "sfb_cols": sfb_cols,
                "total_cols": total_cols,
                "recommended_pow2": alloc_pow2,
                "recommended_aligned": alloc_aligned,
            }
        )

    # Human report
    print(
        "shape(m,n,k,l) | acc_cols | sfa_cols | sfb_cols | total_cols | "
        "recommended_pow2 | recommended_aligned"
    )
    for row in rows:
        print(
            f"{row['shape']} | {row['acc_cols']} | {row['sfa_cols']} | {row['sfb_cols']} "
            f"| {row['total_cols']} | {row['recommended_pow2']} | {row['recommended_aligned']}"
        )

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "shape",
                    "acc_cols",
                    "sfa_cols",
                    "sfb_cols",
                    "total_cols",
                    "recommended_pow2",
                    "recommended_aligned",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Wrote CSV: {args.csv_out}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "mma_tiler_mnk": list(config.mma_tiler_mnk),
                "mma_inst_shape_k": config.mma_inst_shape_k,
                "sf_vec_size": config.sf_vec_size,
                "num_ab_stage": config.num_ab_stage,
                "threads_per_cta": config.threads_per_cta,
                "alignment": args.alignment,
            },
            "rows": rows,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
