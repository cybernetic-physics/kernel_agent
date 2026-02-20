#!/usr/bin/env python3
"""Proof + simplification report for blockscaled SFA/SFB layouts."""

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
    benchmark_problem_sizes_flat,
    build_layout_bundle,
    parse_mma_tiler,
    parse_shape_spec,
)
from tools.tract_layout_utils import (
    assert_layout_equivalent,
    canonicalize_layout,
    canonicalize_layout_report,
    collect_zero_stride_paths,
    is_tractable,
    layout_digest,
    unique_offsets_count,
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
        help="Skip benchmark-distribution shapes.",
    )
    p.add_argument("--mma-tiler", default="128,128,256")
    p.add_argument("--mma-inst-shape-k", type=int, default=64)
    p.add_argument("--sf-vec-size", type=int, default=16)
    p.add_argument("--num-ab-stage", type=int, default=1)
    p.add_argument("--threads-per-cta", type=int, default=128)
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exact-threshold", type=int, default=65536)
    p.add_argument("--json-out", type=Path)
    return p.parse_args()


def _default_shapes(include_benchmarks: bool) -> list[tuple[int, int, int, int]]:
    shapes = [
        (64, 64, 256, 1),
        (64, 128, 256, 1),
        (128, 64, 512, 1),
        (128, 128, 512, 1),
    ]
    if include_benchmarks:
        shapes.extend(benchmark_problem_sizes_flat())

    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []
    for shape in shapes:
        if shape not in seen:
            seen.add(shape)
            out.append(shape)
    return out


def _analyze_layout(
    name: str,
    layout: Any,
    *,
    samples: int,
    seed: int,
    exact_threshold: int,
) -> dict[str, Any]:
    tractable = bool(is_tractable(layout))
    if not tractable:
        raise AssertionError(f"{name}: non-tractable layout, refusing to continue")

    canonical = canonicalize_layout(layout, coalesce_mode="tract")
    eq_stats = assert_layout_equivalent(
        layout,
        canonical,
        samples=samples,
        exhaustive_if_small=True,
        exhaustive_threshold=exact_threshold,
        seed=seed,
    )

    digest = layout_digest(layout)
    zero_stride_paths = collect_zero_stride_paths(digest["stride"])
    unique_count, unique_exact, logical_size = unique_offsets_count(
        layout,
        samples=samples,
        exact_if_small=True,
        exact_threshold=exact_threshold,
        seed=seed,
    )
    duplication_factor = (
        float(logical_size) / float(unique_count) if unique_count > 0 else float("inf")
    )
    filter_zeros_required = len(zero_stride_paths) > 0 and duplication_factor > 1.0

    report = {
        "name": name,
        "tractable": tractable,
        "canonicalization": canonicalize_layout_report(layout),
        "equivalence": {
            "checked": eq_stats.checked,
            "exhaustive": eq_stats.exhaustive,
        },
        "broadcast_analysis": {
            "zero_stride_paths": [list(p) for p in zero_stride_paths],
            "unique_offsets": unique_count,
            "unique_offsets_exact": unique_exact,
            "logical_size": logical_size,
            "duplication_factor": duplication_factor,
            "filter_zeros_required": filter_zeros_required,
        },
        "digest": digest,
    }
    return report


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

    # Stable de-dup preserving order.
    dedup_shapes: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for shape in shapes:
        if shape not in seen:
            seen.add(shape)
            dedup_shapes.append(shape)

    all_reports: list[dict[str, Any]] = []

    for shape in dedup_shapes:
        try:
            bundle = build_layout_bundle(shape, config=config)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

        per_shape: dict[str, Any] = {
            "shape": {
                "m": shape[0],
                "n": shape[1],
                "k": shape[2],
                "l": shape[3],
            },
            "layouts": [],
            "rewrite_candidates": [],
        }

        layout_items = [
            ("SFA.gmem.helper", bundle["sfa_gmem_layout_helper"]),
            ("SFA.gmem.explicit", bundle["sfa_gmem_layout_explicit"]),
            ("SFB.gmem.helper", bundle["sfb_gmem_layout_helper"]),
            ("SFB.gmem.explicit", bundle["sfb_gmem_layout_explicit"]),
            ("SFA.smem.staged", bundle["sfa_smem_layout_staged"]),
            ("SFB.smem.staged", bundle["sfb_smem_layout_staged"]),
            ("SFA.tmem", bundle["tCtSFA_layout"]),
            ("SFB.tmem", bundle["tCtSFB_layout"]),
        ]

        for name, layout in layout_items:
            rep = _analyze_layout(
                name,
                layout,
                samples=args.samples,
                seed=args.seed,
                exact_threshold=args.exact_threshold,
            )
            per_shape["layouts"].append(rep)

            bcast = rep["broadcast_analysis"]
            print(
                f"[PASS] {name} shape={shape} "
                f"tractable={rep['tractable']} "
                f"checked={rep['equivalence']['checked']} "
                f"dup={bcast['duplication_factor']:.4f} "
                f"zeros={len(bcast['zero_stride_paths'])} "
                f"filter_zeros_required={bcast['filter_zeros_required']}"
            )

        # Explicit vs helper rewrite candidates
        eq_sfa = assert_layout_equivalent(
            bundle["sfa_gmem_layout_helper"],
            bundle["sfa_gmem_layout_explicit"],
            samples=args.samples,
            exhaustive_if_small=True,
            exhaustive_threshold=args.exact_threshold,
            seed=args.seed,
        )
        eq_sfb = assert_layout_equivalent(
            bundle["sfb_gmem_layout_helper"],
            bundle["sfb_gmem_layout_explicit"],
            samples=args.samples,
            exhaustive_if_small=True,
            exhaustive_threshold=args.exact_threshold,
            seed=args.seed,
        )
        per_shape["rewrite_candidates"].append(
            {
                "candidate": "SFA gmem: helper <-> explicit atom tile_to_shape",
                "equivalent": True,
                "checked": eq_sfa.checked,
                "exhaustive": eq_sfa.exhaustive,
            }
        )
        per_shape["rewrite_candidates"].append(
            {
                "candidate": "SFB gmem: helper <-> explicit atom tile_to_shape",
                "equivalent": True,
                "checked": eq_sfb.checked,
                "exhaustive": eq_sfb.exhaustive,
            }
        )

        all_reports.append(per_shape)

    safe_transformations = [
        "Tract coalesce canonicalization of SFA/SFB layouts (proof-guarded).",
        "Relative/pure CuTe coalesce for readability only after tract equivalence checks.",
        "Helper <-> explicit atom-based SFA/SFB gmem layout rewrites when equivalence proof passes.",
    ]

    action = {
        "safe_transformations": safe_transformations,
        "rewrite_candidates": [
            "Prefer layout rewrites only when helper-vs-explicit equivalence passes for all target shapes.",
            "Retain `filter_zeros` in copy partitions for layouts with duplication_factor>1 and zero-stride leaves.",
        ],
    }

    result = {
        "config": {
            "mma_tiler_mnk": list(config.mma_tiler_mnk),
            "mma_inst_shape_k": config.mma_inst_shape_k,
            "sf_vec_size": config.sf_vec_size,
            "num_ab_stage": config.num_ab_stage,
            "threads_per_cta": config.threads_per_cta,
        },
        "num_shapes": len(all_reports),
        "reports": all_reports,
        "actionable_recommendations": action,
    }

    print("\nActionable recommendations:")
    for item in action["safe_transformations"]:
        print(f"  - SAFE: {item}")
    for item in action["rewrite_candidates"]:
        print(f"  - CANDIDATE: {item}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[INFO] Wrote proof report: {args.json_out}")


if __name__ == "__main__":
    main()
