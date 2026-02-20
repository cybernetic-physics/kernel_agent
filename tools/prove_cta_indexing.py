#!/usr/bin/env python3
"""Proof harness for CTA linear index mapping rewrites."""

from __future__ import annotations

import argparse
import bisect
import random
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.nvfp4_layout_model import (
    benchmark_problem_groups,
    build_cta_mn_list,
    parse_shape_spec,
)


def old_mapping(bidz: int, cta_mn_list: Sequence[tuple[int, int]]) -> tuple[int, int, int]:
    """Exact translation of the legacy O(G) loop mapping."""
    group_idx = 0
    found = False
    coord_x = 0
    coord_y = 0
    cta_rest = bidz

    for cta_m, cta_n in cta_mn_list:
        tiles = cta_m * cta_n
        if cta_rest >= tiles:
            group_idx += 1
            cta_rest -= tiles
        else:
            if not found:
                coord_y = cta_rest // cta_m
                coord_x = cta_rest % cta_m
                cta_rest -= tiles
                found = True

    return group_idx, coord_x, coord_y


def build_prefix_sums(cta_mn_list: Sequence[tuple[int, int]]) -> list[int]:
    prefix = [0]
    for cta_m, cta_n in cta_mn_list:
        prefix.append(prefix[-1] + cta_m * cta_n)
    return prefix


def new_mapping(
    bidz: int,
    cta_mn_list: Sequence[tuple[int, int]],
    prefix_sums: Sequence[int],
) -> tuple[int, int, int]:
    """Prefix-sum + binary search mapping."""
    group_idx = bisect.bisect_right(prefix_sums, bidz) - 1
    if group_idx < 0 or group_idx >= len(cta_mn_list):
        raise IndexError(
            f"bidz {bidz} out of range [0, {prefix_sums[-1]}) for prefix={prefix_sums}"
        )

    local_tile = bidz - prefix_sums[group_idx]
    cta_m, _cta_n = cta_mn_list[group_idx]
    coord_y = local_tile // cta_m
    coord_x = local_tile % cta_m
    return group_idx, coord_x, coord_y


def _verify_case(cta_mn_list: Sequence[tuple[int, int]], case_name: str) -> tuple[int, int]:
    prefix = build_prefix_sums(cta_mn_list)
    total = prefix[-1]
    checked = 0

    for bidz in range(total):
        old = old_mapping(bidz, cta_mn_list)
        new = new_mapping(bidz, cta_mn_list, prefix)
        checked += 1
        if old != new:
            raise AssertionError(
                "CTA mapping mismatch\n"
                f"  case={case_name}\n"
                f"  bidz={bidz}\n"
                f"  cta_mn_list={cta_mn_list}\n"
                f"  prefix={prefix}\n"
                f"  old={old}\n"
                f"  new={new}"
            )

    return total, checked


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--problem-size",
        action="append",
        default=[],
        help="Custom shape m,n,k,l (repeatable).",
    )
    p.add_argument("--tile-m", type=int, default=128)
    p.add_argument("--tile-n", type=int, default=128)
    p.add_argument("--random-cases", type=int, default=200)
    p.add_argument("--max-groups", type=int, default=12)
    p.add_argument("--max-tiles-per-axis", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-benchmarks",
        action="store_true",
        help="Skip benchmark distribution cases from context_dump.",
    )
    p.add_argument(
        "--emit-snippet",
        action="store_true",
        help="Print a drop-in kernel-side mapping snippet.",
    )
    return p.parse_args()


def _emit_snippet() -> None:
    print("\nSuggested kernel-side mapping snippet (prefix + binary search):")
    print(
        """
# Inputs:
#   bidz: CTA linear z-index
#   tensor_of_cta_prefix: tensor/list of len(num_groups + 1), prefix[0] == 0
#   tensor_of_problem_sizes[group, 0] -> M
left = cutlass.Int32(0)
right = num_groups
while left < right:
    mid = (left + right) // 2
    if tensor_of_cta_prefix[mid + 1] <= bidz:
        left = mid + 1
    else:
        right = mid
group_idx = left
local_tile = bidz - tensor_of_cta_prefix[group_idx]
cta_m = ceil_div(tensor_of_problem_sizes[group_idx, 0], mma_tiler_mnk[0])
coord_y = local_tile // cta_m
coord_x = local_tile % cta_m
        """.strip()
    )


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    total_cases = 0
    total_bidz = 0

    if not args.no_benchmarks:
        for idx, group_shapes in enumerate(benchmark_problem_groups(), start=1):
            cta_mn = build_cta_mn_list(group_shapes, tile_m=args.tile_m, tile_n=args.tile_n)
            total, checked = _verify_case(cta_mn, case_name=f"benchmark_group_{idx}")
            total_cases += 1
            total_bidz += checked
            print(
                f"[PASS] benchmark_group_{idx}: groups={len(cta_mn)} total_ctas={total} checked={checked}"
            )

    if args.problem_size:
        shapes = [parse_shape_spec(s) for s in args.problem_size]
        cta_mn = build_cta_mn_list(shapes, tile_m=args.tile_m, tile_n=args.tile_n)
        total, checked = _verify_case(cta_mn, case_name="custom_problem_sizes")
        total_cases += 1
        total_bidz += checked
        print(
            f"[PASS] custom_problem_sizes: groups={len(cta_mn)} total_ctas={total} checked={checked}"
        )

    for case_idx in range(args.random_cases):
        groups = rng.randint(1, max(1, args.max_groups))
        cta_mn = [
            (
                rng.randint(1, max(1, args.max_tiles_per_axis)),
                rng.randint(1, max(1, args.max_tiles_per_axis)),
            )
            for _ in range(groups)
        ]
        total, checked = _verify_case(cta_mn, case_name=f"random_{case_idx}")
        total_cases += 1
        total_bidz += checked

    print(
        f"[PASS] CTA indexing proofs complete: cases={total_cases}, checked_bidz={total_bidz}"
    )

    if args.emit_snippet:
        _emit_snippet()


if __name__ == "__main__":
    main()
