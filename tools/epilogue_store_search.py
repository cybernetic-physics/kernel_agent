#!/usr/bin/env python3
"""Search epilogue thread/value layout candidates for coalesced stores."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass
class Candidate:
    thread_layout: tuple[int, int]
    value_layout: tuple[int, int]
    rounds: int
    covered: int
    complete_coverage: bool
    overlap_free: bool
    contiguity_ratio: float
    predicate_vectors: int
    coalescing_bonus: int
    score: float


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a in range(1, int(math.sqrt(n)) + 1):
        if n % a == 0:
            b = n // a
            out.append((a, b))
            if a != b:
                out.append((b, a))
    out.sort()
    return out


def _evaluate(tile_m: int, tile_n: int, threads: int, th_m: int, th_n: int, vec: int) -> Candidate:
    total = tile_m * tile_n
    rounds = math.ceil(total / (threads * vec))

    covered_coords: set[tuple[int, int]] = set()
    contiguous_segments = 0
    total_segments = 0
    predicate_vectors = 0

    for tid in range(threads):
        for r in range(rounds):
            base = (r * threads + tid) * vec
            coords: list[tuple[int, int]] = []
            for lane in range(vec):
                lin = base + lane
                if lin >= total:
                    continue
                row = lin // tile_n
                col = lin % tile_n
                coords.append((row, col))
                covered_coords.add((row, col))

            if not coords:
                continue

            total_segments += 1
            contiguous = True
            for i in range(1, len(coords)):
                pr, pc = coords[i - 1]
                cr, cc = coords[i]
                if cr != pr or cc != pc + 1:
                    contiguous = False
                    break
            if contiguous:
                contiguous_segments += 1

            if len(coords) < vec:
                predicate_vectors += 1

    complete_coverage = len(covered_coords) == total
    overlap_free = len(covered_coords) == total
    contiguity_ratio = (
        float(contiguous_segments) / float(total_segments) if total_segments > 0 else 0.0
    )

    coalescing_bonus = 0
    lane_span = th_n * vec
    if lane_span <= tile_n and tile_n % lane_span == 0:
        coalescing_bonus += 1
    if vec in (2, 4, 8, 16):
        coalescing_bonus += 1

    score = (
        100.0 * contiguity_ratio
        + 6.0 * vec
        + 12.0 * coalescing_bonus
        - 2.0 * predicate_vectors
        - 0.5 * rounds
    )

    return Candidate(
        thread_layout=(th_m, th_n),
        value_layout=(1, vec),
        rounds=rounds,
        covered=len(covered_coords),
        complete_coverage=complete_coverage,
        overlap_free=overlap_free,
        contiguity_ratio=contiguity_ratio,
        predicate_vectors=predicate_vectors,
        coalescing_bonus=coalescing_bonus,
        score=score,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tile-m", type=int, default=128)
    p.add_argument("--tile-n", type=int, default=128)
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--vector-widths", default="1,2,4,8,16")
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    vecs = [int(x.strip()) for x in args.vector_widths.split(",") if x.strip()]

    candidates: list[Candidate] = []
    for th_m, th_n in _factor_pairs(args.threads):
        for vec in vecs:
            if vec <= 0:
                continue
            if vec > args.tile_n:
                continue
            cand = _evaluate(args.tile_m, args.tile_n, args.threads, th_m, th_n, vec)
            if cand.complete_coverage and cand.overlap_free:
                candidates.append(cand)

    if not candidates:
        raise SystemExit("No valid candidate mappings found for the given parameters.")

    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[: max(1, args.top_k)]

    print("rank | thread_layout | value_layout | score | contiguity | predicate_vectors | rounds")
    for idx, cand in enumerate(best, start=1):
        print(
            f"{idx:>4} | {cand.thread_layout} | {cand.value_layout} | {cand.score:7.2f} | "
            f"{cand.contiguity_ratio:9.4f} | {cand.predicate_vectors:17d} | {cand.rounds}"
        )

    top = best[0]
    print("\nSuggested CuTe configuration for top candidate:")
    print(
        "\n".join(
            [
                "simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype, num_bits_per_copy=16)",
                (
                    f"thread_layout = cute.make_layout({top.thread_layout}, "
                    f"stride=({top.thread_layout[1]}, 1))"
                ),
                f"value_layout = cute.make_layout({top.value_layout})",
                "tiled_copy_r2g = cute.make_tiled_copy_tv(simt_atom, thread_layout, value_layout)",
            ]
        )
    )


if __name__ == "__main__":
    main()
