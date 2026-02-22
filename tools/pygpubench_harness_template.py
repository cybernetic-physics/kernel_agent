"""Template harness for PyGPUBench with optional staged early-stop execution.

Copy this file and adapt:
1) `generate_test_case(args, seed)` to build your kernel inputs and expected output.
2) `kernel_generator()` to return the callable from `submission.py`.
3) `TEST_ARGS`, `REPEATS`, `SEED` for your workload.

Runtime controls (env vars):
- PYGPUBENCH_REPEATS: total repeats (int, default REPEATS)
- PYGPUBENCH_STAGE_REPEATS: comma-separated stage sizes (default total only)
- PYGPUBENCH_EARLY_STOP_US: if >0, stop after a stage when median_us exceeds this
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass

import pygpubench


def generate_test_case(args: tuple, seed: int):
    """
    Must return:
      (kernel_inputs_tuple, (expected_output_tensor, rtol, atol))

    Replace this body with your problem-specific input/reference generation.
    """
    raise NotImplementedError("Implement generate_test_case for your kernel/problem.")


def kernel_generator():
    """
    Must return a callable that accepts one argument: kernel_inputs_tuple.
    """
    import submission

    return submission.custom_kernel


TEST_ARGS = ()
REPEATS = 100
SEED = 5


@dataclass
class StageResult:
    repeats: int
    result: object
    median_us: float


def _parse_stage_repeats(total_repeats: int) -> list[int]:
    raw = os.getenv("PYGPUBENCH_STAGE_REPEATS", "").strip()
    if not raw:
        return [total_repeats]

    stage_sizes: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        val = int(item)
        if val > 0:
            stage_sizes.append(val)

    if not stage_sizes:
        return [total_repeats]

    total = sum(stage_sizes)
    if total < total_repeats:
        stage_sizes.append(total_repeats - total)
    elif total > total_repeats:
        # Truncate to total_repeats while preserving order.
        trimmed: list[int] = []
        remaining = total_repeats
        for size in stage_sizes:
            if remaining <= 0:
                break
            used = min(size, remaining)
            if used > 0:
                trimmed.append(used)
                remaining -= used
        stage_sizes = trimmed

    return stage_sizes or [total_repeats]


def _median_us(bench_result) -> float:
    values = [float(x) for x in bench_result.time_us]
    if not values:
        return float("inf")
    return float(statistics.median(values))


def _run_staged_benchmark(
    total_repeats: int,
    early_stop_us: float,
) -> StageResult:
    stage_repeats = _parse_stage_repeats(total_repeats)
    last_stage: StageResult | None = None

    for stage_idx, repeats in enumerate(stage_repeats, start=1):
        bench_result = pygpubench.do_bench_isolated(
            kernel_generator,
            generate_test_case,
            TEST_ARGS,
            repeats,
            SEED,
            discard=True,
        )
        median_us = _median_us(bench_result)
        last_stage = StageResult(repeats=repeats, result=bench_result, median_us=median_us)

        print(
            f"stage={stage_idx}/{len(stage_repeats)} repeats={repeats} "
            f"median_us={median_us:.3f} errors={len(bench_result.errors)}"
        )

        if bench_result.errors:
            return last_stage

        if early_stop_us > 0 and median_us > early_stop_us:
            print(
                "EARLY_STOP: median_us exceeded threshold "
                f"({median_us:.3f} > {early_stop_us:.3f})"
            )
            return last_stage

    if last_stage is None:
        raise RuntimeError("No benchmark stages were executed.")
    return last_stage


if __name__ == "__main__":
    total_repeats = int(os.getenv("PYGPUBENCH_REPEATS", str(REPEATS)))
    early_stop_us = float(os.getenv("PYGPUBENCH_EARLY_STOP_US", "0"))

    stage_result = _run_staged_benchmark(total_repeats=total_repeats, early_stop_us=early_stop_us)
    result = stage_result.result

    print("FAIL" if result.errors else "PASS", pygpubench.basic_stats(result.time_us))
