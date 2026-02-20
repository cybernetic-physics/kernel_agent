#!/usr/bin/env python3
"""Run a CUTLASS DSL kernel on Modal B200 and report execution stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "cutlass-execution-modal-b200-example"
DEFAULT_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
DEFAULT_KERNEL = "kernels/nvfp4_group_gemm/wagmiv67.py"
DEFAULT_PROBLEM_SIZES = "80,4096,7168,1;40,7168,2048,1"

app = modal.App(APP_NAME)
image = modal.Image.from_registry(DEFAULT_IMAGE, add_python="3.11").pip_install(
    "nvidia-cutlass-dsl==4.4.0", "torch"
)


def _parse_problem_sizes(raw: str) -> list[tuple[int, int, int, int]]:
    result: list[tuple[int, int, int, int]] = []
    for group in raw.split(";"):
        parts = [int(x.strip()) for x in group.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Expected m,n,k,l group; got {group!r}")
        result.append((parts[0], parts[1], parts[2], parts[3]))
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kernel",
        type=str,
        default=DEFAULT_KERNEL,
        help=f"Kernel file containing custom_kernel(...) (default: {DEFAULT_KERNEL})",
    )
    p.add_argument(
        "--problem-sizes",
        type=str,
        default=DEFAULT_PROBLEM_SIZES,
        help=(
            "Semicolon-separated m,n,k,l groups. "
            f"Default: {DEFAULT_PROBLEM_SIZES!r}"
        ),
    )
    p.add_argument("--warmup", type=int, default=2, help="Warmup iterations.")
    p.add_argument("--iters", type=int, default=5, help="Timed iterations.")
    p.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/cutlass_execution_modal_b200_example.json"),
        help="Path to write JSON output.",
    )
    # Modal may pass argv fragments into local entrypoint processes.
    args, _unknown = p.parse_known_args()
    return args


@app.function(image=image, gpu="B200", timeout=1800)
def _run_cutlass_kernel(
    kernel_source: str,
    problem_sizes: list[tuple[int, int, int, int]],
    warmup_iters: int,
    timed_iters: int,
) -> dict[str, Any]:
    import importlib.util
    import statistics
    import sys
    import types

    import torch

    kernel_path = "/tmp/modal_cutlass_kernel.py"
    with open(kernel_path, "w", encoding="utf-8") as f:
        f.write(kernel_source)

    # Kernels in this repo import `task.input_t/output_t`; stub it for standalone use.
    task_mod = types.ModuleType("task")
    task_mod.input_t = tuple
    task_mod.output_t = list
    sys.modules["task"] = task_mod

    spec = importlib.util.spec_from_file_location("_modal_cutlass_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create module spec for kernel source")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "custom_kernel"):
        raise RuntimeError("Kernel module must define custom_kernel(data)")

    def make_data() -> tuple[list[Any], None, list[Any], list[tuple[int, int, int, int]]]:
        abc = []
        sfasfb = []
        for m, n, k, l in problem_sizes:
            a = torch.randint(0, 256, (m, k // 2, l), dtype=torch.uint8, device="cuda")
            b = torch.randint(0, 256, (n, k // 2, l), dtype=torch.uint8, device="cuda")
            c = torch.zeros((m, n, l), dtype=torch.float16, device="cuda")
            sfa = torch.randint(0, 256, (m, k // 16, l), dtype=torch.uint8, device="cuda")
            sfb = torch.randint(0, 256, (n, k // 16, l), dtype=torch.uint8, device="cuda")
            abc.append((a, b, c))
            sfasfb.append((sfa, sfb))
        return (abc, None, sfasfb, problem_sizes)

    data = make_data()

    for _ in range(max(0, int(warmup_iters))):
        mod.custom_kernel(data)
    torch.cuda.synchronize()

    times_us: list[float] = []
    out = None
    for _ in range(max(1, int(timed_iters))):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = mod.custom_kernel(data)
        end.record()
        end.synchronize()
        times_us.append(float(start.elapsed_time(end)) * 1000.0)

    # Sanity checks for outputs.
    c_tensors = [triple[2] for triple in data[0]]
    all_finite = True
    checksums = []
    max_abs = []
    for c in c_tensors:
        all_finite = all_finite and bool(torch.isfinite(c).all().item())
        checksums.append(float(c.float().sum().item()))
        max_abs.append(float(c.abs().max().item()))

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    result = {
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "sm_count": int(props.multi_processor_count),
            "total_memory_gb": float(props.total_memory) / float(1024**3),
        },
        "problem_sizes": [list(x) for x in problem_sizes],
        "warmup_iters": int(warmup_iters),
        "timed_iters": int(timed_iters),
        "latency_us": {
            "mean": float(sum(times_us) / len(times_us)),
            "median": float(statistics.median(times_us)),
            "min": float(min(times_us)),
            "max": float(max(times_us)),
            "stdev": float(statistics.pstdev(times_us)) if len(times_us) > 1 else 0.0,
            "samples": times_us,
        },
        "output": {
            "num_groups": len(c_tensors),
            "all_finite": bool(all_finite),
            "checksums": checksums,
            "max_abs": max_abs,
            "returned_type": type(out).__name__ if out is not None else None,
        },
    }
    return result


@app.local_entrypoint()
def main() -> None:
    args = _parse_args()
    kernel_path = Path(args.kernel)
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")

    problem_sizes = _parse_problem_sizes(args.problem_sizes)
    kernel_source = kernel_path.read_text(encoding="utf-8")

    print(
        f"[INFO] Running CUTLASS execution example on B200 with kernel={kernel_path} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(f"[INFO] Problem sizes: {problem_sizes}")

    result = _run_cutlass_kernel.remote(
        kernel_source=kernel_source,
        problem_sizes=problem_sizes,
        warmup_iters=args.warmup,
        timed_iters=args.iters,
    )

    text = json.dumps(result, indent=2)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(text + "\n", encoding="utf-8")
    print(f"[INFO] Wrote: {args.json_out}")
    print(text)
