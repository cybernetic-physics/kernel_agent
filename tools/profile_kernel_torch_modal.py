#!/usr/bin/env python3
"""Profile a CUTLASS kernel file on Modal B200 using torch.profiler (CUPTI).

This is a practical fallback when Nsight Compute counters are unavailable in
the runtime. It captures per-kernel CUDA time aggregates from
``torch.profiler`` and stores structured JSON artifacts.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import modal

from modal_tools_app import APP_NAME as MODAL_APP_NAME
from modal_tools_app import CLASS_NAME as MODAL_CLASS_NAME

BEGIN_MARKER = "__TORCH_PROFILER_MODAL_JSON_BEGIN__"
END_MARKER = "__TORCH_PROFILER_MODAL_JSON_END__"


def _parse_problem_sizes(spec: str) -> list[tuple[int, int, int, int]]:
    groups: list[tuple[int, int, int, int]] = []
    for raw_group in spec.split(";"):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        parts = [p.strip() for p in raw_group.split(",")]
        if len(parts) != 4:
            raise SystemExit(
                f"Invalid problem-size group {raw_group!r}; expected m,n,k,l."
            )
        try:
            m, n, k, l = (int(x) for x in parts)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid problem-size group {raw_group!r}; values must be ints."
            ) from exc
        groups.append((m, n, k, l))
    if not groups:
        raise SystemExit("No valid problem sizes parsed.")
    return groups


def _parse_args() -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", required=True, help="Kernel .py path (repo-relative or absolute).")
    p.add_argument(
        "--problem-sizes",
        default="80,4096,7168,1;40,7168,2048,1",
        help="Semicolon-separated m,n,k,l groups.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo,
        help=f"Repo root (default: {default_repo})",
    )
    p.add_argument("--gpu", default="B200", help="Modal GPU type (default: B200).")
    p.add_argument(
        "--profile",
        default="candidate",
        choices=["smoke", "candidate", "final"],
        help="Deployed runner profile (default: candidate).",
    )
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Override snippet timeout inside deployed runner.",
    )
    p.add_argument("--warmup", type=int, default=2, help="Warmup iterations (default: 2).")
    p.add_argument(
        "--profile-iters",
        type=int,
        default=12,
        help="Profile-region iterations (default: 12).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top events to keep by self CUDA time (default: 30).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Artifact directory (default: artifacts/torch_profiler_<kernel_stem>).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write result JSON.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print modal command and remote stderr stream.",
    )
    return p.parse_args()


def _make_remote_snippet(
    *,
    kernel_rel: str,
    problem_sizes: list[tuple[int, int, int, int]],
    warmup: int,
    profile_iters: int,
    top_k: int,
) -> str:
    cfg_json = json.dumps(
        {
            "kernel_rel": kernel_rel,
            "problem_sizes": problem_sizes,
            "warmup": warmup,
            "profile_iters": profile_iters,
            "top_k": top_k,
        }
    )
    return f"""
import importlib.util
import json
import os
import pathlib
import sys

cfg = json.loads({cfg_json!r})
kernel_rel = cfg["kernel_rel"]
problem_sizes = [tuple(int(x) for x in g) for g in cfg["problem_sizes"]]
warmup = int(cfg["warmup"])
profile_iters = int(cfg["profile_iters"])
top_k = int(cfg["top_k"])

import torch  # noqa: E402
import cutlass  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

print("cuda_available", torch.cuda.is_available(), file=sys.stderr)
print(
    "gpu",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    file=sys.stderr,
)

# Kernels in this repo may import task.input_t/output_t.
import types  # noqa: E402
task_mod = types.ModuleType("task")
task_mod.input_t = tuple
task_mod.output_t = list
sys.modules.setdefault("task", task_mod)

stage_root = pathlib.Path(os.environ.get("MODAL_STAGE_DIR", "."))
kernel_path = stage_root / "repo" / kernel_rel
if not kernel_path.exists():
    raise RuntimeError(f"Kernel file not found: {{kernel_path}}")

spec = importlib.util.spec_from_file_location("modal_torch_prof_kernel", str(kernel_path))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to import kernel module from {{kernel_path}}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if not hasattr(module, "custom_kernel"):
    raise RuntimeError("Kernel module missing custom_kernel(data)")

def make_data():
    abc = []
    sfasfb = []
    for m, n, k, l in problem_sizes:
        a = torch.randint(0, 256, (m, k // 2, l), dtype=torch.uint8, device="cuda")
        b = torch.randint(0, 256, (n, k // 2, l), dtype=torch.uint8, device="cuda")
        c = torch.zeros(m, n, l, dtype=torch.float16, device="cuda")
        abc.append((a, b, c))
        sfa = torch.randint(0, 256, (m, k // 16, l), dtype=torch.uint8, device="cuda")
        sfb = torch.randint(0, 256, (n, k // 16, l), dtype=torch.uint8, device="cuda")
        sfasfb.append((sfa, sfb))
    return (abc, None, sfasfb, problem_sizes)

data = make_data()
for _ in range(warmup):
    module.custom_kernel(data)
    torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(profile_iters):
        module.custom_kernel(data)
    torch.cuda.synchronize()

events = []
for evt in prof.key_averages():
    self_cuda_us = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
    cuda_total_us = float(getattr(evt, "device_time_total", 0.0) or 0.0)
    if self_cuda_us <= 0.0 and cuda_total_us <= 0.0:
        continue
    events.append(
        {{
            "name": evt.key,
            "calls": int(evt.count),
            "self_cuda_us": self_cuda_us,
            "cuda_total_us": cuda_total_us,
            "self_cpu_us": float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0),
        }}
    )
events.sort(key=lambda x: x["self_cuda_us"], reverse=True)

table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=top_k)

result = {{
    "status": "ok",
    "kernel_rel": kernel_rel,
    "problem_sizes": problem_sizes,
    "warmup": warmup,
    "profile_iters": profile_iters,
    "events": events[:top_k],
    "table": table,
}}

print({BEGIN_MARKER!r})
print(json.dumps(result))
print({END_MARKER!r})
"""


def _extract_payload(stdout: str) -> dict[str, Any]:
    start = stdout.rfind(BEGIN_MARKER)
    end = stdout.rfind(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(
            "Profiler payload markers not found in Modal output. "
            "Check modal stderr/stdout logs."
        )
    payload_txt = stdout[start + len(BEGIN_MARKER) : end].strip()
    return json.loads(payload_txt)


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        raise SystemExit(f"Repo root not found: {repo_root}")

    kernel_path = Path(args.kernel)
    if kernel_path.is_absolute():
        kernel_abs = kernel_path.resolve()
        try:
            kernel_rel = str(kernel_abs.relative_to(repo_root))
        except ValueError as exc:
            raise SystemExit(
                f"Absolute kernel path must be under repo root {repo_root}: {kernel_abs}"
            ) from exc
    else:
        kernel_rel = str(kernel_path)
        kernel_abs = (repo_root / kernel_path).resolve()
    if not kernel_abs.exists():
        raise SystemExit(f"Kernel file not found: {kernel_abs}")

    problem_sizes = _parse_problem_sizes(args.problem_sizes)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (repo_root / "artifacts" / f"torch_profiler_{kernel_abs.stem}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    snippet = _make_remote_snippet(
        kernel_rel=kernel_rel,
        problem_sizes=problem_sizes,
        warmup=args.warmup,
        profile_iters=args.profile_iters,
        top_k=args.top_k,
    )
    if args.gpu != "B200":
        raise SystemExit("profile_kernel_torch_modal.py is B200-only with deployed modal tools.")

    runner = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)()
    files_b64 = {
        f"repo/{kernel_rel}": base64.b64encode(kernel_abs.read_bytes()).decode("ascii")
    }
    if args.verbose:
        print(
            f"Running deployed modal tools app={MODAL_APP_NAME} class={MODAL_CLASS_NAME} "
            f"profile={args.profile}"
        )
    remote = runner.run_python.remote(
        snippet_source=snippet,
        files_b64=files_b64,
        profile=args.profile,
        timeout_seconds=args.timeout_seconds,
    )

    stdout_log = out_dir / "modal_stdout.log"
    stderr_log = out_dir / "modal_stderr.log"
    stdout_log.write_text(str(remote.get("stdout", "") or ""), encoding="utf-8")
    stderr_log.write_text(str(remote.get("stderr", "") or ""), encoding="utf-8")

    payload = _extract_payload(str(remote.get("stdout", "") or ""))
    payload["gpu"] = args.gpu
    payload["kernel"] = kernel_rel
    payload["modal_returncode"] = int(remote.get("exit_code", 1))
    payload["modal_timed_out"] = bool(remote.get("timed_out", False))
    payload["modal_stdout_log"] = str(stdout_log)
    payload["modal_stderr_log"] = str(stderr_log)
    payload["artifacts_dir"] = str(out_dir)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "table.txt").write_text(str(payload.get("table", "")) + "\n", encoding="utf-8")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
