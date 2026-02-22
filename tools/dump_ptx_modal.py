#!/usr/bin/env python3
"""Dump CUTLASS DSL PTX (and optional CUBIN) for a kernel via Modal B200.

This tool compiles a kernel module remotely on Modal by calling its
``compile_kernel(problem_sizes)`` entrypoint with PTX/CUBIN dumps enabled.
It then returns the dumped artifacts to the local machine.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import modal

from modal_tools_app import APP_NAME as MODAL_APP_NAME
from modal_tools_app import CLASS_NAME as MODAL_CLASS_NAME

BEGIN_MARKER = "__PTX_DUMP_JSON_BEGIN__"
END_MARKER = "__PTX_DUMP_JSON_END__"


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
                f"Invalid problem-size group {raw_group!r}; all values must be ints."
            ) from exc
        groups.append((m, n, k, l))
    if not groups:
        raise SystemExit("No valid problem sizes parsed.")
    return groups


def _parse_args() -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kernel",
        required=True,
        help="Kernel Python file path (repo-relative or absolute), e.g. kernels/nvfp4_group_gemm/wagmiv67.py",
    )
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
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Primary PTX output path (default: artifacts/<kernel_stem>.ptx).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write all dumped artifacts (default: artifacts/ptx_dump_<kernel_stem>).",
    )
    p.add_argument(
        "--keep-cubin",
        dest="keep_cubin",
        action="store_true",
        default=True,
        help="Request and save CUBIN dumps (default: enabled).",
    )
    p.add_argument(
        "--no-cubin",
        dest="keep_cubin",
        action="store_false",
        help="Disable CUBIN dumping (PTX only).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional metadata JSON output path.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print modal command and stderr stream.",
    )
    return p.parse_args()


def _make_remote_snippet(
    kernel_rel: str,
    problem_sizes: list[tuple[int, int, int, int]],
    keep_cubin: bool,
) -> str:
    cfg_json = json.dumps(
        {
            "kernel_rel": kernel_rel,
            "problem_sizes": problem_sizes,
            "keep_cubin": keep_cubin,
        }
    )
    return f"""
import base64
import glob
import importlib.util
import json
import os
import pathlib
import shutil
import sys

cfg = json.loads({cfg_json!r})
kernel_rel = cfg["kernel_rel"]
problem_sizes = [tuple(int(x) for x in g) for g in cfg["problem_sizes"]]
keep_cubin = bool(cfg["keep_cubin"])

dump_dir = "/tmp/cute_ptx_dump"
shutil.rmtree(dump_dir, ignore_errors=True)
os.makedirs(dump_dir, exist_ok=True)

# Must be set before CUTLASS DSL compiles.
os.environ["CUTE_DSL_KEEP_PTX"] = "1"
os.environ["CUTE_DSL_DUMP_DIR"] = dump_dir
if keep_cubin:
    os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"

import torch  # noqa: E402
import cutlass  # noqa: E402

# Required sanity prints for CUTLASS execution checks.
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

spec = importlib.util.spec_from_file_location("modal_ptx_kernel_module", str(kernel_path))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to import kernel module from {{kernel_path}}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if not hasattr(module, "compile_kernel"):
    raise RuntimeError("Kernel module missing compile_kernel(problem_sizes)")

if hasattr(module, "_compiled_kernel_cache"):
    try:
        module._compiled_kernel_cache.clear()
    except Exception:
        pass

module.compile_kernel(problem_sizes)

ptx_files = sorted(glob.glob(os.path.join(dump_dir, "*.ptx")))
cubin_files = sorted(glob.glob(os.path.join(dump_dir, "*.cubin")))

payload = {{
    "kernel_rel": kernel_rel,
    "problem_sizes": problem_sizes,
    "dump_dir": dump_dir,
    "ptx_files": [],
    "cubin_files": [],
}}
for p in ptx_files:
    payload["ptx_files"].append({{
        "name": os.path.basename(p),
        "b64": base64.b64encode(pathlib.Path(p).read_bytes()).decode("ascii"),
    }})
for p in cubin_files:
    payload["cubin_files"].append({{
        "name": os.path.basename(p),
        "b64": base64.b64encode(pathlib.Path(p).read_bytes()).decode("ascii"),
    }})

print({BEGIN_MARKER!r})
print(json.dumps(payload))
print({END_MARKER!r})
"""


def _extract_payload(stdout: str) -> dict:
    start = stdout.rfind(BEGIN_MARKER)
    end = stdout.rfind(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(
            "PTX payload markers not found in modal output. "
            "Check stderr/logs for compile failures."
        )
    payload_txt = stdout[start + len(BEGIN_MARKER) : end].strip()
    return json.loads(payload_txt)


def _write_b64_items(items: list[dict], out_dir: Path) -> list[Path]:
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        name = item["name"]
        b64 = item["b64"]
        data = base64.b64decode(b64.encode("ascii"))
        dst = out_dir / name
        dst.write_bytes(data)
        written.append(dst)
    return written


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
    kernel_stem = kernel_abs.stem

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (repo_root / "artifacts" / f"ptx_dump_{kernel_stem}").resolve()
    )
    primary_out = (
        args.out.resolve()
        if args.out is not None
        else (repo_root / "artifacts" / f"{kernel_stem}.ptx").resolve()
    )

    snippet = _make_remote_snippet(
        kernel_rel=kernel_rel,
        problem_sizes=problem_sizes,
        keep_cubin=args.keep_cubin,
    )

    if args.gpu != "B200":
        raise SystemExit("dump_ptx_modal.py is B200-only with deployed modal tools.")

    runner = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)()
    files_b64 = {
        f"repo/{kernel_rel}": base64.b64encode(kernel_abs.read_bytes()).decode("ascii")
    }
    remote = runner.run_python.remote(
        snippet_source=snippet,
        files_b64=files_b64,
        profile=args.profile,
        timeout_seconds=args.timeout_seconds,
    )

    if args.verbose and remote.get("stderr"):
        print(remote["stderr"], file=sys.stderr, end="")
    if int(remote.get("exit_code", 1)) != 0:
        if remote.get("stderr"):
            print(remote["stderr"], file=sys.stderr, end="")
        raise SystemExit(int(remote.get("exit_code", 1)))

    payload = _extract_payload(str(remote.get("stdout", "")))
    ptx_written = _write_b64_items(payload.get("ptx_files", []), out_dir)
    cubin_written = _write_b64_items(payload.get("cubin_files", []), out_dir)

    if not ptx_written:
        print(proc.stderr, file=sys.stderr, end="")
        raise SystemExit("No PTX files were dumped. Check compile errors in Modal output.")

    primary_out.parent.mkdir(parents=True, exist_ok=True)
    primary_out.write_bytes(ptx_written[0].read_bytes())

    meta = {
        "kernel": kernel_rel,
        "problem_sizes": problem_sizes,
        "gpu": args.gpu,
        "out_dir": str(out_dir),
        "primary_ptx": str(primary_out),
        "ptx_files": [str(p) for p in ptx_written],
        "cubin_files": [str(p) for p in cubin_written],
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
