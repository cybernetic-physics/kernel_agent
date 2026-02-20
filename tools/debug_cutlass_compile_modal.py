#!/usr/bin/env python3
"""Debug CuTe/CUTLASS DSL compile failures on Modal with rich artifacts.

This tool runs a kernel module's ``compile_kernel(problem_sizes)`` remotely on
Modal and captures:
  - CuTe DSL logs (configurable verbosity)
  - full Python traceback on failure
  - dumped IR/PTX/CUBIN files (via CUTE_DSL_* env vars)
  - optional compile option overrides (e.g. "--opt-level 0")

It is designed for cases where popcorn-cli reports only a terse compilation
failure and you need deeper diagnostics.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

BEGIN_MARKER = "__CUTLASS_COMPILE_DEBUG_JSON_BEGIN__"
END_MARKER = "__CUTLASS_COMPILE_DEBUG_JSON_END__"


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
        "--compile-options",
        default="",
        help="Extra cute.compile options to append, e.g. '--opt-level 0 --enable-device-assertions'.",
    )
    p.add_argument(
        "--log-level",
        type=int,
        default=10,
        help="CUTE_DSL_LOG_LEVEL (0,10,20,30,40,50; default: 10).",
    )
    p.add_argument(
        "--print-ir",
        action="store_true",
        help="Set CUTE_DSL_PRINT_IR=1 (very verbose).",
    )
    p.add_argument(
        "--no-lineinfo",
        action="store_true",
        help="Disable CUTE_DSL_LINEINFO=1.",
    )
    p.add_argument(
        "--no-keep-ir",
        action="store_true",
        help="Disable CUTE_DSL_KEEP_IR=1.",
    )
    p.add_argument(
        "--no-keep-ptx",
        action="store_true",
        help="Disable CUTE_DSL_KEEP_PTX=1.",
    )
    p.add_argument(
        "--no-keep-cubin",
        action="store_true",
        help="Disable CUTE_DSL_KEEP_CUBIN=1.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Artifact directory (default: artifacts/compile_debug_<kernel_stem>).",
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
    compile_options: str,
    log_level: int,
    print_ir: bool,
    lineinfo: bool,
    keep_ir: bool,
    keep_ptx: bool,
    keep_cubin: bool,
) -> str:
    cfg_json = json.dumps(
        {
            "kernel_rel": kernel_rel,
            "problem_sizes": problem_sizes,
            "compile_options": compile_options,
            "log_level": log_level,
            "print_ir": print_ir,
            "lineinfo": lineinfo,
            "keep_ir": keep_ir,
            "keep_ptx": keep_ptx,
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
import traceback

cfg = json.loads({cfg_json!r})
kernel_rel = cfg["kernel_rel"]
problem_sizes = [tuple(int(x) for x in g) for g in cfg["problem_sizes"]]
compile_options = (cfg.get("compile_options") or "").strip()

dump_dir = "/tmp/cute_compile_debug_dump"
shutil.rmtree(dump_dir, ignore_errors=True)
os.makedirs(dump_dir, exist_ok=True)

os.environ["CUTE_DSL_LOG_TO_CONSOLE"] = "1"
os.environ["CUTE_DSL_LOG_LEVEL"] = str(int(cfg.get("log_level", 10)))
os.environ["CUTE_DSL_DUMP_DIR"] = dump_dir
if cfg.get("print_ir"):
    os.environ["CUTE_DSL_PRINT_IR"] = "1"
if cfg.get("lineinfo"):
    os.environ["CUTE_DSL_LINEINFO"] = "1"
if cfg.get("keep_ir"):
    os.environ["CUTE_DSL_KEEP_IR"] = "1"
if cfg.get("keep_ptx"):
    os.environ["CUTE_DSL_KEEP_PTX"] = "1"
if cfg.get("keep_cubin"):
    os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"

import torch  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402

print("cuda_available", torch.cuda.is_available(), file=sys.stderr)
print(
    "gpu",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    file=sys.stderr,
)

if compile_options:
    _orig_compile = cute.compile

    def _compile_with_extra(*args, **kwargs):
        cur = str(kwargs.get("options", "") or "").strip()
        merged = (cur + " " + compile_options).strip() if cur else compile_options
        kwargs["options"] = merged
        return _orig_compile(*args, **kwargs)

    cute.compile = _compile_with_extra

result = {{
    "status": "unknown",
    "kernel_rel": kernel_rel,
    "problem_sizes": problem_sizes,
    "compile_options": compile_options,
    "dsl_env": {{
        "CUTE_DSL_LOG_TO_CONSOLE": os.environ.get("CUTE_DSL_LOG_TO_CONSOLE"),
        "CUTE_DSL_LOG_LEVEL": os.environ.get("CUTE_DSL_LOG_LEVEL"),
        "CUTE_DSL_DUMP_DIR": os.environ.get("CUTE_DSL_DUMP_DIR"),
        "CUTE_DSL_PRINT_IR": os.environ.get("CUTE_DSL_PRINT_IR", "0"),
        "CUTE_DSL_LINEINFO": os.environ.get("CUTE_DSL_LINEINFO", "0"),
        "CUTE_DSL_KEEP_IR": os.environ.get("CUTE_DSL_KEEP_IR", "0"),
        "CUTE_DSL_KEEP_PTX": os.environ.get("CUTE_DSL_KEEP_PTX", "0"),
        "CUTE_DSL_KEEP_CUBIN": os.environ.get("CUTE_DSL_KEEP_CUBIN", "0"),
    }},
    "compiled_attrs": {{}},
    "dump_files": [],
}}

try:
    kernel_path = pathlib.Path("/workspace/repo") / kernel_rel
    if not kernel_path.exists():
        raise RuntimeError(f"Kernel file not found: {{kernel_path}}")

    spec = importlib.util.spec_from_file_location("modal_compile_debug_kernel", str(kernel_path))
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

    compiled = module.compile_kernel(problem_sizes)
    result["status"] = "ok"
    result["compiled_type"] = type(compiled).__name__

    for attr in ("__mlir__", "__ptx__", "__cubin__"):
        try:
            val = getattr(compiled, attr)
        except Exception:
            continue
        if val is None:
            continue
        if isinstance(val, bytes):
            result["compiled_attrs"][attr] = {{
                "kind": "bytes",
                "b64": base64.b64encode(val).decode("ascii"),
            }}
        else:
            txt = str(val)
            result["compiled_attrs"][attr] = {{
                "kind": "text",
                "b64": base64.b64encode(txt.encode("utf-8")).decode("ascii"),
            }}
except Exception as exc:
    result["status"] = "error"
    result["error_type"] = type(exc).__name__
    result["error_message"] = str(exc)
    result["traceback"] = traceback.format_exc()

for p in sorted(glob.glob(os.path.join(dump_dir, "*"))):
    pp = pathlib.Path(p)
    if not pp.is_file():
        continue
    raw = pp.read_bytes()
    result["dump_files"].append(
        {{
            "name": pp.name,
            "size": len(raw),
            "b64": base64.b64encode(raw).decode("ascii"),
        }}
    )

print({BEGIN_MARKER!r})
print(json.dumps(result))
print({END_MARKER!r})
"""


def _extract_payload(stdout: str) -> dict[str, Any]:
    start = stdout.rfind(BEGIN_MARKER)
    end = stdout.rfind(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(
            "Debug payload markers not found in Modal output. "
            "Check modal stderr/stdout logs."
        )
    payload_txt = stdout[start + len(BEGIN_MARKER) : end].strip()
    return json.loads(payload_txt)


def _decode_b64_to_file(b64_data: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(base64.b64decode(b64_data.encode("ascii")))


def _derive_hints(payload: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    trace = str(payload.get("traceback", "") or "")
    message = str(payload.get("error_message", "") or "")
    text = trace + "\n" + message

    if "structured different after this `while`" in text:
        hints.append(
            "Dynamic while/if loop changed value structure across iterations. Keep tensor/value "
            "type+structure identical through loop-carried variables, or use constexpr control flow."
        )
    if "changing type of a variable" in text or "Dependent Type" in text:
        hints.append(
            "CuTe DSL requires static type consistency in dynamic control flow. Avoid dependent-type "
            "expressions and type changes inside loops/branches."
        )
    if "Operation creation failed" in text:
        hints.append(
            "CUTLASS op creation failed. Validate TMA atom layouts, tile shapes, and CTA mapping compatibility."
        )
    if "MLIRError" in text:
        hints.append(
            "MLIR lowering failed. Dump and inspect __mlir__ / kept IR files to isolate the first invalid op."
        )
    if "Failed to import" in text or "Kernel file not found" in text:
        hints.append(
            "Import/path issue. Confirm --repo-root and --kernel resolve to the intended file in /workspace/repo."
        )
    if not hints and payload.get("status") == "error":
        hints.append(
            "Re-run with --print-ir and --compile-options '--opt-level 0 --enable-device-assertions' for more diagnostics."
        )
    return hints


def _truncate(text: str | None, limit: int = 3000) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated, {len(text) - limit} chars omitted>"


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
        else (repo_root / "artifacts" / f"compile_debug_{kernel_stem}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    snippet = _make_remote_snippet(
        kernel_rel=kernel_rel,
        problem_sizes=problem_sizes,
        compile_options=args.compile_options,
        log_level=args.log_level,
        print_ir=bool(args.print_ir),
        lineinfo=not args.no_lineinfo,
        keep_ir=not args.no_keep_ir,
        keep_ptx=not args.no_keep_ptx,
        keep_cubin=not args.no_keep_cubin,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(snippet)
        code_file = Path(tf.name)

    cmd = [
        "uv",
        "run",
        "--with",
        "modal",
        "python",
        "tools/modal_python_exec.py",
        "--gpu",
        args.gpu,
        "--mount-repo",
        "--allow-nonzero",
        "--code-file",
        str(code_file),
    ]

    try:
        if args.verbose:
            print("Running:", " ".join(cmd), file=sys.stderr)
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
    finally:
        code_file.unlink(missing_ok=True)

    stdout_log = out_dir / "modal_stdout.log"
    stderr_log = out_dir / "modal_stderr.log"
    stdout_log.write_text(proc.stdout or "", encoding="utf-8")
    stderr_log.write_text(proc.stderr or "", encoding="utf-8")

    payload: dict[str, Any]
    try:
        payload = _extract_payload(proc.stdout or "")
    except Exception as exc:
        raise SystemExit(
            f"Failed to extract debug payload: {exc}\n"
            f"See logs:\n  {stdout_log}\n  {stderr_log}"
        ) from exc

    extracted_dir = out_dir / "remote_dump"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    extracted_dump_files: list[dict[str, Any]] = []
    # Write runtime dump files captured from CUTE_DSL_DUMP_DIR.
    for item in payload.get("dump_files", []):
        name = str(item.get("name", "unnamed.bin"))
        dst = extracted_dir / name
        _decode_b64_to_file(str(item.get("b64", "")), dst)
        extracted_dump_files.append(
            {
                "name": name,
                "size": int(item.get("size", 0)),
                "path": str(dst),
            }
        )

    # Write in-memory compiled attrs (if available).
    compiled_dir = out_dir / "compiled_attrs"
    compiled_dir.mkdir(parents=True, exist_ok=True)
    extracted_compiled_attrs: dict[str, Any] = {}
    for attr, meta in payload.get("compiled_attrs", {}).items():
        b64_data = str(meta.get("b64", ""))
        ext = ".bin" if meta.get("kind") == "bytes" else ".txt"
        safe_name = attr.strip("_") + ext
        dst = compiled_dir / safe_name
        _decode_b64_to_file(b64_data, dst)
        extracted_compiled_attrs[attr] = {
            "kind": meta.get("kind"),
            "path": str(dst),
        }

    full_error_message = payload.get("error_message")
    full_traceback = payload.get("traceback")
    error_message_file = out_dir / "error_message_full.txt"
    traceback_file = out_dir / "traceback_full.txt"
    if full_error_message:
        error_message_file.write_text(str(full_error_message), encoding="utf-8")
    if full_traceback:
        traceback_file.write_text(str(full_traceback), encoding="utf-8")

    summary = {
        "status": payload.get("status"),
        "error_type": payload.get("error_type"),
        "error_message": _truncate(str(full_error_message), limit=1200)
        if full_error_message is not None
        else None,
        "traceback": _truncate(str(full_traceback), limit=1800)
        if full_traceback is not None
        else None,
        "error_message_file": str(error_message_file) if full_error_message else None,
        "traceback_file": str(traceback_file) if full_traceback else None,
        "kernel_rel": payload.get("kernel_rel", kernel_rel),
        "compiled_type": payload.get("compiled_type"),
        "compile_options": payload.get("compile_options", ""),
        "dsl_env": payload.get("dsl_env", {}),
        "hints": _derive_hints(payload),
        "kernel": kernel_rel,
        "gpu": args.gpu,
        "problem_sizes": problem_sizes,
        "artifacts_dir": str(out_dir),
        "modal_returncode": proc.returncode,
        "modal_stdout_log": str(stdout_log),
        "modal_stderr_log": str(stderr_log),
        "dump_files": extracted_dump_files,
        "compiled_attrs": extracted_compiled_attrs,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if summary.get("status") != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
