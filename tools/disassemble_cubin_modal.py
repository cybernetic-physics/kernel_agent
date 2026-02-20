#!/usr/bin/env python3
"""Disassemble CUBIN files to SASS on Modal.

Runs `nvdisasm` remotely on Modal so macOS hosts can inspect generated SASS
without a local CUDA toolkit installation.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

BEGIN_MARKER = "__NV_DISASM_JSON_BEGIN__"
END_MARKER = "__NV_DISASM_JSON_END__"


def _parse_args() -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cubin",
        required=True,
        help="Input .cubin file or directory containing .cubin files.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo,
        help=f"Repo root for resolving relative paths (default: {default_repo})",
    )
    p.add_argument("--gpu", default="B200", help="Modal GPU type (default: B200).")
    p.add_argument(
        "--nvdisasm-args",
        default="-g -hex -c",
        help="Extra args passed to nvdisasm (default: '-g -hex -c').",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir for .sass files (default: artifacts/cubin_disasm_<stem>).",
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
        help="Print modal command and remote stderr stream.",
    )
    return p.parse_args()


def _resolve_cubin_inputs(cubin_arg: str, repo_root: Path) -> tuple[Path, list[str], str]:
    raw = Path(cubin_arg)
    cubin_path = raw if raw.is_absolute() else (repo_root / raw)
    cubin_path = cubin_path.resolve()
    if not cubin_path.exists():
        raise SystemExit(f"CUBIN path not found: {cubin_path}")

    if cubin_path.is_file():
        if cubin_path.suffix != ".cubin":
            raise SystemExit(f"Expected .cubin file, got: {cubin_path}")
        mount_dir = cubin_path.parent
        filenames = [cubin_path.name]
        stem = cubin_path.stem
    else:
        files = sorted(p.name for p in cubin_path.glob("*.cubin") if p.is_file())
        if not files:
            raise SystemExit(f"No .cubin files found under: {cubin_path}")
        mount_dir = cubin_path
        filenames = files
        stem = cubin_path.name
    return mount_dir, filenames, stem


def _make_remote_snippet(filenames: list[str], nvdisasm_args: str) -> str:
    cfg_json = json.dumps({"filenames": filenames, "nvdisasm_args": nvdisasm_args})
    return f"""
import base64
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

cfg = json.loads({cfg_json!r})
filenames = list(cfg.get("filenames", []))
extra_args = shlex.split(cfg.get("nvdisasm_args", ""))

import torch  # noqa: E402
import cutlass  # noqa: E402
print("cuda_available", torch.cuda.is_available(), file=sys.stderr)
print(
    "gpu",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    file=sys.stderr,
)

nvdisasm = shutil.which("nvdisasm") or "/usr/local/cuda/bin/nvdisasm"
if not os.path.exists(nvdisasm):
    raise RuntimeError("nvdisasm not found in Modal container.")

base_dir = pathlib.Path("/workspace/cubin_in")
result = {{
    "status": "ok",
    "nvdisasm": nvdisasm,
    "files": [],
}}

for name in filenames:
    src = base_dir / name
    entry = {{
        "name": name,
        "exists": src.exists(),
    }}
    if not src.exists():
        entry["status"] = "missing"
        result["files"].append(entry)
        continue

    cmd = [nvdisasm] + extra_args + [str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    entry["status"] = "ok" if proc.returncode == 0 else "error"
    entry["returncode"] = proc.returncode
    entry["cmd"] = " ".join(cmd)
    entry["stderr_tail"] = (proc.stderr or "")[-4000:]
    if proc.returncode == 0:
        entry["sass_b64"] = base64.b64encode(proc.stdout.encode("utf-8")).decode("ascii")
    result["files"].append(entry)

if any(f.get("status") == "error" for f in result["files"]):
    result["status"] = "partial_error"

print({BEGIN_MARKER!r})
print(json.dumps(result))
print({END_MARKER!r})
"""


def _extract_payload(stdout: str) -> dict[str, Any]:
    start = stdout.rfind(BEGIN_MARKER)
    end = stdout.rfind(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(
            "Disassembly payload markers not found in Modal output. "
            "Check modal stderr/stdout logs."
        )
    payload_txt = stdout[start + len(BEGIN_MARKER) : end].strip()
    return json.loads(payload_txt)


def _decode_to_text(b64_data: str) -> str:
    return base64.b64decode(b64_data.encode("ascii")).decode("utf-8", errors="replace")


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        raise SystemExit(f"Repo root not found: {repo_root}")

    mount_dir, filenames, stem = _resolve_cubin_inputs(args.cubin, repo_root)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (repo_root / "artifacts" / f"cubin_disasm_{stem}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    snippet = _make_remote_snippet(
        filenames=filenames,
        nvdisasm_args=args.nvdisasm_args,
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
        "--mount-local-dir",
        f"{mount_dir}:/workspace/cubin_in",
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
            f"Failed to extract disassembly payload: {exc}\n"
            f"See logs:\n  {stdout_log}\n  {stderr_log}"
        ) from exc

    results: list[dict[str, Any]] = []
    for item in payload.get("files", []):
        row = dict(item)
        if item.get("status") == "ok" and "sass_b64" in item:
            src_name = str(item["name"])
            sass_name = src_name.replace(".cubin", ".sass")
            sass_text = _decode_to_text(str(item["sass_b64"]))
            sass_path = out_dir / sass_name
            sass_path.write_text(sass_text, encoding="utf-8")
            row["sass_file"] = str(sass_path)
            row.pop("sass_b64", None)
        results.append(row)

    summary = {
        "status": payload.get("status", "unknown"),
        "gpu": args.gpu,
        "nvdisasm": payload.get("nvdisasm"),
        "input_dir": str(mount_dir),
        "out_dir": str(out_dir),
        "modal_returncode": proc.returncode,
        "modal_stdout_log": str(stdout_log),
        "modal_stderr_log": str(stderr_log),
        "files": results,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if summary["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
