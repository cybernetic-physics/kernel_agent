#!/usr/bin/env python3
"""Run a PyGPUBench harness on Modal GPU with a selected submission file.

This runner is self-contained: the remote image installs PyGPUBench directly
from GitHub and does not require a local `pygpubench` checkout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import modal

APP_NAME = "pygpubench-modal-runner"
DEFAULT_IMAGE = "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04"
PYGPUBENCH_GIT_URL = "git+https://github.com/ngc92/pygpubench.git"


def _find_workspace_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "tools").exists():
            return parent
    raise RuntimeError(f"Could not find workspace root from {start}")


THIS_FILE = Path(__file__).resolve()
try:
    WORKSPACE_ROOT = _find_workspace_root(THIS_FILE)
except RuntimeError:
    WORKSPACE_ROOT = Path("/workspace/kernel_agents")

base_image = (
    modal.Image.from_registry(DEFAULT_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential", "cmake", "ninja-build", "git")
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("nvidia-cutlass-dsl==4.4.0")
    .uv_pip_install("scikit-build-core>=0.11", "nanobind")
)

image = base_image.run_commands(
    "CC=gcc CXX=g++ CUDACXX=/usr/local/cuda/bin/nvcc "
    f"python -m pip install -v {PYGPUBENCH_GIT_URL}"
)

app = modal.App(APP_NAME, image=image)


def _load_sources(
    harness: Path, submission: Path
) -> tuple[Path, Path, str, str]:
    harness_path = harness.resolve()
    submission_path = submission.resolve()

    if not harness_path.exists():
        raise FileNotFoundError(f"Harness file not found: {harness_path}")
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    return (
        harness_path,
        submission_path,
        harness_path.read_text(encoding="utf-8"),
        submission_path.read_text(encoding="utf-8"),
    )


@app.function(gpu="B200", timeout=60 * 60)
def run_harness(
    harness_source: str,
    submission_source: str,
    harness_name: str,
) -> dict[str, Any]:
    stage_dir = Path(tempfile.mkdtemp(prefix="modal-pygpubench-"))
    try:
        harness_path = stage_dir / harness_name
        harness_path.write_text(harness_source, encoding="utf-8")
        (stage_dir / "submission.py").write_text(submission_source, encoding="utf-8")
        # Some kernels import typing aliases from task.py.
        (stage_dir / "task.py").write_text(
            "input_t = tuple\noutput_t = list\n", encoding="utf-8"
        )

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Ensure local staged submission is importable first.
        env["PYTHONPATH"] = f"{stage_dir}:{env.get('PYTHONPATH', '')}"

        proc = subprocess.run(
            [sys.executable, harness_path.name],
            cwd=stage_dir,
            env=env,
            text=True,
            capture_output=True,
        )

        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "harness_name": harness_name,
            "stage_dir": str(stage_dir),
        }
    finally:
        # Keep staged files out of remote container after run.
        subprocess.run(["rm", "-rf", str(stage_dir)], check=False)


@app.local_entrypoint()
def main(
    harness: str = str(WORKSPACE_ROOT / "tools" / "pygpubench_harness_template.py"),
    submission: str = "",
    gpu: str = "B200",
    json_out: str = str(WORKSPACE_ROOT / "artifacts" / "pygpubench_modal_last_run.json"),
    print_log: bool = False,
) -> None:
    if not submission:
        raise ValueError("--submission is required.")

    harness_path = Path(harness).resolve()
    submission_path = Path(submission).resolve()
    harness_path, submission_path, harness_source, submission_source = _load_sources(
        harness_path, submission_path
    )

    print(f"[INFO] pygpubench source: {PYGPUBENCH_GIT_URL}")
    print(f"[INFO] harness: {harness_path}")
    print(f"[INFO] submission: {submission_path}")
    print(f"[INFO] gpu: {gpu}")

    # Modal function decorator uses static gpu value. Keep this guard explicit.
    if gpu != "B200":
        raise ValueError("This runner is currently pinned to gpu='B200'.")

    result = run_harness.remote(
        harness_source=harness_source,
        submission_source=submission_source,
        harness_name=harness_path.name,
    )

    json_out_path = Path(json_out).resolve()
    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    json_out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] wrote {json_out_path}")
    print(f"[INFO] exit_code={result['exit_code']}")
    if print_log:
        if result.get("stdout"):
            print("=== stdout ===")
            print(result["stdout"], end="")
        if result.get("stderr"):
            print("=== stderr ===", file=sys.stderr)
            print(result["stderr"], end="", file=sys.stderr)

    if result["exit_code"] != 0:
        raise SystemExit(result["exit_code"])
