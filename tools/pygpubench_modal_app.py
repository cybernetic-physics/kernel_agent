#!/usr/bin/env python3
"""Deployed Modal app for running PyGPUBench harnesses on B200.

This app is designed for reuse via `modal.Cls.from_name(...)` to avoid per-run
cold startup overhead from `modal run`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "pygpubench-modal-runner"
CLASS_NAME = "PyGPUBenchRunner"
DEFAULT_IMAGE = "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04"
PYGPUBENCH_GIT_URL = "git+https://github.com/ngc92/pygpubench.git"
CACHE_VOLUME_NAME = "pygpubench-cache-v1"
CACHE_MOUNT_PATH = "/cache"

PROFILE_TIMEOUTS_SECONDS = {
    "smoke": 180,
    "candidate": 420,
    "final": 900,
}

# Conservative defaults; can be overridden by caller through `extra_env`.
PROFILE_ENV_DEFAULTS = {
    "smoke": {
        "PYGPUBENCH_REPEATS": "20",
        "PYGPUBENCH_STAGE_REPEATS": "8,12",
        "PYGPUBENCH_EARLY_STOP_US": "1200",
    },
    "candidate": {
        "PYGPUBENCH_REPEATS": "60",
        "PYGPUBENCH_STAGE_REPEATS": "12,24,24",
        "PYGPUBENCH_EARLY_STOP_US": "0",
    },
    "final": {
        "PYGPUBENCH_REPEATS": "100",
        "PYGPUBENCH_STAGE_REPEATS": "20,40,40",
        "PYGPUBENCH_EARLY_STOP_US": "0",
    },
}


def _build_image() -> modal.Image:
    base_image = (
        modal.Image.from_registry(DEFAULT_IMAGE, add_python="3.12")
        .entrypoint([])
        .apt_install("build-essential", "cmake", "ninja-build", "git")
        .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
        .uv_pip_install("nvidia-cutlass-dsl==4.4.0")
        .uv_pip_install("scikit-build-core>=0.11", "nanobind")
    )
    return base_image.run_commands(
        "CC=gcc CXX=g++ CUDACXX=/usr/local/cuda/bin/nvcc "
        f"python -m pip install -v {PYGPUBENCH_GIT_URL}"
    )


image = _build_image()
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu="B200",
    timeout=max(PROFILE_TIMEOUTS_SECONDS.values()) + 120,
    scaledown_window=300,
    volumes={CACHE_MOUNT_PATH: cache_volume},
)
class PyGPUBenchRunner:
    """Runs a staged harness/submission pair in an isolated Modal container."""

    @modal.enter()
    def _setup_cache_dirs(self) -> None:
        for rel in [
            "torch",
            "torchinductor",
            "triton",
            "cuda",
            "xdg",
            "pycache",
        ]:
            (Path(CACHE_MOUNT_PATH) / rel).mkdir(parents=True, exist_ok=True)

    @modal.method()
    def run_harness(
        self,
        harness_source: str,
        submission_source: str,
        harness_name: str,
        profile: str = "candidate",
        timeout_seconds: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if profile not in PROFILE_TIMEOUTS_SECONDS:
            raise ValueError(
                f"Unknown profile '{profile}'. Expected one of: "
                + ", ".join(PROFILE_TIMEOUTS_SECONDS.keys())
            )

        process_timeout = int(timeout_seconds or PROFILE_TIMEOUTS_SECONDS[profile])
        stage_dir = Path(tempfile.mkdtemp(prefix="modal-pygpubench-"))
        start = time.time()

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
            env["PYTHONPATH"] = f"{stage_dir}:{env.get('PYTHONPATH', '')}"

            # Persistent caches cut repeated compile cost across runs.
            env["XDG_CACHE_HOME"] = f"{CACHE_MOUNT_PATH}/xdg"
            env["TORCHINDUCTOR_CACHE_DIR"] = f"{CACHE_MOUNT_PATH}/torchinductor"
            env["TRITON_CACHE_DIR"] = f"{CACHE_MOUNT_PATH}/triton"
            env["CUDA_CACHE_PATH"] = f"{CACHE_MOUNT_PATH}/cuda"
            env["PYTHONPYCACHEPREFIX"] = f"{CACHE_MOUNT_PATH}/pycache"

            env.update(PROFILE_ENV_DEFAULTS.get(profile, {}))
            if extra_env:
                env.update({k: str(v) for k, v in extra_env.items()})

            try:
                proc = subprocess.run(
                    ["python", harness_path.name],
                    cwd=stage_dir,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=process_timeout,
                )
                result = {
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "timed_out": False,
                }
            except subprocess.TimeoutExpired as exc:
                timeout_stderr = (exc.stderr or "") + (
                    f"\n[TIMEOUT] Harness exceeded {process_timeout}s "
                    f"(profile={profile})."
                )
                result = {
                    "exit_code": 124,
                    "stdout": exc.stdout or "",
                    "stderr": timeout_stderr,
                    "timed_out": True,
                }

            result.update(
                {
                    "harness_name": harness_name,
                    "profile": profile,
                    "timeout_seconds": process_timeout,
                    "elapsed_seconds": round(time.time() - start, 3),
                    "stage_dir": str(stage_dir),
                    "cache_volume": CACHE_VOLUME_NAME,
                }
            )
            return result
        finally:
            subprocess.run(["rm", "-rf", str(stage_dir)], check=False)
            cache_volume.commit()
