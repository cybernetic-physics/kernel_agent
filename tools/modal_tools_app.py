#!/usr/bin/env python3
"""Deployed Modal app for general B200 tool snippet execution.

Used by multiple tools under `tools/*_modal.py` to avoid repeated modal CLI
startup and to share persistent compile/runtime caches across runs.
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile
import time
import tarfile
from pathlib import Path
from typing import Any

import modal

APP_NAME = "kernel-agents-modal-tools"
CLASS_NAME = "ModalToolsRunner"
DEFAULT_IMAGE = "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04"
CACHE_VOLUME_NAME = "modal-tools-cache-v1"
CACHE_MOUNT_PATH = "/cache"

PROFILE_TIMEOUTS_SECONDS = {
    "smoke": 180,
    "candidate": 600,
    "final": 1500,
}

base_image = (
    modal.Image.from_registry(DEFAULT_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install(
        "build-essential",
        "cmake",
        "ninja-build",
        "git",
        "nsight-systems-2025.5.2",
        "nsight-compute-2025.4.0",
    )
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("nvidia-cutlass-dsl==4.4.0")
)

app = modal.App(APP_NAME, image=base_image)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


@app.cls(
    gpu="B200",
    timeout=max(PROFILE_TIMEOUTS_SECONDS.values()) + 120,
    scaledown_window=300,
    volumes={CACHE_MOUNT_PATH: cache_volume},
)
class ModalToolsRunner:
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
    def run_python(
        self,
        snippet_source: str,
        files_b64: dict[str, str] | None = None,
        archives_b64: dict[str, str] | None = None,
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
        stage_dir = Path(tempfile.mkdtemp(prefix="modal-tools-snippet-"))
        start = time.time()

        try:
            (stage_dir / "snippet.py").write_text(snippet_source, encoding="utf-8")

            for rel, b64_data in (files_b64 or {}).items():
                rel_path = Path(rel)
                if rel_path.is_absolute() or ".." in rel_path.parts:
                    raise ValueError(f"Invalid file path in files_b64: {rel!r}")
                dst = stage_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(base64.b64decode(b64_data.encode("ascii")))

            for rel_dir, b64_data in (archives_b64 or {}).items():
                rel_path = Path(rel_dir)
                if rel_path.is_absolute() or ".." in rel_path.parts:
                    raise ValueError(f"Invalid archive target in archives_b64: {rel_dir!r}")
                dst_dir = stage_dir / rel_path
                dst_dir.mkdir(parents=True, exist_ok=True)
                raw = base64.b64decode(b64_data.encode("ascii"))
                with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
                    tf.extractall(path=dst_dir)

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["MODAL_STAGE_DIR"] = str(stage_dir)

            env["XDG_CACHE_HOME"] = f"{CACHE_MOUNT_PATH}/xdg"
            env["TORCHINDUCTOR_CACHE_DIR"] = f"{CACHE_MOUNT_PATH}/torchinductor"
            env["TRITON_CACHE_DIR"] = f"{CACHE_MOUNT_PATH}/triton"
            env["CUDA_CACHE_PATH"] = f"{CACHE_MOUNT_PATH}/cuda"
            env["PYTHONPYCACHEPREFIX"] = f"{CACHE_MOUNT_PATH}/pycache"

            if extra_env:
                env.update({k: str(v) for k, v in extra_env.items()})

            try:
                proc = subprocess.run(
                    ["python", "snippet.py"],
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
                    f"\n[TIMEOUT] Snippet exceeded {process_timeout}s "
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
