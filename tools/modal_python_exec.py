#!/usr/bin/env python3
"""Execute arbitrary Python code in a Modal Sandbox (default GPU: B200).

This is a convenience runner for Codex-style snippets such as:
  python3 - <<'PY'
  import pkgutil, cutlass
  ...
  PY

You can pass code directly with --code, as a file with --code-file,
or via stdin (pipe/heredoc).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import modal

DEFAULT_APP_NAME = "codex-modal-python-exec"
DEFAULT_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
DEFAULT_GPU = "B200"
DEFAULT_PYTHON = "3.11"
DEFAULT_PIP = ("nvidia-cutlass-dsl==4.4.0", "torch")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "snippet",
        nargs="?",
        default=None,
        help="Python snippet to execute remotely (alternative to --code).",
    )
    p.add_argument("--code", type=str, default=None, help="Python code string.")
    p.add_argument(
        "--code-file",
        type=Path,
        default=None,
        help="Path to a .py file whose contents will be executed remotely.",
    )
    p.add_argument(
        "--app-name",
        type=str,
        default=DEFAULT_APP_NAME,
        help=f"Modal app name for sandbox lifecycle (default: {DEFAULT_APP_NAME}).",
    )
    p.add_argument("--gpu", type=str, default=DEFAULT_GPU, help="GPU type (default: B200).")
    p.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help=f"Base container image (default: {DEFAULT_IMAGE}).",
    )
    p.add_argument(
        "--python-version",
        type=str,
        default=DEFAULT_PYTHON,
        help=f"Python version in remote image (default: {DEFAULT_PYTHON}).",
    )
    p.add_argument(
        "--mount-repo",
        action="store_true",
        help="Mount the local repo into the sandbox and prepend it to sys.path.",
    )
    p.add_argument(
        "--repo-local-path",
        type=Path,
        default=Path("."),
        help="Local repo path to mount when --mount-repo is set (default: .).",
    )
    p.add_argument(
        "--repo-remote-path",
        type=str,
        default="/workspace/repo",
        help="Remote mount path for --mount-repo (default: /workspace/repo).",
    )
    p.add_argument(
        "--no-repo-snapshot",
        action="store_true",
        help=(
            "Disable repo snapshot copy before mounting. "
            "By default a temp snapshot is used to avoid 'file modified during build' races."
        ),
    )
    p.add_argument(
        "--mount-local-dir",
        action="append",
        default=[],
        help=(
            "Extra local:remote directory mounts. Repeatable. "
            "Example: --mount-local-dir '/tmp/data:/workspace/data'"
        ),
    )
    p.add_argument(
        "--pip",
        action="append",
        default=[],
        help="Extra pip package(s) to install in remote image. Repeatable.",
    )
    p.add_argument(
        "--no-default-pip",
        action="store_true",
        help="Do not install default packages (nvidia-cutlass-dsl, torch).",
    )
    p.add_argument(
        "--sandbox-timeout",
        type=int,
        default=900,
        help="Sandbox lifetime timeout in seconds (default: 900).",
    )
    p.add_argument(
        "--exec-timeout",
        type=int,
        default=600,
        help="Command execution timeout in seconds (default: 600).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write JSON result.",
    )
    p.add_argument(
        "--verbose-modal",
        action="store_true",
        help="Enable Modal output stream (image build/sandbox lifecycle logs).",
    )
    p.add_argument(
        "--allow-nonzero",
        action="store_true",
        help="Do not exit non-zero when remote python exits non-zero.",
    )
    # Modal CLI may inject argv fragments.
    args, _unknown = p.parse_known_args()
    return args


def _read_code(args: argparse.Namespace) -> str:
    sources = [
        args.code_file is not None,
        args.code is not None,
        args.snippet is not None,
    ]
    if sum(1 for x in sources if x) > 1:
        raise SystemExit("Use only one of: positional snippet, --code, --code-file.")

    if args.code_file is not None:
        if not args.code_file.exists():
            raise SystemExit(f"Code file not found: {args.code_file}")
        return args.code_file.read_text(encoding="utf-8")

    if args.code is not None:
        return args.code

    if args.snippet is not None:
        return args.snippet

    if not sys.stdin.isatty():
        return sys.stdin.read()

    raise SystemExit("No code provided. Pass --code / --code-file / snippet or pipe stdin.")


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _parse_mount(spec: str) -> tuple[str, str]:
    parts = spec.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise SystemExit(
            f"Invalid --mount-local-dir spec {spec!r}; expected 'local_path:remote_path'."
        )
    local = str(Path(parts[0]).resolve())
    remote = parts[1]
    return local, remote


def _snapshot_repo(local_repo: Path) -> tuple[tempfile.TemporaryDirectory[str], str]:
    tmp = tempfile.TemporaryDirectory(prefix="modal_repo_snapshot_")
    dst = Path(tmp.name) / "repo"
    ignore = shutil.ignore_patterns(
        ".git",
        "__pycache__",
        "*.pyc",
        ".venv",
        "artifacts",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    )
    shutil.copytree(local_repo, dst, ignore=ignore)
    return tmp, str(dst)


def main() -> None:
    args = _parse_args()
    code = _read_code(args)
    if not code.endswith("\n"):
        code += "\n"

    # If requested, make mounted repo importable without user boilerplate.
    if args.mount_repo:
        repo = args.repo_remote_path
        prelude = (
            "import os, sys\n"
            f"_repo = {repo!r}\n"
            "if _repo not in sys.path:\n"
            "    sys.path.insert(0, _repo)\n"
            "os.chdir(_repo)\n"
            "try:\n"
            "    import task as _task  # noqa: F401\n"
            "except Exception:\n"
            "    import types\n"
            "    _task_mod = types.ModuleType('task')\n"
            "    _task_mod.input_t = tuple\n"
            "    _task_mod.output_t = list\n"
            "    sys.modules['task'] = _task_mod\n"
        )
        code = prelude + code

    packages: list[str] = []
    if not args.no_default_pip:
        packages.extend(DEFAULT_PIP)
    packages.extend(args.pip)

    image = modal.Image.from_registry(args.image, add_python=args.python_version)
    if packages:
        image = image.pip_install(*packages)

    snapshot_tmp: tempfile.TemporaryDirectory[str] | None = None

    if args.mount_repo:
        repo_local = args.repo_local_path.resolve()
        if args.no_repo_snapshot:
            local_repo = str(repo_local)
        else:
            snapshot_tmp, local_repo = _snapshot_repo(repo_local)
        image = image.add_local_dir(local_repo, remote_path=args.repo_remote_path)

    for mount_spec in args.mount_local_dir:
        local_path, remote_path = _parse_mount(mount_spec)
        image = image.add_local_dir(local_path, remote_path=remote_path)

    app = modal.App.lookup(args.app_name, create_if_missing=True)

    ctx = modal.enable_output() if args.verbose_modal else contextlib.nullcontext()
    with ctx:
        sb = modal.Sandbox.create(
            app=app,
            image=image,
            gpu=args.gpu,
            timeout=args.sandbox_timeout,
        )

    try:
        proc = sb.exec("python3", "-", timeout=args.exec_timeout)
        proc.stdin.write(code.encode("utf-8"))
        proc.stdin.write_eof()
        proc.stdin.drain()
        exit_code = int(proc.wait())
        stdout = _to_text(proc.stdout.read())
        stderr = _to_text(proc.stderr.read())
    finally:
        sb.terminate()
        if snapshot_tmp is not None:
            snapshot_tmp.cleanup()

    result = {
        "app_name": args.app_name,
        "gpu": args.gpu,
        "image": args.image,
        "python_version": args.python_version,
        "packages": packages,
        "sandbox_timeout_s": args.sandbox_timeout,
        "exec_timeout_s": args.exec_timeout,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote: {args.json_out}", file=sys.stderr)

    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)

    if exit_code != 0 and not args.allow_nonzero:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
