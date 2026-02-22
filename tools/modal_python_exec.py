#!/usr/bin/env python3
"""Execute arbitrary Python code in deployed ModalToolsRunner (default GPU: B200).

This runner uses the deployed app in `tools/modal_tools_app.py` and supports
optional directory mounts by shipping tar.gz archives into the remote stage dir.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import modal

from modal_tools_app import APP_NAME as DEFAULT_APP_NAME
from modal_tools_app import CLASS_NAME as DEFAULT_CLASS_NAME

DEFAULT_GPU = "B200"


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
        help=f"Deployed Modal app name (default: {DEFAULT_APP_NAME}).",
    )
    p.add_argument(
        "--class-name",
        type=str,
        default=DEFAULT_CLASS_NAME,
        help=f"Deployed Modal class name (default: {DEFAULT_CLASS_NAME}).",
    )
    p.add_argument("--gpu", type=str, default=DEFAULT_GPU, help="GPU type (default: B200).")
    p.add_argument(
        "--profile",
        default="candidate",
        choices=["smoke", "candidate", "final"],
        help="Deployed runner profile (default: candidate).",
    )
    p.add_argument(
        "--mount-repo",
        action="store_true",
        help="Mount the local repo into the remote stage dir and prepend it to sys.path.",
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
            "Disable repo snapshot copy before archiving. "
            "By default a temp snapshot is used to avoid file-modified races."
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
        "--exec-timeout",
        type=int,
        default=600,
        help="Snippet execution timeout in seconds (default: 600).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write JSON result.",
    )
    p.add_argument(
        "--allow-nonzero",
        action="store_true",
        help="Do not exit non-zero when remote python exits non-zero.",
    )
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


def _parse_mount(spec: str) -> tuple[Path, str]:
    parts = spec.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise SystemExit(
            f"Invalid --mount-local-dir spec {spec!r}; expected 'local_path:remote_path'."
        )
    local = Path(parts[0]).expanduser().resolve()
    if not local.exists() or not local.is_dir():
        raise SystemExit(f"Mount local dir not found or not dir: {local}")
    remote = parts[1]
    return local, remote


def _snapshot_repo(local_repo: Path) -> tuple[tempfile.TemporaryDirectory[str], Path]:
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
    return tmp, dst


def _norm_remote_dir(remote: str) -> str:
    # Stage dir relative target; keep compatibility with absolute user paths.
    rel = remote.lstrip("/")
    if not rel or rel.startswith("../") or "/../" in rel:
        raise SystemExit(f"Invalid remote mount path: {remote!r}")
    return rel


def _add_dir_files_b64(files_b64: dict[str, str], src_dir: Path, dst_root_rel: str) -> None:
    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_dir).as_posix()
        remote_rel = f"{dst_root_rel}/{rel}" if rel else dst_root_rel
        files_b64[remote_rel] = base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    args = _parse_args()
    if args.gpu != "B200":
        raise SystemExit("modal_python_exec.py is B200-only with deployed ModalToolsRunner.")

    code = _read_code(args)
    if not code.endswith("\n"):
        code += "\n"

    snapshot_tmp: tempfile.TemporaryDirectory[str] | None = None
    files_b64: dict[str, str] = {}

    try:
        if args.mount_repo:
            repo_local = args.repo_local_path.resolve()
            if not repo_local.exists() or not repo_local.is_dir():
                raise SystemExit(f"Repo local path not found or not dir: {repo_local}")
            if args.no_repo_snapshot:
                src_repo = repo_local
            else:
                snapshot_tmp, src_repo = _snapshot_repo(repo_local)

            repo_remote_rel = _norm_remote_dir(args.repo_remote_path)
            _add_dir_files_b64(files_b64, src_repo, repo_remote_rel)

            # Make mounted repo importable without user boilerplate.
            prelude = (
                "import os, sys\n"
                f"_repo = os.path.join(os.environ.get('MODAL_STAGE_DIR', ''), {repo_remote_rel!r})\n"
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

        for mount_spec in args.mount_local_dir:
            local_dir, remote_dir = _parse_mount(mount_spec)
            remote_rel = _norm_remote_dir(remote_dir)
            _add_dir_files_b64(files_b64, local_dir, remote_rel)

        cls = modal.Cls.from_name(args.app_name, args.class_name)
        runner = cls()
        remote = runner.run_python.remote(
            snippet_source=code,
            files_b64=files_b64 or None,
            profile=args.profile,
            timeout_seconds=args.exec_timeout,
            extra_env=None,
        )
    finally:
        if snapshot_tmp is not None:
            snapshot_tmp.cleanup()

    exit_code = int(remote.get("exit_code", 1))
    stdout = str(remote.get("stdout", "") or "")
    stderr = str(remote.get("stderr", "") or "")

    result = {
        "app_name": args.app_name,
        "class_name": args.class_name,
        "gpu": args.gpu,
        "profile": args.profile,
        "exec_timeout_s": args.exec_timeout,
        "exit_code": exit_code,
        "timed_out": bool(remote.get("timed_out", False)),
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
