#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
import shlex
import subprocess
from pathlib import Path


def _default_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()


DEFAULT_REPO_ROOT = _default_repo_root()
DEFAULT_MODE = "benchmark"


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _directive_value(kernel_text: str, key: str) -> str | None:
    pat = re.compile(rf"^#!POPCORN\s+{re.escape(key)}\s+(.+)$", re.MULTILINE)
    m = pat.search(kernel_text)
    return m.group(1).strip() if m else None


def _safe_tag(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    return out.strip("._-") or "run"


def main() -> int:
    p = argparse.ArgumentParser(description=f"Run popcorn-cli in {DEFAULT_MODE} mode with git context artifact.")
    p.add_argument("--kernel", required=True, help="Kernel file path (absolute or repo-relative).")
    p.add_argument("--leaderboard", default="", help="Leaderboard name. Defaults from kernel directives.")
    p.add_argument("--gpu", default="", help="GPU target. Defaults from kernel directives, then B200.")
    p.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Git repo root.")
    p.add_argument("--artifact", default="", help="Artifact output path. Defaults under repo_root/artifacts/.")
    p.add_argument("--tag", default="", help="Tag added to default artifact filename.")
    p.add_argument("--artifact-only", action="store_true", help="Print only artifact path on success.")
    args = p.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.exists():
        raise SystemExit(f"Repo root does not exist: {repo_root}")

    kernel_in = Path(args.kernel).expanduser()
    kernel_abs = (repo_root / kernel_in).resolve() if not kernel_in.is_absolute() else kernel_in.resolve()
    if not kernel_abs.exists():
        raise SystemExit(f"Kernel file not found: {kernel_abs}")

    try:
        kernel_rel = kernel_abs.relative_to(repo_root)
    except ValueError as e:
        raise SystemExit(f"Kernel file must be under repo root ({repo_root}): {kernel_abs}") from e

    if not (repo_root / ".git").exists():
        raise SystemExit(f"Repo root is not a git repository: {repo_root}")

    kernel_text = kernel_abs.read_text(encoding="utf-8", errors="ignore")
    leaderboard = args.leaderboard or _directive_value(kernel_text, "leaderboard")
    gpu = args.gpu or _directive_value(kernel_text, "gpu") or "B200"
    if not leaderboard:
        raise SystemExit("Leaderboard not provided and not found in kernel directives.")

    git_head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if git_head.returncode != 0:
        raise SystemExit(f"Unable to read git HEAD:\n{git_head.stderr}")
    git_hash = git_head.stdout.strip()

    if args.artifact:
        artifact_path = Path(args.artifact).expanduser().resolve()
    else:
        ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        tag = _safe_tag(args.tag) if args.tag else ts
        artifact_path = (repo_root / "artifacts" / f"{kernel_abs.stem}.{DEFAULT_MODE}.{tag}.txt").resolve()

    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    tracked = _run(["git", "ls-files", "--error-unmatch", str(kernel_rel)], cwd=repo_root)
    if tracked.returncode == 0:
        diff_cmd = ["git", "--no-pager", "diff", git_hash, "--", str(kernel_rel)]
    else:
        diff_cmd = ["git", "--no-pager", "diff", "--no-index", "--", "/dev/null", str(kernel_abs)]
    diff_res = _run(diff_cmd, cwd=repo_root)
    diff_text = diff_res.stdout.strip()
    if not diff_text:
        diff_text = "(no kernel diff vs git hash)"

    submit_cmd = [
        "uv", "run", "popcorn-cli", "submit", "--no-tui",
        "--gpu", gpu,
        "--leaderboard", leaderboard,
        "--mode", DEFAULT_MODE,
        str(kernel_rel),
    ]

    now = dt.datetime.now(dt.UTC).isoformat()
    header = [
        "# POPCORN Submission Artifact",
        f"timestamp_utc: {now}",
        f"mode: {DEFAULT_MODE}",
        f"repo_root: {repo_root}",
        f"git_hash: {git_hash}",
        f"kernel_file: {kernel_abs}",
        f"kernel_file_relative: {kernel_rel}",
        f"leaderboard: {leaderboard}",
        f"gpu: {gpu}",
        f"command: {' '.join(shlex.quote(x) for x in submit_cmd)}",
        "",
        "## Kernel Diff (against git_hash)",
        "```diff",
        diff_text,
        "```",
        "",
        "## popcorn-cli output",
    ]

    with artifact_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")

    run = _run(submit_cmd, cwd=repo_root)

    with artifact_path.open("a", encoding="utf-8") as f:
        if run.stdout:
            f.write(run.stdout)
            if not run.stdout.endswith("\n"):
                f.write("\n")
        if run.stderr:
            f.write("\n## stderr\n")
            f.write(run.stderr)
            if not run.stderr.endswith("\n"):
                f.write("\n")
        f.write(f"\n## exit_code\n{run.returncode}\n")

    if args.artifact_only:
        print(str(artifact_path))
    else:
        print(f"ARTIFACT_PATH={artifact_path}")

    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
