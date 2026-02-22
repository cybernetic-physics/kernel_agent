#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
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
DEFAULT_PROFILE = "candidate"


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _directive_value(kernel_text: str, key: str) -> str | None:
    patterns = [
        rf"^#!AFTERHOURS\s+{re.escape(key)}\s+(.+)$",
        rf"^#!PYGPUBENCH\s+{re.escape(key)}\s+(.+)$",
        rf"^#!POPCORN\s+{re.escape(key)}\s+(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, kernel_text, re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


def _safe_tag(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    return out.strip("._-") or "run"


def main() -> int:
    p = argparse.ArgumentParser(
        description=f"Run deployed PyGPUBench in {DEFAULT_MODE} mode with git context artifact."
    )
    p.add_argument("--kernel", required=True, help="Kernel file path (absolute or repo-relative).")
    p.add_argument("--harness", default="", help="Harness file path (absolute or repo-relative).")
    p.add_argument("--profile", default=DEFAULT_PROFILE, help="Runner profile: smoke|candidate|final.")
    p.add_argument("--gpu", default="B200", help="GPU target (must be B200 for this runner).")
    p.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Git repo root.")
    p.add_argument("--artifact", default="", help="Artifact output path. Defaults under repo_root/artifacts/.")
    p.add_argument("--json-out", default="", help="JSON result output path (optional).")
    p.add_argument("--tag", default="", help="Tag added to default artifact filename.")
    p.add_argument("--timeout-seconds", type=int, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--stage-repeats", default=None)
    p.add_argument("--early-stop-us", type=float, default=None)
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

    harness_val = args.harness or _directive_value(kernel_text, "harness")
    if not harness_val:
        raise SystemExit("Harness not provided. Pass --harness or add '#!AFTERHOURS harness <path>' in kernel.")

    harness_in = Path(harness_val).expanduser()
    harness_abs = (repo_root / harness_in).resolve() if not harness_in.is_absolute() else harness_in.resolve()
    if not harness_abs.exists():
        raise SystemExit(f"Harness file not found: {harness_abs}")

    if args.gpu != "B200":
        raise SystemExit("afterhours runner is B200-only. Use --gpu B200.")

    git_head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if git_head.returncode != 0:
        raise SystemExit(f"Unable to read git HEAD:\n{git_head.stderr}")
    git_hash = git_head.stdout.strip()

    if args.artifact:
        artifact_path = Path(args.artifact).expanduser().resolve()
    else:
        ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        tag = _safe_tag(args.tag) if args.tag else ts
        artifact_path = (repo_root / "artifacts" / f"{kernel_abs.stem}.afterhours_{DEFAULT_MODE}.{tag}.txt").resolve()

    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    if args.json_out:
        json_out_path = Path(args.json_out).expanduser().resolve()
    else:
        json_out_path = artifact_path.with_suffix(".json")

    tracked = _run(["git", "ls-files", "--error-unmatch", str(kernel_rel)], cwd=repo_root)
    if tracked.returncode == 0:
        diff_cmd = ["git", "--no-pager", "diff", git_hash, "--", str(kernel_rel)]
    else:
        diff_cmd = ["git", "--no-pager", "diff", "--no-index", "--", "/dev/null", str(kernel_abs)]
    diff_res = _run(diff_cmd, cwd=repo_root)
    diff_text = diff_res.stdout.strip() or "(no kernel diff vs git hash)"

    submit_cmd = [
        "bash",
        "skills/codex/pygpubench-modal-run/scripts/run_pygpubench_modal.sh",
        "--harness",
        str(harness_abs),
        "--submission",
        str(kernel_rel),
        "--profile",
        args.profile,
        "--json-out",
        str(json_out_path),
        "--print-log",
    ]

    if args.timeout_seconds is not None:
        submit_cmd += ["--timeout-seconds", str(args.timeout_seconds)]
    if args.repeats is not None:
        submit_cmd += ["--repeats", str(args.repeats)]
    if args.stage_repeats is not None:
        submit_cmd += ["--stage-repeats", str(args.stage_repeats)]
    if args.early_stop_us is not None:
        submit_cmd += ["--early-stop-us", str(args.early_stop_us)]

    now = dt.datetime.now(dt.UTC).isoformat()
    header = [
        "# AFTERHOURS PyGPUBench Artifact",
        f"timestamp_utc: {now}",
        f"mode: {DEFAULT_MODE}",
        f"repo_root: {repo_root}",
        f"git_hash: {git_hash}",
        f"kernel_file: {kernel_abs}",
        f"kernel_file_relative: {kernel_rel}",
        f"harness_file: {harness_abs}",
        f"profile: {args.profile}",
        f"gpu: B200",
        f"json_out: {json_out_path}",
        f"command: {' '.join(shlex.quote(x) for x in submit_cmd)}",
        "",
        "## Kernel Diff (against git_hash)",
        "```diff",
        diff_text,
        "```",
        "",
        "## runner output",
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

        if json_out_path.exists():
            f.write("\n## json_result\n")
            try:
                obj = json.loads(json_out_path.read_text(encoding="utf-8"))
                f.write("```json\n")
                f.write(json.dumps(obj, indent=2))
                f.write("\n```\n")
            except Exception as exc:  # noqa: BLE001
                f.write(f"(failed to parse JSON result: {exc})\n")

    if args.artifact_only:
        print(str(artifact_path))
    else:
        print(f"ARTIFACT_PATH={artifact_path}")

    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
