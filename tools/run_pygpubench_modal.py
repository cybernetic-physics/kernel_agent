#!/usr/bin/env python3
"""Invoke deployed PyGPUBench Modal runner class.

Usage:
  1) Deploy once:
       uv run --with modal modal deploy tools/pygpubench_modal_app.py
  2) Run harnesses repeatedly with low client overhead:
       uv run --with modal python tools/run_pygpubench_modal.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

from pygpubench_modal_app import (
    APP_NAME,
    CLASS_NAME,
    PROFILE_TIMEOUTS_SECONDS,
    PYGPUBENCH_GIT_URL,
)


def _find_workspace_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "tools").exists():
            return parent
    raise RuntimeError(f"Could not find workspace root from {start}")


def _load_sources(harness: Path, submission: Path) -> tuple[Path, Path, str, str]:
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


def _parse_env_pairs(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --env '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env '{item}'. Empty KEY.")
        env[key] = value
    return env


def _build_parser(default_harness: str, default_json_out: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deployed PyGPUBench harness on Modal B200.")
    parser.add_argument("--harness", default=default_harness, help="Absolute/relative harness path.")
    parser.add_argument("--submission", required=True, help="Absolute/relative submission path.")
    parser.add_argument(
        "--profile",
        default="candidate",
        choices=sorted(PROFILE_TIMEOUTS_SECONDS.keys()),
        help="Execution profile (controls timeout + harness env defaults).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Override process timeout inside container.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override PYGPUBENCH_REPEATS for harnesses honoring env controls.",
    )
    parser.add_argument(
        "--stage-repeats",
        default=None,
        help="Override PYGPUBENCH_STAGE_REPEATS (comma-separated ints).",
    )
    parser.add_argument(
        "--early-stop-us",
        type=float,
        default=None,
        help="Override PYGPUBENCH_EARLY_STOP_US (0 disables early stop).",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Additional env in KEY=VALUE form (repeatable).",
    )
    parser.add_argument(
        "--json-out",
        default=default_json_out,
        help="Path to write JSON result.",
    )
    parser.add_argument("--print-log", action="store_true", help="Print remote stdout/stderr.")
    parser.add_argument("--app-name", default=APP_NAME, help="Deployed Modal app name.")
    parser.add_argument("--class-name", default=CLASS_NAME, help="Deployed Modal class name.")
    return parser


def main() -> int:
    this_file = Path(__file__).resolve()
    workspace_root = _find_workspace_root(this_file)

    parser = _build_parser(
        default_harness=str(workspace_root / "tools" / "pygpubench_harness_template.py"),
        default_json_out=str(workspace_root / "artifacts" / "pygpubench_modal_last_run.json"),
    )
    args = parser.parse_args()

    harness_path, submission_path, harness_source, submission_source = _load_sources(
        Path(args.harness), Path(args.submission)
    )

    extra_env = _parse_env_pairs(args.env)
    if args.repeats is not None:
        extra_env["PYGPUBENCH_REPEATS"] = str(args.repeats)
    if args.stage_repeats is not None:
        extra_env["PYGPUBENCH_STAGE_REPEATS"] = args.stage_repeats
    if args.early_stop_us is not None:
        extra_env["PYGPUBENCH_EARLY_STOP_US"] = str(args.early_stop_us)

    print(f"[INFO] mode: deployed")
    print(f"[INFO] app: {args.app_name}")
    print(f"[INFO] class: {args.class_name}")
    print("[INFO] gpu: B200 (fixed)")
    print(f"[INFO] profile: {args.profile}")
    print(f"[INFO] timeout_seconds: {args.timeout_seconds or PROFILE_TIMEOUTS_SECONDS[args.profile]}")
    print(f"[INFO] pygpubench source: {PYGPUBENCH_GIT_URL}")
    print(f"[INFO] harness: {harness_path}")
    print(f"[INFO] submission: {submission_path}")

    cls = modal.Cls.from_name(args.app_name, args.class_name)
    runner = cls()

    result = runner.run_harness.remote(
        harness_source=harness_source,
        submission_source=submission_source,
        harness_name=harness_path.name,
        profile=args.profile,
        timeout_seconds=args.timeout_seconds,
        extra_env=extra_env,
    )

    json_out_path = Path(args.json_out).resolve()
    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    json_out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"[INFO] wrote {json_out_path}")
    print(f"[INFO] exit_code={result['exit_code']}")
    print(f"[INFO] timed_out={result.get('timed_out', False)}")
    if args.print_log:
        if result.get("stdout"):
            print("=== stdout ===")
            print(result["stdout"], end="")
        if result.get("stderr"):
            print("=== stderr ===", file=sys.stderr)
            print(result["stderr"], end="", file=sys.stderr)

    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
