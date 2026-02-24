#!/usr/bin/env python3
"""Run one worker Claude iteration and enforce worker_result.json output."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from loop_verdict_schema import validate_worker_result
except ImportError:  # pragma: no cover
    from tools.loop_verdict_schema import validate_worker_result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--kernel-path", type=str, required=True)
    p.add_argument("--metric-name", default="gmean_us")
    p.add_argument("--timeout-seconds", type=int, default=7200)
    p.add_argument("--model", default="")
    return p.parse_args()


def _fallback_payload(args: argparse.Namespace, errors: list[str]) -> dict[str, Any]:
    return {
        "iteration": args.iteration,
        "kernel_path": args.kernel_path,
        "tests_passed": False,
        "benchmark_passed": False,
        "metric_name": args.metric_name,
        "metric_value": 1.0e18,
        "decision": "REVERT",
        "artifacts": [],
        "errors": errors,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    prompt_file = args.prompt_file.resolve()
    output_json = args.output_json.resolve()
    output_dir = output_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "worker_claude_output.txt"
    stderr_path = output_dir / "worker_claude_error.txt"

    base_prompt = prompt_file.read_text(encoding="utf-8")
    extra_prompt = f"""

Coordinator constraints for this single iteration:
- Perform exactly one kernel-side change at most.
- You must execute afterhours validation flow from the prompt and collect evidence.
- You must finish by writing JSON to: {output_json}
- JSON schema required:
  {{
    "iteration": <int>,
    "kernel_path": "<string>",
    "tests_passed": <bool>,
    "benchmark_passed": <bool>,
    "metric_name": "<string>",
    "metric_value": <number>,
    "decision": "KEEP|REVERT",
    "artifacts": ["<path>", ...],
    "errors": ["<error>", ...]
  }}
- Use absolute or repo-relative artifact paths.
- If benchmark/test cannot run, set decision to REVERT and include errors.
- Return after writing the JSON file.
"""
    full_prompt = base_prompt.rstrip() + "\n" + extra_prompt.strip() + "\n"

    cmd = [
        "claude",
        "-p",
        "--permission-mode",
        "bypassPermissions",
        "--dangerously-skip-permissions",
    ]
    if args.model.strip():
        cmd.extend(["--model", args.model.strip()])
    cmd.extend(["--add-dir", str(repo_root)])

    try:
        proc = subprocess.run(
            cmd,
            input=full_prompt,
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        payload = _fallback_payload(
            args,
            [f"worker Claude timeout after {args.timeout_seconds} seconds"],
        )
        _write_json(output_json, payload)
        transcript_path.write_text("", encoding="utf-8")
        stderr_path.write_text(
            f"timeout after {args.timeout_seconds} seconds\n", encoding="utf-8"
        )
        return 124

    transcript_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    if not output_json.exists():
        payload = _fallback_payload(
            args, [f"worker did not produce required JSON ({output_json})"]
        )
        _write_json(output_json, payload)

    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        payload = _fallback_payload(args, [f"worker JSON parse failure: {exc}"])
        _write_json(output_json, payload)
        return proc.returncode if proc.returncode != 0 else 1

    errs = validate_worker_result(payload)
    if errs:
        payload = _fallback_payload(args, [f"worker schema validation failed: {e}" for e in errs])
        _write_json(output_json, payload)
        return proc.returncode if proc.returncode != 0 else 1

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
