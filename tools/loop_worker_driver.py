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


def _run_progress_report(
    repo_root: Path,
    progress_json: Path,
    *,
    status: str,
    note: str,
    metric_name: str,
    iteration: int,
    time_us: float | None = None,
    artifact: str = "",
    errors: list[str] | None = None,
) -> None:
    cmd = [
        "python3",
        "tools/loop_best_time_report.py",
        "--progress-json",
        str(progress_json),
        "--status",
        status,
        "--note",
        note,
        "--metric-name",
        metric_name,
        "--iteration",
        str(iteration),
    ]
    if time_us is not None:
        cmd.extend(["--time-us", str(float(time_us))])
    if artifact:
        cmd.extend(["--artifact", artifact])
    for err in errors or []:
        cmd.extend(["--error", err])
    subprocess.run(
        cmd,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )


def _render_reporting_prompt(repo_root: Path, progress_json: Path) -> str:
    template = repo_root / "prompts" / "worker_best_time_reporting_prompt.md"
    if not template.exists():
        return (
            "Use skill `loop-best-time-report` and report live updates via:\n"
            f"python3 tools/loop_best_time_report.py --progress-json \"{progress_json}\" ..."
        )
    text = template.read_text(encoding="utf-8")
    return text.replace("<progress_json_path>", str(progress_json))


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    prompt_file = args.prompt_file.resolve()
    output_json = args.output_json.resolve()
    output_dir = output_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_json = output_dir / "worker_progress.json"
    transcript_path = output_dir / "worker_claude_output.txt"
    stderr_path = output_dir / "worker_claude_error.txt"
    _run_progress_report(
        repo_root,
        progress_json,
        status="running",
        note="worker iteration started",
        metric_name=args.metric_name,
        iteration=args.iteration,
    )

    base_prompt = prompt_file.read_text(encoding="utf-8")
    reporting_prompt = _render_reporting_prompt(repo_root, progress_json)
    extra_prompt = f"""

Coordinator constraints for this single iteration:
- Perform exactly one kernel-side change at most.
- You must execute afterhours validation flow from the prompt and collect evidence.
- Required skill to use for live progress: `loop-best-time-report`.
- Progress file for loop TUI: {progress_json}
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

Live best-time reporting prompt:
{reporting_prompt}
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
        _run_progress_report(
            repo_root,
            progress_json,
            status="blocked",
            note=f"worker timeout after {args.timeout_seconds}s",
            metric_name=args.metric_name,
            iteration=args.iteration,
            errors=payload["errors"],
        )
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
        _run_progress_report(
            repo_root,
            progress_json,
            status="blocked",
            note="worker missing required worker_result.json",
            metric_name=args.metric_name,
            iteration=args.iteration,
            errors=payload["errors"],
        )

    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        payload = _fallback_payload(args, [f"worker JSON parse failure: {exc}"])
        _write_json(output_json, payload)
        _run_progress_report(
            repo_root,
            progress_json,
            status="blocked",
            note="worker_result.json parse failure",
            metric_name=args.metric_name,
            iteration=args.iteration,
            errors=payload["errors"],
        )
        return proc.returncode if proc.returncode != 0 else 1

    errs = validate_worker_result(payload)
    if errs:
        payload = _fallback_payload(args, [f"worker schema validation failed: {e}" for e in errs])
        _write_json(output_json, payload)
        _run_progress_report(
            repo_root,
            progress_json,
            status="blocked",
            note="worker_result schema validation failed",
            metric_name=args.metric_name,
            iteration=args.iteration,
            errors=payload["errors"],
        )
        return proc.returncode if proc.returncode != 0 else 1

    artifact = ""
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list):
        for item in artifacts:
            if isinstance(item, str) and item.strip():
                artifact = item
                break

    metric_value = payload.get("metric_value")
    metric_time: float | None = None
    if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
        # Ignore sentinel fallback-like values.
        if float(metric_value) < 1.0e17:
            metric_time = float(metric_value)

    status = "completed"
    errors_raw = payload.get("errors")
    report_errors: list[str] = []
    if isinstance(errors_raw, list):
        report_errors = [str(e) for e in errors_raw if isinstance(e, str)]
    if payload.get("decision") == "REVERT" and report_errors:
        status = "blocked"
    _run_progress_report(
        repo_root,
        progress_json,
        status=status,
        note=f"worker finished with decision={payload.get('decision')}",
        metric_name=str(payload.get("metric_name", args.metric_name)),
        iteration=args.iteration,
        time_us=metric_time,
        artifact=artifact,
        errors=report_errors,
    )

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
