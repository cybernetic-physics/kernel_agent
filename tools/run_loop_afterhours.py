#!/usr/bin/env python3
"""Launch afterhours loop through loop_tui with key=value overrides."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DEFAULTS = {
    "kernel": "kernels/nvfp4_group_gemm_001.py",
    "harness": "tools/pygpubench_nvfp4_silu_dual_gemm_harness.py",
    "target_threshold": "10",
    "max_iterations": "400",
    "max_wall_clock_minutes": "1440",
    "no_progress_limit": "20",
    "infra_failure_limit": "5",
}


def _parse_overrides(items: list[str]) -> dict[str, str]:
    values = dict(DEFAULTS)
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected key=value."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in values:
            allowed = ", ".join(sorted(values.keys()))
            raise ValueError(f"Unknown override '{key}'. Allowed keys: {allowed}")
        if not value:
            raise ValueError(f"Override '{key}' must have non-empty value.")
        values[key] = value
    return values


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Optional key=value overrides. "
            "Allowed: kernel, harness, target_threshold, max_iterations, "
            "max_wall_clock_minutes, no_progress_limit, infra_failure_limit."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command without executing.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    values = _parse_overrides(args.overrides)

    cmd = [
        "python3",
        "tools/loop_tui.py",
        "--run",
        "--",
        "python3",
        "tools/loop_coordinator.py",
        "--fresh-start",
        "--execution-mode",
        "command",
        "--worker-prompt-template",
        "prompts/kernel_optimization_prompt_afterhours_001.md",
        "--reviewer-prompt-template",
        "prompts/reviewer_prompt_001.md",
        "--worker-cmd-template",
        (
            "python3 tools/loop_worker_driver.py "
            "--repo-root {repo_root} "
            "--prompt-file {worker_prompt} "
            "--output-json {worker_result} "
            "--iteration {iteration} "
            f"--kernel-path {values['kernel']} "
            "--metric-name gmean_us "
            "--timeout-seconds 1800"
        ),
        "--reviewer-cmd-template",
        (
            "python3 tools/loop_reviewer_driver.py "
            "--repo-root {repo_root} "
            "--prompt-file {reviewer_prompt} "
            "--worker-result {worker_result} "
            "--output-json {reviewer_verdict} "
            "--iteration {iteration} "
            "--timeout-seconds 900"
        ),
        "--kernel-path",
        values["kernel"],
        "--harness-path",
        values["harness"],
        "--target-metric-name",
        "gmean_us",
        "--target-metric-threshold",
        values["target_threshold"],
        "--metric-direction",
        "min",
        "--max-iterations",
        values["max_iterations"],
        "--max-wall-clock-minutes",
        values["max_wall_clock_minutes"],
        "--target-confirmations",
        "2",
        "--no-progress-limit",
        values["no_progress_limit"],
        "--infra-failure-limit",
        values["infra_failure_limit"],
        "--command-timeout-seconds",
        "1900",
    ]

    if args.dry_run:
        print("Resolved command:")
        print(" ".join(cmd))
        return 0

    os.chdir(repo_root)
    os.execvp(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
