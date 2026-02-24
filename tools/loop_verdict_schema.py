#!/usr/bin/env python3
"""Validate worker/reviewer JSON payloads for the loop coordinator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VALID_DECISIONS = {"KEEP", "REVERT"}
VALID_VERDICTS = {"CONTINUE", "STOP_TARGET_REACHED", "STOP_NO_PROGRESS", "STOP_BLOCKED"}
VALID_CONFIDENCE = {"low", "medium", "high"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expect_object(payload: Any, errors: list[str], prefix: str = "payload") -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        errors.append(f"{prefix} must be a JSON object.")
        return None
    return payload


def _require_key(data: dict[str, Any], key: str, errors: list[str], prefix: str = "payload") -> None:
    if key not in data:
        errors.append(f"{prefix}.{key} is required.")


def _require_int_ge(data: dict[str, Any], key: str, min_value: int, errors: list[str], prefix: str = "payload") -> None:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{prefix}.{key} must be an integer.")
        return
    if value < min_value:
        errors.append(f"{prefix}.{key} must be >= {min_value}.")


def _require_bool(data: dict[str, Any], key: str, errors: list[str], prefix: str = "payload") -> None:
    if not isinstance(data.get(key), bool):
        errors.append(f"{prefix}.{key} must be a boolean.")


def _require_str_nonempty(data: dict[str, Any], key: str, errors: list[str], prefix: str = "payload") -> None:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}.{key} must be a non-empty string.")


def _require_string_array(data: dict[str, Any], key: str, errors: list[str], prefix: str = "payload") -> None:
    value = data.get(key)
    if not isinstance(value, list):
        errors.append(f"{prefix}.{key} must be an array of strings.")
        return
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{prefix}.{key}[{idx}] must be a string.")


def validate_worker_result(payload: Any) -> list[str]:
    """Return validation errors for a worker_result payload."""
    errors: list[str] = []
    data = _expect_object(payload, errors, prefix="worker_result")
    if data is None:
        return errors

    required_keys = (
        "iteration",
        "kernel_path",
        "tests_passed",
        "benchmark_passed",
        "metric_name",
        "metric_value",
        "decision",
        "artifacts",
        "errors",
    )
    for key in required_keys:
        _require_key(data, key, errors, prefix="worker_result")

    if "iteration" in data:
        _require_int_ge(data, "iteration", 1, errors, prefix="worker_result")
    if "kernel_path" in data:
        _require_str_nonempty(data, "kernel_path", errors, prefix="worker_result")
    if "tests_passed" in data:
        _require_bool(data, "tests_passed", errors, prefix="worker_result")
    if "benchmark_passed" in data:
        _require_bool(data, "benchmark_passed", errors, prefix="worker_result")
    if "metric_name" in data:
        _require_str_nonempty(data, "metric_name", errors, prefix="worker_result")
    if "metric_value" in data and not _is_number(data.get("metric_value")):
        errors.append("worker_result.metric_value must be a number.")

    decision = data.get("decision")
    if decision is not None and decision not in VALID_DECISIONS:
        errors.append(
            f"worker_result.decision must be one of {sorted(VALID_DECISIONS)}."
        )
    if "artifacts" in data:
        _require_string_array(data, "artifacts", errors, prefix="worker_result")
    if "errors" in data:
        _require_string_array(data, "errors", errors, prefix="worker_result")
    return errors


def validate_reviewer_verdict(payload: Any) -> list[str]:
    """Return validation errors for a reviewer_verdict payload."""
    errors: list[str] = []
    data = _expect_object(payload, errors, prefix="reviewer_verdict")
    if data is None:
        return errors

    required_keys = (
        "iteration",
        "verdict",
        "confidence",
        "reason",
        "next_change_hint",
        "requires_revert",
    )
    for key in required_keys:
        _require_key(data, key, errors, prefix="reviewer_verdict")

    if "iteration" in data:
        _require_int_ge(data, "iteration", 1, errors, prefix="reviewer_verdict")

    verdict = data.get("verdict")
    if verdict is not None and verdict not in VALID_VERDICTS:
        errors.append(
            f"reviewer_verdict.verdict must be one of {sorted(VALID_VERDICTS)}."
        )

    confidence = data.get("confidence")
    if confidence is not None and confidence not in VALID_CONFIDENCE:
        errors.append(
            f"reviewer_verdict.confidence must be one of {sorted(VALID_CONFIDENCE)}."
        )

    if "reason" in data:
        _require_str_nonempty(data, "reason", errors, prefix="reviewer_verdict")
    if "next_change_hint" in data:
        _require_str_nonempty(data, "next_change_hint", errors, prefix="reviewer_verdict")
    if "requires_revert" in data:
        _require_bool(data, "requires_revert", errors, prefix="reviewer_verdict")
    return errors


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kind",
        required=True,
        choices=["worker", "reviewer"],
        help="Schema kind to validate against.",
    )
    p.add_argument(
        "--json-file",
        type=Path,
        required=True,
        help="Path to JSON file to validate.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = json.loads(args.json_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"[ERROR] file not found: {args.json_file}")
        return 2
    except json.JSONDecodeError as exc:
        print(f"[ERROR] invalid JSON in {args.json_file}: {exc}")
        return 2

    if args.kind == "worker":
        errors = validate_worker_result(payload)
    else:
        errors = validate_reviewer_verdict(payload)

    if errors:
        print(f"[INVALID] {args.kind} payload failed validation ({len(errors)} issue(s))")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"[VALID] {args.kind} payload: {args.json_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
