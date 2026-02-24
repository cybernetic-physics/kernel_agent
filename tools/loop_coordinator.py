#!/usr/bin/env python3
"""External worker/reviewer loop coordinator for kernel optimization runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from loop_verdict_schema import validate_reviewer_verdict, validate_worker_result
except ImportError:  # pragma: no cover - fallback import path
    from tools.loop_verdict_schema import validate_reviewer_verdict, validate_worker_result


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_timestamp(value: str) -> datetime | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized).astimezone(UTC)
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{_utc_iso()}] {message}\n")


def _is_better(candidate: float, best: float | None, direction: str) -> bool:
    if best is None:
        return True
    if direction == "min":
        return candidate < best
    return candidate > best


def _meets_target(candidate: float, threshold: float | None, direction: str) -> bool:
    if threshold is None:
        return False
    if direction == "min":
        return candidate <= threshold
    return candidate >= threshold


def _parse_key_value_pairs(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --var value '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --var value '{item}'. KEY must be non-empty.")
        out[key] = value
    return out


def _apply_angle_placeholders(template: str, values: dict[str, str]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace(f"<{key}>", str(value))
    return out


def _load_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_prompt(template_text: str, values: dict[str, str], iteration: int) -> str:
    rendered = _apply_angle_placeholders(template_text, values)
    header = (
        f"# loop iteration: {iteration}\n"
        f"# generated_utc: {_utc_iso()}\n\n"
    )
    return header + rendered.strip() + "\n"


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "tools").exists():
            return parent
    raise RuntimeError(f"Could not determine repo root from: {start}")


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _run_command(
    cmd: str,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int | None,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        return proc
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        timeout_note = (
            f"\n[loop_coordinator] command timed out after {timeout_seconds} seconds."
        )
        stdout_path.write_text(out, encoding="utf-8")
        stderr_path.write_text(err + timeout_note, encoding="utf-8")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=out,
            stderr=err + timeout_note,
        )


def _wait_for_file(path: Path, timeout_seconds: int, poll_seconds: int = 2) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(poll_seconds)
    return path.exists()


def _capture_git_diff(repo_root: Path, patch_path: Path, kernel_path: Path | None) -> None:
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "-C", str(repo_root), "diff"]
    if kernel_path is not None:
        rel = _relative_to_repo(kernel_path, repo_root)
        cmd.extend(["--", rel])
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        patch_path.write_text(
            f"[git diff error]\nreturncode={proc.returncode}\n{proc.stderr}",
            encoding="utf-8",
        )
        return
    patch_path.write_text(proc.stdout, encoding="utf-8")


def _load_and_validate_worker_result(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"worker_result missing: {path}"]
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        return None, [f"worker_result invalid JSON: {exc}"]
    except OSError as exc:
        return None, [f"worker_result read error: {exc}"]
    errors = validate_worker_result(payload)
    return payload, errors


def _load_and_validate_reviewer_verdict(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"reviewer_verdict missing: {path}"]
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        return None, [f"reviewer_verdict invalid JSON: {exc}"]
    except OSError as exc:
        return None, [f"reviewer_verdict read error: {exc}"]
    errors = validate_reviewer_verdict(payload)
    return payload, errors


def _build_cmd_context(
    iteration: int,
    state_root: Path,
    iter_dir: Path,
    worker_prompt_path: Path,
    reviewer_prompt_path: Path,
    worker_result_path: Path,
    reviewer_verdict_path: Path,
    status_path: Path,
    repo_root: Path,
) -> dict[str, str]:
    return {
        "iteration": str(iteration),
        "state_root": str(state_root),
        "iter_dir": str(iter_dir),
        "worker_prompt": str(worker_prompt_path),
        "reviewer_prompt": str(reviewer_prompt_path),
        "worker_result": str(worker_result_path),
        "reviewer_verdict": str(reviewer_verdict_path),
        "status_file": str(status_path),
        "repo_root": str(repo_root),
    }


def _render_cmd_template(template: str, context: dict[str, str]) -> str:
    missing: list[str] = []

    class _Safe(dict[str, str]):
        def __missing__(self, key: str) -> str:
            missing.append(key)
            return "{" + key + "}"

    rendered = template.format_map(_Safe(context))
    if missing:
        keys = ", ".join(sorted(set(missing)))
        raise KeyError(f"Missing placeholders in command template: {keys}")
    return rendered


def _simulate_worker_result(
    iteration: int,
    metric_name: str,
    kernel_path: str,
    worker_result_path: Path,
    target_threshold: float | None,
    direction: str,
) -> dict[str, Any]:
    baseline = 500.0
    metric_value = max(0.5, baseline - (iteration * 15.0))
    if target_threshold is not None and _meets_target(metric_value, target_threshold, direction):
        decision = "KEEP"
    else:
        decision = "KEEP" if iteration % 7 != 0 else "REVERT"

    payload: dict[str, Any] = {
        "iteration": iteration,
        "kernel_path": kernel_path,
        "tests_passed": True,
        "benchmark_passed": True,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "decision": decision,
        "artifacts": [],
        "errors": [],
    }
    _write_json(worker_result_path, payload)
    return payload


def _simulate_reviewer_verdict(
    iteration: int,
    worker_payload: dict[str, Any],
    reviewer_verdict_path: Path,
    target_threshold: float | None,
    direction: str,
) -> dict[str, Any]:
    metric_value = float(worker_payload["metric_value"])
    if target_threshold is not None and _meets_target(metric_value, target_threshold, direction):
        verdict = "STOP_TARGET_REACHED"
        reason = "Target threshold reached in dry-run mode."
    else:
        verdict = "CONTINUE"
        reason = "Continue iterating."
    payload = {
        "iteration": iteration,
        "verdict": verdict,
        "confidence": "high",
        "reason": reason,
        "next_change_hint": "Try one additional kernel-side optimization.",
        "requires_revert": worker_payload.get("decision") == "REVERT",
    }
    _write_json(reviewer_verdict_path, payload)
    return payload


def _ensure_control_file(control_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if control_path.exists():
        try:
            payload = _read_json(control_path)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    payload = {
        "stop": False,
        "reason": "",
        "created_at": _utc_iso(),
        "updated_at": _utc_iso(),
        "config": config,
    }
    _write_json(control_path, payload)
    return payload


def _update_control(control_path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    payload = _read_json(control_path) if control_path.exists() else {}
    payload.update(updates)
    payload["updated_at"] = _utc_iso()
    _write_json(control_path, payload)
    return payload


def _read_control(control_path: Path) -> dict[str, Any]:
    if not control_path.exists():
        return {"stop": False}
    try:
        payload = _read_json(control_path)
    except Exception:
        return {"stop": False}
    if not isinstance(payload, dict):
        return {"stop": False}
    return payload


@dataclass
class LoopState:
    current_iteration: int
    best_metric_value: float | None
    best_iteration: int | None
    no_progress_count: int
    infra_failure_count: int
    target_confirmation_count: int
    last_metric_value: float | None
    phase: str


class CoordinatorLock:
    def __init__(self, lock_file: Path, heartbeat_file: Path, stale_after_seconds: int) -> None:
        self.lock_file = lock_file
        self.heartbeat_file = heartbeat_file
        self.stale_after_seconds = stale_after_seconds
        self.acquired = False

    def _is_stale(self) -> bool:
        if not self.heartbeat_file.exists():
            return True
        try:
            payload = _read_json(self.heartbeat_file)
        except Exception:
            return True
        ts = _parse_iso_timestamp(payload.get("updated_at", ""))
        if ts is None:
            return True
        age = (_utc_now() - ts).total_seconds()
        return age > self.stale_after_seconds

    def acquire(self) -> None:
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_file.exists():
            if self._is_stale():
                self.lock_file.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    "Coordinator lock already active and heartbeat is fresh. "
                    f"Lock: {self.lock_file}"
                )
        fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "created_at": _utc_iso(),
                    },
                    indent=2,
                )
            )
            f.write("\n")
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.lock_file.unlink(missing_ok=True)
            self.acquired = False


def _load_resume_state(heartbeat_path: Path) -> LoopState:
    if heartbeat_path.exists():
        try:
            hb = _read_json(heartbeat_path)
            return LoopState(
                current_iteration=int(hb.get("current_iteration", 0)),
                best_metric_value=(
                    float(hb["best_metric_value"])
                    if hb.get("best_metric_value") is not None
                    else None
                ),
                best_iteration=(
                    int(hb["best_iteration"]) if hb.get("best_iteration") is not None else None
                ),
                no_progress_count=int(hb.get("no_progress_count", 0)),
                infra_failure_count=int(hb.get("infra_failure_count", 0)),
                target_confirmation_count=int(hb.get("target_confirmation_count", 0)),
                last_metric_value=(
                    float(hb["last_metric_value"]) if hb.get("last_metric_value") is not None else None
                ),
                phase=str(hb.get("phase", "INIT")),
            )
        except Exception:
            pass
    return LoopState(
        current_iteration=0,
        best_metric_value=None,
        best_iteration=None,
        no_progress_count=0,
        infra_failure_count=0,
        target_confirmation_count=0,
        last_metric_value=None,
        phase="INIT",
    )


def _write_heartbeat(
    heartbeat_path: Path,
    state: LoopState,
    metric_name: str,
    stop_reason: str,
) -> None:
    payload = {
        "updated_at": _utc_iso(),
        "current_iteration": state.current_iteration,
        "phase": state.phase,
        "metric_name": metric_name,
        "last_metric_value": state.last_metric_value,
        "best_metric_value": state.best_metric_value,
        "best_iteration": state.best_iteration,
        "no_progress_count": state.no_progress_count,
        "infra_failure_count": state.infra_failure_count,
        "target_confirmation_count": state.target_confirmation_count,
        "stop_reason": stop_reason,
    }
    _write_json(heartbeat_path, payload)


def _parse_args() -> argparse.Namespace:
    default_repo = _find_repo_root(Path(__file__).resolve())
    default_state = default_repo / "artifacts" / "loop_coordinator"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=default_repo)
    p.add_argument("--state-root", type=Path, default=default_state)
    p.add_argument(
        "--execution-mode",
        choices=["command", "manual", "dry-run"],
        default="manual",
        help=(
            "command: run worker/reviewer command templates, "
            "manual: wait for JSON files produced by external sessions, "
            "dry-run: synthesize outputs."
        ),
    )
    p.add_argument(
        "--worker-cmd-template",
        default="",
        help=(
            "Shell template for worker execution in command mode. "
            "Available keys: {iteration}, {iter_dir}, {worker_prompt}, {worker_result}, "
            "{state_root}, {repo_root}, {status_file}."
        ),
    )
    p.add_argument(
        "--reviewer-cmd-template",
        default="",
        help=(
            "Shell template for reviewer execution in command mode. "
            "Available keys: {iteration}, {iter_dir}, {reviewer_prompt}, {reviewer_verdict}, "
            "{worker_result}, {state_root}, {repo_root}, {status_file}."
        ),
    )
    p.add_argument(
        "--worker-prompt-template",
        type=Path,
        default=default_repo / "prompts" / "kernel_optimization_prompt_001.md",
    )
    p.add_argument(
        "--reviewer-prompt-template",
        type=Path,
        default=default_repo / "prompts" / "reviewer_prompt_001.md",
    )
    p.add_argument("--max-iterations", type=int, default=40)
    p.add_argument("--max-wall-clock-minutes", type=int, default=360)
    p.add_argument("--target-confirmations", type=int, default=2)
    p.add_argument("--no-progress-limit", type=int, default=6)
    p.add_argument("--infra-failure-limit", type=int, default=3)
    p.add_argument("--wait-timeout-seconds", type=int, default=3600)
    p.add_argument("--command-timeout-seconds", type=int, default=3600)
    p.add_argument("--lock-timeout-minutes", type=int, default=10)
    p.add_argument("--metric-direction", choices=["min", "max"], default="min")
    p.add_argument("--target-metric-name", default="gmean_us")
    p.add_argument("--target-metric-threshold", type=float, default=None)
    p.add_argument("--kernel-path", type=Path, default=None)
    p.add_argument("--harness-path", type=Path, default=None)
    p.add_argument("--gpu-type", default="B200")
    p.add_argument(
        "--var",
        action="append",
        default=[],
        help="Additional prompt variables in KEY=VALUE format (repeatable).",
    )
    p.add_argument(
        "--capture-git-diff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture git diff patch for each iteration.",
    )
    p.add_argument(
        "--fresh-start",
        action="store_true",
        help="Delete existing state root before starting a new run.",
    )
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    state_root = args.state_root.resolve()
    control_path = state_root / "control.json"
    heartbeat_path = state_root / "heartbeat.json"
    run_log_path = state_root / "run.log"
    lock_file = state_root / "lock" / "active.lock"

    if args.max_iterations <= 0:
        raise SystemExit("--max-iterations must be > 0")

    if args.fresh_start and state_root.exists():
        shutil.rmtree(state_root)

    if args.execution_mode == "command":
        if not args.worker_cmd_template.strip():
            raise SystemExit("--worker-cmd-template is required in command mode")
        if not args.reviewer_cmd_template.strip():
            raise SystemExit("--reviewer-cmd-template is required in command mode")

    extra_vars = _parse_key_value_pairs(args.var)
    worker_template_text = _load_template(args.worker_prompt_template)
    reviewer_template_text = _load_template(args.reviewer_prompt_template)

    config = {
        "repo_root": str(repo_root),
        "state_root": str(state_root),
        "execution_mode": args.execution_mode,
        "max_iterations": args.max_iterations,
        "max_wall_clock_minutes": args.max_wall_clock_minutes,
        "target_confirmations": args.target_confirmations,
        "no_progress_limit": args.no_progress_limit,
        "infra_failure_limit": args.infra_failure_limit,
        "target_metric_name": args.target_metric_name,
        "target_metric_threshold": args.target_metric_threshold,
        "metric_direction": args.metric_direction,
    }
    _ensure_control_file(control_path, config)
    # Always clear stale stop state at startup. Runtime manual stop can set this again.
    _update_control(control_path, {"stop": False, "reason": ""})

    lock = CoordinatorLock(
        lock_file=lock_file,
        heartbeat_file=heartbeat_path,
        stale_after_seconds=args.lock_timeout_minutes * 60,
    )

    state = _load_resume_state(heartbeat_path)
    start_time = _utc_now()
    stop_reason = ""

    try:
        lock.acquire()
        _append_log(run_log_path, f"loop start execution_mode={args.execution_mode}")
        _write_heartbeat(heartbeat_path, state, args.target_metric_name, stop_reason)

        while state.current_iteration < args.max_iterations:
            control = _read_control(control_path)
            if bool(control.get("stop")):
                stop_reason = f"manual stop set in {control_path}"
                _append_log(run_log_path, stop_reason)
                break

            elapsed_minutes = (_utc_now() - start_time).total_seconds() / 60.0
            if elapsed_minutes >= args.max_wall_clock_minutes:
                stop_reason = "max wall clock limit reached"
                _append_log(run_log_path, stop_reason)
                break

            iteration = state.current_iteration + 1
            iter_dir = state_root / f"iter_{iteration:04d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            status_path = iter_dir / "status.json"
            worker_prompt_path = iter_dir / "worker_prompt.txt"
            reviewer_prompt_path = iter_dir / "reviewer_prompt.txt"
            worker_result_path = iter_dir / "worker_result.json"
            reviewer_verdict_path = iter_dir / "reviewer_verdict.json"
            metrics_snapshot_path = iter_dir / "metrics_snapshot.json"
            git_diff_path = iter_dir / "git_diff.patch"

            state.current_iteration = iteration
            state.phase = "RUN_WORKER"
            _write_heartbeat(heartbeat_path, state, args.target_metric_name, stop_reason)

            kernel_path = args.kernel_path.resolve() if args.kernel_path else None
            harness_path = args.harness_path.resolve() if args.harness_path else None
            prompt_values = {
                "repo_root": str(repo_root),
                "kernel_path": str(kernel_path) if kernel_path else "",
                "harness_path": str(harness_path) if harness_path else "",
                "gpu_type": args.gpu_type,
                "target_metric": args.target_metric_name,
                "target_metric_name": args.target_metric_name,
                "target_metric_threshold": (
                    str(args.target_metric_threshold)
                    if args.target_metric_threshold is not None
                    else ""
                ),
                "target_gmean_us": (
                    str(args.target_metric_threshold)
                    if args.target_metric_name == "gmean_us"
                    and args.target_metric_threshold is not None
                    else ""
                ),
                "iteration": str(iteration),
                "iteration_dir": str(iter_dir),
                "worker_result_path": str(worker_result_path),
                "reviewer_verdict_path": str(reviewer_verdict_path),
            }
            prompt_values.update(extra_vars)

            worker_prompt_text = _render_prompt(worker_template_text, prompt_values, iteration)
            worker_prompt_path.write_text(worker_prompt_text, encoding="utf-8")

            cmd_context = _build_cmd_context(
                iteration=iteration,
                state_root=state_root,
                iter_dir=iter_dir,
                worker_prompt_path=worker_prompt_path,
                reviewer_prompt_path=reviewer_prompt_path,
                worker_result_path=worker_result_path,
                reviewer_verdict_path=reviewer_verdict_path,
                status_path=status_path,
                repo_root=repo_root,
            )

            worker_payload: dict[str, Any] | None = None
            worker_errors: list[str] = []
            for attempt in (1, 2):
                _append_log(run_log_path, f"iteration={iteration} worker attempt={attempt}")
                if args.execution_mode == "dry-run":
                    worker_payload = _simulate_worker_result(
                        iteration=iteration,
                        metric_name=args.target_metric_name,
                        kernel_path=str(kernel_path or ""),
                        worker_result_path=worker_result_path,
                        target_threshold=args.target_metric_threshold,
                        direction=args.metric_direction,
                    )
                elif args.execution_mode == "manual":
                    found = _wait_for_file(worker_result_path, args.wait_timeout_seconds)
                    if not found:
                        worker_errors = [f"timed out waiting for {worker_result_path}"]
                else:
                    cmd_context["attempt"] = str(attempt)
                    command = _render_cmd_template(args.worker_cmd_template, cmd_context)
                    stdout_path = iter_dir / f"worker_attempt_{attempt}.stdout.log"
                    stderr_path = iter_dir / f"worker_attempt_{attempt}.stderr.log"
                    proc = _run_command(
                        command,
                        cwd=repo_root,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        timeout_seconds=args.command_timeout_seconds,
                    )
                    _append_log(
                        run_log_path,
                        f"iteration={iteration} worker attempt={attempt} returncode={proc.returncode} cmd={shlex.quote(command)}",
                    )

                if worker_payload is None:
                    worker_payload, worker_errors = _load_and_validate_worker_result(worker_result_path)

                if worker_errors:
                    _append_log(
                        run_log_path,
                        f"iteration={iteration} worker validation failed attempt={attempt}: "
                        + "; ".join(worker_errors),
                    )
                    if attempt == 1:
                        time.sleep(args.sleep_seconds)
                        continue
                break

            if worker_payload is None or worker_errors:
                state.infra_failure_count += 1
                state.phase = "APPLY_VERDICT"
                status_payload = {
                    "iteration": iteration,
                    "state": "worker_failed",
                    "errors": worker_errors or ["unknown worker failure"],
                    "updated_at": _utc_iso(),
                }
                _write_json(status_path, status_payload)
                _write_heartbeat(heartbeat_path, state, args.target_metric_name, stop_reason)
                if state.infra_failure_count >= args.infra_failure_limit:
                    stop_reason = "infra failure limit reached after worker failures"
                    _append_log(run_log_path, stop_reason)
                    break
                _append_log(run_log_path, "worker failed; continuing to next iteration")
                time.sleep(args.sleep_seconds)
                continue

            state.infra_failure_count = 0
            metric_name = str(worker_payload.get("metric_name", args.target_metric_name))
            metric_value = float(worker_payload["metric_value"])
            state.last_metric_value = metric_value

            if _is_better(metric_value, state.best_metric_value, args.metric_direction):
                state.best_metric_value = metric_value
                state.best_iteration = iteration
                state.no_progress_count = 0
            else:
                state.no_progress_count += 1

            if _meets_target(metric_value, args.target_metric_threshold, args.metric_direction):
                state.target_confirmation_count += 1
            else:
                state.target_confirmation_count = 0

            if args.capture_git_diff:
                _capture_git_diff(repo_root, git_diff_path, kernel_path)

            _write_json(
                metrics_snapshot_path,
                {
                    "iteration": iteration,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "best_metric_value": state.best_metric_value,
                    "best_iteration": state.best_iteration,
                    "no_progress_count": state.no_progress_count,
                    "target_confirmation_count": state.target_confirmation_count,
                    "updated_at": _utc_iso(),
                },
            )

            reviewer_values = dict(prompt_values)
            reviewer_values.update(
                {
                    "worker_result_path": str(worker_result_path),
                    "git_diff_patch_path": str(git_diff_path),
                    "metrics_snapshot_path": str(metrics_snapshot_path),
                    "best_metric_value": (
                        str(state.best_metric_value) if state.best_metric_value is not None else ""
                    ),
                }
            )
            reviewer_prompt_text = _render_prompt(
                reviewer_template_text, reviewer_values, iteration
            )
            reviewer_prompt_path.write_text(reviewer_prompt_text, encoding="utf-8")

            state.phase = "RUN_REVIEWER"
            _write_heartbeat(heartbeat_path, state, metric_name, stop_reason)

            reviewer_payload: dict[str, Any] | None = None
            reviewer_errors: list[str] = []
            for attempt in (1, 2):
                _append_log(run_log_path, f"iteration={iteration} reviewer attempt={attempt}")
                if args.execution_mode == "dry-run":
                    reviewer_payload = _simulate_reviewer_verdict(
                        iteration=iteration,
                        worker_payload=worker_payload,
                        reviewer_verdict_path=reviewer_verdict_path,
                        target_threshold=args.target_metric_threshold,
                        direction=args.metric_direction,
                    )
                elif args.execution_mode == "manual":
                    found = _wait_for_file(reviewer_verdict_path, args.wait_timeout_seconds)
                    if not found:
                        reviewer_errors = [f"timed out waiting for {reviewer_verdict_path}"]
                else:
                    cmd_context["attempt"] = str(attempt)
                    command = _render_cmd_template(args.reviewer_cmd_template, cmd_context)
                    stdout_path = iter_dir / f"reviewer_attempt_{attempt}.stdout.log"
                    stderr_path = iter_dir / f"reviewer_attempt_{attempt}.stderr.log"
                    proc = _run_command(
                        command,
                        cwd=repo_root,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        timeout_seconds=args.command_timeout_seconds,
                    )
                    _append_log(
                        run_log_path,
                        f"iteration={iteration} reviewer attempt={attempt} returncode={proc.returncode} cmd={shlex.quote(command)}",
                    )

                if reviewer_payload is None:
                    reviewer_payload, reviewer_errors = _load_and_validate_reviewer_verdict(
                        reviewer_verdict_path
                    )

                if reviewer_errors:
                    _append_log(
                        run_log_path,
                        f"iteration={iteration} reviewer validation failed attempt={attempt}: "
                        + "; ".join(reviewer_errors),
                    )
                    if attempt == 1:
                        time.sleep(args.sleep_seconds)
                        continue
                break

            if reviewer_payload is None or reviewer_errors:
                state.infra_failure_count += 1
                stop_reason = "reviewer verdict missing/invalid after retry"
                status_payload = {
                    "iteration": iteration,
                    "state": "reviewer_failed",
                    "errors": reviewer_errors or ["unknown reviewer failure"],
                    "updated_at": _utc_iso(),
                }
                _write_json(status_path, status_payload)
                _append_log(run_log_path, stop_reason)
                break

            state.infra_failure_count = 0
            verdict = str(reviewer_payload["verdict"])
            state.phase = "APPLY_VERDICT"

            status_payload = {
                "iteration": iteration,
                "state": "apply_verdict",
                "worker_metric_name": metric_name,
                "worker_metric_value": metric_value,
                "reviewer_verdict": verdict,
                "reviewer_confidence": reviewer_payload.get("confidence"),
                "requires_revert": reviewer_payload.get("requires_revert"),
                "updated_at": _utc_iso(),
            }
            _write_json(status_path, status_payload)
            _write_heartbeat(heartbeat_path, state, metric_name, stop_reason)

            if verdict != "CONTINUE":
                stop_reason = f"reviewer requested stop: {verdict}"
                _append_log(run_log_path, stop_reason)
                break

            if state.no_progress_count >= args.no_progress_limit:
                stop_reason = "no progress limit reached"
                _append_log(run_log_path, stop_reason)
                break

            if state.target_confirmation_count >= args.target_confirmations:
                stop_reason = "target confirmed for required consecutive iterations"
                _append_log(run_log_path, stop_reason)
                break

            time.sleep(args.sleep_seconds)

        if not stop_reason and state.current_iteration >= args.max_iterations:
            stop_reason = "max iterations reached"

        _update_control(control_path, {"stop": True, "reason": stop_reason})
        state.phase = "STOPPED"
        _write_heartbeat(heartbeat_path, state, args.target_metric_name, stop_reason)
        _append_log(run_log_path, f"loop stopped reason={stop_reason}")
        print(json.dumps({"status": "stopped", "reason": stop_reason, "iteration": state.current_iteration}, indent=2))
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
