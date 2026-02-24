#!/usr/bin/env python3
"""Tiny terminal UI for monitoring (and optionally launching) loop_coordinator."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    if len(lines) <= limit:
        return lines
    return lines[-limit:]


def _iter_dirs(state_root: Path) -> list[Path]:
    out: list[Path] = []
    if not state_root.exists():
        return out
    for p in state_root.iterdir():
        if p.is_dir() and p.name.startswith("iter_"):
            out.append(p)
    return sorted(out)


def _best_time_snapshot(state_root: Path) -> tuple[float | None, str]:
    best: float | None = None
    best_src = ""
    for it in _iter_dirs(state_root):
        progress = _read_json(it / "worker_progress.json") or {}
        p_val = progress.get("best_time_us")
        if isinstance(p_val, (int, float)) and not isinstance(p_val, bool):
            pv = float(p_val)
            if best is None or pv < best:
                best = pv
                best_src = f"{it.name}/worker_progress.json"
        wr = _read_json(it / "worker_result.json") or {}
        wv = wr.get("metric_value")
        if isinstance(wv, (int, float)) and not isinstance(wv, bool):
            vv = float(wv)
            if vv < 1.0e17 and (best is None or vv < best):
                best = vv
                best_src = f"{it.name}/worker_result.json"
    return best, best_src


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _request_stop(control_path: Path, reason: str) -> None:
    payload = _read_json(control_path) or {}
    payload["stop"] = True
    payload["reason"] = reason
    payload["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    control_path.parent.mkdir(parents=True, exist_ok=True)
    control_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass
class UiState:
    state_root: Path
    refresh_seconds: float
    tail_lines: int
    no_clear: bool
    child: subprocess.Popen[str] | None

    @property
    def control_path(self) -> Path:
        return self.state_root / "control.json"

    @property
    def heartbeat_path(self) -> Path:
        return self.state_root / "heartbeat.json"

    @property
    def run_log_path(self) -> Path:
        return self.state_root / "run.log"

    @property
    def lock_path(self) -> Path:
        return self.state_root / "lock" / "active.lock"


def _render(ui: UiState) -> str:
    hb = _read_json(ui.heartbeat_path) or {}
    ctl = _read_json(ui.control_path) or {}
    lock = _read_json(ui.lock_path) or {}
    log_tail = _tail_lines(ui.run_log_path, ui.tail_lines)

    current_iteration = int(hb.get("current_iteration", 0))
    phase = str(hb.get("phase", "INIT"))
    metric_name = str(hb.get("metric_name", ""))
    last_metric = hb.get("last_metric_value")
    best_metric = hb.get("best_metric_value")
    best_iteration = hb.get("best_iteration")
    no_progress_count = hb.get("no_progress_count", 0)
    infra_failure_count = hb.get("infra_failure_count", 0)
    target_conf = hb.get("target_confirmation_count", 0)
    hb_updated = hb.get("updated_at", "n/a")
    global_best, global_best_src = _best_time_snapshot(ui.state_root)

    lock_pid = lock.get("pid")
    lock_alive = _is_pid_alive(int(lock_pid)) if isinstance(lock_pid, int) else False

    child_status = "n/a"
    if ui.child is not None:
        if ui.child.poll() is None:
            child_status = f"running (pid={ui.child.pid})"
        else:
            child_status = f"exited (code={ui.child.returncode})"

    iter_dir = ui.state_root / f"iter_{current_iteration:04d}" if current_iteration > 0 else None
    file_checks: list[tuple[str, bool]] = []
    if iter_dir is not None:
        file_checks = [
            ("worker_prompt.txt", (iter_dir / "worker_prompt.txt").exists()),
            ("worker_progress.json", (iter_dir / "worker_progress.json").exists()),
            ("worker_result.json", (iter_dir / "worker_result.json").exists()),
            ("reviewer_prompt.txt", (iter_dir / "reviewer_prompt.txt").exists()),
            ("reviewer_verdict.json", (iter_dir / "reviewer_verdict.json").exists()),
            ("metrics_snapshot.json", (iter_dir / "metrics_snapshot.json").exists()),
            ("status.json", (iter_dir / "status.json").exists()),
        ]
    progress = _read_json(iter_dir / "worker_progress.json") if iter_dir is not None else {}
    if progress is None:
        progress = {}

    lines: list[str] = []
    lines.append("Loop Coordinator TUI")
    lines.append("=" * 80)
    lines.append(f"UTC now: {_utc_now()}")
    lines.append(f"state_root: {ui.state_root}")
    lines.append(f"coordinator lock pid: {lock_pid} ({'alive' if lock_alive else 'not-alive'})")
    lines.append(f"launcher child: {child_status}")
    lines.append(
        f"phase={phase} iteration={current_iteration} heartbeat_updated={hb_updated} stop={bool(ctl.get('stop', False))}"
    )
    lines.append(
        f"metric={metric_name} last={last_metric} best={best_metric} best_iter={best_iteration}"
    )
    lines.append(
        f"no_progress={no_progress_count} infra_failures={infra_failure_count} target_confirmations={target_conf}"
    )
    lines.append(
        f"global_best_time_us={global_best} source={global_best_src if global_best_src else 'n/a'}"
    )
    if ctl.get("reason"):
        lines.append(f"control.reason: {ctl.get('reason')}")

    if iter_dir is not None:
        lines.append("-" * 80)
        lines.append(f"Iteration files: {iter_dir}")
        for name, exists in file_checks:
            status = "[x]" if exists else "[ ]"
            lines.append(f"{status} {name}")
        lines.append(
            "worker_progress: "
            f"status={progress.get('status', 'n/a')} "
            f"last_time_us={progress.get('last_time_us')} "
            f"best_time_us={progress.get('best_time_us')} "
            f"updated_at={progress.get('updated_at', 'n/a')}"
        )
        note = progress.get("note")
        if isinstance(note, str) and note.strip():
            lines.append(f"worker_progress.note: {note.strip()}")

    lines.append("-" * 80)
    lines.append(f"Recent run log ({ui.tail_lines} lines)")
    if log_tail:
        lines.extend(log_tail)
    else:
        lines.append("(no log output yet)")

    lines.append("-" * 80)
    lines.append("Controls: Ctrl+C to exit UI. If running child, UI requests graceful stop.")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1]
    default_state = default_repo / "artifacts" / "loop_coordinator"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-root", type=Path, default=default_state)
    p.add_argument("--refresh-seconds", type=float, default=1.0)
    p.add_argument("--tail-lines", type=int, default=30)
    p.add_argument("--no-clear", action="store_true")
    p.add_argument(
        "--run",
        action="store_true",
        help="Run command after '--' and monitor it.",
    )
    p.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run when using --run. Example: --run -- python3 tools/loop_coordinator.py ...",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    state_root = args.state_root.resolve()
    child: subprocess.Popen[str] | None = None

    if args.run:
        cmd = list(args.cmd)
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            raise SystemExit("--run requires a command after '--'")
        child = subprocess.Popen(cmd, text=True, start_new_session=True)

    ui = UiState(
        state_root=state_root,
        refresh_seconds=max(0.2, args.refresh_seconds),
        tail_lines=max(1, args.tail_lines),
        no_clear=args.no_clear,
        child=child,
    )

    try:
        while True:
            if not ui.no_clear:
                # ANSI clear + cursor home.
                sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(_render(ui))
            sys.stdout.flush()

            if ui.child is not None and ui.child.poll() is not None:
                return int(ui.child.returncode or 0)
            time.sleep(ui.refresh_seconds)
    except KeyboardInterrupt:
        if ui.child is not None and ui.child.poll() is None:
            _request_stop(ui.control_path, "UI interrupted by operator")
            try:
                os.killpg(ui.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                ui.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(ui.child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
