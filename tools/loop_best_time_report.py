#!/usr/bin/env python3
"""Update worker best-time progress JSON consumed by loop_tui."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--progress-json", type=Path, required=True)
    p.add_argument("--status", default="update")
    p.add_argument("--note", default="")
    p.add_argument("--metric-name", default="gmean_us")
    p.add_argument("--time-us", type=float, default=None)
    p.add_argument("--artifact", default="")
    p.add_argument("--iteration", type=int, default=None)
    p.add_argument("--source", default="worker")
    p.add_argument("--error", action="append", default=[])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    path = args.progress_json.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = _read_json(path)
    if not payload:
        payload = {
            "metric_name": args.metric_name,
            "best_time_us": None,
            "last_time_us": None,
            "best_artifact": "",
            "last_artifact": "",
            "status": "init",
            "note": "",
            "errors": [],
            "updates": 0,
            "source": args.source,
            "updated_at": _utc_iso(),
        }

    payload["metric_name"] = args.metric_name
    payload["source"] = args.source
    payload["status"] = args.status
    payload["note"] = args.note
    payload["updated_at"] = _utc_iso()
    payload["updates"] = int(payload.get("updates", 0)) + 1
    if args.iteration is not None:
        payload["iteration"] = int(args.iteration)

    if args.error:
        errs = list(payload.get("errors", []))
        errs.extend([str(e) for e in args.error])
        payload["errors"] = errs[-20:]

    if args.time_us is not None:
        t = float(args.time_us)
        payload["last_time_us"] = t
        if args.artifact:
            payload["last_artifact"] = args.artifact
        best = payload.get("best_time_us")
        if best is None or t < float(best):
            payload["best_time_us"] = t
            if args.artifact:
                payload["best_artifact"] = args.artifact

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
