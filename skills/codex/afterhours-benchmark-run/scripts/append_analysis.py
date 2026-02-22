#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Append Codex analysis to an AFTERHOURS artifact file.")
    p.add_argument("--artifact", required=True, help="Path to artifact file.")
    p.add_argument("--decision", default="", help="Decision label, e.g., KEEP/REVERT.")
    p.add_argument("--summary", required=True, help="Short result summary.")
    p.add_argument("--feedback", default="", help="Key feedback/signals from run output.")
    p.add_argument("--next-step", default="", help="Proposed next step.")
    args = p.parse_args()

    artifact = Path(args.artifact).expanduser().resolve()
    if not artifact.exists():
        raise SystemExit(f"Artifact not found: {artifact}")

    now = dt.datetime.now(dt.UTC).isoformat()
    lines = [
        "",
        "## Codex Analysis",
        f"timestamp_utc: {now}",
        f"decision: {args.decision or 'N/A'}",
        f"summary: {args.summary}",
    ]
    if args.feedback:
        lines.append(f"feedback: {args.feedback}")
    if args.next_step:
        lines.append(f"next_step: {args.next_step}")
    lines.append("")

    with artifact.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(str(artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
