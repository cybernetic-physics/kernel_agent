#!/usr/bin/env python3
"""Diff two layout snapshot manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True, help="Old snapshot JSON")
    p.add_argument("--new", type=Path, required=True, help="New snapshot JSON")
    p.add_argument(
        "--protected",
        action="append",
        default=[],
        help="Protected layout key (repeatable).",
    )
    p.add_argument(
        "--allow-protected-changes",
        action="store_true",
        help="Do not fail non-zero when protected layouts changed.",
    )
    p.add_argument("--json-out", type=Path)
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _layout_digest(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    return record.get("digest")


def _changed_fields(old_d: dict[str, Any], new_d: dict[str, Any]) -> list[str]:
    fields = [
        "shape",
        "stride",
        "flat_shape",
        "flat_stride",
        "flat_rank",
        "flat_size",
        "flat_cosize",
        "tractable",
        "canonical",
        "hash",
    ]
    return [f for f in fields if old_d.get(f) != new_d.get(f)]


def main() -> None:
    args = _parse_args()
    old_m = _load(args.old)
    new_m = _load(args.new)

    old_layouts = old_m.get("layouts", {})
    new_layouts = new_m.get("layouts", {})

    names = sorted(set(old_layouts.keys()) | set(new_layouts.keys()))

    changes: list[dict[str, Any]] = []
    protected_changed: list[str] = []

    for name in names:
        old_rec = old_layouts.get(name)
        new_rec = new_layouts.get(name)

        if old_rec is None:
            changes.append({"layout": name, "status": "added"})
            if name in args.protected:
                protected_changed.append(name)
            continue

        if new_rec is None:
            changes.append({"layout": name, "status": "removed"})
            if name in args.protected:
                protected_changed.append(name)
            continue

        old_d = _layout_digest(old_rec)
        new_d = _layout_digest(new_rec)

        if old_d is None or new_d is None:
            if old_rec != new_rec:
                changes.append(
                    {
                        "layout": name,
                        "status": "changed_non_digest",
                        "old_repr": old_rec.get("repr"),
                        "new_repr": new_rec.get("repr"),
                    }
                )
                if name in args.protected:
                    protected_changed.append(name)
            continue

        if old_d.get("hash") != new_d.get("hash"):
            diff_fields = _changed_fields(old_d, new_d)
            changes.append(
                {
                    "layout": name,
                    "status": "changed",
                    "changed_fields": diff_fields,
                    "old_hash": old_d.get("hash"),
                    "new_hash": new_d.get("hash"),
                    "old_canonical": old_d.get("canonical"),
                    "new_canonical": new_d.get("canonical"),
                }
            )
            if name in args.protected:
                protected_changed.append(name)

    if not changes:
        print("[PASS] No layout manifest differences.")
    else:
        print(f"[INFO] Found {len(changes)} layout change(s):")
        for ch in changes:
            status = ch["status"]
            name = ch["layout"]
            print(f"- {name}: {status}")
            if status == "changed":
                fields = ", ".join(ch.get("changed_fields", []))
                print(f"  fields: {fields}")
                print(f"  old_hash: {ch.get('old_hash')}")
                print(f"  new_hash: {ch.get('new_hash')}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "old": str(args.old),
            "new": str(args.new),
            "num_changes": len(changes),
            "changes": changes,
            "protected": args.protected,
            "protected_changed": sorted(set(protected_changed)),
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote diff JSON: {args.json_out}")

    if protected_changed and not args.allow_protected_changes:
        print(
            "[FAIL] Protected layouts changed: "
            + ", ".join(sorted(set(protected_changed)))
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
