#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/skills/codex"
DST_DIR="$ROOT_DIR/.claude/skills"

mkdir -p "$DST_DIR"
find "$DST_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

for d in "$SRC_DIR"/*; do
  base="$(basename "$d")"
  if [[ "$base" == .* ]]; then
    continue
  fi
  if [[ -d "$d" ]]; then
    ln -s "../../skills/codex/$base" "$DST_DIR/$base"
  fi
done

echo "Synced Claude skills from $SRC_DIR -> $DST_DIR"
ls -la "$DST_DIR"
