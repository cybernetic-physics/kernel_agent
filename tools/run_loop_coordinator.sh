#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="$ROOT_DIR/artifacts/loop_coordinator"

mkdir -p "$STATE_ROOT"

uv run python "$ROOT_DIR/tools/loop_coordinator.py" \
  --repo-root "$ROOT_DIR" \
  --state-root "$STATE_ROOT" \
  "$@"
