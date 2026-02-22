#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${1:-}" == "--deploy" ]]; then
  shift
  uv run --with modal modal deploy tools/pygpubench_modal_app.py "$@"
  exit 0
fi

uv run --with modal python tools/run_pygpubench_modal.py "$@"
