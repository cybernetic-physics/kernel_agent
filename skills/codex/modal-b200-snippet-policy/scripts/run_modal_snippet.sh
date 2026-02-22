#!/usr/bin/env bash
set -euo pipefail

# Enforce repository + environment defaults for kernel_agents Modal snippet runs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd -- "$SCRIPT_DIR/../../../../" && pwd)"
REPO_ROOT="${REPO_ROOT:-$REPO_ROOT_DEFAULT}"
GPU="${GPU:-B200}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "error: repo root not found: $REPO_ROOT" >&2
  exit 2
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: env file not found: $ENV_FILE" >&2
  exit 2
fi

run_once() {
  (
    cd "$REPO_ROOT"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    uv run --with modal python tools/modal_python_exec.py --gpu "$GPU" --mount-repo "$@"
  )
}

cmd_desc="uv run --with modal python tools/modal_python_exec.py --gpu $GPU --mount-repo $*"

if ! run_once "$@"; then
  echo "first attempt failed; retrying once with exact same command..." >&2
  echo "command: $cmd_desc" >&2
  run_once "$@"
fi
