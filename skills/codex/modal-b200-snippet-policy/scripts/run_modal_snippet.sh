#!/usr/bin/env bash
set -euo pipefail

# Enforce repository + environment defaults for lean4real Modal snippet runs.
REPO_ROOT="${REPO_ROOT:-/Users/cuboniks/Projects/kernel_projects/lean4real}"
GPU="${GPU:-B200}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/../kernel_rl/.env}"

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
