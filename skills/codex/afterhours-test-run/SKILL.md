---
name: afterhours-test-run
description: Run deployed PyGPUBench smoke/profile test mode for a kernel with a standardized artifact that includes repo git hash, kernel diff, command output, and JSON result, then append Codex analysis.
---

# Afterhours Test Run

Run deployed PyGPUBench in `test`-style mode (default profile: `smoke`) with a standardized artifact that includes:
- repo root
- git HEAD hash
- kernel diff against that hash
- raw command output
- captured JSON result

Default repo root is auto-detected from this repository's git root.

## Command

```bash
ARTIFACT="$(python scripts/run_submission.py \
  --kernel kernels/nvfp4_group_gemm/wagmiv67.py \
  --harness /abs/path/to/harness.py \
  --artifact-only)"
```

Optional overrides:

```bash
ARTIFACT="$(python scripts/run_submission.py \
  --kernel kernels/nvfp4_group_gemm/wagmiv67.py \
  --harness /abs/path/to/harness.py \
  --profile smoke \
  --repeats 20 \
  --stage-repeats 8,12 \
  --early-stop-us 1200 \
  --tag v1 \
  --artifact-only)"
```

## Follow-up analysis step (required)

After reading the artifact output, append Codex analysis/thoughts:

```bash
python scripts/append_analysis.py \
  --artifact "$ARTIFACT" \
  --decision KEEP \
  --summary "Single-change result summary." \
  --feedback "What feedback/output mattered and why." \
  --next-step "What to try next."
```

If run failed, set `--decision REVERT` and explain failure cause in `--feedback`.
