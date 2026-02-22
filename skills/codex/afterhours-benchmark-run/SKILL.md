---
name: afterhours-benchmark-run
description: Run deployed PyGPUBench candidate/final benchmark mode for a kernel with a standardized artifact that includes repo git hash, kernel diff, command output, and JSON result, then append Codex analysis.
---

# Afterhours Benchmark Run

Run deployed PyGPUBench in `benchmark`-style mode (default profile: `candidate`) with a standardized artifact that includes:
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
  --profile final \
  --repeats 100 \
  --stage-repeats 20,40,40 \
  --tag v1 \
  --artifact-only)"
```

## Follow-up analysis step (required)

After reading benchmark feedback, append Codex analysis/thoughts:

```bash
python scripts/append_analysis.py \
  --artifact "$ARTIFACT" \
  --decision KEEP \
  --summary "Benchmark means/geomean summary." \
  --feedback "What benchmark feedback/outliers indicate." \
  --next-step "Next single lever to test."
```

If performance regressed or correctness failed, set `--decision REVERT`.
