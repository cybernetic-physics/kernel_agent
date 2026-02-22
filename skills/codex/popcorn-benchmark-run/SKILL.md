---
name: popcorn-benchmark-run
description: Run POPCORN benchmark mode for a kernel with a foolproof artifact flow that records repo git hash and kernel diff against that hash, then append Codex analysis/thoughts from benchmark feedback.
---

# Popcorn Benchmark Run

Run POPCORN `benchmark` mode with a standardized artifact that always includes:
- repo root
- git HEAD hash
- kernel diff against that hash
- raw `popcorn-cli` output

Default repo root is auto-detected from this repository's git root.

## Command

```bash
ARTIFACT="$(python scripts/run_submission.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --artifact-only)"
```

Optional overrides:

```bash
ARTIFACT="$(python scripts/run_submission.py \
  --kernel kernels/nvfp4_group_gemm/wagmiv67.py \
  --leaderboard nvfp4_group_gemm \
  --gpu B200 \
  --tag v49 \
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
