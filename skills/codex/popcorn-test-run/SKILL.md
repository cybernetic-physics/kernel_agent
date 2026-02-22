---
name: popcorn-test-run
description: Run POPCORN test mode for a kernel with a foolproof artifact flow that records repo git hash and kernel diff against that hash, then append Codex analysis/thoughts from the run feedback.
---

# Popcorn Test Run

Run POPCORN `test` mode with a standardized artifact that always includes:
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
