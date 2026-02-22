---
name: popcorn-leaderboard-run
description: Run POPCORN leaderboard mode for a kernel with a foolproof artifact flow that records repo git hash and kernel diff against that hash, then append Codex analysis/thoughts including ranked feedback and outliers.
---

# Popcorn Leaderboard Run

Run POPCORN `leaderboard` mode with a standardized artifact that always includes:
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
  --tag ranked_try_1 \
  --artifact-only)"
```

## Follow-up analysis step (required)

After reading ranked feedback, append Codex analysis/thoughts:

```bash
python scripts/append_analysis.py \
  --artifact "$ARTIFACT" \
  --decision KEEP \
  --summary "Ranked means and notable deltas." \
  --feedback "Outlier behavior and ranked stability notes." \
  --next-step "Retry/next lever based on ranked feedback."
```

If ranked run shows unsafe behavior or severe outliers, set `--decision REVERT`.
