---
name: loop-best-time-report
description: Report live best-time progress from worker iterations to artifacts/loop_coordinator/iter_XXXX/worker_progress.json so loop_tui can display current and best gmean_us.
---

# Loop Best-Time Report

Use this skill during worker iterations to stream timing updates to the loop TUI.

## Command

```bash
python3 tools/loop_best_time_report.py \
  --progress-json artifacts/loop_coordinator/iter_0001/worker_progress.json \
  --status benchmark \
  --time-us 312.4 \
  --artifact artifacts/nvfp4_group_gemm_001.afterhours_benchmark.20260224T000000Z.txt \
  --note "candidate benchmark"
```

## Typical statuses

- `running`: worker iteration started
- `benchmark`: new measured timing (include `--time-us`)
- `blocked`: temporary blocker/error (include `--error`)
- `completed`: worker finished iteration

## Notes

- Metric is expected to be `gmean_us` (lower is better).
- Always report after each candidate/final benchmark so the TUI reflects live best time.
