Use skill `loop-best-time-report` so the loop TUI can show live timing progress.

Live reporting requirements:
- Progress JSON path: `<progress_json_path>`
- Metric: `gmean_us` (lower is better)

At start of the iteration, run:
```bash
python3 tools/loop_best_time_report.py \
  --progress-json "<progress_json_path>" \
  --status running \
  --note "worker iteration started"
```

Every time you obtain a benchmark `gmean_us` (candidate/final), immediately run:
```bash
python3 tools/loop_best_time_report.py \
  --progress-json "<progress_json_path>" \
  --status benchmark \
  --time-us <gmean_us_value> \
  --artifact "<artifact_path>" \
  --note "candidate_or_final result"
```

On any blocker/failure, report it:
```bash
python3 tools/loop_best_time_report.py \
  --progress-json "<progress_json_path>" \
  --status blocked \
  --note "<short blocker>" \
  --error "<error summary>"
```

Before finishing, write a completion update:
```bash
python3 tools/loop_best_time_report.py \
  --progress-json "<progress_json_path>" \
  --status completed \
  --note "worker iteration completed"
```
