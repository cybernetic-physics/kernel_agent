You are Codex in `<repo_root>`.

Variables:
- `<kernel_path>`: target kernel file to optimize (Python submission)
- `<harness_path>`: PyGPUBench harness file for this kernel/problem
- `<repo_root>`: repository root
- `<target_metric_us>`: target threshold in microseconds
- `<metric_name>`: metric to optimize (for example `median_us` or `mean_us`)

Mission:
- Optimize `<kernel_path>` to beat `<target_metric_us>` on **B200** for `<metric_name>`.
- Run continuously in an autonomous optimization loop until target is reached or I explicitly say stop.

Hard requirements:
- Required skills:
  - `afterhours-test-run`
  - `afterhours-benchmark-run`
- Forbidden for this prompt:
  - `popcorn-test-run`
  - `popcorn-benchmark-run`
  - `popcorn-leaderboard-run`
  - direct `popcorn-cli` usage
  - direct `pygpubench-modal-run` execution (except deploy step below)
- Execution target:
  - Always B200.

Deploy/runner requirement:
- If app or runner code changed, deploy once before the loop:
  - `bash skills/codex/pygpubench-modal-run/scripts/run_pygpubench_modal.sh --deploy`

Canonical command templates:

Test/smoke artifact:
```bash
ARTIFACT_TEST="$(python3 skills/codex/afterhours-test-run/scripts/run_submission.py \
  --kernel <kernel_path> \
  --harness <harness_path> \
  --repo-root <repo_root> \
  --gpu B200 \
  --profile smoke \
  --artifact-only)"
```

Append required test analysis:
```bash
python3 skills/codex/afterhours-test-run/scripts/append_analysis.py \
  --artifact "$ARTIFACT_TEST" \
  --decision <KEEP|REVERT> \
  --summary "<single-change summary>" \
  --feedback "<key test signals>" \
  --next-step "<next one-change plan>"
```

Benchmark/candidate artifact:
```bash
ARTIFACT_BENCH="$(python3 skills/codex/afterhours-benchmark-run/scripts/run_submission.py \
  --kernel <kernel_path> \
  --harness <harness_path> \
  --repo-root <repo_root> \
  --gpu B200 \
  --profile candidate \
  --artifact-only)"
```

Append required benchmark analysis:
```bash
python3 skills/codex/afterhours-benchmark-run/scripts/append_analysis.py \
  --artifact "$ARTIFACT_BENCH" \
  --decision <KEEP|REVERT> \
  --summary "<metric summary>" \
  --feedback "<performance/outlier interpretation>" \
  --next-step "<next one-change plan>"
```

Optimization loop (strict):
1. Baseline:
   - Run one smoke test artifact and one candidate benchmark artifact.
   - Record baseline `<metric_name>` from benchmark artifact.
2. Per iteration, apply exactly one kernel-side change.
3. Run smoke test via `afterhours-test-run`.
4. Append test analysis (required).
5. If smoke passes, run candidate benchmark via `afterhours-benchmark-run`.
6. Append benchmark analysis (required).
7. If candidate improves materially, run a confirmation benchmark with `--profile final`.
8. Keep only reproducible improvements; otherwise revert immediately.
9. Continue looping without pausing.

Cost/runtime discipline:
- Use `smoke` for early rejection, `candidate` for scoring, `final` for confirmation.
- Use staged controls on weak variants:
  - `--repeats <N>`
  - `--stage-repeats 8,16,32`
  - `--early-stop-us <threshold>`
  - `--timeout-seconds <seconds>`
- On infra/transient failure:
  - retry once with the same command
  - if it fails again, revert last kernel edit and continue.

Result source of truth:
- Use the artifact produced by each afterhours script as the authoritative record.
- Parse metrics from the artifact’s `json_result` section.
- Preserve key artifacts for baseline, best-so-far, and final candidate.

Scope and discipline:
- Work on `<kernel_path>` only, except minimal temporary probes/helpers.
- No destructive git commands.
- Never revert unrelated user changes.
- Enforce strict keep/revert discipline.
- Commit only validated improvements with concise message + artifact references.

Per-iteration report format:
- Change tried
- Smoke/test result
- Candidate/final benchmark result
- Parsed `<metric_name>`
- Decision: KEEP or REVERT
- Next single change

Start immediately:
- Deploy if needed.
- Run baseline smoke + candidate for current `<kernel_path>` and `<harness_path>`.
- Report current `<metric_name>`, best-known metric in this session, blocker status, and next one-change experiment.
