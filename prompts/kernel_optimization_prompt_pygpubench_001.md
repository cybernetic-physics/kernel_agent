You are Codex in `<repo_root>`.

Variables:
- `<kernel_path>`: target kernel file to optimize (Python submission)
- `<harness_path>`: PyGPUBench harness file for this kernel/problem
- `<repo_root>`: repository root
- `<target_metric>`: target performance threshold (for example median_us or mean_us)

Mission:
- Optimize `<kernel_path>` to beat `<target_metric>` on **B200**.
- Evaluate kernel performance using the `pygpubench-modal-run` skill only.
- Run continuously in an autonomous optimization loop until target is reached or I explicitly say stop.

Hard requirements:
- Required skill:
  - `pygpubench-modal-run`
- Forbidden skills/tools for this prompt:
  - `popcorn-test-run`
  - `popcorn-benchmark-run`
  - `popcorn-leaderboard-run`
  - direct `popcorn-cli` usage
- Execution target:
  - Always B200 (no GPU down-tiering).
- Use deployed Modal app path (lower overhead):
  1) Deploy once when app/runner changes:
     `bash skills/codex/pygpubench-modal-run/scripts/run_pygpubench_modal.sh --deploy`
  2) Run via deployed caller:
     `bash skills/codex/pygpubench-modal-run/scripts/run_pygpubench_modal.sh --harness <harness_path> --submission <kernel_path> --profile <smoke|candidate|final> --print-log`

Optimization loop (strict):
1. Establish baseline (candidate profile):
   `bash skills/codex/pygpubench-modal-run/scripts/run_pygpubench_modal.sh --harness <harness_path> --submission <kernel_path> --profile candidate --print-log`
2. Apply exactly one kernel-side change per iteration.
3. Run smoke validation (fast gate):
   `bash skills/codex/pygpubench-modal-run/scripts/run_pygpubench_modal.sh --harness <harness_path> --submission <kernel_path> --profile smoke --print-log`
4. If smoke passes, run candidate measurement:
   `bash skills/codex/pygpubench-modal-run/scripts/run_pygpubench_modal.sh --harness <harness_path> --submission <kernel_path> --profile candidate --print-log`
5. If candidate improves materially, confirm with final:
   `bash skills/codex/pygpubench-modal-run/scripts/run_pygpubench_modal.sh --harness <harness_path> --submission <kernel_path> --profile final --print-log`
6. Keep only changes with reproducible improvement on candidate/final. Otherwise revert immediately.
7. Continue looping without pausing.

Cost and runtime discipline:
- Prefer smoke profile for early rejection.
- Use staged repeats and early-stop for weak variants:
  - `--stage-repeats 8,16,32`
  - `--early-stop-us <threshold>`
  - `--repeats <N>`
- Tighten `--timeout-seconds` when kernels are unstable/hanging.
- Re-run once on infra/transient failures; on second failure, keep current code and move to next change.

Result source of truth:
- Parse and report metrics from:
  - `artifacts/pygpubench_modal_last_run.json`
- Preserve the JSON artifact for each key iteration (copy/rename as needed).

Scope and discipline:
- Work on `<kernel_path>` only, except minimal temporary probes/helpers.
- No destructive git commands.
- Never revert unrelated user changes.
- Enforce strict keep/revert discipline.
- Commit only validated improvements with concise message + artifact references.

Per-iteration report format:
- Change tried
- Smoke result
- Candidate/final result
- Parsed metric(s)
- Decision: KEEP or REVERT
- Next single change

Start immediately:
- Deploy app if needed.
- Run baseline candidate for current `<kernel_path>` with `<harness_path>`.
- Report current metric, best-known metric in this session, blocker status, and next one-change experiment.
