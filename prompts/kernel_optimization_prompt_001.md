You are Codex in `<repo_root>`.

Variables:
- `<kernel_path>`: target kernel file to optimize
- `<repo_root>`: repository root
- `<gpu_type>`: target GPU (e.g., `B200`)
- `<target_gmean_us>`: target gmean in microseconds

Mission:
- Optimize `<kernel_path>` to achieve **< <target_gmean_us> us gmean** on **<gpu_type>**.
- Run continuously in an autonomous optimization loop until target is reached or I explicitly say stop.

Hard requirements:
- Use these skills exactly when relevant:
  - `popcorn-test-run`
  - `popcorn-benchmark-run`
  - `popcorn-leaderboard-run`
  - `modal-b200-snippet-policy`
  - `cutlass-compile-debug`
  - `ptx-dump`
  - `cubin-disasm`
  - `torch-profiler-modal`
  - `gpu-profiler`
- POPCORN submission mechanics:
  - Never use ad-hoc `popcorn-cli`.
  - Always use the skill scripts.
  - Always pass `--gpu <gpu_type>`.
- Python/CUTLASS/GPU snippets:
  - Must follow modal policy:
    - `set -a; source .env; set +a`
    - `uv run --with modal python tools/modal_python_exec.py --gpu <gpu_type> ...`
  - On snippet failure: show stderr, retry once with exact same command, if it fails again revert last kernel edit.

Optimization loop (strict):
1. Establish clean baseline on current target kernel (test + benchmark).
2. Apply exactly one kernel-side change per iteration.
3. Run test:
   `python3 skills/codex/popcorn-test-run/scripts/run_submission.py --kernel <kernel_path> --repo-root <repo_root> --gpu <gpu_type> --artifact-only`
4. Append required test analysis:
   `python3 skills/codex/popcorn-test-run/scripts/append_analysis.py --artifact <artifact> --decision <KEEP|REVERT> --summary "<...>" --feedback "<...>" --next-step "<...>"`
5. If test passes, run benchmark:
   `python3 skills/codex/popcorn-benchmark-run/scripts/run_submission.py --kernel <kernel_path> --repo-root <repo_root> --gpu <gpu_type> --artifact-only`
6. Append required benchmark analysis with KEEP/REVERT.
7. If benchmark fails: include stderr, retry same command once, if it fails again revert last change.
8. Keep only changes that improve benchmark gmean (or very strong leaderboard evidence). Otherwise revert immediately.
9. Run leaderboard periodically and for candidates:
   `python3 skills/codex/popcorn-leaderboard-run/scripts/run_submission.py --kernel <kernel_path> --repo-root <repo_root> --gpu <gpu_type> --artifact-only`
   then append required leaderboard analysis.
10. Continue looping without pausing.

Debug escalation policy:
- Opaque compile/test failures: use `cutlass-compile-debug`.
- Codegen/ISA investigations: use `ptx-dump` then `cubin-disasm`.
- Bottleneck diagnosis: use `torch-profiler-modal` as primary, `gpu-profiler` as supplemental.
- Search official docs/web as needed; cite sources in decisions.
- Do not guess if evidence can be collected.

Scope and discipline:
- Work on `<kernel_path>` only, except minimal temporary probes/helpers.
- No destructive git commands.
- Never revert unrelated user changes.
- Enforce strict keep/revert discipline.
- Commit only validated improvements with concise message + artifact references.

Per-iteration report format:
- Change tried
- Test result
- Benchmark/leaderboard result
- gmean (computed)
- Decision: KEEP or REVERT
- Next single change

Operator intent refinements applied during the run:
- Do not inspect or optimize other kernels.
- Make large algorithmic changes and push aggressively when needed.
- Temporary regressions are acceptable if they support a promising thread.
- Stay on a thread of thought long enough to validate it before abandoning.
- When things fail, debug why first (do not blindly churn changes).
- If plateaued, search the web and use external evidence.
- Keep going continuously until explicitly told to stop.
- Stretch target aggressively when requested, while the primary hard target remains `<target_gmean_us>`.

Start immediately:
- Run baseline test + benchmark for current `<kernel_path>`.
- Report current gmean, best-known gmean in this session, blocker status, and next one-change experiment.
