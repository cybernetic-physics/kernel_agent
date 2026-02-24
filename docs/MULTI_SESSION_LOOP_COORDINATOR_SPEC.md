# Multi-Session Loop Coordinator Spec (Worker + Reviewer)

This spec defines a safe continuous optimization loop for `kernel_agents` using:
- Session A: worker (edits kernel + runs benchmarks)
- Session B: reviewer (analyzes A's artifacts/diff and produces a verdict)
- External coordinator: the only process that decides whether to continue

## Why external coordinator

Use hooks for local gating, not orchestration.

Reason:
- hook logic is event-driven and can run in parallel
- recursive continuation logic is easy to misconfigure
- spawning nested `claude` sessions from hooks makes failure handling hard

Design rule:
- Stop hook in A may read verdict state only
- Stop hook in A must not launch another `claude` process

## Scope

In scope:
- autonomous optimization loop with guardrails
- two independent Claude sessions
- continuous run until explicit stop condition

Out of scope:
- direct hook-to-hook recursive process spawning
- unbounded/no-limit loops

## Directory layout

Use this state root:
- `artifacts/loop_coordinator/`

Per iteration:
- `artifacts/loop_coordinator/iter_0001/`
- `artifacts/loop_coordinator/iter_0002/`

Required files per iteration:
- `worker_prompt.txt`
- `worker_result.json`
- `reviewer_prompt.txt`
- `reviewer_verdict.json`
- `metrics_snapshot.json`
- `git_diff.patch`
- `status.json`

Global files:
- `artifacts/loop_coordinator/control.json`
- `artifacts/loop_coordinator/heartbeat.json`
- `artifacts/loop_coordinator/lock/` (lock dir)

## Roles and contracts

### Session A (worker)

Input:
- `worker_prompt.txt` (target, constraints, accepted tools/skills)

Responsibilities:
- apply at most one meaningful kernel change per iteration
- run required validation/benchmark flow
- write structured result to `worker_result.json`

Required `worker_result.json` fields:
- `iteration` (int)
- `kernel_path` (string)
- `tests_passed` (bool)
- `benchmark_passed` (bool)
- `metric_name` (string)
- `metric_value` (number)
- `decision` (`KEEP` or `REVERT`)
- `artifacts` (array of paths)
- `errors` (array of strings)

### Session B (reviewer)

Input:
- `reviewer_prompt.txt`
- `git_diff.patch`
- `worker_result.json`
- relevant benchmark/profiler artifacts

Responsibilities:
- validate evidence quality
- detect regressions/risk
- emit continuation verdict

Required `reviewer_verdict.json` fields:
- `iteration` (int)
- `verdict` (`CONTINUE`, `STOP_TARGET_REACHED`, `STOP_NO_PROGRESS`, `STOP_BLOCKED`)
- `confidence` (`low`, `medium`, `high`)
- `reason` (string)
- `next_change_hint` (string)
- `requires_revert` (bool)

## Coordinator state machine

States:
- `INIT`
- `RUN_WORKER`
- `RUN_REVIEWER`
- `APPLY_VERDICT`
- `STOPPED`

Loop:
1. `INIT`
2. Create `iter_N` directory and prompts.
3. `RUN_WORKER`: execute worker session A, collect `worker_result.json` + patch + metrics.
4. `RUN_REVIEWER`: execute reviewer session B with A artifacts.
5. `APPLY_VERDICT`:
   - if `STOP_*`: set `control.json.stop=true`, transition `STOPPED`
   - if `CONTINUE`: generate next `worker_prompt.txt`, increment iteration
6. Repeat until stop.

## Stop conditions (must enforce all)

- `max_iterations` hard cap (required)
- `max_wall_clock_minutes` hard cap
- target reached and confirmed `N` consecutive iterations (for stability)
- `no_progress_limit`: stop after K iterations with no metric improvement
- `infra_failure_limit`: stop after M consecutive infra/tool failures
- manual stop: set `control.json.stop=true`

## Suggested defaults

- `max_iterations`: 40
- `max_wall_clock_minutes`: 360
- `target_confirmations`: 2
- `no_progress_limit`: 6
- `infra_failure_limit`: 3

## Locking and concurrency

Single coordinator instance only.

Lock protocol:
- coordinator creates `artifacts/loop_coordinator/lock/active.lock` atomically
- if lock exists and heartbeat is fresh, second coordinator exits
- stale lock recovery allowed only if heartbeat older than timeout (for example 10 minutes)

Session isolation:
- keep A and B in separate working shells/sessions
- coordinator serializes iteration transitions (no overlapping iteration commits)

## Prompt strategy

Worker prompt source:
- `prompts/kernel_optimization_prompt_001.md` (POPCORN flow), or
- `prompts/kernel_optimization_prompt_pygpubench_001.md` (PyGPUBench flow)

Reviewer prompt requirements:
- analyze only current iteration evidence plus best-so-far baseline
- output strict JSON verdict only
- reject conclusions without artifact evidence

## Hook integration (optional)

If using a Stop hook in worker session A:
- hook reads latest `reviewer_verdict.json`
- if verdict says continue, allow continuation
- if verdict says stop, halt loop
- include recursion guard (for example `stop_hook_active`)

Do not place orchestration logic in hooks. Coordinator owns orchestration.

## Failure handling

- Missing/invalid `worker_result.json`: mark infra failure and retry iteration once.
- Missing/invalid `reviewer_verdict.json`: retry reviewer once, else stop blocked.
- Repeated compile failures: force debug path before next change.
- Repeated benchmark instability: require confirmation runs before KEEP.

## Observability

Write `heartbeat.json` every loop step with:
- current iteration
- state
- last metric
- best metric
- consecutive no-progress count
- consecutive infra-failure count
- last update timestamp

Keep append-only run log:
- `artifacts/loop_coordinator/run.log`

## Minimal implementation plan

1. Create coordinator script:
   - `tools/loop_coordinator.py`
2. Add reviewer output schema validator:
   - `tools/loop_verdict_schema.py`
3. Add starter prompts:
   - `prompts/reviewer_prompt_001.md`
4. Add run wrapper:
   - `tools/run_loop_coordinator.sh`
5. Dry-run mode with mocked worker/reviewer outputs before real sessions.

## Implementation status in this repo

- Implemented coordinator:
  - `tools/loop_coordinator.py`
- Implemented payload validator:
  - `tools/loop_verdict_schema.py`
- Implemented reviewer prompt starter:
  - `prompts/reviewer_prompt_001.md`
- Implemented wrapper:
  - `tools/run_loop_coordinator.sh`

## How to run

Dry-run validation:

```bash
bash tools/run_loop_coordinator.sh \
  --fresh-start \
  --execution-mode dry-run \
  --kernel-path kernels/nvfp4_group_gemm_001.py \
  --target-metric-name gmean_us \
  --target-metric-threshold 320 \
  --max-iterations 5
```

Manual two-session mode:

```bash
bash tools/run_loop_coordinator.sh \
  --fresh-start \
  --execution-mode manual \
  --kernel-path kernels/nvfp4_group_gemm_001.py \
  --target-metric-name gmean_us \
  --target-metric-threshold 320 \
  --max-iterations 40
```

In manual mode, Session A/B must write:
- `artifacts/loop_coordinator/iter_XXXX/worker_result.json`
- `artifacts/loop_coordinator/iter_XXXX/reviewer_verdict.json`

Schema check:

```bash
uv run python tools/loop_verdict_schema.py --kind worker --json-file artifacts/loop_coordinator/iter_0001/worker_result.json
uv run python tools/loop_verdict_schema.py --kind reviewer --json-file artifacts/loop_coordinator/iter_0001/reviewer_verdict.json
```

## References

- Claude Code hooks reference: <https://code.claude.com/docs/en/hooks>
- Claude Code hooks guide: <https://code.claude.com/docs/en/hooks-guide>
- Agent SDK hooks (recursive loop cautions): <https://platform.claude.com/docs/en/agent-sdk/hooks>
- Ralph Loop plugin: <https://claude.com/plugins/ralph-loop>
