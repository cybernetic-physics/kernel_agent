# Ralph Loop for Kernel Optimization

This guide shows how to run autonomous kernel optimization loops in `kernel_agents` using Claude Code + Ralph Loop, while keeping runs bounded and debuggable.

## Why Ralph Loop here

Ralph Loop is a Stop-hook based continuation loop for Claude Code. It is useful for this repo because kernel tuning usually requires many test/benchmark iterations.

Use it to:
- keep optimization sessions running without manual reprompting
- enforce iteration caps and explicit completion criteria
- preserve artifacts and decisions from each iteration

## Prerequisites

1. Open a Claude Code session in this directory:

```bash
cd /Users/cuboniks/Projects/kernel_projects/kernel_agents
```

2. Ensure repo dependencies are installed:

```bash
uv sync
```

3. Ensure env is available for Modal/POPCORN flows:

```bash
set -a; source .env; set +a
```

4. Install/enable Ralph Loop in Claude Code (via `/plugin`, select Ralph Loop).

## Recommended loop shape for this repo

Use one optimization objective per loop:
- one target kernel file
- one metric target
- one hard completion promise string

Do not set unbounded loops. Always use `--max-iterations`.

## POPCORN-based optimization example

Use the repo prompt template:
- `prompts/kernel_optimization_prompt_001.md`

Example command in Claude Code:

```text
/ralph-loop "
Optimize kernels/nvfp4_group_gemm_001.py on B200.
Use the exact workflow and constraints in prompts/kernel_optimization_prompt_001.md.
Set <kernel_path>=kernels/nvfp4_group_gemm_001.py
Set <repo_root>=/Users/cuboniks/Projects/kernel_projects/kernel_agents
Set <gpu_type>=B200
Set <target_gmean_us>=320
Completion condition: output TARGET_GMEAN_REACHED only after benchmark evidence.
" --max-iterations 24 --completion-promise TARGET_GMEAN_REACHED
```

## PyGPUBench-based optimization example

Use the repo prompt template:
- `prompts/kernel_optimization_prompt_pygpubench_001.md`

Example command in Claude Code:

```text
/ralph-loop "
Optimize kernels/nvfp4_group_gemm_001.py on B200 with PyGPUBench only.
Use prompts/kernel_optimization_prompt_pygpubench_001.md exactly.
Set <kernel_path>=kernels/nvfp4_group_gemm_001.py
Set <harness_path>=tools/pygpubench_nvfp4_dual_gemm_harness.py
Set <repo_root>=/Users/cuboniks/Projects/kernel_projects/kernel_agents
Set <target_metric>=median_us<320
Completion condition: output PYGPUBENCH_TARGET_REACHED only after final profile confirms.
" --max-iterations 30 --completion-promise PYGPUBENCH_TARGET_REACHED
```

## Tips and tricks (high impact)

- Use milestone promises, not one giant run.
  - Example phase promises: `PHASE_1_CORRECTNESS`, `PHASE_2_REGRESSION_FREE`, `PHASE_3_TARGET_REACHED`.
- Keep prompts strict about keep/revert discipline.
  - This repo already has templates enforcing one change per iteration.
- Keep context short and artifact-driven.
  - Refer to prior artifact files, not long chat history.
- Add explicit blocked-state output.
  - Example: `BLOCKED:<reason>` when compile/profiler infra repeatedly fails.
- Reset stalled loops early.
  - If 3-5 iterations show no measurable improvement, cancel and restart with narrower search scope.
- Separate exploration and confirmation.
  - Use fast gates first, then candidate/final confirmation before KEEP.

## Guardrails and failure control

- Always set `--max-iterations` (never `0`/unlimited).
- Use specific completion promises that are hard to accidentally emit.
- Keep a manual stop path ready:
  - `/cancel-ralph`
- If loops become repetitive, restart with:
  - tighter objective
  - explicit forbidden-change list
  - stronger acceptance criteria

## Hook safety notes

Ralph Loop is built on Claude Code hooks. For custom Stop hooks, avoid recursive loops:
- include a `stop_hook_active` guard
- avoid spawning nested `claude` processes from inside hooks
- keep hook logic small and deterministic

For this repo, prefer:
- Ralph Loop for single-session continuation
- external coordinator for multi-session orchestration (worker/reviewer)

## Suggested run log convention

Store per-run notes under:
- `artifacts/ralph_loop/<timestamp>.md`

Include:
- loop goal
- command used
- max iterations
- completion promise
- per-iteration KEEP/REVERT decisions
- final metric summary

## References

- Claude Code hooks reference: <https://code.claude.com/docs/en/hooks>
- Claude Code hooks guide (loop guard examples): <https://code.claude.com/docs/en/hooks-guide>
- Ralph Loop plugin page: <https://claude.com/plugins/ralph-loop>
- Community Ralph loop manager patterns: <https://github.com/iannuttall/ralph>
- Community long-run guardrail patterns: <https://github.com/frankbria/ralph-claude-code>
