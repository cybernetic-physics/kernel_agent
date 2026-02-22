---
name: pygpubench-modal-run
description: Run PyGPUBench harnesses on Modal B200 against arbitrary submission kernels using the repository runner script. Use when you need adversarial benchmark validation for CUDA kernels with isolated execution.
---

# PyGPUBench Modal Run

Run PyGPUBench on Modal B200 using:
- `tools/run_pygpubench_modal.py`
- `tools/pygpubench_harness_template.py` (template for custom harnesses)

This flow is self-contained:
- Modal image installs `pygpubench` from GitHub (`ngc92/pygpubench`) at build time.
- No local `pygpubench` clone is required.

## Required steps

1. Source environment first:
```bash
set -a; source .env; set +a
```

2. Run the wrapper script:
```bash
bash skills/codex/pygpubench-modal-run/scripts/run_pygpubench_modal.sh \
  --harness /abs/path/to/harness.py \
  --submission /abs/path/to/submission.py \
  --print-log
```

## Wrapper defaults

- GPU is fixed to `B200`.
- Output JSON defaults to:
  - `artifacts/pygpubench_modal_last_run.json`

## Harness contract

Your harness must:
- define `kernel_generator()` returning a callable from `submission.py`
- define `generate_test_case(args, seed)` returning:
  - `(kernel_input_tuple, (expected_tensor, rtol, atol))`
- call `pygpubench.do_bench_isolated(...)` in `if __name__ == "__main__":`

Use:
- `tools/pygpubench_harness_template.py`
as the starting point.

## Failure handling

- If Modal run fails, re-run once with same command.
- If second run fails, treat as hard failure and preserve JSON/log output for debugging.
