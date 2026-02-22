---
name: torch-profiler-modal
description: Profile CUTLASS kernels on Modal B200 using torch.profiler CUDA activity to get per-kernel timing when Nsight counters are unavailable.
---

# Torch Profiler Modal

Use this as the default performance profiler on Modal B200.

## Command

1. Source env:
```bash
set -a; source ../kernel_rl/.env; set +a
```
2. Run torch profiler:
```bash
uv run python tools/profile_kernel_torch_modal.py \
  --kernel kernels/nvfp4_group_gemm/wagmiv67.py \
  --gpu B200 \
  --warmup 2 \
  --profile-iters 12 \
  --top-k 20 \
  --json-out artifacts/torch_profiler_wagmiv67.json
```

## Output

1. `artifacts/torch_profiler_<kernel>/summary.json`
2. `artifacts/torch_profiler_<kernel>/table.txt`
3. `artifacts/torch_profiler_<kernel>/modal_stdout.log`
4. `artifacts/torch_profiler_<kernel>/modal_stderr.log`

## Interpretation

Focus on top kernel event:
1. `events[0].name`
2. `events[0].self_cuda_us`
3. `events[0].calls`
4. compare `self_cuda_us / calls` across revisions
