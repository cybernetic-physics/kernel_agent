---
name: gpu-profiler
description: Profile CUTLASS kernels on Modal B200 with working paths. Prefer torch.profiler kernel-time breakdown; use nsys only as supplemental range/timeline signal.
---

# Gpu Profiler

Use the profiler flow that works reliably on Modal B200.

## Commands

1. Source env:
```bash
set -a; source .env; set +a
```
2. Primary path (recommended): torch.profiler per-kernel CUDA times:
```bash
uv run python tools/profile_kernel_torch_modal.py \
  --kernel kernels/nvfp4_group_gemm/wagmiv67.py \
  --gpu B200 \
  --warmup 2 \
  --profile-iters 12 \
  --json-out artifacts/torch_profiler_wagmiv67.json
```
3. Optional nsys context (range/timeline only):
```bash
uv run --with modal python tools/profile_kernel.py \
  --kernel kernels/nvfp4_group_gemm/wagmiv67.py \
  --mode nsys --remote --gpu B200 \
  --warmup 1 --profile-iters 1 \
  --json-out artifacts/profile_nsys_modal.json
```

## Interpretation Rules

1. Treat `profile_kernel_torch_modal.py` as the main performance evidence for kernel iteration.
2. In torch profiler output, focus on:
   - `events[0].name`
   - `events[0].self_cuda_us`
   - `events[0].calls`
3. Use nsys output as secondary context only.
4. If nsys reports NVTX fallback, treat it as range-level timing only.

## Modal Caveats

1. Nsight Compute counters (`ncu`) can fail in Modal runtime with `LibraryNotLoaded`.
2. Nsight Systems CUDA traces can fail; NVTX-only fallback may still work.
3. Always keep `--gpu B200` for leaderboard parity.

## Escalation

If compile/profiler output is insufficient, switch to skill:
- `cutlass-compile-debug` for full compile diagnostics and IR/PTX/CUBIN bundle.
