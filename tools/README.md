# Tools README

This directory contains CLI utilities and helper modules for NVFP4 Group GEMM layout extraction, tract-style analysis, and regression checks.

## Prerequisites

- From repo root:
  - `uv sync`
- For local CUTLASS-dependent scripts:
  - Use an environment where `cutlass.cute` is importable.
- For Modal B200 scripts:
  - Export Modal credentials from this repo env file (`.env`).

Example Modal env setup:

```bash
set -a; source .env; set +a
```

## Common workflows

1. Collect B200 internal layout dump:
   - `uv run --with modal modal deploy tools/dump_internal_layout_tuples_modal_b200.py`
   - `uv run --with modal python tools/dump_internal_layout_tuples_modal_b200.py`
2. Collect B200 epilogue TMEM->register dump:
   - `uv run --with modal modal deploy tools/dump_epilogue_t2r_layouts_modal_b200.py`
   - `uv run --with modal python tools/dump_epilogue_t2r_layouts_modal_b200.py`
3. Fold both dumps into tract report:
   - `uv run python tools/tract_dump_insights.py --dump-json artifacts/wagmiv67_b200_dump.json --t2r-json artifacts/wagmiv67_t2r_layouts.json --json-out artifacts/wagmiv67_tract_insights.json`
4. Snapshot/diff layout regressions:
   - `uv run python tools/layout_snapshot.py --json-out artifacts/snapshot.json --md-out artifacts/snapshot.md`
   - `uv run python tools/layout_diff.py --old artifacts/snapshot_old.json --new artifacts/snapshot_new.json`
5. Profile a kernel (nsys / ncu / compute-sanitizer):
   - `uv run python tools/profile_kernel.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --mode ncu --json-out artifacts/profile_ncu.json`
   - `uv run python tools/profile_kernel.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --mode nsys --json-out artifacts/profile_nsys.json`
   - `uv run python tools/profile_kernel.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --mode sanitizer --json-out artifacts/profile_sanitizer.json`
   - Remote (Modal B200): add `--remote --gpu B200`
6. Run a CUTLASS execution smoke example on Modal B200:
   - `uv run --with modal modal deploy tools/cutlass_execution_modal_b200_example.py`
   - `uv run --with modal python tools/cutlass_execution_modal_b200_example.py`
7. Execute arbitrary Python snippets on Modal B200 (Codex-friendly):
   - `uv run --with modal python tools/modal_python_exec.py --code "import torch; print(torch.__version__)"`
   - `cat <<'PY' | uv run --with modal python tools/modal_python_exec.py`
     `import pkgutil, cutlass`
     `print(cutlass.__name__)`
     `PY`
8. Wafer B200 quickstart (workspace + profiling):
   - Login + confirm:
     - `wafer settings login`
     - `wafer settings whoami`
   - Create a workspace target:
     - `wafer target init workspace b200-dev --wait`
     - `wafer target list --json`
   - Smoke test CUDA on target:
     - `wafer target run --name b200-dev -- python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
   - NCU run (recommended currently):
     - `wafer tool ncu run --dir /tmp/wafer_ncu --ncu-args "--set launchstats" --output /tmp/wafer_ncu/launchstats.ncu-rep python ncu_launchstats.py`
   - NCU analyze (remote API):
     - `wafer tool ncu analyze /tmp/wafer_ncu/launchstats.ncu-rep --remote --json`
   - Cleanup workspace:
     - `wafer target remove b200-dev`
9. Debug a CUTLASS compile failure with rich Modal artifacts:
   - `uv run python tools/debug_cutlass_compile_modal.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --gpu B200 --json-out artifacts/compile_debug_wagmiv67.json`
10. Disassemble dumped CUBIN(s) into SASS via Modal:
   - `uv run python tools/disassemble_cubin_modal.py --cubin artifacts/ptx_dump_wagmiv67 --gpu B200 --json-out artifacts/cubin_disasm_wagmiv67.json`
11. Profile per-kernel CUDA time with torch.profiler on Modal:
   - `uv run python tools/profile_kernel_torch_modal.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --gpu B200 --json-out artifacts/torch_profiler_wagmiv67.json`

## Script catalog

### `loop_coordinator.py`
- What it does: runs an external state machine for worker/reviewer optimization loops with lockfile, heartbeat, stop conditions, and per-iteration artifacts.
- Modes:
  - `manual`: waits for `worker_result.json` / `reviewer_verdict.json` from external sessions.
  - `command`: runs worker/reviewer command templates each iteration.
  - `dry-run`: synthesizes outputs to validate wiring.
- Example:
  - `uv run python tools/loop_coordinator.py --fresh-start --execution-mode dry-run --kernel-path kernels/nvfp4_group_gemm_001.py --target-metric-name gmean_us --target-metric-threshold 320 --max-iterations 5`
- State root (default): `artifacts/loop_coordinator/`

### `loop_verdict_schema.py`
- What it does: validates iteration payload JSON against coordinator-required schema.
- Examples:
  - `uv run python tools/loop_verdict_schema.py --kind worker --json-file artifacts/loop_coordinator/iter_0001/worker_result.json`
  - `uv run python tools/loop_verdict_schema.py --kind reviewer --json-file artifacts/loop_coordinator/iter_0001/reviewer_verdict.json`

### `run_loop_coordinator.sh`
- What it does: thin wrapper around `tools/loop_coordinator.py` with repo-root defaults.
- Example:
  - `bash tools/run_loop_coordinator.sh --execution-mode dry-run --max-iterations 3`

### `loop_tui.py`
- What it does: small terminal UI that shows live coordinator heartbeat, iteration file status, and recent run log.
- Modes:
  - monitor only: `python3 tools/loop_tui.py`
  - launch + monitor: `python3 tools/loop_tui.py --run -- python3 tools/loop_coordinator.py ...`
- Displays worker-reported best-time updates from `iter_XXXX/worker_progress.json`.

### `run_loop_afterhours.py`
- What it does: convenience launcher for the afterhours worker/reviewer loop with key=value overrides.
- Example:
  - `python3 tools/run_loop_afterhours.py target_threshold=9.5 max_iterations=600`

### `loop_best_time_report.py`
- What it does: update `worker_progress.json` with live benchmark timing and best-time tracking (`gmean_us`, lower is better).
- Example:
  - `python3 tools/loop_best_time_report.py --progress-json artifacts/loop_coordinator/iter_0001/worker_progress.json --status benchmark --time-us 312.4 --artifact artifacts/run.txt --note "candidate benchmark"`

### `dump_internal_layout_tuples.py`
- What it does: local dump of core internal layouts (`a/b smem`, `sfa/sfb smem`, `tCtSFA/SFB`).
- Needs: local CUTLASS.
- Example:
  - `uv run python tools/dump_internal_layout_tuples.py --shape 80,4096,7168,1 --json-out artifacts/local_layout_dump.json`

### `dump_internal_layout_tuples_modal_b200.py`
- What it does: remote B200 dump of internal layouts, `thr_mma_partitions`, `tcgen05_s2t_partitions`, and cpasync input records.
- Needs: Modal + B200.
- Stable default: runs only `with_module_ip`.
- Example:
  - `uv run --with modal modal deploy tools/dump_internal_layout_tuples_modal_b200.py`
  - `uv run --with modal python tools/dump_internal_layout_tuples_modal_b200.py`
- Optional env toggles:
  - `NVFP4_PROBE_MODES=1` to run all context probe modes.
  - `NVFP4_TRY_CPASYNC_PARTITION=1` to best-effort call `cpasync.tma_partition`.

### `dump_epilogue_t2r_layouts_modal_b200.py`
- What it does: remote B200 dump of epilogue TMEM->register/store layouts:
  - `tCtAcc_fake.layout`
  - `tDtAcc.layout`
  - `tDgC.layout`
- Needs: Modal + B200.
- Example:
  - `uv run --with modal modal deploy tools/dump_epilogue_t2r_layouts_modal_b200.py`
  - `uv run --with modal python tools/dump_epilogue_t2r_layouts_modal_b200.py`

### `tract_dump_insights.py`
- What it does: offline tract-style analysis over dumped layout JSON.
- Input: dump JSON from modal/local; optional epilogue t2r JSON.
- Output: per-layout tractability/compactness/coalesce signals + optional `epilogue_t2r` fold checks.
- Example:
  - `uv run python tools/tract_dump_insights.py --dump-json artifacts/wagmiv67_b200_dump.json --t2r-json artifacts/wagmiv67_t2r_layouts.json --json-out artifacts/wagmiv67_tract_insights.json`

### `layout_snapshot.py`
- What it does: generates a digest snapshot manifest (JSON/Markdown) for key layouts.
- Needs: local CUTLASS.
- Example:
  - `uv run python tools/layout_snapshot.py --shape 80,4096,7168,1 --kernel-file kernels/nvfp4_group_gemm/wagmiv67.py --json-out artifacts/snapshot.json --md-out artifacts/snapshot.md`

### `layout_diff.py`
- What it does: diffs two snapshot manifests and optionally enforces protected-layout invariants.
- Example:
  - `uv run python tools/layout_diff.py --old artifacts/snapshot_old.json --new artifacts/snapshot_new.json --protected a_smem_layout_staged.outer`

### `prove_blockscaled_sf_layouts.py`
- What it does: proof-oriented checks and simplification report for SFA/SFB layouts (helper vs explicit forms, tract equivalence, broadcast analysis).
- Needs: local CUTLASS + tract.
- Example:
  - `uv run python tools/prove_blockscaled_sf_layouts.py --json-out artifacts/sf_layout_proof.json`

### `prove_cta_indexing.py`
- What it does: verifies old CTA mapping vs prefix-sum/binary-search mapping on benchmark/custom/random cases.
- Example:
  - `uv run python tools/prove_cta_indexing.py --emit-snippet`

### `tmem_footprint_report.py`
- What it does: reports acc/sfa/sfb TMEM columns and allocator recommendations for shapes.
- Needs: local CUTLASS.
- Example:
  - `uv run python tools/tmem_footprint_report.py --json-out artifacts/tmem_footprint.json --csv-out artifacts/tmem_footprint.csv`

### `tmem_packing_search.py`
- What it does: explores segment order/alignment for TMEM packing slack.
- Needs:
  - local CUTLASS if using `--shape`
  - no CUTLASS if using explicit `--acc-cols --sfa-cols --sfb-cols`.
- Example:
  - `uv run python tools/tmem_packing_search.py --shape 80,4096,7168,1 --top-k 8`

### `epilogue_store_search.py`
- What it does: enumerates thread/value layout candidates for epilogue store coalescing quality.
- Example:
  - `uv run python tools/epilogue_store_search.py --tile-m 128 --tile-n 128 --threads 128 --vector-widths 1,2,4,8,16 --top-k 10`

### `profile_kernel.py`
- What it does: profiles any kernel file with Nsight Systems, Nsight Compute, or Compute Sanitizer.
  Generates a self-contained harness, runs the profiler, parses output into structured JSON with
  bottleneck classification and optimisation hints for AI-agent consumption.
- Modes: `nsys` (timeline), `ncu` (detailed kernel metrics), `sanitizer` (correctness).
- Needs: B200 GPU (local or `--remote` via Modal), nsys/ncu/compute-sanitizer on PATH.
- Examples:
  - Quick timeline: `uv run python tools/profile_kernel.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --mode nsys`
  - Detailed metrics (basic set): `uv run python tools/profile_kernel.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --mode ncu --json-out artifacts/profile_ncu.json`
  - Extended metrics: `uv run python tools/profile_kernel.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --mode ncu --ncu-set extended`
  - Correctness check: `uv run python tools/profile_kernel.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --mode sanitizer --sanitizer-tool racecheck`
  - Remote B200 (recommended path): `uv run --with modal python tools/profile_kernel.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --mode nsys --remote --gpu B200 --json-out artifacts/profile_nsys_modal.json`
  - Custom problem sizes: `--problem-sizes '80,4096,7168,1;40,7168,2048,1'`
- JSON output includes: kernel metadata, GPU info, key performance metrics, classification flags
  (memory_bound/compute_bound/latency_bound/occupancy_limited), optimisation hints, raw profiler
  sections, and stall breakdown.
- Remote B200 notes:
  - Use `python tools/profile_kernel.py --remote ...`; this path uses deployed `ModalToolsRunner`.
  - Deploy once for best latency: `uv run --with modal modal deploy tools/modal_tools_app.py`.
  - `--gpu B200` is recommended to keep leaderboard parity.
  - `nsys` is the currently working remote path; if CUDA event parsing fails (`Unrecognized GPU UUID`),
    the tool automatically retries NVTX-only trace mode and parses `nvtx_sum`.
  - In NVTX fallback mode, metrics are range-level timings (`:profile_region`), not kernel microarchitectural counters.
  - `ncu` on current Modal B200 runtime still fails with counter init error (`LibraryNotLoaded`).
  - `compute-sanitizer` currently reports unsupported device in this environment.
  - If a run appears stuck, check local processes with `pgrep -fl "profile_kernel.py"` and terminate stale runs before retrying.

### `dump_ptx_modal.py`
- What it does: compiles a CUTLASS DSL kernel on Modal and returns dumped PTX and CUBIN to local files.
- Why: gives reproducible raw PTX artifacts for inspection/disassembly without relying on local CUDA toolchain.
- Needs:
  - Modal auth + env sourced (`set -a; source .env; set +a`)
  - Kernel module exposes `compile_kernel(problem_sizes)`.
- Example:
  - `uv run python tools/dump_ptx_modal.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --problem-sizes '80,4096,7168,1;40,7168,2048,1' --out artifacts/wagmiv67.sm100a.ptx --json-out artifacts/wagmiv67.sm100a.ptx.json`
- Notes:
  - Tool sets `CUTE_DSL_KEEP_PTX=1` and `CUTE_DSL_DUMP_DIR=/tmp/...` inside the Modal sandbox before compilation.
  - CUBIN dumping is enabled by default (`CUTE_DSL_KEEP_CUBIN=1`). Use `--no-cubin` for PTX-only output.
  - Saved outputs include:
    - primary PTX at `--out`
    - all dumped PTX/CUBIN files under `artifacts/ptx_dump_<kernel_stem>/` (or `--out-dir`).

### `debug_cutlass_compile_modal.py`
- What it does: compiles a kernel's `compile_kernel(problem_sizes)` in Modal with verbose CuTe DSL diagnostics and writes a full debug bundle:
  - CuTe DSL log config (`CUTE_DSL_LOG_*`)
  - compile traceback/error type
  - dumped IR/PTX/CUBIN from `CUTE_DSL_DUMP_DIR`
  - optional compile option overrides (e.g. `--opt-level 0`)
- Why: popcorn-cli compile errors are often terse; this gives actionable detail for DSL/MLIR failures.
- Example:
  - `uv run python tools/debug_cutlass_compile_modal.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --gpu B200 --compile-options '--opt-level 0 --enable-device-assertions' --json-out artifacts/compile_debug_wagmiv67.json`
- Output:
  - `artifacts/compile_debug_<kernel>/summary.json`
  - `artifacts/compile_debug_<kernel>/modal_stdout.log`
  - `artifacts/compile_debug_<kernel>/modal_stderr.log`
  - `artifacts/compile_debug_<kernel>/remote_dump/*` (IR/PTX/CUBIN and related dump files)
  - `artifacts/compile_debug_<kernel>/compiled_attrs/*` (inline `__mlir__`/`__ptx__`/`__cubin__` when available)

### `disassemble_cubin_modal.py`
- What it does: disassembles one or more `.cubin` files to SASS (`.sass`) on Modal using `nvdisasm`.
- Why: SASS inspection is often required for low-level diagnosis (instruction selection, scheduling, unexpected control flow).
- Examples:
  - Single cubin:
    - `uv run python tools/disassemble_cubin_modal.py --cubin artifacts/ptx_dump_wagmiv67/kernel_0.cubin --gpu B200`
  - Directory of cubins:
    - `uv run python tools/disassemble_cubin_modal.py --cubin artifacts/ptx_dump_wagmiv67 --gpu B200 --json-out artifacts/cubin_disasm_wagmiv67.json`
- Output:
  - `artifacts/cubin_disasm_<name>/*.sass`
  - `artifacts/cubin_disasm_<name>/summary.json`

### `profile_kernel_torch_modal.py`
- What it does: runs a CUTLASS kernel via `custom_kernel(data)` in Modal and captures per-event CUDA timings using `torch.profiler`.
- Why: gives actionable kernel-level timing when Nsight Compute counters are unavailable in the runtime.
- Example:
  - `uv run python tools/profile_kernel_torch_modal.py --kernel kernels/nvfp4_group_gemm/wagmiv67.py --gpu B200 --warmup 2 --profile-iters 12 --json-out artifacts/torch_profiler_wagmiv67.json`
- Output:
  - `artifacts/torch_profiler_<kernel>/summary.json`
  - `artifacts/torch_profiler_<kernel>/table.txt`
  - `artifacts/torch_profiler_<kernel>/modal_stdout.log`
  - `artifacts/torch_profiler_<kernel>/modal_stderr.log`

### `cutlass_execution_modal_b200_example.py`
- What it does: runs a real CUTLASS DSL kernel (`custom_kernel(...)`) on Modal B200 and reports latency + output sanity checks.
- Default kernel: `kernels/nvfp4_group_gemm/wagmiv67.py`.
- Example:
  - `uv run --with modal modal deploy tools/cutlass_execution_modal_b200_example.py`
  - `uv run --with modal python tools/cutlass_execution_modal_b200_example.py`
  - Custom kernel path: `--kernel kernels/nvfp4_group_gemm/wagmiv67.py`
  - Custom shapes: `--problem-sizes '80,4096,7168,1;40,7168,2048,1'`
  - Control timing: `--warmup 2 --iters 5`

### `modal_python_exec.py`
- What it does: runs arbitrary Python code in a Modal Sandbox on GPU (default `B200`) with stdin/code-string/file input modes.
- Why: makes Codex-style snippet execution easy without shell quoting gymnastics.
- Examples:
  - Code as argument:
    - `uv run --with modal python tools/modal_python_exec.py --code "import torch; print(torch.cuda.is_available())"`
  - Code via heredoc/stdin:
    - `cat <<'PY' | uv run --with modal python tools/modal_python_exec.py`
      `import pkgutil, cutlass`
      `print(cutlass.__name__)`
      `PY`
  - Code from file:
    - `uv run --with modal python tools/modal_python_exec.py --code-file /tmp/snippet.py`
  - Import repo modules inside sandbox:
    - `uv run --with modal python tools/modal_python_exec.py --mount-repo --code "import tools.profile_parsers as p; print('ok', p.__name__)"`
- Defaults:
  - GPU: `B200`
  - Image: `nvidia/cuda:12.8.0-devel-ubuntu22.04`
  - Packages: `nvidia-cutlass-dsl==4.4.0`, `torch`
- Mount options:
  - `--mount-repo` mounts local repo (default `.`) to `/workspace/repo`, prepends it to `sys.path`, and `chdir`s there.
  - `--mount-repo` also auto-stubs a minimal `task` module (`input_t`, `output_t`) if missing, for kernel-file imports.
  - `--repo-local-path` / `--repo-remote-path` override default repo mount.
  - `--mount-local-dir local:remote` adds extra directory mounts.

## Helper modules

### `nvfp4_layout_model.py`
- Shared model/build helpers used by multiple tools:
  - benchmark shape sets
  - layout bundle construction
  - TMEM column utilities
- Imported by other scripts, not usually invoked directly.

### `tract_layout_utils.py`
- Shared tract/canonicalization/equivalence/digest utilities.
- Imported by analysis/proof scripts, not usually invoked directly.

### `profile_parsers.py`
- Parsers for nsys/ncu/compute-sanitizer raw output into structured Python dicts.
- Pure Python, no external deps.  Used by `profile_kernel.py`, also usable standalone.

### `profile_analysis.py`
- Bottleneck classification and optimisation hint generation from parsed profiler metrics.
- Thresholds tuned for Blackwell B200 tcgen05/TMA grouped GEMM workloads.
- Pure Python, no external deps.

## Additional reference

- `README_internal_layout_tuples.md`: detailed notes about B200 extraction modes and failure behaviors for internal tuple dumps.

## Wafer notes (current CLI behavior)

- The current CLI namespace is `wafer tool ...` and `wafer target ...` (not older docs that show `wafer nvidia ...` or `wafer workspaces ...`).
- `wafer tool ncu run` can fail with `ERR_NVGPUCTRPERM` on full metric sets. A practical fallback is `--ncu-args "--set launchstats"` which successfully produces a `.ncu-rep`.
- `wafer tool nsys profile --target workspace:...` is currently broken in `wafer-ai 0.0.13` with `ImportError: get_workspace_info`.
- Direct `wafer target run ... /usr/bin/nsys profile ...` currently fails with an internal Wafer session timeout in this environment; use `wafer tool ncu run` as the stable profiling path for now.
