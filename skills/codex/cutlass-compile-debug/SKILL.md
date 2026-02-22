---
name: cutlass-compile-debug
description: Run deep CUTLASS/CuTe compile diagnostics on Modal B200 and collect rich failure artifacts (traceback, IR/PTX/CUBIN, logs) when popcorn-cli output is too terse.
---

# Cutlass Compile Debug

Use this when popcorn compile/test output is not enough to diagnose a failure.

## Command

1. Source env:
```bash
set -a; source .env; set +a
```
2. Run compile-debug tool on B200:
```bash
uv run python tools/debug_cutlass_compile_modal.py \
  --kernel kernels/nvfp4_group_gemm/wagmiv67.py \
  --gpu B200 \
  --json-out artifacts/compile_debug_wagmiv67.json
```

## Higher-detail retry

If needed, rerun with lower optimization and extra assertions:
```bash
uv run python tools/debug_cutlass_compile_modal.py \
  --kernel kernels/nvfp4_group_gemm/wagmiv67.py \
  --gpu B200 \
  --compile-options '--opt-level 0 --enable-device-assertions' \
  --print-ir \
  --json-out artifacts/compile_debug_wagmiv67.opt0.json
```

## Key artifacts

Read these first:
1. `artifacts/compile_debug_<kernel>/summary.json`
2. `artifacts/compile_debug_<kernel>/error_message_full.txt`
3. `artifacts/compile_debug_<kernel>/traceback_full.txt`
4. `artifacts/compile_debug_<kernel>/remote_dump/*.mlir`
5. `artifacts/compile_debug_<kernel>/remote_dump/*.ptx`
6. `artifacts/compile_debug_<kernel>/remote_dump/*.cubin`

## Follow-up

If cubin exists, disassemble with skill `cubin-disasm`.
