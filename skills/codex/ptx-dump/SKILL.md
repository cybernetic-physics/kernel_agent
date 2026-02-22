---
name: ptx-dump
description: Dump raw PTX and CUBIN for CUTLASS DSL kernels on Modal B200 using `tools/dump_ptx_modal.py`. Use when validating generated codegen, sharing compiler artifacts, or comparing emitted PTX/SASS-relevant binaries across kernel changes.
---

# Ptx Dump

Use the reusable PTX dump tool so extraction is deterministic and does not require local CUDA tooling.

## Commands

1. Source env:
```bash
set -a; source .env; set +a
```
2. Dump PTX + CUBIN for a kernel + shape set:
```bash
uv run python tools/dump_ptx_modal.py \
  --kernel kernels/nvfp4_group_gemm/wagmiv67.py \
  --problem-sizes '80,4096,7168,1;40,7168,2048,1' \
  --gpu B200 \
  --out artifacts/wagmiv67.sm100a.ptx \
  --json-out artifacts/wagmiv67.sm100a.ptx.json
```
3. PTX-only (if needed):
```bash
uv run python tools/dump_ptx_modal.py \
  --kernel kernels/nvfp4_group_gemm/wagmiv67.py \
  --problem-sizes '80,4096,7168,1;40,7168,2048,1' \
  --gpu B200 \
  --out artifacts/wagmiv67.sm100a.ptx \
  --no-cubin
```
4. Inspect PTX quickly:
```bash
head -n 60 artifacts/wagmiv67.sm100a.ptx
```
5. Optional SASS disassembly of dumped cubins:
```bash
uv run python tools/disassemble_cubin_modal.py \
  --cubin artifacts/ptx_dump_wagmiv67 \
  --gpu B200 \
  --json-out artifacts/cubin_disasm_wagmiv67.json
```

## Expected Outputs

1. Primary PTX file at `--out`.
2. All dumped files (PTX + CUBIN) under `artifacts/ptx_dump_<kernel_stem>/` (or `--out-dir`).
3. Metadata JSON with file list and parameters at `--json-out`.

## Notes

1. Tool requires kernel module to expose `compile_kernel(problem_sizes)`.
2. PTX/CUBIN dumping is enabled in the remote process via:
   - `CUTE_DSL_KEEP_PTX=1`
   - `CUTE_DSL_KEEP_CUBIN=1` (default on in tool)
   - `CUTE_DSL_DUMP_DIR=/tmp/...`
3. Use `--no-cubin` only when you explicitly need PTX-only output.
4. Keep `--problem-sizes` stable when comparing PTX/CUBIN across revisions.

## Reporting Template

After running, report:
1. kernel file and problem sizes used
2. PTX output path
3. CUBIN output path(s)
4. one short observation from PTX header (`.target`, CUDA toolchain version)
5. if disassembled, one short SASS-level observation
