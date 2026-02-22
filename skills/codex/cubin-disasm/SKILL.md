---
name: cubin-disasm
description: Disassemble generated CUBIN to SASS on Modal B200 with nvdisasm for low-level instruction inspection.
---

# Cubin Disasm

Use this after PTX/CUBIN dump or compile-debug to inspect emitted SASS.

## Commands

1. Source env:
```bash
set -a; source .env; set +a
```
2. Disassemble all cubins in a directory:
```bash
uv run python tools/disassemble_cubin_modal.py \
  --cubin artifacts/ptx_dump_wagmiv67 \
  --gpu B200 \
  --json-out artifacts/cubin_disasm_wagmiv67.json
```
3. Disassemble one cubin:
```bash
uv run python tools/disassemble_cubin_modal.py \
  --cubin artifacts/ptx_dump_wagmiv67/kernel_0.cubin \
  --gpu B200
```

## Output

1. `artifacts/cubin_disasm_<name>/*.sass`
2. `artifacts/cubin_disasm_<name>/summary.json`
3. `artifacts/cubin_disasm_<name>/modal_stdout.log`
4. `artifacts/cubin_disasm_<name>/modal_stderr.log`

## Typical checks

1. Confirm expected tensor core instruction families are present.
2. Compare branch/predicate density between revisions.
3. Compare code size and hot basic blocks across versions.
