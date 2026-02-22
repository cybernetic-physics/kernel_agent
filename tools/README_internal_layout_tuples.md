# Internal Layout Tuple Dump (NVFP4 Group GEMM)

This documents how the concrete internal `shape` / `stride` tuples were extracted for:

- `a_smem_layout_staged` (outer + inner swizzle)
- `b_smem_layout_staged` (outer + inner swizzle)
- `sfa_smem_layout_staged`
- `sfb_smem_layout_staged`
- `tCtSFA_layout`
- `tCtSFB_layout`
- `thr_mma.partition_*` layouts (`tCgA`, `tCgB`, `tCgC`, `tCgSFA`, `tCgSFB`)
- `tcgen05` S2T copy partition layouts
- `cpasync.tma_partition` input layouts (and optional best-effort outputs)

## Why two scripts exist

- `tools/dump_internal_layout_tuples.py`
  - Local extraction path.
  - Uses existing `tools/nvfp4_layout_model.py` and requires local CUTLASS DSL install.

- `tools/dump_internal_layout_tuples_modal_b200.py`
  - Remote extraction path on Modal with `gpu="B200"`.
  - Added because local environments often lack CUTLASS DSL binaries.

## B200 failure mode and fix

On `nvidia-cutlass-dsl==4.4.0`, naive calls fail in two ways:

1. `RuntimeError: ... requires a Context ...`
2. allocator aborts (`unaligned fastbin/tcache chunk`) in some context-only setups

Working approach implemented in the script:

1. Run extraction inside an explicit MLIR `Context`.
2. Establish a default `Location` via `with ir.Location.unknown():`.
3. Create an explicit `ir.Module`.
4. Execute DSL ops under `ir.InsertionPoint(module.body)`.
5. Pass `loc=` to relevant `cute`/helper calls in that path.

This mode is called `with_module_ip` in the script and is the stable path used to get tuples on B200.

## Command used

```bash
set -a; source .env; set +a
uv run --with modal modal deploy tools/dump_internal_layout_tuples_modal_b200.py
uv run --with modal python tools/dump_internal_layout_tuples_modal_b200.py
```

You can also run through `uv`:

```bash
uv run --with modal python tools/dump_internal_layout_tuples_modal_b200.py
```

By default, the script runs only the stable `with_module_ip` mode and does not call
`cpasync.tma_partition` (to avoid backend type errors in non-kernel context).

Optional flags:

- `--probe-modes`: run all context probe modes (`plain`, `with_mlir_context`,
  `with_explicit_loc`, `with_module_ip`)
- `--try-cpasync-partition`: best-effort call to `cpasync.tma_partition`

You can also set env flags:

- `NVFP4_PROBE_MODES=1`
- `NVFP4_TRY_CPASYNC_PARTITION=1`

The script returns a JSON object with:

- `ok: true|false`
- `mode`: successful mode (`with_module_ip` when successful)
- `payload`: extracted tuple data, including:
  - base SMEM/TMEM layouts
  - `thr_mma_partitions`
  - `tcgen05_s2t_partitions`
  - `cpasync_tma_partitions`
- `cpasync_tma_partitions.inputs` is always present (grouped tensor/layout inputs)
- `cpasync_tma_partitions.status` is one of `not_attempted`, `attempted`, `ok`, `failed`
- `cpasync_tma_partitions.outputs` is present when partition construction succeeds
- `cpasync_tma_partitions.error` is populated when the attempted partition fails
- `attempts`: diagnostics for each attempted mode
