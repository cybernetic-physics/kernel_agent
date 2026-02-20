# Session Notes (2026-02-18)

## B200 extraction workflow (stable)

- Preferred command:
  - `set -a; source ../kernel_rl/.env; set +a`
  - `uv run --with modal modal run tools/dump_internal_layout_tuples_modal_b200.py`
- Stable execution mode is `with_module_ip`.
- Default tool behavior now avoids noisy probe paths and runs only `with_module_ip`.

## Known cpasync dump limitation

- In this non-kernel extraction context, `cpasync.tma_partition` may fail when using
  non-exec tiled TMA atoms from CUTLASS DSL wrappers.
- Tool policy now:
  - always dump cpasync input tensors/layouts
  - do **not** attempt `cpasync.tma_partition` unless explicitly enabled
  - use `status` field (`not_attempted|attempted|ok|failed`)

## Epilogue TMEM->register/store data

- New tool:
  - `tools/dump_epilogue_t2r_layouts_modal_b200.py`
- Artifact:
  - `artifacts/wagmiv67_t2r_layouts.json`
- Folded into tract report:
  - `tools/tract_dump_insights.py --t2r-json ...`
  - output includes `epilogue_t2r.cases[*].epilogue_graph.checks`

## Current core artifacts

- `artifacts/wagmiv67_b200_dump.json`
- `artifacts/wagmiv67_t2r_layouts.json`
- `artifacts/wagmiv67_tract_insights.json`
- `artifacts/wagmiv67_tract_findings.md`

## Notes for future runs

- Modal connectivity can fail transiently (`Could not connect to the Modal server`); retry usually works.
- For `modal run`, CLI args may not always propagate to local entrypoint; env toggles are safer.
- For `dump_internal_layout_tuples_modal_b200.py`:
  - `NVFP4_PROBE_MODES=1`
  - `NVFP4_TRY_CPASYNC_PARTITION=1`
