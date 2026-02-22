#!/usr/bin/env python3
"""Dump wagmiv67-style epilogue TMEM->register layouts on deployed Modal B200.

Deploy once:
    uv run --with modal modal deploy tools/dump_epilogue_t2r_layouts_modal_b200.py
Then run repeatedly without `modal run`:
    uv run --with modal python tools/dump_epilogue_t2r_layouts_modal_b200.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "nvfp4-wagmiv67-t2r-layouts-b200"
DEFAULT_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
CACHE_VOLUME_NAME = "modal-tools-cache-v1"
CACHE_MOUNT_PATH = "/cache"

app = modal.App(APP_NAME)
image = modal.Image.from_registry(DEFAULT_IMAGE, add_python="3.11").pip_install(
    "nvidia-cutlass-dsl==4.4.0"
)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Shape m,n,k,l (repeatable). Defaults to two ranked-case representatives.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/wagmiv67_t2r_layouts.json"),
        help="Path to write JSON output.",
    )
    p.add_argument(
        "--app-name",
        default=APP_NAME,
        help=f"Deployed Modal app name (default: {APP_NAME}).",
    )
    p.add_argument(
        "--function-name",
        default="_dump_one",
        help="Deployed Modal function name (default: _dump_one).",
    )
    p.add_argument("--gpu", default="B200", help="GPU type (B200-only).")
    # Modal CLI may inject argv fragments into local entrypoint processes.
    args, _unknown = p.parse_known_args()
    return args


def _parse_shape(spec: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected shape m,n,k,l; got {spec!r}")
    return (parts[0], parts[1], parts[2], parts[3])


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, bool, int, float)) or x is None:
        return x
    if isinstance(x, (tuple, list)):
        return [_to_jsonable(v) for v in x]
    try:
        return int(x)
    except Exception:
        return str(x)


def _layout_record(layout: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"repr": str(layout)}
    if hasattr(layout, "shape"):
        out["shape"] = _to_jsonable(layout.shape)
    if hasattr(layout, "stride"):
        out["stride"] = _to_jsonable(layout.stride)
    return out


@app.function(
    image=image,
    gpu="B200",
    timeout=900,
    volumes={CACHE_MOUNT_PATH: cache_volume},
)
def _dump_one(shape_mnkl: tuple[int, int, int, int]) -> dict[str, Any]:
    import os

    os.environ.setdefault("XDG_CACHE_HOME", f"{CACHE_MOUNT_PATH}/xdg")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{CACHE_MOUNT_PATH}/torchinductor")
    os.environ.setdefault("TRITON_CACHE_DIR", f"{CACHE_MOUNT_PATH}/triton")
    os.environ.setdefault("CUDA_CACHE_PATH", f"{CACHE_MOUNT_PATH}/cuda")

    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir import ir
    from cutlass.cute.nvgpu import tcgen05

    m, n, k, l = [int(x) for x in shape_mnkl]
    mma_tiler_mnk = (128, 128, 256)
    mma_inst_shape_k = 64
    sf_dtype = cutlass.Float8E4M3FN

    with ir.Context():
        with ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                loc = ir.Location.current

                mma_op = tcgen05.MmaMXF4NVF4Op(
                    sf_dtype,
                    (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
                    tcgen05.CtaGroup.ONE,
                    tcgen05.OperandSource.SMEM,
                )
                tiled_mma = cute.make_tiled_mma(mma_op, loc=loc)

                mC_mnl = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float16, 0, cute.AddressSpace.gmem, assumed_align=16
                    ),
                    cute.make_layout((m, n, l), stride=(n, 1, m * n), loc=loc),
                    loc=loc,
                )
                gC_mnl = cute.local_tile(
                    mC_mnl,
                    cute.slice_(mma_tiler_mnk, (None, None, 0), loc=loc),
                    (0, 0, 0),
                    loc=loc,
                )

                thr_mma = tiled_mma.get_slice(0)
                tCgC = thr_mma.partition_C(gC_mnl, loc=loc)

                acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2], loc=loc)
                tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape, loc=loc)

                op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
                copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32, loc=loc)
                tiled_copy_t2r = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tCtAcc_fake[None, 0, 0], loc=loc
                )
                thr_copy_t2r = tiled_copy_t2r.get_slice(0)
                tDtAcc = thr_copy_t2r.partition_S(tCtAcc_fake[None, 0, 0], loc=loc)
                tDgC = thr_copy_t2r.partition_D(tCgC[None, 0, 0], loc=loc)

                return {
                    "shape_mnkl": [m, n, k, l],
                    "tCtAcc_fake_layout": _layout_record(tCtAcc_fake.layout),
                    "tDtAcc_layout": _layout_record(tDtAcc.layout),
                    "tDgC_layout": _layout_record(tDgC.layout),
                }


def main() -> None:
    args = _parse_args()
    if args.gpu != "B200":
        raise SystemExit("dump_epilogue_t2r_layouts_modal_b200.py is B200-only.")

    default_shapes = [
        (80, 4096, 7168, 1),
        (40, 7168, 2048, 1),
    ]
    shapes = [_parse_shape(s) for s in args.shape] if args.shape else default_shapes

    remote_fn = modal.Function.from_name(args.app_name, args.function_name)
    result = {str(tuple(shape)): remote_fn.remote(shape) for shape in shapes}
    text = json.dumps(result, indent=2)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"[INFO] Wrote: {args.json_out}")

    print(text)


if __name__ == "__main__":
    main()
