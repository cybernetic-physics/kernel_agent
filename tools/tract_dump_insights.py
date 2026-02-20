#!/usr/bin/env python3
"""Offline tract-style analysis for dumped CuTe layout dictionaries.

This tool is intentionally CUTLASS-free: it consumes JSON dumps that already
contain nested shape/stride tuples and applies a subset of tract algebra
directly to those tuples for optimization triage.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.tract_layout_utils import (
    as_flat_layout,
    flatten_tuple,
    layout_cosize_flat,
    layout_function,
    layout_size_flat,
)


Nested = Any


@dataclass(frozen=True)
class Flat:
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dump-json",
        type=Path,
        default=Path("artifacts/wagmiv67_b200_dump.json"),
        help="Path to captured layout dump JSON.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=8000,
        help="Sample count for duplicate-offset estimates.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for sampling.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        help="Optional JSON output path for machine-consumable report.",
    )
    p.add_argument(
        "--t2r-json",
        type=Path,
        default=Path("artifacts/wagmiv67_t2r_layouts.json"),
        help=(
            "Optional TMEM->register epilogue dump JSON with tCtAcc/tDtAcc/tDgC "
            "layouts; used to fold epilogue graph into this report."
        ),
    )
    return p.parse_args()


def _load_categories_module(repo_root: Path):
    categories_py = repo_root / "layout-categories" / "tract" / "src" / "tract" / "categories.py"
    spec = importlib.util.spec_from_file_location("tract_categories_local", categories_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tract categories module from: {categories_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flat_from_layout(layout: Any) -> Flat:
    flat = as_flat_layout(layout)
    return Flat(shape=tuple(int(x) for x in flat.shape), stride=tuple(int(x) for x in flat.stride))


def _sort_flat_with_perm(flat: Flat) -> tuple[Flat, list[int]]:
    indexed = list(enumerate(zip(flat.shape, flat.stride), start=1))
    sorted_indexed = sorted(indexed, key=lambda item: (item[1][1], item[1][0]))
    perm = [idx for idx, _ in sorted_indexed]
    sorted_shape = tuple(v[0] for _, v in sorted_indexed)
    sorted_stride = tuple(v[1] for _, v in sorted_indexed)
    return Flat(sorted_shape, sorted_stride), perm


def _is_tractable_flat(flat: Flat) -> bool:
    sorted_flat, _ = _sort_flat_with_perm(flat)
    s = sorted_flat.shape
    d = sorted_flat.stride
    if len(s) <= 1:
        return True
    for i in range(len(s) - 1):
        if d[i] != 0:
            denom = s[i] * d[i]
            if denom == 0 or d[i + 1] % denom != 0:
                return False
    return True


def _compute_tuple_morphism(flat: Flat, tuple_morphism_cls):
    if not _is_tractable_flat(flat):
        raise ValueError("layout is not tractable")

    domain = tuple(flat.shape)
    sorted_flat, perm = _sort_flat_with_perm(flat)
    shape = sorted_flat.shape
    stride = sorted_flat.stride
    m = len(shape)

    # Number of leading zero strides in sorted order.
    k = 0
    for x in stride:
        if x == 0:
            k += 1
        else:
            break

    codomain_list: list[int] = []
    if k < m:
        codomain_list.extend([stride[k], shape[k]])
        for j in range(k + 1, m):
            denom = shape[j - 1] * stride[j - 1]
            factor = (stride[j] // denom) if denom != 0 else 0
            codomain_list.extend([int(factor), shape[j]])
    codomain = tuple(codomain_list)

    alpha_prime = [0] * m
    for j in range(k, m):
        alpha_prime[j] = 2 * (j - k + 1)

    inv_perm = [0] * m
    for i in range(m):
        inv_perm[perm[i] - 1] = i + 1
    alpha = tuple(alpha_prime[inv_perm[i] - 1] for i in range(m))

    morph = tuple_morphism_cls(domain, codomain, alpha)
    keep = tuple(
        i + 1
        for i in range(len(codomain))
        if codomain[i] != 1 or (i + 1) in morph.map
    )
    return morph.factorize(keep)


def _flat_layout_from_morphism(morphism) -> Flat:
    strides = []
    for alpha in morphism.map:
        if alpha == 0:
            strides.append(0)
            continue
        prod = 1
        for j in range(alpha - 1):
            prod *= morphism.codomain[j]
        strides.append(prod)
    return Flat(shape=tuple(int(x) for x in morphism.domain), stride=tuple(int(x) for x in strides))


def _sample_duplication(layout_dict: dict[str, Any], *, samples: int, seed: int) -> float:
    rng = random.Random(seed)
    flat = _flat_from_layout(layout_dict)
    logical_size = layout_size_flat(flat.shape)
    if logical_size <= 0:
        return 1.0
    sample_count = min(samples, logical_size)
    if sample_count == logical_size:
        idxs = range(logical_size)
    else:
        idxs = rng.sample(range(logical_size), sample_count)
    offsets = {layout_function(layout_dict, idx) for idx in idxs}
    return float(sample_count) / float(max(1, len(offsets)))


def _analyze_layout(name: str, layout_dict: dict[str, Any], tuple_morphism_cls, *, samples: int, seed: int) -> dict[str, Any]:
    flat = _flat_from_layout(layout_dict)
    size = int(layout_size_flat(flat.shape))
    cosize = int(layout_cosize_flat(flat.shape, flat.stride))
    compact = bool(size == cosize)
    nonzero_stride_modes = int(sum(1 for x in flat.stride if x != 0))
    zero_stride_modes = int(sum(1 for x in flat.stride if x == 0))
    tractable = _is_tractable_flat(flat)
    sampled_dup = _sample_duplication(layout_dict, samples=samples, seed=seed)

    out: dict[str, Any] = {
        "name": name,
        "flat_shape": list(flat.shape),
        "flat_stride": list(flat.stride),
        "rank": len(flat.shape),
        "size": size,
        "cosize": cosize,
        "size_over_cosize": (float(size) / float(cosize)) if cosize > 0 else 0.0,
        "compact": compact,
        "zero_stride_modes": zero_stride_modes,
        "nonzero_stride_modes": nonzero_stride_modes,
        "tractable": tractable,
        "sampled_duplication": sampled_dup,
        "morphism": None,
        "insights": [],
    }

    if zero_stride_modes > 0 and sampled_dup > 1.5:
        out["insights"].append("broadcast-heavy; filter_zeros-like projection is likely required")
    if not compact and sampled_dup <= 1.2 and cosize > size:
        out["insights"].append("sparse hole-heavy image; consider complement-based packing / remap search")
    if tractable:
        try:
            f = _compute_tuple_morphism(flat, tuple_morphism_cls)
            f_coal = f.coalesce()
            l_coal = _flat_layout_from_morphism(f_coal)
            out["morphism"] = {
                "domain_len": len(f.domain),
                "codomain_len": len(f.codomain),
                "map": list(f.map),
                "coalesced_domain_len": len(f_coal.domain),
                "coalesced_codomain_len": len(f_coal.codomain),
                "coalesced_map": list(f_coal.map),
                "coalesced_layout_shape": list(l_coal.shape),
                "coalesced_layout_stride": list(l_coal.stride),
            }
            if len(f_coal.domain) < len(f.domain):
                out["insights"].append(
                    f"coalesce reduces rank {len(f.domain)} -> {len(f_coal.domain)}; candidate for simpler index arithmetic"
                )
        except Exception as exc:
            out["insights"].append(f"morphism construction failed: {type(exc).__name__}: {exc}")
    else:
        out["insights"].append("non-tractable; avoid algebraic rewrites without explicit equivalence proof")

    # Filtered (non-zero-stride) sublayout approximates cute.filter_zeros behavior at flat level.
    nz = [(s, d) for s, d in zip(flat.shape, flat.stride) if d != 0]
    if nz:
        fshape = tuple(s for s, _ in nz)
        fstride = tuple(d for _, d in nz)
        ftract = _is_tractable_flat(Flat(fshape, fstride))
        out["filtered_nonzero"] = {
            "shape": list(fshape),
            "stride": list(fstride),
            "tractable": ftract,
            "compact": layout_size_flat(fshape) == layout_cosize_flat(fshape, fstride),
        }
        if ftract and zero_stride_modes > 0:
            out["insights"].append("filtered non-zero-stride projection is tractable; good target for tract ops")

    return out


def _collect_layouts(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []

    for key in ("a_smem_layout_staged", "b_smem_layout_staged"):
        if key in payload and "outer" in payload[key]:
            out.append((f"{key}.outer", payload[key]["outer"]))

    for key in ("sfa_smem_layout_staged", "sfb_smem_layout_staged", "tCtSFA_layout", "tCtSFB_layout"):
        if key in payload:
            out.append((key, payload[key]))

    thr = payload.get("thr_mma_partitions", {})
    for k, v in thr.items():
        if isinstance(v, dict) and "shape" in v and "stride" in v:
            out.append((f"thr_mma_partitions.{k}", v))

    s2t = payload.get("tcgen05_s2t_partitions", {})
    for which, rec in s2t.items():
        if not isinstance(rec, dict):
            continue
        for k, v in rec.items():
            if isinstance(v, dict) and "shape" in v and "stride" in v:
                out.append((f"tcgen05_s2t_partitions.{which}.{k}", v))

    return out


def _parse_case_key(key: str) -> list[int]:
    try:
        value = ast.literal_eval(key)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(value, tuple):
        return []
    return [int(x) for x in value]


def _coalesced_flat(report: dict[str, Any]) -> Flat:
    morph = report.get("morphism")
    if isinstance(morph, dict):
        shape = morph.get("coalesced_layout_shape")
        stride = morph.get("coalesced_layout_stride")
        if isinstance(shape, list) and isinstance(stride, list):
            return Flat(tuple(int(x) for x in shape), tuple(int(x) for x in stride))
    return Flat(
        tuple(int(x) for x in report.get("flat_shape", [])),
        tuple(int(x) for x in report.get("flat_stride", [])),
    )


def _fold_epilogue_t2r(
    *,
    t2r_json: Path,
    tuple_morphism_cls,
    samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not t2r_json.exists():
        return [], None

    raw = json.loads(t2r_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return [], None

    folded_reports: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    lane_extent_default = 32
    warp_extent_default = 4

    for i, (case_key, case_rec) in enumerate(raw.items()):
        if not isinstance(case_rec, dict):
            continue
        shape_mnkl = case_rec.get("shape_mnkl")
        if not isinstance(shape_mnkl, list):
            shape_mnkl = _parse_case_key(case_key)

        acc_layout = case_rec.get("tCtAcc_fake_layout")
        dt_layout = case_rec.get("tDtAcc_layout")
        dg_layout = case_rec.get("tDgC_layout")
        if not all(isinstance(x, dict) and "shape" in x and "stride" in x for x in (acc_layout, dt_layout, dg_layout)):
            continue

        rep_acc = _analyze_layout(
            f"epilogue_t2r[{case_key}].tCtAcc_fake_layout",
            acc_layout,
            tuple_morphism_cls,
            samples=samples,
            seed=seed + i * 3 + 0,
        )
        rep_dt = _analyze_layout(
            f"epilogue_t2r[{case_key}].tDtAcc_layout",
            dt_layout,
            tuple_morphism_cls,
            samples=samples,
            seed=seed + i * 3 + 1,
        )
        rep_dg = _analyze_layout(
            f"epilogue_t2r[{case_key}].tDgC_layout",
            dg_layout,
            tuple_morphism_cls,
            samples=samples,
            seed=seed + i * 3 + 2,
        )
        folded_reports.extend([rep_acc, rep_dt, rep_dg])

        coal_acc = _coalesced_flat(rep_acc)
        coal_dt = _coalesced_flat(rep_dt)
        coal_dg = _coalesced_flat(rep_dg)

        # Expected from analysis: coal(tDtAcc) == (V, Lane) : (1, row_stride)
        vector_extent = int(coal_dg.shape[0]) if len(coal_dg.shape) >= 1 else 0
        lane_extent = int(coal_dt.shape[1]) if len(coal_dt.shape) >= 2 else lane_extent_default
        row_stride = int(coal_dt.stride[1]) if len(coal_dt.stride) >= 2 else (
            int(coal_acc.stride[0]) if len(coal_acc.stride) >= 1 else 0
        )
        warp_extent = warp_extent_default if lane_extent > 0 else 0

        expected_dt = Flat(
            shape=(vector_extent, lane_extent),
            stride=(1, row_stride),
        )
        dt_matches_expected = (
            coal_dt.shape == expected_dt.shape and coal_dt.stride == expected_dt.stride
        )

        composed_tmem_thread_vec = {
            "shape": [[warp_extent, lane_extent], vector_extent],
            "stride": [[lane_extent * row_stride, row_stride], 1],
            "repr": (
                f"(({warp_extent},{lane_extent}),{vector_extent})"
                f":(({lane_extent * row_stride},{row_stride}),1)"
            ),
        }
        row_layout = {
            "shape": [warp_extent, lane_extent],
            "stride": [lane_extent, 1],
            "repr": f"({warp_extent},{lane_extent}):({lane_extent},1)",
        }
        expected_dt_dict = {
            "shape": list(expected_dt.shape),
            "stride": list(expected_dt.stride),
            "repr": f"({expected_dt.shape[0]},{expected_dt.shape[1]}):({expected_dt.stride[0]},{expected_dt.stride[1]})",
        }

        cases.append(
            {
                "case_key": case_key,
                "shape_mnkl": shape_mnkl,
                "coalesced": {
                    "tCtAcc_fake_layout": {
                        "shape": list(coal_acc.shape),
                        "stride": list(coal_acc.stride),
                    },
                    "tDtAcc_layout": {
                        "shape": list(coal_dt.shape),
                        "stride": list(coal_dt.stride),
                    },
                    "tDgC_layout": {
                        "shape": list(coal_dg.shape),
                        "stride": list(coal_dg.stride),
                    },
                },
                "epilogue_graph": {
                    "row_from_thread_layout": row_layout,
                    "tmem_addr_from_thread_vec_layout": composed_tmem_thread_vec,
                    "expected_tDtAcc_coalesced_layout": expected_dt_dict,
                    "observed_tDtAcc_coalesced_layout": {
                        "shape": list(coal_dt.shape),
                        "stride": list(coal_dt.stride),
                    },
                    "checks": {
                        "dtacc_matches_expected_v_lane_form": dt_matches_expected,
                        "dgc_is_contiguous_vector": (
                            len(coal_dg.shape) == 1 and coal_dg.stride == (1,)
                        ),
                        "acc_row_stride_matches_dt_lane_stride": (
                            len(coal_acc.stride) >= 1 and row_stride == int(coal_acc.stride[0])
                        ),
                    },
                    "formulas": {
                        "phi_tmem_addr_from_thread_vec": (
                            "Phi(w,lane,v) = (lane_extent*row_stride)*w + row_stride*lane + v"
                        ),
                        "phi_dtacc_coalesced": "Phi(v,lane) = v + row_stride*lane",
                    },
                },
            }
        )

    if not cases:
        return folded_reports, None
    return folded_reports, {"source_json": str(t2r_json), "cases": cases}


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cats = _load_categories_module(repo_root)

    dump = json.loads(args.dump_json.read_text(encoding="utf-8"))
    payload = dump.get("payload", {})
    layouts = _collect_layouts(payload)
    if not layouts:
        raise SystemExit(f"No analyzable layouts found in {args.dump_json}")

    reports: list[dict[str, Any]] = []
    for i, (name, layout_dict) in enumerate(layouts):
        rep = _analyze_layout(
            name,
            layout_dict,
            cats.Tuple_morphism,
            samples=args.samples,
            seed=args.seed + i,
        )
        reports.append(rep)

    epilogue_reports, epilogue_t2r = _fold_epilogue_t2r(
        t2r_json=args.t2r_json,
        tuple_morphism_cls=cats.Tuple_morphism,
        samples=args.samples,
        seed=args.seed + len(reports),
    )
    reports.extend(epilogue_reports)

    # Human summary
    print("name | tractable | compact | size/cosize | rank | zero_modes | dup(sampled)")
    for rep in reports:
        print(
            f"{rep['name']} | {rep['tractable']} | {rep['compact']} | "
            f"{rep['size_over_cosize']:.6f} | {rep['rank']} | "
            f"{rep['zero_stride_modes']} | {rep['sampled_duplication']:.3f}"
        )

    print("\nTop actionable signals:")
    for rep in reports:
        for insight in rep["insights"][:2]:
            print(f"- {rep['name']}: {insight}")
    if epilogue_t2r is not None:
        print("\nEpilogue fold checks:")
        for case in epilogue_t2r["cases"]:
            checks = case["epilogue_graph"]["checks"]
            print(
                f"- {case['case_key']}: dtacc_v_lane={checks['dtacc_matches_expected_v_lane_form']}, "
                f"dgc_vec={checks['dgc_is_contiguous_vector']}, "
                f"acc_row_stride_match={checks['acc_row_stride_matches_dt_lane_stride']}"
            )

    result = {
        "dump_json": str(args.dump_json),
        "num_layouts": len(reports),
        "reports": reports,
    }
    if epilogue_t2r is not None:
        result["epilogue_t2r"] = epilogue_t2r
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[INFO] Wrote JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
