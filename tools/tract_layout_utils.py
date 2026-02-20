#!/usr/bin/env python3
"""Shared layout/morphism utilities for tract-based nvfp4 tooling."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACT_SRC = REPO_ROOT / "layout-categories" / "tract" / "src"
if str(TRACT_SRC) not in sys.path:
    sys.path.insert(0, str(TRACT_SRC))

try:
    import cutlass.cute as cute  # type: ignore
except Exception:
    cute = None

TRACT_IMPORT_ERROR: Exception | None = None
try:
    import tract  # type: ignore
except Exception as exc:  # pragma: no cover - dependency depends on local env
    tract = None
    TRACT_IMPORT_ERROR = exc


NESTED = Any


@dataclass(frozen=True)
class FlatLayout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass(frozen=True)
class EquivalenceStats:
    checked: int
    exhaustive: bool


def _is_intlike(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    try:
        int(x)
        return True
    except Exception:
        return False


def _to_int(x: Any) -> int:
    return int(x)


def to_nested_int_tuple(x: Any) -> NESTED:
    """Convert nested tuple/list/int-like structures to nested tuples of ints."""
    if _is_intlike(x):
        return _to_int(x)
    if isinstance(x, (tuple, list)):
        return tuple(to_nested_int_tuple(v) for v in x)
    raise TypeError(f"Expected int-like or nested tuple/list, got {type(x)!r}")


def flatten_tuple(x: Any) -> list[int]:
    """Flatten nested tuple/list of ints into a flat list."""
    if _is_intlike(x):
        return [_to_int(x)]
    if not isinstance(x, (tuple, list)):
        raise TypeError(f"Expected int-like or tuple/list, got {type(x)!r}")
    out: list[int] = []
    for item in x:
        out.extend(flatten_tuple(item))
    return out


def _unflatten_like(template: Any, values_iter: Iterable[int]) -> Any:
    it = iter(values_iter)

    def _build(node: Any) -> Any:
        if _is_intlike(node):
            return next(it)
        return tuple(_build(child) for child in node)

    built = _build(template)
    return built


def tuple_size(shape: Any) -> int:
    """Product of all leaves in nested shape."""
    size = 1
    for dim in flatten_tuple(shape):
        size *= int(dim)
    return size


def colex_rank(shape: Any, coord_nested: Any) -> int:
    """Colexicographic rank of nested coordinate under nested shape."""
    flat_shape = flatten_tuple(shape)
    flat_coord = flatten_tuple(coord_nested)
    if len(flat_shape) != len(flat_coord):
        raise ValueError(
            f"Coord rank mismatch: shape has {len(flat_shape)} leaves, "
            f"coord has {len(flat_coord)} leaves"
        )

    rank = 0
    stride = 1
    for c, s in zip(flat_coord, flat_shape):
        if c < 0 or c >= s:
            raise ValueError(f"Coordinate {c} out of bounds for extent {s}")
        rank += c * stride
        stride *= s
    return rank


def colex_unrank(shape: Any, idx: int) -> Any:
    """Colexicographic inverse rank for nested shape."""
    flat_shape = flatten_tuple(shape)
    total = tuple_size(flat_shape)
    if idx < 0 or idx >= total:
        raise ValueError(f"Index {idx} out of bounds for logical size {total}")

    rem = idx
    flat_coord: list[int] = []
    for dim in flat_shape:
        flat_coord.append(rem % dim)
        rem //= dim

    return _unflatten_like(shape, flat_coord)


def flatten_like(shape: Any, coord_linear: int) -> Any:
    """Alias for colex_unrank, kept for workflow readability."""
    return colex_unrank(shape, coord_linear)


def _extract_shape_stride(layout: Any) -> tuple[NESTED, NESTED]:
    if isinstance(layout, dict) and "shape" in layout and "stride" in layout:
        return to_nested_int_tuple(layout["shape"]), to_nested_int_tuple(layout["stride"])

    if isinstance(layout, tuple) and len(layout) == 2:
        return to_nested_int_tuple(layout[0]), to_nested_int_tuple(layout[1])

    if hasattr(layout, "shape") and hasattr(layout, "stride"):
        return to_nested_int_tuple(layout.shape), to_nested_int_tuple(layout.stride)

    raise TypeError(
        "layout must be a CuTe layout/tensor, a {'shape','stride'} dict, "
        "or (shape, stride) tuple"
    )


def _maybe_to_cute_layout(layout: Any) -> Any:
    if hasattr(layout, "shape") and hasattr(layout, "stride"):
        return layout
    if cute is None:
        return layout
    shape, stride = _extract_shape_stride(layout)
    return cute.make_layout(shape, stride=stride)


def layout_offset(shape: Any, stride: Any, coord_nested: Any) -> int:
    """Compute offset using recursive CuTe dot-product semantics."""
    shape_n = to_nested_int_tuple(shape)
    stride_n = to_nested_int_tuple(stride)
    coord_n = to_nested_int_tuple(coord_nested)

    if _is_intlike(shape_n):
        return int(coord_n) * int(stride_n)

    if not (isinstance(shape_n, tuple) and isinstance(stride_n, tuple) and isinstance(coord_n, tuple)):
        raise TypeError("shape, stride, coord must be congruent nested tuples")

    if not (len(shape_n) == len(stride_n) == len(coord_n)):
        raise ValueError("shape/stride/coord tuple arity mismatch")

    return sum(layout_offset(s, d, c) for s, d, c in zip(shape_n, stride_n, coord_n))


def layout_function(layout: Any, linear_idx: int) -> int:
    """Evaluate Phi_L(p) = offset(unrank(shape(L), p))."""
    shape, stride = _extract_shape_stride(layout)
    coord = colex_unrank(shape, linear_idx)
    return layout_offset(shape, stride, coord)


def as_flat_layout(layout: Any) -> FlatLayout:
    shape, stride = _extract_shape_stride(layout)
    return FlatLayout(tuple(flatten_tuple(shape)), tuple(flatten_tuple(stride)))


def layout_rank_flat(shape_flat: Sequence[int]) -> int:
    return len(shape_flat)


def layout_size_flat(shape_flat: Sequence[int]) -> int:
    size = 1
    for s in shape_flat:
        size *= int(s)
    return size


def layout_cosize_flat(shape_flat: Sequence[int], stride_flat: Sequence[int]) -> int:
    if len(shape_flat) != len(stride_flat):
        raise ValueError("shape/stride rank mismatch")
    return 1 + sum((int(s) - 1) * int(d) for s, d in zip(shape_flat, stride_flat))


def layout_mode_pairs_flat(
    shape_flat: Sequence[int], stride_flat: Sequence[int]
) -> list[tuple[int, int]]:
    if len(shape_flat) != len(stride_flat):
        raise ValueError("shape/stride rank mismatch")
    return [(int(s), int(d)) for s, d in zip(shape_flat, stride_flat)]


def _require_tract() -> Any:
    if tract is None:
        hint = ""
        if TRACT_IMPORT_ERROR is not None:
            hint = f" Original import error: {TRACT_IMPORT_ERROR}"
        raise RuntimeError(
            "tract is unavailable. Ensure CUTLASS Python DSL and tract dependencies are installed."
            + hint
        )
    return tract


def is_tractable(layout: Any) -> bool:
    t = _require_tract()
    return bool(t.is_tractable(_maybe_to_cute_layout(layout)))


def to_morphism(layout: Any) -> Any:
    t = _require_tract()
    return t.compute_morphism(_maybe_to_cute_layout(layout))


def to_layout(morphism: Any) -> Any:
    t = _require_tract()
    return t.compute_layout(morphism)


def make_morphism(domain: Any, codomain: Any, map_: Any) -> Any:
    t = _require_tract()
    return t.make_morphism(domain=domain, codomain=codomain, map_=map_)


def compose(f: Any, g: Any) -> Any:
    t = _require_tract()
    return t.compose(f, g)


def coalesce_morphism(f: Any) -> Any:
    t = _require_tract()
    return t.coalesce(f)


def logical_divide(f: Any, g: Any) -> Any:
    t = _require_tract()
    return t.logical_divide(f, g)


def logical_product(f: Any, g: Any) -> Any:
    t = _require_tract()
    return t.logical_product(f, g)


def complement(f: Any) -> Any:
    t = _require_tract()
    return t.complement(f)


def _layout_string(shape: Any, stride: Any) -> str:
    return f"{shape}:{stride}"


def canonicalize_layout(
    layout: Any,
    *,
    profile: Any | None = None,
    coalesce_mode: str = "tract",
    readable_coalesce: bool = True,
    max_fixed_point_iters: int = 8,
) -> Any:
    """Canonicalize a layout with optional relative-coalesce and tract coalesce."""
    lay = _maybe_to_cute_layout(layout)

    if profile is not None and cute is not None and hasattr(lay, "shape"):
        lay = cute.coalesce(lay, target_profile=profile)

    if coalesce_mode not in {"tract", "none"}:
        raise ValueError("coalesce_mode must be one of {'tract', 'none'}")

    if coalesce_mode == "tract":
        try:
            if is_tractable(lay):
                f = to_morphism(lay)
                prev = repr(f)
                for _ in range(max_fixed_point_iters):
                    f2 = coalesce_morphism(f)
                    now = repr(f2)
                    f = f2
                    if now == prev:
                        break
                    prev = now
                lay = to_layout(f)
        except RuntimeError:
            # Dependency not available; leave layout unchanged.
            pass

    if readable_coalesce and cute is not None and hasattr(lay, "shape"):
        try:
            lay = cute.coalesce(lay)
        except Exception:
            pass

    return lay


def canonicalize_layout_report(
    layout: Any,
    *,
    profile: Any | None = None,
    coalesce_mode: str = "tract",
) -> dict[str, Any]:
    shape, stride = _extract_shape_stride(layout)
    flat = as_flat_layout(layout)

    canonical = canonicalize_layout(layout, profile=profile, coalesce_mode=coalesce_mode)
    can_shape, can_stride = _extract_shape_stride(canonical)
    can_flat = as_flat_layout(canonical)

    return {
        "original": {
            "shape": shape,
            "stride": stride,
            "layout": _layout_string(shape, stride),
            "rank": layout_rank_flat(flat.shape),
            "size": layout_size_flat(flat.shape),
            "cosize": layout_cosize_flat(flat.shape, flat.stride),
        },
        "canonical": {
            "shape": can_shape,
            "stride": can_stride,
            "layout": _layout_string(can_shape, can_stride),
            "rank": layout_rank_flat(can_flat.shape),
            "size": layout_size_flat(can_flat.shape),
            "cosize": layout_cosize_flat(can_flat.shape, can_flat.stride),
        },
    }


def _structured_points(size: int) -> set[int]:
    if size <= 0:
        return set()
    points = {0, size - 1, size // 2}
    quarter = size // 4
    points.add(quarter)
    points.add(min(size - 1, quarter * 3))
    return {p for p in points if 0 <= p < size}


def _sample_points(size: int, samples: int, seed: int) -> set[int]:
    points = _structured_points(size)
    if size <= 0:
        return points
    rng = random.Random(seed)
    draws = min(size, max(0, samples))
    while len(points) < min(size, draws + len(points)):
        points.add(rng.randrange(size))
        if len(points) >= size:
            break
    return points


def assert_layout_equivalent(
    layout_a: Any,
    layout_b: Any,
    *,
    samples: int = 10_000,
    exhaustive_if_small: bool = True,
    exhaustive_threshold: int = 65_536,
    seed: int = 0,
) -> EquivalenceStats:
    """Assert two layouts have identical layout function over logical domain."""
    flat_a = as_flat_layout(layout_a)
    flat_b = as_flat_layout(layout_b)
    size_a = layout_size_flat(flat_a.shape)
    size_b = layout_size_flat(flat_b.shape)
    if size_a != size_b:
        raise AssertionError(f"Logical-size mismatch: {size_a} != {size_b}")

    exhaustive = exhaustive_if_small and size_a <= exhaustive_threshold
    if exhaustive:
        points = range(size_a)
        checked = size_a
    else:
        sampled = _sample_points(size_a, samples, seed)
        points = sampled
        checked = len(sampled)

    for p in points:
        off_a = layout_function(layout_a, int(p))
        off_b = layout_function(layout_b, int(p))
        if off_a != off_b:
            raise AssertionError(
                "Layout mismatch at linear index "
                f"p={p}: Phi_A={off_a}, Phi_B={off_b}"
            )

    return EquivalenceStats(checked=checked, exhaustive=exhaustive)


def assert_morphism_equivalent(
    f: Any,
    g: Any,
    *,
    samples: int = 10_000,
    exhaustive_if_small: bool = True,
    exhaustive_threshold: int = 65_536,
    seed: int = 0,
) -> EquivalenceStats:
    layout_f = to_layout(f)
    layout_g = to_layout(g)
    return assert_layout_equivalent(
        layout_f,
        layout_g,
        samples=samples,
        exhaustive_if_small=exhaustive_if_small,
        exhaustive_threshold=exhaustive_threshold,
        seed=seed,
    )


def collect_zero_stride_paths(stride_nested: Any) -> list[tuple[int, ...]]:
    stride = to_nested_int_tuple(stride_nested)
    out: list[tuple[int, ...]] = []

    def _walk(node: Any, path: tuple[int, ...]) -> None:
        if _is_intlike(node):
            if int(node) == 0:
                out.append(path)
            return
        for idx, child in enumerate(node):
            _walk(child, path + (idx,))

    _walk(stride, ())
    return out


def unique_offsets_count(
    layout: Any,
    *,
    samples: int = 10_000,
    exact_if_small: bool = True,
    exact_threshold: int = 65_536,
    seed: int = 0,
) -> tuple[int, bool, int]:
    """Return (unique_offsets, exact, logical_size)."""
    flat = as_flat_layout(layout)
    size = layout_size_flat(flat.shape)

    if exact_if_small and size <= exact_threshold:
        offsets = {layout_function(layout, p) for p in range(size)}
        return len(offsets), True, size

    points = sorted(_sample_points(size, samples, seed))
    offsets = {layout_function(layout, p) for p in points}
    return len(offsets), False, size


def layout_digest(layout: Any, *, profile: Any | None = None) -> dict[str, Any]:
    """Stable digest record for snapshot/diff workflows."""
    shape, stride = _extract_shape_stride(layout)
    flat = as_flat_layout(layout)

    tractable: bool | None
    try:
        tractable = is_tractable(layout)
    except Exception:
        tractable = None

    canonical = canonicalize_layout(layout, profile=profile, coalesce_mode="tract")
    can_shape, can_stride = _extract_shape_stride(canonical)
    canonical_str = _layout_string(can_shape, can_stride)

    mode_pairs = layout_mode_pairs_flat(flat.shape, flat.stride)
    digest_src = {
        "canonical": canonical_str,
        "modes": mode_pairs,
    }
    stable_hash = hashlib.sha1(
        json.dumps(digest_src, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "shape": shape,
        "stride": stride,
        "flat_shape": list(flat.shape),
        "flat_stride": list(flat.stride),
        "flat_rank": layout_rank_flat(flat.shape),
        "flat_size": layout_size_flat(flat.shape),
        "flat_cosize": layout_cosize_flat(flat.shape, flat.stride),
        "mode_pairs": mode_pairs,
        "tractable": tractable,
        "canonical": canonical_str,
        "hash": stable_hash,
    }
