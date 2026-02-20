"""Bottleneck classification and optimization hints from profiler metrics.

Thresholds are tuned for Blackwell B200 tcgen05/TMA-heavy grouped GEMM kernels.
Pure Python, no external dependencies.
"""

from __future__ import annotations

from typing import Any

# Threshold knobs -- adjust based on experience.
_T = {
    "memory_bound_dram_pct": 60.0,
    "compute_bound_sm_pct": 60.0,
    "latency_bound_ceiling_pct": 40.0,
    "occupancy_limited_pct": 50.0,
    "l2_hit_rate_low_pct": 50.0,
    "tensor_pipe_underused_pct": 50.0,
    "smem_conflicts_threshold": 1000,
    "stall_significant_pct": 10.0,
}


def classify_bottlenecks(metrics: dict[str, Any]) -> dict[str, bool]:
    """Return boolean classification flags from extracted ncu key metrics."""
    sm = metrics.get("sm_throughput_pct", 0) or 0
    dram = metrics.get("dram_throughput_pct", 0) or 0
    mem = metrics.get("mem_throughput_pct", dram) or 0
    occ = metrics.get("achieved_occupancy_pct", 100) or 100

    return {
        "memory_bound": mem > _T["memory_bound_dram_pct"] and sm < _T["compute_bound_sm_pct"],
        "compute_bound": sm > _T["compute_bound_sm_pct"] and mem < _T["memory_bound_dram_pct"],
        "latency_bound": (
            sm < _T["latency_bound_ceiling_pct"]
            and mem < _T["latency_bound_ceiling_pct"]
        ),
        "occupancy_limited": occ < _T["occupancy_limited_pct"],
    }


def generate_hints(
    metrics: dict[str, Any],
    classification: dict[str, bool],
) -> list[dict[str, str]]:
    """Produce prioritised, actionable optimisation hints.

    Each hint carries *priority* (high / medium / low), *category*,
    and a human-readable *hint* string.
    """
    hints: list[dict[str, str]] = []

    sm = metrics.get("sm_throughput_pct", 0) or 0
    dram = metrics.get("dram_throughput_pct", 0) or 0
    tensor = metrics.get("tensor_pipe_utilization_pct", 0) or 0
    occ = metrics.get("achieved_occupancy_pct", 0) or 0
    l2_hit = metrics.get("l2_hit_rate_pct")
    smem_conf = metrics.get("smem_bank_conflicts")
    stalls = metrics.get("warp_stall_breakdown") or {}

    # -- High-priority bottleneck hints ----------------------------------

    if classification.get("memory_bound"):
        hints.append({
            "priority": "high",
            "category": "memory",
            "hint": (
                f"Memory-bound (DRAM {dram:.1f}% vs SM {sm:.1f}%). "
                "Consider: increase pipeline depth (more AB stages), "
                "TMA multicast/clustering to reduce redundant DRAM traffic, "
                "improve data reuse in shared memory, TMA store for epilogue."
            ),
        })

    if classification.get("compute_bound"):
        hints.append({
            "priority": "high",
            "category": "compute",
            "hint": (
                f"Compute-bound (SM {sm:.1f}% vs DRAM {dram:.1f}%). "
                "Consider: warp specialisation (TMA warp + MMA warp) to hide "
                "load latency behind MMA, reduce control flow in hot loop, "
                "reduce epilogue instruction count."
            ),
        })

    if classification.get("latency_bound"):
        hints.append({
            "priority": "high",
            "category": "latency",
            "hint": (
                f"Latency/stall-bound (SM {sm:.1f}%, DRAM {dram:.1f}% -- both low). "
                "Investigate warp stall breakdown.  Common fixes: warp specialisation, "
                "shorten dependency chains, increase ILP, reduce barrier overhead."
            ),
        })

    # -- Medium-priority hints -------------------------------------------

    if classification.get("occupancy_limited"):
        regs = metrics.get("registers_per_thread")
        smem = metrics.get("smem_per_block_bytes")
        detail = ""
        if regs:
            detail += f"Regs/thread: {int(regs)}.  "
        if smem:
            detail += f"SMEM/block: {int(smem)}B.  "
        hints.append({
            "priority": "medium",
            "category": "occupancy",
            "hint": (
                f"Low occupancy ({occ:.1f}%). {detail}"
                "Consider: reduce register pressure, optimise SMEM usage, "
                "or verify low occupancy is acceptable for this pipeline design."
            ),
        })

    if 0 < tensor < _T["tensor_pipe_underused_pct"]:
        hints.append({
            "priority": "medium",
            "category": "tensor_core",
            "hint": (
                f"Tensor pipe utilisation only {tensor:.1f}%. "
                "MMA may be stalling on operand availability.  "
                "Increase pipeline depth, prefetch next tile during current MMA, "
                "or adopt warp specialisation for TMA/MMA overlap."
            ),
        })

    if l2_hit is not None and l2_hit < _T["l2_hit_rate_low_pct"]:
        hints.append({
            "priority": "medium",
            "category": "cache",
            "hint": (
                f"L2 hit rate low ({l2_hit:.1f}%). "
                "TMA multicast, CTA clustering for L2 locality, "
                "or tile-size tuning may help."
            ),
        })

    if smem_conf is not None and smem_conf > _T["smem_conflicts_threshold"]:
        hints.append({
            "priority": "medium",
            "category": "shared_memory",
            "hint": (
                f"Shared memory bank conflicts: {int(smem_conf)}.  "
                "Verify swizzle patterns for A/B SMEM layouts."
            ),
        })

    if stalls:
        top = sorted(stalls.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if top and top[0][1] > _T["stall_significant_pct"]:
            summary = ", ".join(
                f"{k.rsplit('.', 1)[-1]}={v:.1f}%" for k, v in top
            )
            hints.append({
                "priority": "medium",
                "category": "stalls",
                "hint": f"Top warp stall reasons: {summary}.",
            })

    # -- Fallback --------------------------------------------------------
    if not hints:
        hints.append({
            "priority": "low",
            "category": "general",
            "hint": (
                "No clear single bottleneck.  Kernel may be well-balanced.  "
                "Focus on micro-optimisations: epilogue store coalescing, "
                "boundary predicate reduction, synchronisation minimisation."
            ),
        })

    return hints
