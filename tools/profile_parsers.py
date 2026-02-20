"""Parsers for NVIDIA profiler tool output (nsys, ncu, compute-sanitizer).

Pure Python, no external dependencies.  Consumed by profile_kernel.py and
usable standalone for re-parsing saved profiler output.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def _safe_int(s: str | None) -> int | None:
    if s is None:
        return None
    try:
        return int(float(s.replace(",", "")))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Nsight Compute (ncu) CSV parsing
# ---------------------------------------------------------------------------

def parse_ncu_csv(raw: str) -> dict[str, Any]:
    """Parse ``ncu --csv`` output into structured metrics grouped by section.

    Returns::

        {
            "kernels": [
                {
                    "name": str,
                    "id": int,
                    "sections": {
                        "<Section Name>": {
                            "<metric.name>": {"value": float|str, "unit": str}
                        }
                    }
                }
            ]
        }
    """
    lines = raw.splitlines()

    # Locate CSV header (skip ==PROF== preamble).
    header_idx = None
    for i, line in enumerate(lines):
        if line.lstrip('"').startswith("ID"):
            header_idx = i
            break
    if header_idx is None:
        return {"kernels": [], "parse_error": "No CSV header found in ncu output"}

    # Keep only lines that look like CSV data (skip ==PROF== noise after header).
    csv_lines = [lines[header_idx]]
    for line in lines[header_idx + 1 :]:
        if line.startswith("==") or not line.strip():
            continue
        csv_lines.append(line)
    csv_text = "\n".join(csv_lines)
    reader = csv.DictReader(io.StringIO(csv_text))

    kernels: dict[tuple[str, str], dict] = {}
    for row in reader:
        kid = row.get("ID", "0")
        kname = row.get("Kernel Name", "unknown")
        # Skip malformed rows from profiler preamble/epilogue.
        if not kid or not kid.strip().isdigit():
            continue
        section = row.get("Section Name", "")
        metric_name = row.get("Metric Name", "")
        metric_unit = row.get("Metric Unit", "")
        metric_value_raw = row.get("Metric Value", "")

        key = (kid, kname)
        if key not in kernels:
            kernels[key] = {"name": kname, "id": _safe_int(kid) or 0, "sections": {}}

        kernel = kernels[key]
        if section not in kernel["sections"]:
            kernel["sections"][section] = {}

        parsed = _safe_float(metric_value_raw)
        kernel["sections"][section][metric_name] = {
            "value": parsed if parsed is not None else metric_value_raw,
            "unit": metric_unit,
        }

    return {"kernels": list(kernels.values())}


def extract_ncu_key_metrics(ncu_data: dict[str, Any]) -> dict[str, Any]:
    """Extract key performance metrics from parsed ncu data.

    Focuses on metrics relevant to Blackwell tcgen05 / TMA kernels.
    Metric lookup uses regex pattern matching against section and metric names
    so it stays resilient across ncu versions.
    """
    if not ncu_data.get("kernels"):
        return {}

    kernel = ncu_data["kernels"][0]
    sections = kernel.get("sections", {})

    def _get(sec_pat: str, met_pat: str) -> float | None:
        for sname, metrics in sections.items():
            if re.search(sec_pat, sname, re.I):
                for mname, mdata in metrics.items():
                    if re.search(met_pat, mname, re.I):
                        v = mdata.get("value")
                        if isinstance(v, (int, float)):
                            return float(v)
        return None

    def _get_str(sec_pat: str, met_pat: str) -> str | None:
        for sname, metrics in sections.items():
            if re.search(sec_pat, sname, re.I):
                for mname, mdata in metrics.items():
                    if re.search(met_pat, mname, re.I):
                        return str(mdata.get("value", ""))
        return None

    r: dict[str, Any] = {}

    # -- Duration --------------------------------------------------------
    r["duration_us"] = _get(r"speed.?of.?light|sol|launch", r"gpu__time_duration\.sum")
    dur_ns = _get(r"speed.?of.?light|sol|launch", r"gpu__time_duration")
    if r["duration_us"] is None and dur_ns is not None:
        r["duration_us"] = dur_ns / 1000.0

    # -- Speed-of-light --------------------------------------------------
    r["sm_throughput_pct"] = _get(r"speed.?of.?light|sol", r"sm__throughput.*pct")
    r["dram_throughput_pct"] = _get(r"speed.?of.?light|sol", r"dram.*throughput.*pct")
    r["mem_throughput_pct"] = _get(r"speed.?of.?light|sol", r"mem.*throughput.*pct")

    # -- Tensor pipe -----------------------------------------------------
    r["tensor_pipe_utilization_pct"] = _get(
        r"compute|instruction|speed|sol",
        r"pipe_tensor.*pct|tensor.*pipe.*pct|hmma.*active.*pct",
    )

    # -- Memory hierarchy ------------------------------------------------
    r["l2_hit_rate_pct"] = _get(r"memory|cache", r"l2.*hit.*rate|lts.*hit.*rate")
    r["dram_bytes_read"] = _get(r"memory", r"dram.*bytes.*read")
    r["dram_bytes_write"] = _get(r"memory", r"dram.*bytes.*write")
    r["smem_bank_conflicts"] = _get(r"memory|shared", r"bank.*conflict|shared.*conflict")

    # -- Occupancy -------------------------------------------------------
    r["achieved_occupancy_pct"] = _get(r"occupancy", r"achieved.*occupancy|warps_active.*pct")
    r["theoretical_occupancy_pct"] = _get(r"occupancy", r"theoretical.*occupancy|max.*warps")
    r["registers_per_thread"] = _get(r"launch|occupancy", r"registers.*thread")
    r["smem_per_block_bytes"] = _get(r"launch", r"shared.*memory|smem.*block")

    # -- Launch config ---------------------------------------------------
    r["grid_size"] = _get_str(r"launch", r"grid.*size")
    r["block_size"] = _get_str(r"launch", r"block.*size|threads.*per.*block")
    r["waves_per_sm"] = _get(r"launch|occupancy", r"waves.*sm")

    # -- Stall breakdown -------------------------------------------------
    stalls: dict[str, float] = {}
    for sname, metrics in sections.items():
        if re.search(r"scheduler|stall|warp.?state", sname, re.I):
            for mname, mdata in metrics.items():
                if "stall" in mname.lower() or "warp" in mname.lower():
                    v = mdata.get("value")
                    if isinstance(v, (int, float)) and float(v) > 0:
                        stalls[mname] = float(v)
    if stalls:
        r["warp_stall_breakdown"] = stalls

    return {k: v for k, v in r.items() if v is not None}


# ---------------------------------------------------------------------------
# Nsight Systems (nsys) CSV parsing
# ---------------------------------------------------------------------------

def parse_nsys_gpu_kern_sum(raw: str) -> list[dict[str, Any]]:
    """Parse ``nsys stats --format csv --report gpukernsum`` output."""
    lines = raw.splitlines()

    # Locate CSV header.
    header_idx = None
    for i, line in enumerate(lines):
        if "Time" in line and ("Name" in line or "Kernel" in line or "Range" in line):
            header_idx = i
            break
    if header_idx is None:
        return []

    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_text))

    results: list[dict[str, Any]] = []
    for row in reader:
        rec: dict[str, Any] = {}
        for key, val in row.items():
            if key is None:
                continue
            ck = key.strip().strip('"')
            cv = (val or "").strip().strip('"')

            if "Time (%)" in ck:
                rec["time_pct"] = _safe_float(cv)
            elif "Total Time" in ck:
                rec["total_time_ns"] = _safe_int(cv)
            elif "Instances" in ck:
                rec["instances"] = _safe_int(cv)
            elif "Avg" in ck and "ns" in ck:
                rec["avg_ns"] = _safe_int(cv)
            elif "Med" in ck:
                rec["med_ns"] = _safe_int(cv)
            elif "Min" in ck:
                rec["min_ns"] = _safe_int(cv)
            elif "Max" in ck:
                rec["max_ns"] = _safe_int(cv)
            elif "StdDev" in ck:
                rec["stddev_ns"] = _safe_int(cv)
            elif "Name" in ck or "Kernel" in ck or "Range" in ck:
                rec["name"] = cv

        if rec.get("name"):
            results.append(rec)
    return results


def extract_nsys_summary(kern_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise nsys kernel timing; highlight dominant kernel."""
    if not kern_records:
        return {}
    dominant = max(kern_records, key=lambda r: r.get("time_pct", 0) or 0)
    return {
        "dominant_kernel": dominant.get("name", "unknown"),
        "avg_us": (dominant.get("avg_ns") or 0) / 1000.0,
        "min_us": (dominant.get("min_ns") or 0) / 1000.0,
        "max_us": (dominant.get("max_ns") or 0) / 1000.0,
        "med_us": (dominant.get("med_ns") or 0) / 1000.0,
        "stddev_us": (dominant.get("stddev_ns") or 0) / 1000.0,
        "instances": dominant.get("instances", 0),
        "time_pct": dominant.get("time_pct", 0),
        "all_kernels": kern_records,
    }


# ---------------------------------------------------------------------------
# Compute Sanitizer parsing
# ---------------------------------------------------------------------------

def parse_sanitizer_output(raw: str) -> dict[str, Any]:
    """Parse ``compute-sanitizer`` text output.

    Returns::

        {
            "tool": str,       # memcheck / racecheck / synccheck / initcheck
            "clean": bool,
            "error_count": int,
            "errors": [str],
            "warnings": [str],
        }
    """
    lines = raw.splitlines()

    # Detect sub-tool.
    tool = "memcheck"
    for line in lines[:20]:
        low = line.lower()
        for t in ("racecheck", "synccheck", "initcheck", "memcheck"):
            if t in low:
                tool = t
                break

    errors: list[str] = []
    warnings: list[str] = []
    error_count = 0

    i = 0
    while i < len(lines):
        stripped = lines[i].replace("=========", "").strip()
        if not stripped:
            i += 1
            continue

        m = re.match(r"ERROR SUMMARY:\s*(\d+)\s*error", stripped)
        if m:
            error_count = int(m.group(1))
            i += 1
            continue

        if re.match(
            r"(Invalid|Uninitialized|Race|Hazard|Error|Access)", stripped, re.I
        ):
            msg = stripped
            i += 1
            while i < len(lines):
                nxt = lines[i].replace("=========", "").strip()
                if not nxt or re.match(
                    r"(Invalid|Uninitialized|Race|Hazard|Error|Access|"
                    r"ERROR|COMPUTE|No errors|Target|Host)",
                    nxt,
                    re.I,
                ):
                    break
                msg += "\n    " + nxt
                i += 1
            errors.append(msg)
            continue

        if "warning" in stripped.lower():
            warnings.append(stripped)

        i += 1

    return {
        "tool": tool,
        "clean": error_count == 0 and len(errors) == 0,
        "error_count": error_count,
        "errors": errors,
        "warnings": warnings,
    }
