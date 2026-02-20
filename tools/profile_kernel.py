#!/usr/bin/env python3
"""Profile a CUDA kernel with Nsight Systems, Nsight Compute, or Compute Sanitizer.

Generates a self-contained profiling harness, executes the requested profiler
tool, parses the output into structured JSON suitable for AI-agent consumption,
and prints a human-readable summary.

Local usage (B200 present):

    uv run python tools/profile_kernel.py \\
        --kernel kernels/nvfp4_group_gemm/wagmiv67.py --mode ncu

Remote usage (Modal B200):

    set -a; source ../kernel_rl/.env; set +a
    uv run --with modal python tools/profile_kernel.py \\
        --kernel kernels/nvfp4_group_gemm/wagmiv67.py --mode ncu --remote

Probe tool availability on Modal B200:

    uv run --with modal python tools/profile_kernel.py \\
        --kernel kernels/nvfp4_group_gemm/wagmiv67.py --mode ncu --remote --probe
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Sibling modules -----------------------------------------------------------
_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR.parent))
from tools.profile_parsers import (
    extract_ncu_key_metrics,
    extract_nsys_summary,
    parse_ncu_csv,
    parse_nsys_gpu_kern_sum,
    parse_sanitizer_output,
)
from tools.profile_analysis import classify_bottlenecks, generate_hints

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2

# Leaderboard-representative problem sizes (2-group case).
DEFAULT_PROBLEM_SIZES = [(80, 4096, 7168, 1), (40, 7168, 2048, 1)]

# ncu section presets keyed by detail level.
_NCU_SECTION_SETS: dict[str, str | list[str]] = {
    "basic": "basic",
    "detailed": "detailed",
    "full": "full",
    # Curated set: fast (few replay passes) but covers SOL + occupancy.
    "curated": [
        "SpeedOfLight",
        "SpeedOfLightThroughput",
        "LaunchStatistics",
        "Occupancy",
    ],
    # Extended set: adds memory + scheduler + instruction stats.
    "extended": [
        "SpeedOfLight",
        "SpeedOfLightThroughput",
        "LaunchStatistics",
        "Occupancy",
        "MemoryWorkloadAnalysis",
        "MemoryWorkloadAnalysis_Chart",
        "Scheduler",
        "SchedulerStatistics",
        "WarpStateStatistics",
        "InstructionStats",
        "InstructionStatistics",
    ],
}


def _resolve_ncu_section_args(ncu_set: str) -> list[str]:
    """Convert an ncu preset name into CLI args (e.g. ['--set', 'basic']).

    Resolved locally so the remote function never needs _NCU_SECTION_SETS.
    """
    preset = _NCU_SECTION_SETS.get(ncu_set)
    if isinstance(preset, str):
        return ["--set", preset]
    if isinstance(preset, list):
        args: list[str] = []
        for sec in preset:
            args += ["--section", sec]
        return args
    return ["--set", ncu_set]


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def _find_tool(name: str) -> str | None:
    """Locate an NVIDIA profiler executable on PATH or common install dirs."""
    path = shutil.which(name)
    if path:
        return path
    import glob as _glob
    search_patterns = [
        f"/usr/local/cuda/bin/{name}",
        f"/usr/local/cuda/extras/compute-sanitizer/{name}",
        f"/opt/nvidia/nsight-systems/*/bin/{name}",
        f"/opt/nvidia/nsight-compute/*/bin/{name}",
        f"/usr/local/cuda-*/bin/{name}",
        f"/usr/local/cuda/nsight-compute/ncu",
        f"/usr/local/cuda/nsight-systems/bin/nsys",
    ]
    for pat in search_patterns:
        for candidate in sorted(_glob.glob(pat), reverse=True):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def _require_tool(name: str) -> str:
    path = _find_tool(name)
    if path is None:
        raise SystemExit(
            f"[ERROR] '{name}' not found on PATH or common install locations.\n"
            f"Install the NVIDIA profiler tools or add them to PATH."
        )
    return path


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

def _collect_gpu_info() -> dict[str, Any]:
    """Collect GPU metadata via nvidia-smi."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total,clocks.sm,clocks.mem,"
                "driver_version,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return {"error": "nvidia-smi not available"}
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[:500]}
    parts = [p.strip() for p in proc.stdout.strip().split(",")]
    return {
        "name": parts[0] if len(parts) > 0 else "unknown",
        "compute_cap": parts[1] if len(parts) > 1 else "unknown",
        "memory_total_mb": _try_int(parts[2]) if len(parts) > 2 else None,
        "sm_clock_mhz": _try_int(parts[3]) if len(parts) > 3 else None,
        "mem_clock_mhz": _try_int(parts[4]) if len(parts) > 4 else None,
        "driver_version": parts[5] if len(parts) > 5 else "unknown",
        "pci_bus_id": parts[6] if len(parts) > 6 else "unknown",
    }


def _try_int(s: str) -> int | None:
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Harness generation
# ---------------------------------------------------------------------------

_HARNESS_TEMPLATE = r'''#!/usr/bin/env python3
"""Auto-generated profiling harness.  Do not edit."""
import sys, types, os, importlib.util, time

print("[HARNESS] Starting...", flush=True)

# Stub the Popcorn 'task' module so the kernel can import input_t / output_t.
_task = types.ModuleType("task")
_task.input_t = tuple
_task.output_t = list
sys.modules["task"] = _task

# Ensure kernel directory is importable (for any sibling modules).
_kernel_path = {kernel_path!r}
sys.path.insert(0, os.path.dirname(os.path.abspath(_kernel_path)))

print(f"[HARNESS] Importing kernel from {{_kernel_path}}...", flush=True)
_t0 = time.time()

# Import the kernel module.
_spec = importlib.util.spec_from_file_location("_profiled_kernel", _kernel_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

print(f"[HARNESS] Kernel imported in {{time.time() - _t0:.1f}}s", flush=True)

import torch
print(f"[HARNESS] CUDA device: {{torch.cuda.get_device_name(0)}}", flush=True)

# ---------- Test data ----------
_problem_sizes = {problem_sizes!r}

def _make_test_data():
    abc = []
    sfasfb = []
    for m, n, k, l in _problem_sizes:
        a = torch.randint(0, 256, (m, k // 2, l), dtype=torch.uint8, device="cuda")
        b = torch.randint(0, 256, (n, k // 2, l), dtype=torch.uint8, device="cuda")
        c = torch.zeros(m, n, l, dtype=torch.float16, device="cuda")
        abc.append((a, b, c))
        sfa = torch.randint(0, 256, (m, k // 16, l), dtype=torch.uint8, device="cuda")
        sfb = torch.randint(0, 256, (n, k // 16, l), dtype=torch.uint8, device="cuda")
        sfasfb.append((sfa, sfb))
    return (abc, None, sfasfb, _problem_sizes)

print("[HARNESS] Allocating test tensors...", flush=True)
_data = _make_test_data()

# ---------- Warmup (includes JIT compile) ----------
print("[HARNESS] Warmup ({warmup_iters} iters, includes JIT)...", flush=True)
{nvtx_push_warmup}
_t_warmup = time.time()
for _i in range({warmup_iters}):
    try:
        _mod.custom_kernel(_data)
        torch.cuda.synchronize()
        print(f"[HARNESS]   warmup iter {{_i}} done ({{time.time() - _t_warmup:.1f}}s elapsed)", flush=True)
    except Exception as _e:
        print(f"HARNESS_ERROR(warmup iter {{_i}}): {{type(_e).__name__}}: {{_e}}", file=sys.stderr, flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)
{nvtx_pop_warmup}
print(f"[HARNESS] Warmup done in {{time.time() - _t_warmup:.1f}}s", flush=True)

# ---------- Profiled region ----------
print("[HARNESS] Profiled region ({profile_iters} iters)...", flush=True)
{nvtx_push_profile}
_t_profile = time.time()
for _i in range({profile_iters}):
    try:
        _mod.custom_kernel(_data)
    except Exception as _e:
        print(f"HARNESS_ERROR(profile iter {{_i}}): {{type(_e).__name__}}: {{_e}}", file=sys.stderr, flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)
torch.cuda.synchronize()
{nvtx_pop_profile}
print(f"[HARNESS] Profile region done in {{time.time() - _t_profile:.1f}}s", flush=True)

print("HARNESS_COMPLETE", flush=True)
'''


def _generate_harness(
    kernel_path: str,
    problem_sizes: list[tuple[int, ...]],
    warmup_iters: int,
    profile_iters: int,
    use_nvtx: bool,
) -> str:
    """Return the harness source code as a string."""
    if use_nvtx:
        push_w = 'torch.cuda.nvtx.range_push("warmup")'
        pop_w = 'torch.cuda.nvtx.range_pop()'
        push_p = 'torch.cuda.nvtx.range_push("profile_region")'
        pop_p = 'torch.cuda.nvtx.range_pop()'
    else:
        push_w = pop_w = push_p = pop_p = "pass"

    # The template has {warmup_iters} and {profile_iters} as format placeholders,
    # but also uses them inside nested f-strings with {{...}} escaping.
    return _HARNESS_TEMPLATE.format(
        kernel_path=kernel_path,
        problem_sizes=problem_sizes,
        warmup_iters=warmup_iters,
        profile_iters=profile_iters,
        nvtx_push_warmup=push_w,
        nvtx_pop_warmup=pop_w,
        nvtx_push_profile=push_p,
        nvtx_pop_profile=pop_p,
    )


# ---------------------------------------------------------------------------
# Profiler execution (local)
# ---------------------------------------------------------------------------

def _run_subprocess(
    cmd: list[str],
    timeout: int,
    label: str,
) -> dict[str, Any]:
    """Run a command and return structured result."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"{label} timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"{label} failed: {type(e).__name__}: {e}",
        }


def _run_nsys(
    harness_path: str,
    output_dir: str,
    python_exe: str,
    warmup_iters: int,
    profile_iters: int,
    use_nvtx: bool,
    timeout: int,
) -> dict[str, Any]:
    nsys = _require_tool("nsys")
    report_base = os.path.join(output_dir, "profile_output")
    nsys_cfg_info: dict[str, Any] = {}

    # Work around known QuadD time conversion failures seen on some setups by
    # disabling raw GPU timestamps in the Nsight Systems user config.
    z_res = _run_subprocess([nsys, "-z"], 30, "nsys -z")
    cfg_path = ""
    if z_res.get("returncode") == 0:
        lines = [ln.strip() for ln in (z_res.get("stdout", "") or "").splitlines() if ln.strip()]
        if lines:
            cfg_path = lines[-1]
    nsys_cfg_info["cfg_path"] = cfg_path
    nsys_cfg_info["z_returncode"] = z_res.get("returncode")
    nsys_cfg_info["z_stderr_tail"] = (z_res.get("stderr", "") or "")[-1000:]
    if cfg_path:
        try:
            cfg_parent = Path(cfg_path).parent
            cfg_parent.mkdir(parents=True, exist_ok=True)
            existing = ""
            if os.path.exists(cfg_path):
                existing = Path(cfg_path).read_text(encoding="utf-8", errors="ignore")
            if "CuptiUseRawGpuTimestamps=false" not in existing:
                with open(cfg_path, "a", encoding="utf-8") as cfgf:
                    if existing and not existing.endswith("\n"):
                        cfgf.write("\n")
                    cfgf.write("CuptiUseRawGpuTimestamps=false\n")
            nsys_cfg_info["timestamp_workaround"] = "applied"
        except Exception as e:
            nsys_cfg_info["timestamp_workaround"] = f"failed: {type(e).__name__}: {e}"
    else:
        nsys_cfg_info["timestamp_workaround"] = "skipped: nsys -z path unavailable"

    cmd = [
        nsys, "profile",
        "-o", report_base,
        "--force-overwrite", "true",
        "--trace", "cuda,nvtx",
        "--sample", "none",
        "--cpuctxsw", "none",
    ]
    if use_nvtx:
        cmd += ["--nvtx-capture", "range@profile_region"]
    cmd += [python_exe, harness_path]

    capture = _run_subprocess(cmd, timeout, "nsys profile")
    used_nvtx_only_fallback = False
    if capture.get("returncode") == 0 and "Unrecognized GPU UUID" in (capture.get("stderr", "") or ""):
        # Modal B200 can hit Nsight parser failures for CUDA events. Retry with
        # NVTX-only tracing to still get region-level timing data.
        report_base = os.path.join(output_dir, "profile_output_nvtx")
        nvtx_cmd = [
            nsys, "profile",
            "-o", report_base,
            "--force-overwrite", "true",
            "--trace", "nvtx",
            "--sample", "none",
            "--cpuctxsw", "none",
        ]
        if use_nvtx:
            nvtx_cmd += ["--nvtx-capture", "range@profile_region"]
        nvtx_cmd += [python_exe, harness_path]
        cap2 = _run_subprocess(nvtx_cmd, timeout, "nsys profile (nvtx fallback)")
        if cap2.get("returncode") == 0:
            capture = cap2
            used_nvtx_only_fallback = True
    if capture["returncode"] != 0:
        return {"error": "nsys profile failed", "detail": capture}

    # Nsight Systems output naming differs by version/runtime. Prefer an actual
    # generated file over the historical ".nsys-rep" default.
    rep_file = report_base + ".nsys-rep"
    if not os.path.exists(rep_file):
        candidates = sorted(Path(output_dir).glob("profile_output*"))
        preferred = [
            c for c in candidates if c.name.endswith(".nsys-rep") or c.name.endswith(".qdrep")
        ]
        if preferred:
            rep_file = str(preferred[0])
        elif candidates:
            rep_file = str(candidates[0])

    report_candidates = (
        ["nvtx_sum"]
        if used_nvtx_only_fallback
        else ["cuda_gpu_kern_sum", "cuda_api_sum", "nvtx_sum", "cuda_gpu_kernel_sum", "gpukernsum"]
    )
    stats_attempts: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "returncode": -1,
        "stdout": "",
        "stderr": "nsys stats not attempted",
    }
    selected_report: str | None = None

    for report_name in report_candidates:
        stats_cmd = [
            nsys, "stats",
            "--format", "csv",
            "--output", "-",
            "--force-export", "true",
            "--report", report_name,
            rep_file,
        ]
        attempt = _run_subprocess(stats_cmd, 60, f"nsys stats ({report_name})")
        stats_attempts.append(
            {
                "report": report_name,
                "cmd": " ".join(stats_cmd),
                "returncode": attempt.get("returncode"),
                "stdout_tail": (attempt.get("stdout", "") or "")[-1000:],
                "stderr_tail": (attempt.get("stderr", "") or "")[-1000:],
            }
        )
        stderr_txt = (attempt.get("stderr", "") or "")
        stdout_txt = (attempt.get("stdout", "") or "")
        parseable = bool(parse_nsys_gpu_kern_sum(stdout_txt))
        has_report_error = "error: report" in stderr_txt.lower() or "could not be found" in stderr_txt.lower()
        if attempt.get("returncode") == 0 and parseable and not has_report_error:
            stats = attempt
            selected_report = report_name
            break
        stats = attempt

    help_reports = None
    if selected_report is None:
        help_cmd = [nsys, "stats", "--help-reports"]
        help_reports = _run_subprocess(help_cmd, 60, "nsys stats --help-reports")

    return {
        "capture": capture,
        "stats": stats,
        "report_file": rep_file,
        "report_candidates": [str(p) for p in sorted(Path(output_dir).glob("profile_output*"))],
        "nsys_config": nsys_cfg_info,
        "used_nvtx_only_fallback": used_nvtx_only_fallback,
        "stats_selected_report": selected_report,
        "stats_attempts": stats_attempts,
        "help_reports": help_reports,
    }


def _run_ncu(
    harness_path: str,
    output_dir: str,
    python_exe: str,
    ncu_section_args: list[str],
    launch_skip: int | None,
    launch_count: int,
    use_nvtx: bool,
    timeout: int,
    extra_args: list[str] | None,
) -> dict[str, Any]:
    ncu = _require_tool("ncu")
    cmd = [ncu, "--csv"] + ncu_section_args

    if use_nvtx:
        cmd += ["--nvtx", "--nvtx-include", "profile_region/"]
    if launch_skip is not None:
        cmd += ["--launch-skip", str(launch_skip)]
    cmd += ["--launch-count", str(launch_count)]
    cmd += ["--target-processes", "all"]

    if extra_args:
        cmd.extend(extra_args)

    cmd += [python_exe, harness_path]
    result = _run_subprocess(cmd, timeout, "ncu")
    return {"ncu_result": result, "cmd": " ".join(cmd)}


def _run_sanitizer(
    harness_path: str,
    output_dir: str,
    python_exe: str,
    sanitizer_tool: str,
    timeout: int,
) -> dict[str, Any]:
    san = _require_tool("compute-sanitizer")
    cmd = [san, "--tool", sanitizer_tool, python_exe, harness_path]
    result = _run_subprocess(cmd, timeout, "compute-sanitizer")
    return {"sanitizer_result": result, "cmd": " ".join(cmd)}


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def _build_output(
    mode: str,
    raw_result: dict[str, Any],
    config: dict[str, Any],
    kernel_path: str,
    gpu_info: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the final JSON output document."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    kernel_abs = os.path.abspath(kernel_path)

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": now,
        "mode": mode,
        "kernel": {
            "name": Path(kernel_path).stem,
            "path": kernel_path,
            "file_hash": _file_hash(kernel_abs) if os.path.isfile(kernel_abs) else None,
        },
        "gpu": gpu_info,
        "profile_config": config,
    }

    if mode == "ncu":
        ncu_res = raw_result.get("ncu_result", {})
        stdout = ncu_res.get("stdout", "")
        stderr = ncu_res.get("stderr", "")

        ncu_data = parse_ncu_csv(stdout)
        key_metrics = extract_ncu_key_metrics(ncu_data)
        classification = classify_bottlenecks(key_metrics) if key_metrics else {}
        hints = generate_hints(key_metrics, classification) if key_metrics else []

        output["metrics"] = key_metrics
        output["classification"] = classification
        output["hints"] = hints

        if ncu_data.get("kernels"):
            output["raw_sections"] = ncu_data["kernels"][0].get("sections", {})

        output["profiler_output"] = {
            "returncode": ncu_res.get("returncode"),
            "stderr_tail": stderr[-3000:] if stderr else "",
            "cmd": raw_result.get("cmd", ""),
        }
        if ncu_data.get("parse_error"):
            output["profiler_output"]["parse_error"] = ncu_data["parse_error"]

    elif mode == "nsys":
        stats_res = raw_result.get("stats", {})
        stats_stdout = stats_res.get("stdout", "")
        kern_records = parse_nsys_gpu_kern_sum(stats_stdout)
        summary = extract_nsys_summary(kern_records)
        selected_report = raw_result.get("stats_selected_report")
        if selected_report and summary:
            summary["source_report"] = selected_report
            if selected_report == "cuda_api_sum":
                summary["dominant_api_call"] = summary.get("dominant_kernel")
            if selected_report == "nvtx_sum":
                summary["dominant_nvtx_range"] = summary.get("dominant_kernel")
                profile_rec = None
                for rec in kern_records:
                    name = str(rec.get("name", ""))
                    if "profile_region" in name:
                        profile_rec = rec
                        break
                if profile_rec is not None:
                    summary["dominant_kernel"] = profile_rec.get("name", summary.get("dominant_kernel"))
                    summary["avg_us"] = (profile_rec.get("avg_ns") or 0) / 1000.0
                    summary["min_us"] = (profile_rec.get("min_ns") or 0) / 1000.0
                    summary["max_us"] = (profile_rec.get("max_ns") or 0) / 1000.0
                    summary["med_us"] = (profile_rec.get("med_ns") or 0) / 1000.0
                    summary["stddev_us"] = (profile_rec.get("stddev_ns") or 0) / 1000.0
                    summary["instances"] = profile_rec.get("instances", 0)
                    summary["time_pct"] = profile_rec.get("time_pct", 0)
                    summary["profile_region_us"] = (profile_rec.get("total_time_ns") or 0) / 1000.0

        output["metrics"] = summary
        output["profiler_output"] = {
            "capture_returncode": raw_result.get("capture", {}).get("returncode"),
            "capture_stdout_tail": (raw_result.get("capture", {}).get("stdout", "") or "")[-2000:],
            "capture_stderr_tail": (raw_result.get("capture", {}).get("stderr", "") or "")[-2000:],
            "stats_returncode": stats_res.get("returncode"),
            "report_file": raw_result.get("report_file", ""),
            "report_candidates": raw_result.get("report_candidates", []),
            "nsys_config": raw_result.get("nsys_config", {}),
            "used_nvtx_only_fallback": raw_result.get("used_nvtx_only_fallback", False),
            "stats_selected_report": selected_report,
            "stats_attempts": raw_result.get("stats_attempts", []),
            "stats_stdout_tail": stats_stdout[-3000:] if stats_stdout else "",
            "stderr_tail": stats_res.get("stderr", "")[-3000:],
        }
        help_reports = raw_result.get("help_reports", {})
        if isinstance(help_reports, dict):
            output["profiler_output"]["help_reports_tail"] = (
                help_reports.get("stdout", "")[-3000:] if help_reports.get("stdout") else ""
            )
            output["profiler_output"]["help_reports_stderr_tail"] = (
                help_reports.get("stderr", "")[-3000:] if help_reports.get("stderr") else ""
            )

    elif mode == "sanitizer":
        san_res = raw_result.get("sanitizer_result", {})
        stdout = san_res.get("stdout", "")
        stderr = san_res.get("stderr", "")
        parsed = parse_sanitizer_output(stdout + "\n" + stderr)

        output["sanitizer"] = parsed
        output["profiler_output"] = {
            "returncode": san_res.get("returncode"),
            "stderr_tail": stderr[-3000:] if stderr else "",
            "cmd": raw_result.get("cmd", ""),
        }

    return output


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def _print_summary(output: dict[str, Any]) -> None:
    mode = output.get("mode", "unknown")
    kernel_name = output.get("kernel", {}).get("name", "?")
    gpu_name = output.get("gpu", {}).get("name", "?")

    print(f"\n{'=' * 72}")
    print(f"  Profiler: {mode.upper()}  |  Kernel: {kernel_name}  |  GPU: {gpu_name}")
    print(f"{'=' * 72}")

    if mode == "ncu":
        m = output.get("metrics", {})
        if not m:
            print("  [No metrics extracted -- check profiler_output for errors]")
        else:
            dur = m.get("duration_us")
            if dur is not None:
                print(f"\n  Duration:            {dur:.2f} us")
            _pval("  SM throughput:       ", m.get("sm_throughput_pct"), "%")
            _pval("  DRAM throughput:     ", m.get("dram_throughput_pct"), "%")
            _pval("  Tensor pipe util:    ", m.get("tensor_pipe_utilization_pct"), "%")
            _pval("  L2 hit rate:         ", m.get("l2_hit_rate_pct"), "%")
            _pval("  Achieved occupancy:  ", m.get("achieved_occupancy_pct"), "%")
            _pval("  Registers/thread:    ", m.get("registers_per_thread"), "")
            _pval("  SMEM/block:          ", m.get("smem_per_block_bytes"), " B")

        cls = output.get("classification", {})
        if cls:
            flags = [k for k, v in cls.items() if v]
            print(f"\n  Classification:  {', '.join(flags) if flags else 'balanced'}")

        for h in output.get("hints", []):
            prio = h["priority"].upper()
            print(f"\n  [{prio}] {h['category']}: {h['hint']}")

    elif mode == "nsys":
        m = output.get("metrics", {})
        if m:
            print(f"\n  Dominant kernel:  {m.get('dominant_kernel', '?')}")
            print(f"  Avg:   {m.get('avg_us', 0):.2f} us")
            print(f"  Med:   {m.get('med_us', 0):.2f} us")
            print(f"  Min:   {m.get('min_us', 0):.2f} us")
            print(f"  Max:   {m.get('max_us', 0):.2f} us")
            print(f"  Stdev: {m.get('stddev_us', 0):.2f} us")
            print(f"  Instances: {m.get('instances', 0)}")
        else:
            print("  [No kernel timing extracted]")

    elif mode == "sanitizer":
        s = output.get("sanitizer", {})
        status = "CLEAN" if s.get("clean") else f"ERRORS: {s.get('error_count', '?')}"
        print(f"\n  Tool:   {s.get('tool', '?')}")
        print(f"  Status: {status}")
        for err in s.get("errors", [])[:5]:
            print(f"  - {err}")
        for w in s.get("warnings", [])[:5]:
            print(f"  [WARN] {w}")

    print(f"\n{'=' * 72}\n")


def _pval(label: str, val: Any, unit: str) -> None:
    if val is not None:
        print(f"{label}{val:.1f}{unit}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--kernel", required=True, type=str,
        help="Path to kernel Python file (e.g. kernels/nvfp4_group_gemm/wagmiv67.py)",
    )
    p.add_argument(
        "--mode", required=True, choices=["nsys", "ncu", "sanitizer"],
        help="Profiling mode.",
    )
    p.add_argument(
        "--json-out", type=Path, default=None,
        help="Write JSON output to this file.",
    )

    # Problem sizes.
    p.add_argument(
        "--problem-sizes", type=str, default=None,
        help=(
            "Problem sizes as semicolon-separated groups, e.g. "
            "'80,4096,7168,1;40,7168,2048,1'.  "
            "Defaults to leaderboard 2-group case."
        ),
    )

    # Profiler tuning.
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3).")
    p.add_argument("--profile-iters", type=int, default=1, help="Profiled iterations (default: 1).")
    p.add_argument("--timeout", type=int, default=600, help="Subprocess timeout in seconds.")
    p.add_argument("--python-exe", type=str, default=sys.executable)

    # NVTX.
    p.add_argument("--nvtx", action="store_true", default=True, help="Use NVTX markers (default: on).")
    p.add_argument("--no-nvtx", action="store_true", help="Disable NVTX markers.")

    # ncu-specific.
    p.add_argument(
        "--ncu-set", type=str, default="basic",
        help="ncu section set: basic, detailed, full, curated, extended (default: basic).",
    )
    p.add_argument("--launch-skip", type=int, default=None, help="ncu --launch-skip value.")
    p.add_argument("--launch-count", type=int, default=1, help="ncu --launch-count (default: 1).")
    p.add_argument("--ncu-extra", type=str, default=None, help="Extra ncu args (space-separated).")

    # sanitizer-specific.
    p.add_argument(
        "--sanitizer-tool", type=str, default="memcheck",
        choices=["memcheck", "racecheck", "synccheck", "initcheck"],
        help="compute-sanitizer sub-tool (default: memcheck).",
    )

    # Remote execution.
    p.add_argument(
        "--remote", action="store_true",
        help="Run on Modal B200 instead of locally.",
    )
    p.add_argument("--gpu", type=str, default="B200", help="Modal GPU type (default: B200).")
    p.add_argument(
        "--probe", action="store_true",
        help="(Remote only) Quick check of tool availability without profiling.",
    )

    # Output control.
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress human-readable summary.")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print generated harness source and exit without profiling.",
    )

    args = p.parse_args()
    if args.no_nvtx:
        args.nvtx = False
    return args


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------

def _run_local(args: argparse.Namespace) -> dict[str, Any]:
    """Execute profiling locally."""
    kernel_path = os.path.abspath(args.kernel)
    if not os.path.isfile(kernel_path):
        raise SystemExit(f"[ERROR] Kernel file not found: {kernel_path}")

    problem_sizes = _parse_problem_sizes(args.problem_sizes)
    ncu_section_args = _resolve_ncu_section_args(args.ncu_set)

    gpu_info = _collect_gpu_info()

    config: dict[str, Any] = {
        "mode": args.mode,
        "problem_sizes": problem_sizes,
        "warmup_iters": args.warmup,
        "profile_iters": args.profile_iters,
        "nvtx": args.nvtx,
    }
    if args.mode == "ncu":
        config.update({
            "ncu_set": args.ncu_set,
            "launch_skip": args.launch_skip,
            "launch_count": args.launch_count,
        })
    elif args.mode == "sanitizer":
        config["sanitizer_tool"] = args.sanitizer_tool

    with tempfile.TemporaryDirectory(prefix="profile_kernel_") as tmpdir:
        harness_src = _generate_harness(
            kernel_path, problem_sizes,
            args.warmup, args.profile_iters, args.nvtx,
        )
        harness_path = os.path.join(tmpdir, "harness.py")
        with open(harness_path, "w") as f:
            f.write(harness_src)

        print(f"[INFO] Mode={args.mode}  Kernel={args.kernel}  GPU={gpu_info.get('name', '?')}")
        print(f"[INFO] Problem sizes: {problem_sizes}")
        print(f"[INFO] Warmup={args.warmup}  Profile iters={args.profile_iters}  NVTX={args.nvtx}")

        if args.mode == "nsys":
            raw = _run_nsys(
                harness_path, tmpdir, args.python_exe,
                args.warmup, args.profile_iters,
                args.nvtx, args.timeout,
            )
        elif args.mode == "ncu":
            raw = _run_ncu(
                harness_path, tmpdir, args.python_exe,
                ncu_section_args, args.launch_skip, args.launch_count,
                args.nvtx, args.timeout,
                args.ncu_extra.split() if args.ncu_extra else None,
            )
        elif args.mode == "sanitizer":
            raw = _run_sanitizer(
                harness_path, tmpdir, args.python_exe,
                args.sanitizer_tool, args.timeout,
            )
        else:
            raise SystemExit(f"Unknown mode: {args.mode}")

    return _build_output(args.mode, raw, config, args.kernel, gpu_info)


# ---------------------------------------------------------------------------
# Modal remote execution
# ---------------------------------------------------------------------------

def _generate_modal_runner(
    gpu: str,
    probe_only: bool,
) -> str:
    """Generate a self-contained Modal script for ``modal run``.

    The generated script embeds the remote function body and uses
    ``@app.function`` + ``@app.local_entrypoint()`` -- the only pattern
    that works reliably with the Modal CLI (``app.run()`` programmatic API
    hangs on ``.remote()`` calls with modal 1.3.x).
    """
    # We embed _profile_on_modal_impl's logic directly rather than
    # importing it, so the generated script is fully self-contained.
    return f'''\
#!/usr/bin/env python3
"""Auto-generated Modal runner for profile_kernel.py. Do not edit."""
from __future__ import annotations

import json
import sys

import modal

app = modal.App("profile-kernel-b200")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install("nvidia-cutlass-dsl==4.4.0", "torch")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends "
        "nsight-systems-2025.5.2 nsight-compute-2025.4.0 "
        "&& rm -rf /var/lib/apt/lists/*"
    )
)


@app.function(image=image, gpu="{gpu}", timeout=1800)
def profile_remote(kern_src: str, harness_src: str, cfg: dict) -> str:
    """Runs inside Modal container.  Returns JSON string to avoid pickle issues."""
    import glob
    import json as _json
    import os
    import shutil
    import subprocess as sp
    import sys as _sys
    import time
    import traceback

    def log(msg: str) -> None:
        print(f"[PROFILE-REMOTE] {{msg}}", flush=True)

    try:
        log("=== Remote profiler function starting ===")

        # -- Probe-only mode -----------------------------------------------
        if cfg.get("probe_only"):
            log("Probe mode: checking tool availability")

            def _find(name: str) -> str | None:
                for pat in [
                    f"/opt/nvidia/nsight-systems/*/bin/{{name}}",
                    f"/opt/nvidia/nsight-compute/*/bin/{{name}}",
                    f"/usr/local/cuda/bin/{{name}}",
                    f"/usr/local/cuda/extras/compute-sanitizer/{{name}}",
                    f"/usr/local/cuda/nsight-compute/ncu",
                    f"/usr/local/cuda/nsight-systems/bin/nsys",
                ]:
                    for c in sorted(glob.glob(pat), reverse=True):
                        if os.path.isfile(c) and os.access(c, os.X_OK):
                            return c
                p = shutil.which(name)
                if p:
                    return p
                return None

            tools = {{
                "ncu": _find("ncu"),
                "nsys": _find("nsys"),
                "compute-sanitizer": _find("compute-sanitizer"),
            }}
            log(f"Tools found: {{tools}}")

            try:
                gpu_proc = sp.run(
                    ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                gpu_raw = gpu_proc.stdout.strip()
            except Exception:
                gpu_raw = "nvidia-smi failed"

            versions: dict[str, str] = {{"python": _sys.version}}
            try:
                import torch
                versions["torch"] = torch.__version__
                versions["cuda_available"] = str(torch.cuda.is_available())
                if torch.cuda.is_available():
                    versions["cuda_device"] = torch.cuda.get_device_name(0)
            except Exception as e:
                versions["torch_error"] = str(e)
            try:
                import cutlass
                versions["cutlass"] = getattr(cutlass, "__version__", "unknown")
            except Exception as e:
                versions["cutlass_error"] = str(e)

            log("Probe complete")
            return _json.dumps({{
                "probe": True,
                "tools": tools,
                "gpu_info_raw": gpu_raw,
                "versions": versions,
            }})

        # -- Full profiling mode -------------------------------------------
        log("Writing kernel and harness to /tmp")
        with open("/tmp/kernel.py", "w") as f:
            f.write(kern_src)
        with open("/tmp/harness.py", "w") as f:
            f.write(harness_src)

        python = _sys.executable
        mode = cfg["mode"]
        timeout_ = min(cfg.get("timeout", 600), 900)

        def _find(name: str) -> str | None:
            for pat in [
                f"/opt/nvidia/nsight-systems/*/bin/{{name}}",
                f"/opt/nvidia/nsight-compute/*/bin/{{name}}",
                f"/usr/local/cuda/bin/{{name}}",
                f"/usr/local/cuda/extras/compute-sanitizer/{{name}}",
                f"/usr/local/cuda/nsight-compute/ncu",
                f"/usr/local/cuda/nsight-systems/bin/nsys",
            ]:
                for c in sorted(glob.glob(pat), reverse=True):
                    if os.path.isfile(c) and os.access(c, os.X_OK):
                        return c
            p = shutil.which(name)
            if p:
                return p
            return None

        def _run(cmd: list[str], timeout_s: int = timeout_, label: str = "cmd") -> dict:
            log(f"Running: {{' '.join(cmd[:6])}}... (timeout={{timeout_s}}s)")
            t0 = time.time()
            try:
                proc = sp.run(cmd, capture_output=True, text=True, timeout=timeout_s)
                elapsed = time.time() - t0
                log(f"  {{label}} finished in {{elapsed:.1f}}s (rc={{proc.returncode}})")
                return {{
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-100_000:],
                    "stderr": proc.stderr[-20_000:],
                    "elapsed_s": round(elapsed, 1),
                }}
            except sp.TimeoutExpired:
                elapsed = time.time() - t0
                log(f"  {{label}} TIMED OUT after {{elapsed:.1f}}s")
                return {{"returncode": -1, "stdout": "", "stderr": f"{{label}} timed out after {{timeout_s}}s"}}
            except Exception as e:
                log(f"  {{label}} EXCEPTION: {{e}}")
                return {{"returncode": -1, "stdout": "", "stderr": f"{{label}}: {{e}}"}}

        result: dict = {{}}

        if mode == "ncu":
            ncu_bin = _find("ncu")
            if not ncu_bin:
                return _json.dumps({{"error": "ncu not found in container. Run --probe to check available tools."}})
            log(f"Using ncu at: {{ncu_bin}}")
            cmd = [ncu_bin, "--csv"]
            cmd += cfg.get("ncu_section_args", ["--set", "basic"])
            if cfg.get("nvtx"):
                cmd += ["--nvtx", "--nvtx-include", "profile_region/"]
            if cfg.get("launch_skip") is not None:
                cmd += ["--launch-skip", str(cfg["launch_skip"])]
            cmd += ["--launch-count", str(cfg.get("launch_count", 1))]
            cmd += ["--target-processes", "all"]
            if cfg.get("ncu_extra"):
                cmd += cfg["ncu_extra"].split()
            cmd += [python, "/tmp/harness.py"]
            result["ncu_result"] = _run(cmd, label="ncu")
            result["cmd"] = " ".join(cmd)

        elif mode == "nsys":
            nsys_bin = _find("nsys")
            if not nsys_bin:
                return _json.dumps({{"error": "nsys not found in container. Run --probe to check available tools."}})
            log(f"Using nsys at: {{nsys_bin}}")
            nsys_cfg_info = {{}}
            z_res = _run([nsys_bin, "-z"], timeout_s=30, label="nsys -z")
            cfg_path = ""
            if z_res.get("returncode", -1) == 0:
                lines = [ln.strip() for ln in (z_res.get("stdout", "") or "").splitlines() if ln.strip()]
                if lines:
                    cfg_path = lines[-1]
            nsys_cfg_info["cfg_path"] = cfg_path
            nsys_cfg_info["z_returncode"] = z_res.get("returncode")
            nsys_cfg_info["z_stderr_tail"] = (z_res.get("stderr", "") or "")[-1000:]
            if cfg_path:
                try:
                    cfg_parent = os.path.dirname(cfg_path)
                    if cfg_parent:
                        os.makedirs(cfg_parent, exist_ok=True)
                    existing = ""
                    if os.path.exists(cfg_path):
                        with open(cfg_path, "r", encoding="utf-8", errors="ignore") as f:
                            existing = f.read()
                    if "CuptiUseRawGpuTimestamps=false" not in existing:
                        with open(cfg_path, "a", encoding="utf-8") as f:
                            if existing and not existing.endswith("\\n"):
                                f.write("\\n")
                            f.write("CuptiUseRawGpuTimestamps=false\\n")
                    nsys_cfg_info["timestamp_workaround"] = "applied"
                except Exception as e:
                    nsys_cfg_info["timestamp_workaround"] = f"failed: {{type(e).__name__}}: {{e}}"
            else:
                nsys_cfg_info["timestamp_workaround"] = "skipped: nsys -z path unavailable"

            cap_cmd = [
                nsys_bin, "profile", "-o", "/tmp/prof_out",
                "--force-overwrite", "true",
                "--trace", "cuda,nvtx",
                "--sample", "none", "--cpuctxsw", "none",
            ]
            if cfg.get("nvtx"):
                cap_cmd += ["--nvtx-capture", "range@profile_region"]
            cap_cmd += [python, "/tmp/harness.py"]
            result["capture"] = _run(cap_cmd, label="nsys profile")
            used_nvtx_only_fallback = False
            if result["capture"].get("returncode", -1) == 0 and "Unrecognized GPU UUID" in (result["capture"].get("stderr", "") or ""):
                cap_nvtx = [
                    nsys_bin, "profile", "-o", "/tmp/prof_out_nvtx",
                    "--force-overwrite", "true",
                    "--trace", "nvtx",
                    "--sample", "none", "--cpuctxsw", "none",
                ]
                if cfg.get("nvtx"):
                    cap_nvtx += ["--nvtx-capture", "range@profile_region"]
                cap_nvtx += [python, "/tmp/harness.py"]
                cap2 = _run(cap_nvtx, label="nsys profile (nvtx fallback)")
                if cap2.get("returncode", -1) == 0:
                    result["capture"] = cap2
                    used_nvtx_only_fallback = True

            if result["capture"].get("returncode", -1) == 0:
                out_prefix = "/tmp/prof_out_nvtx" if used_nvtx_only_fallback else "/tmp/prof_out"
                all_outputs = sorted(glob.glob(out_prefix + "*"))
                preferred = [
                    p for p in all_outputs if p.endswith(".nsys-rep") or p.endswith(".qdrep")
                ]
                if preferred:
                    rep = preferred[0]
                elif all_outputs:
                    rep = all_outputs[0]
                else:
                    rep = out_prefix + ".nsys-rep"
                if used_nvtx_only_fallback:
                    report_candidates = ["nvtx_sum"]
                else:
                    report_candidates = ["cuda_gpu_kern_sum", "cuda_api_sum", "nvtx_sum", "cuda_gpu_kernel_sum", "gpukernsum"]
                stats_attempts: list[dict] = []
                selected_report = None
                stats_res = {{"returncode": -1, "stdout": "", "stderr": "nsys stats not attempted"}}
                for report_name in report_candidates:
                    stats_cmd = [
                        nsys_bin, "stats", "--format", "csv", "--output", "-",
                        "--force-export", "true",
                        "--report", report_name, rep,
                    ]
                    attempt = _run(stats_cmd, timeout_s=60, label=f"nsys stats ({{report_name}})")
                    stats_attempts.append({{
                        "report": report_name,
                        "cmd": " ".join(stats_cmd),
                        "returncode": attempt.get("returncode"),
                        "stdout_tail": (attempt.get("stdout", "") or "")[-1000:],
                        "stderr_tail": (attempt.get("stderr", "") or "")[-1000:],
                    }})
                    stderr_txt = (attempt.get("stderr", "") or "").lower()
                    stdout_txt = (attempt.get("stdout", "") or "")
                    has_report_error = ("error: report" in stderr_txt) or ("could not be found" in stderr_txt)
                    looks_csv = (("Time (%)" in stdout_txt) and ("," in stdout_txt)) or (("Kernel Name" in stdout_txt) and ("," in stdout_txt))
                    if attempt.get("returncode", -1) == 0 and looks_csv and not has_report_error:
                        selected_report = report_name
                        stats_res = attempt
                        break
                    stats_res = attempt

                result["stats"] = stats_res
                result["report_file"] = rep
                result["report_candidates"] = all_outputs
                result["nsys_config"] = nsys_cfg_info
                result["used_nvtx_only_fallback"] = used_nvtx_only_fallback
                result["stats_selected_report"] = selected_report
                result["stats_attempts"] = stats_attempts
                if selected_report is None:
                    result["help_reports"] = _run(
                        [nsys_bin, "stats", "--help-reports"],
                        timeout_s=60,
                        label="nsys stats --help-reports",
                    )
            else:
                result["stats"] = {{"returncode": -1, "stdout": "", "stderr": "skipped: capture failed"}}
                result["nsys_config"] = nsys_cfg_info

        elif mode == "sanitizer":
            san_bin = _find("compute-sanitizer")
            if not san_bin:
                return _json.dumps({{"error": "compute-sanitizer not found in container. Run --probe to check available tools."}})
            log(f"Using compute-sanitizer at: {{san_bin}}")
            san_tool = cfg.get("sanitizer_tool", "memcheck")
            cmd = [san_bin, "--tool", san_tool, python, "/tmp/harness.py"]
            result["sanitizer_result"] = _run(cmd, label="compute-sanitizer")
            result["cmd"] = " ".join(cmd)

        # GPU info
        log("Collecting GPU info...")
        try:
            gpu_proc = sp.run(
                ["nvidia-smi",
                 "--query-gpu=name,compute_cap,memory.total,clocks.sm,clocks.mem,"
                 "driver_version,pci.bus_id",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            result["gpu_info_raw"] = gpu_proc.stdout.strip()
        except Exception:
            result["gpu_info_raw"] = ""

        log("=== Remote profiler function complete ===")
        return _json.dumps(result)

    except Exception:
        tb = traceback.format_exc()
        print(f"[PROFILE-REMOTE] FATAL: {{tb}}", flush=True)
        return _json.dumps({{"error": f"Remote function crashed:\\n{{tb}}"}})


@app.local_entrypoint()
def main(config_json: str, kernel_file: str, harness_file: str = ""):
    with open(config_json) as f:
        cfg = json.load(f)

    kern_src = ""
    if kernel_file and kernel_file != "__probe__":
        with open(kernel_file) as f:
            kern_src = f.read()

    harness_src = ""
    if harness_file:
        with open(harness_file) as f:
            harness_src = f.read()

    result_json = profile_remote.remote(kern_src, harness_src, cfg)

    # Output JSON on a line prefixed with RESULT_JSON: for easy parsing.
    # result_json is already a JSON string from the remote function.
    print("RESULT_JSON:" + result_json)
'''


def _run_remote(args: argparse.Namespace) -> dict[str, Any]:
    """Execute profiling on Modal B200 via ``modal run``.

    Generates a self-contained Modal script in a temp directory, invokes it
    with ``modal run``, and parses the JSON result from stdout.  This avoids
    the ``app.run()`` programmatic API which hangs with modal 1.3.x.
    """
    kernel_path = os.path.abspath(args.kernel)
    if not os.path.isfile(kernel_path):
        raise SystemExit(f"[ERROR] Kernel file not found: {kernel_path}")

    problem_sizes = _parse_problem_sizes(args.problem_sizes)
    ncu_section_args = _resolve_ncu_section_args(args.ncu_set)

    with tempfile.TemporaryDirectory(prefix="profile_remote_") as tmpdir:
        # Write the Modal runner script.
        runner_path = os.path.join(tmpdir, "modal_runner.py")
        with open(runner_path, "w") as f:
            f.write(_generate_modal_runner(gpu=args.gpu, probe_only=args.probe))

        # Write config JSON.
        remote_config: dict[str, Any] = {
            "mode": args.mode,
            "problem_sizes": problem_sizes,
            "warmup_iters": args.warmup,
            "profile_iters": args.profile_iters,
            "nvtx": args.nvtx,
            "timeout": args.timeout,
            "ncu_section_args": ncu_section_args,
            "launch_skip": args.launch_skip,
            "launch_count": args.launch_count,
            "ncu_extra": args.ncu_extra,
            "sanitizer_tool": args.sanitizer_tool,
        }
        if args.probe:
            remote_config["probe_only"] = True

        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(remote_config, f)

        # Write harness (for non-probe).
        harness_path = ""
        if not args.probe:
            harness_src = _generate_harness(
                "/tmp/kernel.py", problem_sizes,
                args.warmup, args.profile_iters, args.nvtx,
            )
            harness_path = os.path.join(tmpdir, "harness.py")
            with open(harness_path, "w") as f:
                f.write(harness_src)

        # Build modal run command.
        cmd = [
            sys.executable, "-m", "modal", "run", runner_path,
            "--config-json", config_path,
            "--kernel-file", kernel_path if not args.probe else "__probe__",
        ]
        if harness_path:
            cmd += ["--harness-file", harness_path]

        if args.probe:
            print(f"[INFO] Probing tool availability on Modal {args.gpu}...")
        else:
            print(f"[INFO] Launching {args.mode} profile on Modal {args.gpu}...")
            print(f"[INFO] Problem sizes: {problem_sizes}")
        print("[INFO] (First run builds image -- may take a few minutes)")

        # Run via subprocess to use the reliable `modal run` CLI path.
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout + 300,  # Extra margin for image build.
            )
        except subprocess.TimeoutExpired:
            return {
                "schema_version": SCHEMA_VERSION,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "mode": args.mode,
                "kernel": {"name": Path(kernel_path).stem, "path": args.kernel},
                "error": "modal run timed out",
            }

        # Print Modal CLI output (progress messages go to stderr).
        if proc.stderr:
            for line in proc.stderr.strip().splitlines():
                print(f"  [modal] {line}")

        # Parse RESULT_JSON: line from stdout.
        raw: dict[str, Any] = {}
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                try:
                    raw = json.loads(line[len("RESULT_JSON:"):])
                except json.JSONDecodeError as e:
                    raw = {"error": f"JSON decode failed: {e}", "raw_line": line[:2000]}
                break
        else:
            if proc.returncode != 0:
                raw = {
                    "error": f"modal run exited {proc.returncode}",
                    "stdout_tail": proc.stdout[-3000:],
                    "stderr_tail": proc.stderr[-3000:],
                }
            else:
                raw = {
                    "error": "No RESULT_JSON line in modal output",
                    "stdout_tail": proc.stdout[-3000:],
                }

    # -- Probe mode: just print and exit -----------------------------------
    if args.probe:
        print(json.dumps(raw, indent=2))
        raise SystemExit(0)

    # -- Handle errors -----------------------------------------------------
    if "error" in raw:
        err_msg = raw["error"]
        print(f"\n[ERROR] Remote profiling failed:\n{err_msg}", file=sys.stderr)
        return {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": args.mode,
            "kernel": {"name": Path(kernel_path).stem, "path": args.kernel},
            "error": err_msg,
            "raw_remote_result": raw,
        }

    # -- Parse GPU info from raw nvidia-smi output -------------------------
    gpu_info: dict[str, Any] = {}
    gpu_raw = raw.pop("gpu_info_raw", "")
    if gpu_raw:
        parts = [p.strip() for p in gpu_raw.split(",")]
        gpu_info = {
            "name": parts[0] if len(parts) > 0 else "unknown",
            "compute_cap": parts[1] if len(parts) > 1 else "unknown",
            "memory_total_mb": _try_int(parts[2]) if len(parts) > 2 else None,
            "sm_clock_mhz": _try_int(parts[3]) if len(parts) > 3 else None,
            "mem_clock_mhz": _try_int(parts[4]) if len(parts) > 4 else None,
            "driver_version": parts[5] if len(parts) > 5 else "unknown",
            "pci_bus_id": parts[6] if len(parts) > 6 else "unknown",
        }

    config: dict[str, Any] = {
        "mode": args.mode,
        "problem_sizes": problem_sizes,
        "warmup_iters": args.warmup,
        "profile_iters": args.profile_iters,
        "nvtx": args.nvtx,
        "execution": "modal",
        "gpu_type": args.gpu,
    }
    if args.mode == "ncu":
        config["ncu_set"] = args.ncu_set

    return _build_output(args.mode, raw, config, args.kernel, gpu_info)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_problem_sizes(raw: str | None) -> list[tuple[int, ...]]:
    if raw is None:
        return DEFAULT_PROBLEM_SIZES
    groups = raw.split(";")
    result = []
    for g in groups:
        parts = [int(x.strip()) for x in g.split(",")]
        if len(parts) != 4:
            raise SystemExit(
                f"Each problem size must have 4 ints (m,n,k,l), got: {g!r}"
            )
        result.append(tuple(parts))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Force line-buffered stdout so progress appears immediately when piped.
    sys.stdout.reconfigure(line_buffering=True)

    args = _parse_args()

    # -- Dry run: just print the harness and exit --------------------------
    if args.dry_run:
        kernel_path = os.path.abspath(args.kernel)
        if args.remote:
            kernel_path = "/tmp/kernel.py"
        problem_sizes = _parse_problem_sizes(args.problem_sizes)
        harness = _generate_harness(
            kernel_path, problem_sizes,
            args.warmup, args.profile_iters, args.nvtx,
        )
        print(harness)
        return

    # -- Real execution ----------------------------------------------------
    if args.remote:
        output = _run_remote(args)
    else:
        output = _run_local(args)

    # Write JSON.
    json_text = json.dumps(output, indent=2, default=str)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text + "\n", encoding="utf-8")
        print(f"[INFO] Wrote: {args.json_out}")

    if not args.quiet:
        _print_summary(output)

    # Also print raw JSON to stdout if no --json-out specified.
    if not args.json_out:
        print(json_text)


if __name__ == "__main__":
    main()
