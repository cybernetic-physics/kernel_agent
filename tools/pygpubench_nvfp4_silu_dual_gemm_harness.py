"""PyGPUBench harness for nvfp4_dual_gemm (silu-gated) kernel.

Computes C = silu(A @ B1) * (A @ B2) with NVFP4 block-scaled quantization.

Kernel interface:
  custom_kernel(data) where data is a 10-tuple:
    (a, b1, b2, sfa_ref_cpu, sfb1_ref_cpu, sfb2_ref_cpu,
     sfa_permuted, sfb1_permuted, sfb2_permuted, c)

Runtime controls (env vars):
- PYGPUBENCH_REPEATS: total repeats (int, default REPEATS)
- PYGPUBENCH_STAGE_REPEATS: comma-separated stage sizes
- PYGPUBENCH_EARLY_STOP_US: if >0, stop after a stage when median_us exceeds this
- PYGPUBENCH_PROBLEM_SIZE: comma-separated m,n,k override (default: 256,4096,7168)
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass

import torch
import pygpubench

SF_VEC_SIZE = 16


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix: torch.Tensor):
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def create_reordered_scale_factor_tensor(l, mn, k, ref_f8_tensor):
    sf_k = ceil_div(k, SF_VEC_SIZE)
    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    rand_int_tensor = torch.randint(1, 3, mma_shape, dtype=torch.int8, device="cuda")
    reordered_f8_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
    reordered_f8_tensor = reordered_f8_tensor.permute(*mma_permute_order)

    if ref_f8_tensor.device.type == "cpu":
        ref_f8_tensor = ref_f8_tensor.cuda()

    i_idx = torch.arange(mn, device="cuda")
    j_idx = torch.arange(sf_k, device="cuda")
    b_idx = torch.arange(l, device="cuda")
    i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing="ij")

    mm = i_grid // (atom_m[0] * atom_m[1])
    mm32 = i_grid % atom_m[0]
    mm4 = (i_grid % 128) // atom_m[0]
    kk = j_grid // atom_k
    kk4 = j_grid % atom_k

    reordered_f8_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_tensor[
        i_grid, j_grid, b_grid
    ]
    return reordered_f8_tensor


def _create_fp4_tensors(l, mn, k):
    ref_i8 = torch.randint(255, size=(l, mn, k // 2), dtype=torch.uint8, device="cuda")
    ref_i8 = ref_i8 & 0b1011_1011
    return ref_i8.permute(1, 2, 0).contiguous().view(torch.float4_e2m1fn_x2)


def generate_input(m, n, k, seed):
    torch.manual_seed(seed)
    l = 1

    a = _create_fp4_tensors(l, m, k)
    b1 = _create_fp4_tensors(l, n, k)
    b2 = _create_fp4_tensors(l, n, k)

    c = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)

    sf_k = ceil_div(k, SF_VEC_SIZE)
    sfa_ref_cpu = (
        torch.randint(1, 3, (l, m, sf_k), dtype=torch.int8)
        .to(dtype=torch.float8_e4m3fn)
        .permute(1, 2, 0)
    )
    sfb1_ref_cpu = (
        torch.randint(1, 3, (l, n, sf_k), dtype=torch.int8)
        .to(dtype=torch.float8_e4m3fn)
        .permute(1, 2, 0)
    )
    sfb2_ref_cpu = (
        torch.randint(1, 3, (l, n, sf_k), dtype=torch.int8)
        .to(dtype=torch.float8_e4m3fn)
        .permute(1, 2, 0)
    )

    sfa_permuted = create_reordered_scale_factor_tensor(l, m, k, sfa_ref_cpu)
    sfb1_permuted = create_reordered_scale_factor_tensor(l, n, k, sfb1_ref_cpu)
    sfb2_permuted = create_reordered_scale_factor_tensor(l, n, k, sfb2_ref_cpu)

    return (a, b1, b2, sfa_ref_cpu, sfb1_ref_cpu, sfb2_ref_cpu,
            sfa_permuted, sfb1_permuted, sfb2_permuted, c)


def ref_kernel(data):
    """Reference: C = silu(A @ B1) * (A @ B2)"""
    a_ref, b1_ref, b2_ref, sfa_ref_cpu, sfb1_ref_cpu, sfb2_ref_cpu, _, _, _, c_ref = data
    m, n, l = c_ref.shape

    ref1 = torch.empty((l, m, n), dtype=torch.float32, device="cuda").permute(1, 2, 0)
    ref2 = torch.empty((l, m, n), dtype=torch.float32, device="cuda").permute(1, 2, 0)

    for l_idx in range(l):
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b1 = to_blocked(sfb1_ref_cpu[:, :, l_idx])
        scale_b2 = to_blocked(sfb2_ref_cpu[:, :, l_idx])

        res1 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b1_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b1.cuda(),
            bias=None,
            out_dtype=torch.float32,
        )
        ref1[:, :, l_idx] = res1

        res2 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b2_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b2.cuda(),
            bias=None,
            out_dtype=torch.float32,
        )
        ref2[:, :, l_idx] = res2

    return (torch.nn.functional.silu(ref1) * ref2).to(torch.float16)


def generate_test_case(args, seed):
    m, n, k = args
    data = generate_input(m, n, k, seed)
    expected = ref_kernel(data)
    # Debug: verify shapes and values
    print(f"[DBG] m={m} n={n} k={k} l={data[9].shape[2]}")
    print(f"[DBG] a.shape={data[0].shape} dtype={data[0].dtype}")
    print(f"[DBG] sfa_cpu.shape={data[3].shape} dtype={data[3].dtype}")
    print(f"[DBG] sfa_perm.shape={data[6].shape} dtype={data[6].dtype}")
    print(f"[DBG] c.shape={data[9].shape} expected.shape={expected.shape}")
    print(f"[DBG] expected stats: min={expected.min().item():.4f} max={expected.max().item():.4f} mean={expected.float().mean().item():.4f}")
    out_c = data[9]
    return (out_c, data), (expected.clone(), 0.1, 0.5)


def kernel_generator():
    import submission
    _call_count = [0]

    def _wrapped(args):
        out_c, data = args
        result = submission.custom_kernel(data)
        if isinstance(result, torch.Tensor):
            out_c.copy_(result)
        _call_count[0] += 1
        if _call_count[0] <= 2:
            print(f"[DBG] call={_call_count[0]} out_c stats: min={out_c.min().item():.4f} max={out_c.max().item():.4f} mean={out_c.float().mean().item():.4f}")

    return _wrapped


# Default problem size; override with PYGPUBENCH_PROBLEM_SIZE env var
_ps_raw = os.getenv("PYGPUBENCH_PROBLEM_SIZE", "256,4096,7168")
TEST_ARGS = tuple(int(x.strip()) for x in _ps_raw.split(","))
REPEATS = 40
SEED = 1111


@dataclass
class StageResult:
    repeats: int
    result: object
    median_us: float


def _parse_stage_repeats(total_repeats):
    raw = os.getenv("PYGPUBENCH_STAGE_REPEATS", "").strip()
    if not raw:
        return [total_repeats]
    stage_sizes = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        val = int(item)
        if val > 0:
            stage_sizes.append(val)
    if not stage_sizes:
        return [total_repeats]
    total = sum(stage_sizes)
    if total < total_repeats:
        stage_sizes.append(total_repeats - total)
    elif total > total_repeats:
        trimmed = []
        remaining = total_repeats
        for size in stage_sizes:
            if remaining <= 0:
                break
            used = min(size, remaining)
            if used > 0:
                trimmed.append(used)
                remaining -= used
        stage_sizes = trimmed
    return stage_sizes or [total_repeats]


def _median_us(bench_result):
    values = [float(x) for x in bench_result.time_us]
    if not values:
        return float("inf")
    return float(statistics.median(values))


def _error_count(bench_result) -> int:
    errs = bench_result.errors
    if errs is None:
        return 0
    if isinstance(errs, int):
        return errs
    return len(errs)


def _run_staged_benchmark(total_repeats, early_stop_us):
    stage_repeats = _parse_stage_repeats(total_repeats)
    last_stage = None
    for stage_idx, repeats in enumerate(stage_repeats, start=1):
        bench_result = pygpubench.do_bench_isolated(
            kernel_generator,
            generate_test_case,
            TEST_ARGS,
            repeats,
            SEED,
            discard=True,
        )
        median_us = _median_us(bench_result)
        last_stage = StageResult(repeats=repeats, result=bench_result, median_us=median_us)
        num_errors = _error_count(bench_result)
        print(
            f"stage={stage_idx}/{len(stage_repeats)} repeats={repeats} "
            f"median_us={median_us:.3f} errors={num_errors}"
        )
        if num_errors:
            return last_stage
        if early_stop_us > 0 and median_us > early_stop_us:
            print(
                "EARLY_STOP: median_us exceeded threshold "
                f"({median_us:.3f} > {early_stop_us:.3f})"
            )
            return last_stage
    if last_stage is None:
        raise RuntimeError("No benchmark stages were executed.")
    return last_stage


if __name__ == "__main__":
    total_repeats = int(os.getenv("PYGPUBENCH_REPEATS", str(REPEATS)))
    early_stop_us = float(os.getenv("PYGPUBENCH_EARLY_STOP_US", "0"))
    stage_result = _run_staged_benchmark(total_repeats=total_repeats, early_stop_us=early_stop_us)
    result = stage_result.result
    num_errors = _error_count(result)
    print("FAIL" if num_errors else "PASS", pygpubench.basic_stats(result.time_us))
