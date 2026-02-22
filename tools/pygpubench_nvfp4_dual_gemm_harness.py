"""PyGPUBench harness aligned to reference-kernels nvfp4_group_gemm task.

This mirrors the grouped GEMM data contract used by:
  reference-kernels/problems/nvidia/nvfp4_group_gemm/{task.py,reference.py,task.yml}

Kernel contract:
  custom_kernel(data) where:
    data = (
      abc_tensors,               # list[(a, b, c)]
      sfasfb_tensors,            # list[(sfa_ref_cpu, sfb_ref_cpu)]
      sfasfb_reordered_tensors,  # list[(sfa_reordered, sfb_reordered)]
      problem_sizes,             # list[(m, n, k, l)]
    )

Runtime controls (env vars):
  PYGPUBENCH_CASE_SET: "benchmarks" (default) or "tests"
  PYGPUBENCH_CASE_INDEX: index into selected case set (default: 0)
  PYGPUBENCH_REPEATS, PYGPUBENCH_STAGE_REPEATS, PYGPUBENCH_EARLY_STOP_US
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass

import pygpubench
import torch

SF_VEC_SIZE = 16

TEST_CASES = [
    {"m": [96, 128], "n": [128, 256], "k": [256, 512], "g": 2, "seed": 1111},
    {"m": [256, 72], "n": [512, 384], "k": [256, 256], "g": 2, "seed": 1111},
    {"m": [128, 128], "n": [128, 256], "k": [512, 256], "g": 2, "seed": 1111},
    {"m": [80, 128, 256], "n": [384, 256, 128], "k": [256, 512, 256], "g": 3, "seed": 1111},
    {"m": [64, 72, 96], "n": [128, 384, 512], "k": [512, 512, 256], "g": 3, "seed": 1111},
    {"m": [64, 256, 128], "n": [768, 128, 256], "k": [512, 256, 512], "g": 3, "seed": 1111},
    {"m": [128, 128, 64], "n": [256, 512, 512], "k": [768, 256, 768], "g": 3, "seed": 1111},
    {
        "m": [128, 128, 128, 128],
        "n": [128, 128, 128, 128],
        "k": [512, 256, 512, 256],
        "g": 4,
        "seed": 1111,
    },
    {
        "m": [40, 56, 384, 512],
        "n": [512, 384, 256, 128],
        "k": [256, 256, 256, 256],
        "g": 4,
        "seed": 1111,
    },
    {
        "m": [512, 384, 256, 128],
        "n": [256, 256, 256, 256],
        "k": [512, 768, 512, 768],
        "g": 4,
        "seed": 1111,
    },
]

BENCHMARK_CASES = [
    {
        "m": [80, 176, 128, 72, 64, 248, 96, 160],
        "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096],
        "k": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168],
        "g": 8,
        "seed": 1111,
    },
    {
        "m": [40, 76, 168, 72, 164, 148, 196, 160],
        "n": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168],
        "k": [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
        "g": 8,
        "seed": 1111,
    },
    {"m": [192, 320], "n": [3072, 3072], "k": [4096, 4096], "g": 2, "seed": 1111},
    {"m": [128, 384], "n": [4096, 4096], "k": [1536, 1536], "g": 2, "seed": 1111},
]


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix: torch.Tensor):
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    if padded_rows != rows or padded_cols != cols:
        padded = torch.nn.functional.pad(
            input_matrix,
            (0, padded_cols - cols, 0, padded_rows - rows),
            mode="constant",
            value=0,
        )
    else:
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
    return ref_i8.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)


def generate_input(m: tuple, n: tuple, k: tuple, g: int, seed: int):
    torch.manual_seed(seed)
    abc_tensors = []
    sfasfb_tensors = []
    sfasfb_reordered_tensors = []
    problem_sizes = []
    l = 1

    for group_idx in range(g):
        mi = int(m[group_idx])
        ni = int(n[group_idx])
        ki = int(k[group_idx])

        a_ref = _create_fp4_tensors(l, mi, ki)
        b_ref = _create_fp4_tensors(l, ni, ki)
        c_ref = torch.randn((l, mi, ni), dtype=torch.float16, device="cuda").permute(1, 2, 0)

        sf_k = ceil_div(ki, SF_VEC_SIZE)
        sfa_ref_cpu = (
            torch.randint(1, 3, (l, mi, sf_k), dtype=torch.int8)
            .to(dtype=torch.float8_e4m3fn)
            .permute(1, 2, 0)
        )
        sfb_ref_cpu = (
            torch.randint(1, 3, (l, ni, sf_k), dtype=torch.int8)
            .to(dtype=torch.float8_e4m3fn)
            .permute(1, 2, 0)
        )

        sfa_reordered = create_reordered_scale_factor_tensor(l, mi, ki, sfa_ref_cpu)
        sfb_reordered = create_reordered_scale_factor_tensor(l, ni, ki, sfb_ref_cpu)

        abc_tensors.append((a_ref, b_ref, c_ref))
        sfasfb_tensors.append((sfa_ref_cpu, sfb_ref_cpu))
        sfasfb_reordered_tensors.append((sfa_reordered, sfb_reordered))
        problem_sizes.append((mi, ni, ki, l))

    return (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)


def ref_kernel(data):
    abc_tensors, sfasfb_tensors, _, problem_sizes = data
    result_tensors = []

    for (a_ref, b_ref, c_ref), (sfa_ref, sfb_ref), (m, n, k, l) in zip(
        abc_tensors, sfasfb_tensors, problem_sizes
    ):
        for l_idx in range(l):
            scale_a = to_blocked(sfa_ref[:, :, l_idx])
            scale_b = to_blocked(sfb_ref[:, :, l_idx])
            res = torch._scaled_mm(
                a_ref[:, :, l_idx].view(torch.float4_e2m1fn_x2),
                b_ref[:, :, l_idx].transpose(0, 1).view(torch.float4_e2m1fn_x2),
                scale_a.cuda(),
                scale_b.cuda(),
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, :, l_idx] = res
        result_tensors.append(c_ref)

    return result_tensors


def _flatten_tensor_list(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1).to(torch.float16) for t in tensors], dim=0)


def _extract_group_outputs_from_data(data):
    abc_tensors = data[0]
    return [abc_tensors[i][2] for i in range(len(abc_tensors))]


def _resolve_case() -> dict:
    case_set = os.getenv("PYGPUBENCH_CASE_SET", "benchmarks").strip().lower()
    case_index = int(os.getenv("PYGPUBENCH_CASE_INDEX", "0"))
    cases = TEST_CASES if case_set == "tests" else BENCHMARK_CASES
    if case_index < 0 or case_index >= len(cases):
        raise ValueError(
            f"PYGPUBENCH_CASE_INDEX={case_index} out of range for {case_set} "
            f"(size={len(cases)})"
        )
    return cases[case_index]


def generate_test_case(args, seed):
    case = dict(args[0])
    effective_seed = int(case.get("seed", 1111)) + int(seed)
    data = generate_input(
        m=tuple(case["m"]),
        n=tuple(case["n"]),
        k=tuple(case["k"]),
        g=int(case["g"]),
        seed=effective_seed,
    )

    expected = _flatten_tensor_list(ref_kernel(data))
    out_flat = torch.empty_like(expected)
    return (out_flat, data), (expected, 1e-3, 1e-3)


def kernel_generator():
    import submission

    def _wrapped(args):
        out_flat, data = args
        result = submission.custom_kernel(data)
        if result is None:
            result = _extract_group_outputs_from_data(data)

        if isinstance(result, torch.Tensor):
            flat = result.reshape(-1).to(out_flat.dtype)
        elif isinstance(result, (list, tuple)):
            flat = _flatten_tensor_list(list(result)).to(out_flat.dtype)
        else:
            raise TypeError(f"Unsupported kernel return type: {type(result)}")

        if flat.numel() != out_flat.numel():
            raise RuntimeError(
                f"flattened output size mismatch: got {flat.numel()} expected {out_flat.numel()}"
            )
        out_flat.copy_(flat)

    return _wrapped


CASE = _resolve_case()
TEST_ARGS = (CASE,)
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
            f"case_set={os.getenv('PYGPUBENCH_CASE_SET', 'benchmarks')} "
            f"case_index={os.getenv('PYGPUBENCH_CASE_INDEX', '0')} "
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
