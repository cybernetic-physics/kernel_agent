"""Template harness for PyGPUBench.

Copy this file and adapt:
1) `generate_test_case(args, seed)` to build your kernel inputs and expected output.
2) `kernel_generator()` to return the callable from `submission.py`.
3) `TEST_ARGS`, `REPEATS`, `SEED` for your workload.
"""

from __future__ import annotations

import pygpubench


def generate_test_case(args: tuple, seed: int):
    """
    Must return:
      (kernel_inputs_tuple, (expected_output_tensor, rtol, atol))

    Replace this body with your problem-specific input/reference generation.
    """
    raise NotImplementedError("Implement generate_test_case for your kernel/problem.")


def kernel_generator():
    """
    Must return a callable that accepts one argument: kernel_inputs_tuple.
    """
    import submission

    return submission.custom_kernel


TEST_ARGS = ()
REPEATS = 100
SEED = 5


if __name__ == "__main__":
    result = pygpubench.do_bench_isolated(
        kernel_generator,
        generate_test_case,
        TEST_ARGS,
        REPEATS,
        SEED,
        discard=True,
    )
    print("FAIL" if result.errors else "PASS", pygpubench.basic_stats(result.time_us))
