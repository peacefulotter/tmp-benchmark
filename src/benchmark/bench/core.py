import os
import platform
import statistics
import time
from contextlib import contextmanager

import psutil
import torch


def set_cpu_threads(mode: str = "optimized") -> None:
    """
    Controls BLAS threading behavior.

    mode:
        - "single": forces deterministic CPU (no parallel BLAS)
        - "optimized": lets MKL/OpenBLAS decide (real system behavior)
    """

    if mode == "single":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    else:
        del os.environ["OMP_NUM_THREADS"]
        del os.environ["MKL_NUM_THREADS"]
        del os.environ["OPENBLAS_NUM_THREADS"]


def system_info():
    return {
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": psutil.virtual_memory().total / 1e9,
        "platform": platform.platform(),
    }


@contextmanager
def timed():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def run_benchmark(fn, warmup=4, repeat=32):
    # Warmup
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeat):
        with timed() as t:
            fn()
        times.append(t())

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "runs": times,
    }
