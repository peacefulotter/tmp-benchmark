from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from .core import run_benchmark, set_cpu_threads


def numpy_matmul(size: int = 3000, threads: str = "optimized") -> Dict[str, Any]:
    set_cpu_threads(threads)

    a = np.random.rand(size, size)
    b = np.random.rand(size, size)

    def fn():
        np.dot(a, b)

    flops = 2 * size**3
    result = run_benchmark(fn)
    result["flops"] = flops
    result["size"] = size
    return result


def torch_cpu_matmul(size: int = 3000, threads: str = "optimized") -> Dict[str, Any]:
    set_cpu_threads(threads)

    a = torch.rand(size, size)
    b = torch.rand(size, size)

    def fn():
        torch.matmul(a, b)

    flops = 2 * size**3
    result = run_benchmark(fn)
    result["flops"] = flops
    result["size"] = size
    return result


def run() -> Dict[str, Any]:
    return {
        "numpy_matmul": numpy_matmul(),
        "torch_cpu_matmul": torch_cpu_matmul(),
    }
