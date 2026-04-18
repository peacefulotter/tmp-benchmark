from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .core import run_benchmark


def memory_bandwidth(size: int = 200_000_000) -> Dict[str, Any]:
    def fn():
        arr = np.ones(size, dtype=np.float64)
        arr *= 2.0
        return arr

    bytes_used = size * 8
    result = run_benchmark(fn)
    result["bytes"] = bytes_used
    result["size"] = size
    return result


def run() -> Dict[str, Any]:
    return {
        "memory_bandwidth": memory_bandwidth(),
    }
