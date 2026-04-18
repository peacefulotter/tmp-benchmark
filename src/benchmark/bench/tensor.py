from __future__ import annotations

import statistics
from typing import Any, Dict, List

import torch

from .core import run_benchmark, set_cpu_threads

# -----------------------------
# CPU BENCHMARK
# -----------------------------


def torch_cpu_matmul(size: int = 2048, threads: str = "optimized") -> Dict[str, Any]:

    set_cpu_threads(threads)

    a = torch.randn(size, size)
    b = torch.randn(size, size)

    def fn():
        torch.matmul(a, b)

    flops = 2 * size**3

    result = run_benchmark(fn)
    result.update(
        {
            "flops": flops,
            "size": size,
            "mode": threads,
            "device": "cpu",
        }
    )

    return result


# -----------------------------
# CUDA EVENT TIMING
# -----------------------------


def run_cuda_event_benchmark(fn, warmup: int = 4, repeat: int = 16) -> Dict[str, Any]:
    """
    Accurate GPU timing using CUDA events.
    """

    # Warmup
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    times: List[float] = []

    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms / 1000.0)

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "runs": times,
    }


# -----------------------------
# GPU SATURATED MATMUL
# -----------------------------


def torch_cuda_matmul(size: int = 1024, batch: int = 64) -> Dict[str, Any]:

    if not torch.cuda.is_available():
        return {"available": False}

    device = "cuda"

    # Batched matrices (critical for GPU utilization)
    a = torch.randn(batch, size, size, device=device)
    b = torch.randn(batch, size, size, device=device)

    def fn():
        torch.bmm(a, b)

    # FLOPs per matmul * batch
    flops = batch * (2 * size**3)

    result = run_cuda_event_benchmark(fn)

    result.update(
        {
            "flops": flops,
            "size": size,
            "batch": batch,
            "device": device,
            "timing": "cuda_event",
        }
    )

    return result


# -----------------------------
# GPU AUTO-SATURATION TEST
# -----------------------------


def find_saturating_batch(size: int = 1024, max_batch: int = 2048) -> int:
    """
    Heuristic: increase batch until marginal speedup drops.
    """

    def bench(batch: int) -> float:
        a = torch.randn(batch, size, size, device="cuda")
        b = torch.randn(batch, size, size, device="cuda")

        def fn():
            torch.bmm(a, b)

        res = run_cuda_event_benchmark(fn, warmup=1, repeat=3)
        return res["mean"]

    batch = 8
    last = bench(batch)

    while batch < max_batch:
        batch *= 2
        current = bench(batch)

        # diminishing returns → saturation
        if abs(last - current) / last < 0.1:
            break

        last = current

    return batch


# -----------------------------
# MAIN ENTRY
# -----------------------------


def run() -> dict[str, Any]:

    # optional auto-scaling batch for GPU
    batch = 64
    if torch.cuda.is_available():
        try:
            batch = find_saturating_batch()
        except Exception:
            batch = 64

    return {
        "torch_cpu_single": torch_cpu_matmul(threads="single"),
        "torch_cpu_optimized": torch_cpu_matmul(threads="optimized"),
        "torch_cuda": torch_cuda_matmul(batch=batch),
    }
