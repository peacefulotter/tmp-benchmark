from __future__ import annotations

import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict

import psutil
from tqdm import tqdm

from benchmark.bench import cpu, disk, memory, ml, tensor

BenchmarkTask = Callable[[], dict[str, Any]]


def system_info() -> Dict[str, Any]:
    return {
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": psutil.virtual_memory().total / 1e9,
        "platform": platform.platform(),
    }


def build_cpu_tasks() -> dict[str, BenchmarkTask]:
    tasks: dict[str, BenchmarkTask] = {}
    tasks["cpu.numpy_matmul.single"] = lambda: cpu.numpy_matmul(threads="single")
    tasks["cpu.numpy_matmul.optimized"] = lambda: cpu.numpy_matmul(threads="optimized")
    tasks["memory.bandwidth"] = lambda: memory.memory_bandwidth()
    tasks["disk.parquet"] = lambda: disk.disk_io()
    tasks["ml.random_forest"] = lambda: ml.random_forest()
    tasks["torch.cpu"] = lambda: tensor.torch_cpu_matmul()
    return tasks


def build_cuda_tasks() -> dict[str, BenchmarkTask]:
    tasks: dict[str, BenchmarkTask] = {}
    tasks["torch.cuda"] = lambda: tensor.torch_cuda_matmul()
    return tasks


def run_all() -> Dict[str, Any]:
    cpu_tasks = build_cpu_tasks()
    gpu_tasks = build_cuda_tasks()
    tasks = {**cpu_tasks, **gpu_tasks}

    results: Dict[str, Any] = {}
    lock = Lock()

    total = len(tasks) * 2
    completed = 0
    start_time = time.time()

    def update_bar(pbar, name: str, mode: str):
        nonlocal completed
        completed += 1
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        pbar.set_postfix(
            {
                "mode": mode,
                "done": f"{completed}/{total}",
                "rate": f"{rate:.2f} tasks/s",
                "last": name,
            }
        )
        pbar.update(1)

    with tqdm(total=total, desc="Benchmark suite") as pbar:
        # ---------------- CPU + IO SEQUENTIAL ----------------
        mode = "sequential"
        for name, task in tasks.items():
            pbar.set_description(f"[{mode}] Running: {name}")
            value = task()
            results[f"{mode}.{name}"] = value
            update_bar(pbar, name, mode)

        # ---------------- CPU + IO PARALLEL ----------------
        mode = "parallel"
        pbar.set_description(f"[{mode}] Running CPU tasks")
        with ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=True)) as ex:
            future_map = {ex.submit(task): name for name, task in cpu_tasks.items()}

            for future in as_completed(future_map):
                name = future_map[future]
                value = future.result()
                results[f"{mode}.{name}"] = value
                update_bar(pbar, name, mode)

        # ---------------- GPU SERIAL ----------------
        mode = "gpu"
        pbar.set_description(f"[{mode}] Running GPU tasks")
        for name, task_fn in gpu_tasks.items():
            with lock:
                value = task_fn()
                results[f"{mode}.{name}"] = value
                update_bar(pbar, name, mode)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "system": system_info(),
        "benchmarks": results,
    }


if __name__ == "__main__":
    results = run_all()

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
