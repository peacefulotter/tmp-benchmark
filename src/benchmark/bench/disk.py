from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import numpy as np
import pandas as pd

from .core import run_benchmark


def disk_io(rows: int = 2_000_000) -> Dict[str, Any]:
    df = pd.DataFrame(
        {
            "a": np.random.rand(rows),
            "b": np.random.rand(rows),
        }
    )

    def fn():
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        df.to_parquet(path)
        size_bytes = os.path.getsize(path)
        _ = pd.read_parquet(path)
        os.remove(path)
        return size_bytes

    result = run_benchmark(fn)

    # run once to get size
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    df.to_parquet(path)
    size_bytes = os.path.getsize(path)
    os.remove(path)

    result["bytes"] = size_bytes
    result["rows"] = rows
    return result


def run() -> Dict[str, Any]:
    return {
        "disk_io_parquet": disk_io(),
    }
