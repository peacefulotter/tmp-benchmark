from __future__ import annotations

from typing import Any, Dict

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from .core import run_benchmark


def random_forest(samples: int = 300_000, features: int = 50) -> Dict[str, Any]:
    X, y = make_classification(n_samples=samples, n_features=features, random_state=42)

    def fn():
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        model.fit(X, y)

    result = run_benchmark(fn)
    result["samples"] = samples
    result["features"] = features
    return result


def run() -> Dict[str, Any]:
    return {
        "random_forest": random_forest(),
    }
