from __future__ import annotations

import numpy as np


def best_so_far(values: list[float]) -> list[float]:
    return np.maximum.accumulate(np.asarray(values, dtype=float)).tolist()


def first_target_trial(values: list[float], target: float) -> int | None:
    for index, value in enumerate(best_so_far(values), start=1):
        if value >= target:
            return index
    return None


def bootstrap_mean_interval(matrix: np.ndarray, seed: int = 0, samples: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError("expected a seed-by-trial matrix")
    mean = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = np.stack([matrix[rng.integers(0, len(matrix), len(matrix))].mean(axis=0) for _ in range(samples)])
    return mean, np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)

