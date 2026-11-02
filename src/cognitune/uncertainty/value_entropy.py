from __future__ import annotations

import math
from typing import Any

import numpy as np

from cognitune.config.schema import KnobSpec


def value_grid(knob: KnobSpec, size: int = 16) -> np.ndarray:
    if knob.kind not in {"integer", "float"}:
        choices = (False, True) if knob.kind == "bool" else knob.choices
        return np.asarray(choices, dtype=object)
    values = np.geomspace(float(knob.lower), float(knob.upper), size) if knob.log else np.linspace(float(knob.lower), float(knob.upper), size)
    return np.unique(np.rint(values).astype(int)) if knob.kind == "integer" else values


def recommendation_distribution(knob: KnobSpec, recommendation, grid: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    distribution = np.zeros(len(grid), dtype=float)
    values = recommendation.recommended_values
    interval = recommendation.recommended_interval
    if values:
        for value in values:
            index = int(np.argmin([abs(float(cell) - float(value)) if knob.kind in {"integer", "float"} else 0 if cell == value else 1 for cell in grid]))
            distribution[index] += 0.7 / len(values) if interval else 1.0 / len(values)
    if interval:
        covered = np.asarray([interval[0] <= float(cell) <= interval[1] for cell in grid])
        if not covered.any():
            covered[int(np.argmin(np.abs(grid.astype(float) - np.mean(interval))))] = True
        distribution[covered] += (0.3 if values else 1.0) / covered.sum()
    if not values and interval is None:
        distribution[:] = 1 / len(grid)
    distribution += alpha
    return distribution / distribution.sum()


def _entropy(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0]
    return float(-np.sum(positive * np.log(positive)))


def value_statistics(responses: list, knobs: list[KnobSpec], grid_size: int = 16, alpha: float = 0.1) -> dict[str, dict[str, Any]]:
    result = {}
    for knob in knobs:
        grid = value_grid(knob, grid_size)
        items = [(response.recommendations[knob.name], response.recommendations[knob.name].confidence) for response in responses if knob.name in response.recommendations]
        if not items:
            continue
        weights = np.asarray([max(item[1], 1e-9) for item in items], dtype=float)
        weights /= weights.sum()
        distributions = np.stack([recommendation_distribution(knob, item[0], grid, alpha) for item in items])
        average = np.sum(weights[:, None] * distributions, axis=0)
        disagreement = _entropy(average) - sum(weight * _entropy(dist) for weight, dist in zip(weights, distributions))
        result[knob.name] = {"value_entropy": max(0.0, float(disagreement)), "grid": grid.tolist(), "distribution": average.tolist()}
    return result

