from __future__ import annotations

import math
from typing import Any

import numpy as np

from cognitune.config.schema import KnobSpec
from cognitune.search_space.space import SearchSpace


def draw_value(knob: KnobSpec, rng: np.random.Generator) -> Any:
    if knob.kind == "integer":
        if knob.log:
            return int(np.clip(np.rint(math.exp(rng.uniform(math.log(knob.lower), math.log(knob.upper)))), knob.lower, knob.upper))
        return int(rng.integers(int(knob.lower), int(knob.upper) + 1))
    if knob.kind == "float":
        return float(math.exp(rng.uniform(math.log(knob.lower), math.log(knob.upper)))) if knob.log else float(rng.uniform(knob.lower, knob.upper))
    if knob.kind == "bool":
        return bool(rng.integers(0, 2))
    return knob.choices[int(rng.integers(0, len(knob.choices)))]


def local_mutations(
    space: SearchSpace,
    incumbent: dict,
    knob_names: list[str],
    rng: np.random.Generator,
    n: int,
    max_changes: int = 2,
) -> list[dict]:
    candidates = []
    available = [name for name in knob_names if name in space.by_name]
    if not available:
        return candidates
    for _ in range(n):
        candidate = dict(incumbent)
        count = min(len(available), int(rng.integers(1, max_changes + 1)))
        for name in rng.choice(available, size=count, replace=False):
            candidate[str(name)] = draw_value(space.by_name[str(name)], rng)
        candidates.append(space.repair(candidate))
    return candidates


def deduplicate_legal(space: SearchSpace, candidates: list[dict], seen: set[str]) -> list[dict]:
    result, local_seen = [], set()
    for candidate in candidates:
        repaired = space.repair(candidate)
        key = space.config_key(repaired)
        if key not in seen and key not in local_seen and space.validate(repaired):
            result.append(repaired)
            local_seen.add(key)
    return result

