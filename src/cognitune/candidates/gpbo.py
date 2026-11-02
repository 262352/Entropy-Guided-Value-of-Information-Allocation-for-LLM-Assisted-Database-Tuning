from __future__ import annotations

import numpy as np

from cognitune.search_space.space import SearchSpace
from .common import deduplicate_legal, local_mutations


def generate_gpbo_pool(
    space: SearchSpace,
    incumbent: dict,
    seen: set[str],
    seed: int,
    size: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    full_count = max(1, round(size * 0.8))
    candidates = space.sample_iid(seed, full_count)
    candidates.extend(local_mutations(space, incumbent, list(space.by_name), rng, size - full_count))
    legal = deduplicate_legal(space, candidates, seen)
    if not legal:
        legal = deduplicate_legal(space, space.sample_iid(seed + 1_000_003, max(10, size)), seen)
    return legal

