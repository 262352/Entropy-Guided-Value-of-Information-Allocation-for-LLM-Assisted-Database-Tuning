from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cognitune.search_space.space import SearchSpace
from cognitune.state import KnobState, StateManager
from .common import deduplicate_legal, local_mutations


@dataclass(frozen=True)
class CandidatePool:
    configs: list[dict]
    sources: list[str]


def generate_cognitune_pool(
    space: SearchSpace,
    incumbent: dict,
    seen: set[str],
    state: StateManager,
    blocks: list[tuple[str, ...]],
    value_stats: dict,
    seed: int,
    size: int,
) -> CandidatePool:
    rng = np.random.default_rng(seed)
    active = state.names(KnobState.ACTIVE)
    watch = state.names(KnobState.WATCH)
    challenged = state.challenge_names()
    counts = _allocate(size, (0.35, 0.25, 0.20, 0.10, 0.10))
    tagged: list[tuple[dict, str]] = []
    tagged.extend((item, "llm_recommended") for item in _cognitive_samples(space, incumbent, active, value_stats, rng, counts[0]))
    tagged.extend((item, "local_perturbation") for item in local_mutations(space, incumbent, active, rng, counts[1]))
    tagged.extend((item, "coordination_block") for item in _block_samples(space, incumbent, blocks, rng, counts[2]))
    tagged.extend((item, "active_global_random") for item in _active_iid_samples(space, incumbent, active, seed + 7_919, counts[3]))
    challenge_names = challenged or watch
    tagged.extend((item, "challenge") for item in local_mutations(space, incumbent, challenge_names, rng, counts[4]))

    result, sources, local_seen = [], [], set()
    for candidate, source in tagged:
        repaired = space.repair(candidate)
        key = space.config_key(repaired)
        if key not in seen and key not in local_seen and space.validate(repaired):
            result.append(repaired)
            sources.append(source)
            local_seen.add(key)
    if len(result) < max(1, size // 5):
        fallback = deduplicate_legal(space, space.sample_iid(seed + 2_000_003, size), seen | local_seen)
        result.extend(fallback)
        sources.extend(["iid_fallback"] * len(fallback))
    return CandidatePool(result, sources)


def one_knob_probes(space: SearchSpace, incumbent: dict, names: list[str], seed: int, count: int = 4) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    return {name: local_mutations(space, incumbent, [name], rng, count, 1) for name in names}


def _allocate(total: int, ratios: tuple[float, ...]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    for index in np.argsort([value - int(value) for value in raw])[::-1][: total - sum(counts)]:
        counts[int(index)] += 1
    return counts


def _cognitive_samples(space, incumbent, names, stats, rng, count):
    candidates = []
    usable = [name for name in names if name in stats]
    if not usable:
        return local_mutations(space, incumbent, names, rng, count)
    for _ in range(count):
        candidate = dict(incumbent)
        name = usable[int(rng.integers(0, len(usable)))]
        item = stats[name]
        index = int(rng.choice(len(item["grid"]), p=np.asarray(item["distribution"]) / np.sum(item["distribution"])))
        candidate[name] = item["grid"][index]
        candidates.append(space.repair(candidate))
    return candidates


def _active_iid_samples(space, incumbent, active, seed, count):
    """IID samples over ACTIVE knobs, with non-active knobs pinned."""
    candidates = []
    for config in space.sample_iid(seed, count):
        merged = dict(incumbent)
        for name in active:
            if name in config:
                merged[name] = config[name]
        candidates.append(space.repair(merged))
    return candidates


def _block_samples(space, incumbent, blocks, rng, count):
    useful = [block for block in blocks if len(block) > 1]
    if not useful:
        useful = [tuple(space.by_name)]
    candidates = []
    for _ in range(count):
        block = useful[int(rng.integers(0, len(useful)))]
        candidates.extend(local_mutations(space, incumbent, list(block), rng, 1, max_changes=min(3, len(block))))
    return candidates
