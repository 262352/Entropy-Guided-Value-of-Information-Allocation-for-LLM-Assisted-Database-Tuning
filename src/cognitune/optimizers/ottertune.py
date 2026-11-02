from __future__ import annotations

from pathlib import Path

import numpy as np

from cognitune.candidates import generate_gpbo_pool
from cognitune.candidates.common import deduplicate_legal, local_mutations
from cognitune.evaluator import Evaluator
from cognitune.experiment.recorder import RunRecorder
from cognitune.search_space.space import SearchSpace
from cognitune.surrogate import GPPerformanceSurrogate
from .common import evaluate_and_record


def run_ottertune(
    space: SearchSpace,
    evaluator: Evaluator,
    initial_configs: list[dict],
    budget: int,
    seed: int,
    candidate_pool_size: int,
    run_dir: str | Path,
    duplicate_new_best: bool = True,
    top_k: int = 10,
) -> dict:
    """OtterTune-style GP-BO with importance-based knob selection.

    OtterTune fits a GP and restricts exploration to the most influential
    knobs. Here the importance ranks come from the fitted ARD lengthscales
    instead of OtterTune's factor-analysis/k-means pipeline; the rest of the
    loop (shared random design, LogEI over a finite pool) matches this
    benchmark's protocol.
    """
    recorder = RunRecorder(run_dir)
    configs, utilities = [], []
    best_utility = -float("inf")
    for trial, config in enumerate(initial_configs):
        best_utility, result = evaluate_and_record(
            evaluator, recorder, "ottertune", seed, trial, config, "shared_initial", best_utility,
            budget=budget,
        )
        configs.append(config)
        utilities.append(result.utility)

    surrogate = GPPerformanceSurrogate(space)
    while len(configs) < budget:
        trial = len(configs)
        incumbent = configs[int(np.argmax(utilities))]
        surrogate.fit(configs, utilities)
        selected_knobs = _rank_knobs(space, surrogate, incumbent, top_k)
        seen = {space.config_key(config) for config in configs}
        pool = _restricted_pool(space, incumbent, seen, selected_knobs, seed * 100_003 + trial, candidate_pool_size)
        if not pool:
            pool = generate_gpbo_pool(space, incumbent, seen, seed * 100_003 + trial, candidate_pool_size)
        scores = surrogate.log_expected_improvement(pool)
        selected_index = int(np.argmax(scores))
        selected = pool[selected_index]
        best_utility, result = evaluate_and_record(
            evaluator, recorder, "ottertune", seed, trial, selected, "ottertune_logei", best_utility,
            budget=budget, duplicate_new_best=duplicate_new_best,
            acquisition_value=float(scores[selected_index]),
        )
        configs.append(selected)
        utilities.append(result.utility)
    return recorder.finalize()


def _rank_knobs(space: SearchSpace, surrogate: GPPerformanceSurrogate, incumbent: dict, top_k: int) -> list[str]:
    active = [name for name in space.by_name if name in space.get_active_knobs(incumbent)]
    if top_k >= len(active):
        return active
    if surrogate.model is None:
        rng = np.random.default_rng(0)
        return list(rng.choice(active, size=top_k, replace=False))
    try:
        lengthscale = surrogate.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
    except Exception:
        return active[:top_k]
    importance = {}
    for node in space.encoder.nodes:
        if node.knob.name not in active:
            continue
        values = lengthscale[node.value_start:node.value_stop]
        if len(values):
            importance[node.knob.name] = float(np.max(1.0 / np.maximum(values, 1e-8)))
    ranked = sorted(importance, key=lambda name: importance[name], reverse=True)
    for name in active:
        if name not in ranked:
            ranked.append(name)
    return ranked[:top_k]


def _restricted_pool(
    space: SearchSpace,
    incumbent: dict,
    seen: set[str],
    knobs: list[str],
    seed: int,
    size: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    keep = set(knobs)
    random_count = max(1, int(size * 0.75))
    candidates = []
    for config in space.sample_iid(seed, random_count):
        merged = dict(config)
        for name, knob in space.by_name.items():
            if name not in keep:
                merged[name] = incumbent.get(name, knob.default)
        candidates.append(space.repair(merged))
    candidates.extend(local_mutations(space, incumbent, knobs, rng, max(1, size - random_count), max_changes=min(3, len(knobs))))
    return deduplicate_legal(space, candidates, seen)
