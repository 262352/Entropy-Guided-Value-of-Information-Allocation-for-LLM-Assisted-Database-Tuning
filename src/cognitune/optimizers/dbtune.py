from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from cognitune.candidates.common import deduplicate_legal, local_mutations
from cognitune.evaluator import Evaluator
from cognitune.experiment.recorder import RunRecorder
from cognitune.search_space.space import SearchSpace
from .common import evaluate_and_record


def run_dbtune(
    space: SearchSpace,
    evaluator: Evaluator,
    initial_configs: list[dict],
    budget: int,
    seed: int,
    candidate_pool_size: int,
    run_dir: str | Path,
    duplicate_new_best: bool = True,
) -> dict:
    """DBTune-style BO using a probabilistic random-forest surrogate and EI.

    DBTune's automatic BO selection uses a probabilistic random forest plus
    expected improvement for spaces with at least ten hyperparameters (the
    regime of this 60-knob benchmark). The forest mean/std across trees form
    the predictive distribution; candidate generation uses the same finite
    pool protocol as the other BO baselines.
    """
    recorder = RunRecorder(run_dir)
    configs, utilities = [], []
    best_utility = -float("inf")
    for trial, config in enumerate(initial_configs):
        best_utility, result = evaluate_and_record(
            evaluator, recorder, "dbtune", seed, trial, config, "shared_initial", best_utility,
            budget=budget,
        )
        configs.append(config)
        utilities.append(result.utility)

    while len(configs) < budget:
        trial = len(configs)
        incumbent = configs[int(np.argmax(utilities))]
        seen = {space.config_key(config) for config in configs}
        pool = _local_first_pool(space, incumbent, seen, seed * 100_003 + trial, candidate_pool_size)
        mean, std = _forest_posterior(space, configs, utilities, pool, seed, trial)
        best = float(max(utilities))
        scores = _expected_improvement(mean, std, best)
        selected_index = int(np.argmax(scores))
        selected = pool[selected_index]
        best_utility, result = evaluate_and_record(
            evaluator, recorder, "dbtune", seed, trial, selected, "dbtune_prf_ei", best_utility,
            budget=budget, duplicate_new_best=duplicate_new_best,
            acquisition_value=float(scores[selected_index]),
        )
        configs.append(selected)
        utilities.append(result.utility)
    return recorder.finalize()


def _local_first_pool(space, incumbent, seen, seed, size):
    """DBTune's local-random acquisition optimizer, adapted to a finite pool."""
    rng = np.random.default_rng(seed)
    local_count = max(1, int(size * 0.7))
    candidates = local_mutations(space, incumbent, list(space.by_name), rng, local_count, max_changes=5)
    candidates.extend(space.sample_iid(seed, size - local_count))
    legal = deduplicate_legal(space, candidates, seen)
    if not legal:
        legal = deduplicate_legal(space, space.sample_iid(seed + 1_000_003, size), seen)
    return legal


def _forest_posterior(space, configs, utilities, pool, seed, trial):
    x = np.asarray(space.encode_many(configs), dtype=np.float64)
    y = np.asarray(utilities, dtype=np.float64)
    model = RandomForestRegressor(
        n_estimators=128, min_samples_leaf=1, max_features="sqrt",
        random_state=seed * 1_000_003 + trial, n_jobs=1,
    )
    model.fit(x, y)
    pool_x = np.asarray(space.encode_many(pool), dtype=np.float64)
    predictions = np.stack([tree.predict(pool_x) for tree in model.estimators_], axis=0)
    return predictions.mean(axis=0), predictions.std(axis=0) + 1e-9


def _expected_improvement(mean, std, best):
    z = (mean - best) / std
    return (mean - best) * norm.cdf(z) + std * norm.pdf(z)
