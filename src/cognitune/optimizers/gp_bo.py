from __future__ import annotations

from pathlib import Path

import numpy as np

from cognitune.candidates import generate_gpbo_pool
from cognitune.evaluator import Evaluator
from cognitune.experiment.recorder import RunRecorder
from cognitune.search_space.space import SearchSpace
from cognitune.surrogate import GPPerformanceSurrogate
from .common import evaluate_and_record


def run_gp_bo(
    space: SearchSpace,
    evaluator: Evaluator,
    initial_configs: list[dict],
    budget: int,
    seed: int,
    candidate_pool_size: int,
    run_dir: str | Path,
    duplicate_new_best: bool = True,
) -> dict:
    recorder = RunRecorder(run_dir)
    configs, utilities = [], []
    best_utility = -float("inf")
    for trial, config in enumerate(initial_configs):
        best_utility, result = evaluate_and_record(
            evaluator, recorder, "gp_bo", seed, trial, config, "shared_initial", best_utility,
            budget=budget,
        )
        configs.append(config)
        utilities.append(result.utility)
    surrogate = GPPerformanceSurrogate(space)
    while len(configs) < budget:
        surrogate.fit(configs, utilities)
        incumbent = configs[int(np.argmax(utilities))]
        seen = {space.config_key(config) for config in configs}
        trial = len(configs)
        pool = generate_gpbo_pool(space, incumbent, seen, seed * 100_003 + trial, candidate_pool_size)
        scores = surrogate.log_expected_improvement(pool)
        selected_index = int(np.argmax(scores))
        selected = pool[selected_index]
        prediction = surrogate.posterior([selected])
        best_utility, result = evaluate_and_record(
            evaluator, recorder, "gp_bo", seed, trial, selected, "gp_logei", best_utility,
            budget=budget, duplicate_new_best=duplicate_new_best, acquisition_value=float(scores[selected_index]), posterior_mean=float(prediction.mean[0]),
            posterior_std=float(prediction.std[0]),
        )
        configs.append(selected)
        utilities.append(result.utility)
    return recorder.finalize()

