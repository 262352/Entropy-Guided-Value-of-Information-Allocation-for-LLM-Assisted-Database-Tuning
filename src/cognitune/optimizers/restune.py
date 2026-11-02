from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from cognitune.candidates import generate_gpbo_pool
from cognitune.evaluator import Evaluator
from cognitune.experiment.recorder import RunRecorder
from cognitune.search_space.space import SearchSpace
from cognitune.surrogate import GPPerformanceSurrogate
from .common import evaluate_and_record


def run_restune(
    space: SearchSpace,
    evaluator: Evaluator,
    initial_configs: list[dict],
    budget: int,
    seed: int,
    candidate_pool_size: int,
    run_dir: str | Path,
    meta_dir: str | Path | None = None,
    duplicate_new_best: bool = True,
) -> dict:
    """ResTune-style weighted GP ensemble with meta-learned source tasks.

    The target GP is the benchmark's BoTorch surrogate. Each previously tuned
    workload contributes a source GP; source weights are the positive rank
    correlation between source predictions and target observations, while the
    target model keeps weight one. Means and variances are combined as a
    Gaussian mixture before expected-improvement selection. The first workload
    therefore reduces to GP-BO (ResTune's cold start), later workloads reuse
    earlier tasks as meta-data.
    """
    recorder = RunRecorder(run_dir)
    configs, utilities = [], []
    best_utility = -float("inf")
    for trial, config in enumerate(initial_configs):
        best_utility, result = evaluate_and_record(
            evaluator, recorder, "restune", seed, trial, config, "shared_initial", best_utility,
            budget=budget,
        )
        configs.append(config)
        utilities.append(result.utility)

    source_models = _load_sources(space, Path(meta_dir) if meta_dir else None)
    surrogate = GPPerformanceSurrogate(space)
    while len(configs) < budget:
        trial = len(configs)
        incumbent = configs[int(np.argmax(utilities))]
        seen = {space.config_key(config) for config in configs}
        pool = generate_gpbo_pool(space, incumbent, seen, seed * 100_003 + trial, candidate_pool_size)
        surrogate.fit(configs, utilities)
        mean, std = _ensemble_posterior(space, surrogate, source_models, configs, utilities, pool)
        scores = _expected_improvement(mean, std, float(max(utilities)))
        selected_index = int(np.argmax(scores))
        selected = pool[selected_index]
        best_utility, result = evaluate_and_record(
            evaluator, recorder, "restune", seed, trial, selected, "restune_wgpe", best_utility,
            budget=budget, duplicate_new_best=duplicate_new_best,
            acquisition_value=float(scores[selected_index]),
        )
        configs.append(selected)
        utilities.append(result.utility)
    return recorder.finalize()


def _load_sources(space: SearchSpace, meta_dir: Path | None) -> list[tuple[GaussianProcessRegressor, str]]:
    if meta_dir is None or not meta_dir.is_dir():
        return []
    models = []
    for path in sorted(meta_dir.glob("*.json")):
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            configs = [space.repair(item["config"]) for item in records if item.get("status", "SUCCESS") == "SUCCESS"]
            utilities = [float(item["utility"]) for item in records if item.get("status", "SUCCESS") == "SUCCESS"]
        except (ValueError, KeyError, TypeError):
            continue
        if len(configs) < 3:
            continue
        x = np.asarray(space.encode_many(configs), dtype=np.float64)
        y = np.asarray(utilities, dtype=np.float64)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(x.shape[1]), nu=2.5) + WhiteKernel(1e-6, (1e-8, 1e-2))
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6, n_restarts_optimizer=0)
        try:
            model.fit(x, y)
        except Exception:
            continue
        models.append((model, path.stem))
    return models


def _ensemble_posterior(space, surrogate, source_models, configs, utilities, pool):
    target = surrogate.posterior(pool)
    means = [target.mean]
    variances = [np.maximum(target.std, 1e-9) ** 2]
    weights = [1.0]
    if source_models:
        current_x = np.asarray(space.encode_many(configs), dtype=np.float64)
        current_y = np.asarray(utilities, dtype=np.float64)
        pool_x = np.asarray(space.encode_many(pool), dtype=np.float64)
        for model, _ in source_models:
            try:
                fitted = model.predict(current_x, return_std=False)
                correlation = spearmanr(fitted, current_y).statistic
                weight = float(correlation) if np.isfinite(correlation) else 0.0
                if weight <= 0:
                    continue
                mean, std = model.predict(pool_x, return_std=True)
            except Exception:
                continue
            means.append(np.asarray(mean, dtype=np.float64))
            variances.append(np.maximum(np.asarray(std, dtype=np.float64), 1e-9) ** 2)
            weights.append(weight)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    stacked_means = np.stack(means, axis=0)
    stacked_second = np.stack([var + mean ** 2 for mean, var in zip(means, variances)], axis=0)
    mean = np.sum(weights[:, None] * stacked_means, axis=0)
    variance = np.sum(weights[:, None] * stacked_second, axis=0) - mean ** 2
    return mean, np.sqrt(np.maximum(variance, 1e-12))


def _expected_improvement(mean, std, best):
    z = (mean - best) / std
    return (mean - best) * norm.cdf(z) + std * norm.pdf(z)
