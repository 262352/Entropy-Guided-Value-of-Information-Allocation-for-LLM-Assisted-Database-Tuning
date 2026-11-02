from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cognitune.agents import AgentCoordinator
from cognitune.agents.expert import MCTUNER_EXPERTS
from cognitune.candidates.common import deduplicate_legal, local_mutations
from cognitune.evaluator import Evaluator
from cognitune.experiment.recorder import RunRecorder
from cognitune.llm import LLMClient, MockLLMClient
from cognitune.search_space.space import SearchSpace
from cognitune.surrogate import GPPerformanceSurrogate
from cognitune.uncertainty import priority_statistics, value_statistics
from cognitune.uncertainty.value_entropy import recommendation_distribution, value_grid
from .common import evaluate_and_record


def run_mctuner(
    space: SearchSpace,
    evaluator: Evaluator,
    initial_configs: list[dict],
    budget: int,
    seed: int,
    candidate_pool_size: int,
    run_dir: str | Path,
    method_name: str = "mctuner",
    llm_client: LLMClient | None = None,
    cognitive_context: dict | None = None,
    knob_budget: int = 20,
) -> dict:
    """MCTuner-style baseline: seven experts prune knobs/ranges, then GP-BO.

    The baseline keeps MCTuner's LLM-guided knob selection and range pruning
    but uses this repository's finite-pool LogEI loop. It deliberately has no
    expert routing, entropy-gated active set, or lookahead acquisition.
    """
    run_dir = Path(run_dir)
    coordinator = AgentCoordinator(space, llm_client or MockLLMClient(space), run_dir / "llm")
    context = cognitive_context or {"seed": seed, "budget": budget}
    responses = coordinator.collect(context, experts=MCTUNER_EXPERTS, templates=("role_template",))
    priority = priority_statistics(responses, list(space.by_name))
    values = value_statistics(responses, space.knobs)
    protected = {term["knob"] for rule in space.hard_constraints for term in rule.get("terms", [])}
    selected = _select_knobs(space, priority, protected, knob_budget)
    grids, dists = _distributions(responses, space, selected)
    recorder = RunRecorder(run_dir)
    diagnostics_path = run_dir / "round_diagnostics.jsonl"
    diagnostics_path.unlink(missing_ok=True)
    configs, utilities = [], []
    best_utility = -float("inf")
    for trial, config in enumerate(initial_configs):
        best_utility, result = evaluate_and_record(
            evaluator, recorder, method_name, seed, trial, config, "shared_initial", best_utility,
            budget=budget,
        )
        _diag(diagnostics_path, trial, "shared_initial", result.utility, best_utility, selected)
        configs.append(config)
        utilities.append(result.utility)
    surrogate = GPPerformanceSurrogate(space)
    while len(configs) < budget:
        trial = len(configs)
        incumbent = configs[int(np.argmax(utilities))]
        seen = {space.config_key(config) for config in configs}
        pool = _pool(space, incumbent, seen, selected, grids, dists, seed * 100_003 + trial, candidate_pool_size)
        surrogate.fit(configs, utilities)
        scores = surrogate.log_expected_improvement(pool)
        selected_index = int(np.argmax(scores))
        best_utility, result = evaluate_and_record(
            evaluator, recorder, method_name, seed, trial, pool[selected_index], "mctuner_pruned_logei", best_utility,
            budget=budget, acquisition_value=float(scores[selected_index]),
        )
        _diag(diagnostics_path, trial, "mctuner_pruned_logei", result.utility, best_utility, selected)
        configs.append(pool[selected_index])
        utilities.append(result.utility)
    summary = recorder.finalize()
    llm_records = [json.loads(path.read_text()) for path in (run_dir / "llm").glob("*.json") if "_invalid_" not in path.name]
    summary.update({
        "llm_calls": len(responses),
        "llm_tokens": sum(item["input_tokens"] + item["output_tokens"] for item in llm_records),
        "selected_knobs": sorted(selected),
    })
    return summary


def _diag(path, trial, source, utility, best, selected):
    import json as _json
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_json.dumps({
            "trial": trial, "source": source, "utility": utility,
            "best_utility": best, "selected_knobs": sorted(selected),
        }, sort_keys=True) + "\n")


def _select_knobs(space, priority, protected, budget):
    ordered = sorted(space.by_name, key=lambda name: (-priority[name]["priority_score"], name))
    selected = set(protected)
    for name in ordered:
        if len(selected) >= budget:
            break
        selected.add(name)
    return selected


def _distributions(responses, space, selected):
    grids, dists = {}, {}
    for knob in space.knobs:
        grid = value_grid(knob)
        grids[knob.name] = grid
        arrays = [
            recommendation_distribution(knob, response.recommendations[knob.name], grid)
            for response in responses if knob.name in response.recommendations
        ]
        dists[knob.name] = np.mean(arrays, axis=0) if arrays else np.ones(len(grid)) / len(grid)
    return grids, dists


def _sample(rng, grid, distribution):
    probabilities = np.asarray(distribution, dtype=float)
    probabilities = probabilities / probabilities.sum()
    value = grid[int(rng.choice(len(grid), p=probabilities))]
    return value.item() if isinstance(value, np.generic) else value


def _pool(space, incumbent, seen, selected, grids, dists, seed, size):
    rng = np.random.default_rng(seed)
    active = sorted(selected)
    candidates = []
    for _ in range(int(size * 0.8)):
        config = dict(incumbent)
        for name in active:
            config[name] = _sample(rng, grids[name], dists[name])
        candidates.append(space.repair(config))
    candidates.extend(local_mutations(space, incumbent, active, rng, size - len(candidates), max_changes=3))
    legal = deduplicate_legal(space, candidates, seen)
    if not legal:
        legal = deduplicate_legal(space, space.sample_iid(seed + 1_000_003, size), seen)
    return legal
