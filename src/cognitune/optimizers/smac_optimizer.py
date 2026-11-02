from __future__ import annotations

from pathlib import Path

from ConfigSpace import Configuration
from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design.default_design import DefaultInitialDesign
from smac.runhistory.dataclasses import TrialValue

from cognitune.evaluator import Evaluator
from cognitune.experiment.recorder import RunRecorder
from cognitune.search_space.space import SearchSpace
from .common import evaluate_and_record


def run_smac(
    space: SearchSpace,
    evaluator: Evaluator,
    initial_configs: list[dict],
    budget: int,
    seed: int,
    run_dir: str | Path,
    duplicate_new_best: bool = True,
) -> dict:
    run_dir = Path(run_dir)
    recorder = RunRecorder(run_dir)
    configspace = space.to_configspace(seed)
    scenario = Scenario(
        configspace, name=f"smac-seed-{seed}", output_directory=run_dir / "smac3",
        deterministic=True, n_trials=budget, seed=seed, use_default_config=False,
    )
    supplied = [Configuration(configspace, values=config) for config in initial_configs]
    initial_design = DefaultInitialDesign(scenario, n_configs=0, additional_configs=supplied, seed=seed)
    facade = HyperparameterOptimizationFacade(
        scenario, lambda config, seed=0: 0.0, initial_design=initial_design,
        overwrite=True, logging_level=False,
    )
    best_utility = -float("inf")
    for trial in range(budget):
        info = facade.ask()
        proposed = space.repair(dict(info.config))
        if trial < len(initial_configs) and space.config_key(proposed) != space.config_key(initial_configs[trial]):
            raise RuntimeError("SMAC did not preserve the shared initial design order")
        source = "shared_initial" if trial < len(initial_configs) else "smac_ask"
        best_utility, result = evaluate_and_record(
            evaluator, recorder, "smac", seed, trial, proposed, source, best_utility,
            budget=budget,
        )
        facade.tell(info, TrialValue(cost=-result.utility, time=result.elapsed_seconds))
    return recorder.finalize()

