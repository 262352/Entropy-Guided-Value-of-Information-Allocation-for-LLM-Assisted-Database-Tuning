from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitune.evaluator import SyntheticEvaluator
from cognitune.evaluator.base import EvaluationResult
from cognitune.experiment.recorder import RunRecorder
from cognitune.optimizers import run_cdbtune, run_dbtune, run_ottertune, run_restune
from cognitune.optimizers.common import evaluate_and_record
from cognitune.search_space.initial_design import build_initial_design


@pytest.mark.parametrize("optimizer", [run_ottertune, run_dbtune, run_cdbtune, run_restune])
def test_baseline_optimizers_complete_the_budget(space, tmp_path, optimizer):
    initial = build_initial_design(space, 0, 3)["configs"]
    evaluator = SyntheticEvaluator(space, 0, "throughput", 1000.0, 0.01)
    summary = optimizer(
        space, evaluator, initial, budget=7, seed=0,
        candidate_pool_size=64, run_dir=tmp_path / optimizer.__name__,
    )
    assert summary["trials"] == 7
    assert summary["best_raw_metric"] > 0
    assert (tmp_path / optimizer.__name__ / "trials.jsonl").exists()


def test_new_best_is_measured_twice_and_averaged(space, tmp_path):
    class CountingEvaluator:
        def __init__(self):
            self.calls = 0
            self.values = iter([100.0, 300.0, 200.0])

        def evaluate(self, config, trial):
            self.calls += 1
            return EvaluationResult(raw_metric=next(self.values), utility=float(self.calls), status="SUCCESS")

    evaluator = CountingEvaluator()
    recorder = RunRecorder(tmp_path)
    config = space.get_default()
    best = -float("inf")
    best, _ = evaluate_and_record(evaluator, recorder, "gp_bo", 0, 0, config, "shared_initial", best, budget=3, duplicate_new_best=True)
    assert evaluator.calls == 1
    best, result = evaluate_and_record(evaluator, recorder, "gp_bo", 0, 1, config, "gp_logei", best, budget=3, duplicate_new_best=True)
    assert evaluator.calls == 3
    assert result.raw_metric == pytest.approx(250.0)
    assert recorder.records[-1].metadata["duplicate_measurement"] == {
        "first_raw_metric": 300.0, "second_raw_metric": 200.0,
    }


def test_cognitune_style_calls_do_not_duplicate(space, tmp_path):
    class CountingEvaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, config, trial):
            self.calls += 1
            return EvaluationResult(raw_metric=100.0 * self.calls, utility=float(self.calls), status="SUCCESS")

    evaluator = CountingEvaluator()
    recorder = RunRecorder(tmp_path)
    best = -float("inf")
    config = space.get_default()
    for trial in range(3):
        best, _ = evaluate_and_record(evaluator, recorder, "cognitune", 0, trial, config, "shared_initial", best, budget=3)
    assert evaluator.calls == 3
