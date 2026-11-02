from __future__ import annotations

import pytest

from cognitune.evaluator import SyntheticEvaluator
from cognitune.llm.mock import MockLLMClient
from cognitune.optimizers import run_cognitune_v2, run_mctuner
from cognitune.search_space.initial_design import build_initial_design


@pytest.mark.parametrize("optimizer", [run_cognitune_v2, run_mctuner])
def test_v2_and_mctuner_complete_the_budget(space, tmp_path, optimizer):
    initial = build_initial_design(space, 0, 3)["configs"]
    evaluator = SyntheticEvaluator(space, 0, "throughput", 1000.0, 0.01)
    client = MockLLMClient(space)
    summary = optimizer(
        space, evaluator, initial, budget=6, seed=0,
        candidate_pool_size=32, run_dir=tmp_path / optimizer.__name__,
        llm_client=client,
    )
    assert summary["trials"] == 6
    assert (tmp_path / optimizer.__name__ / "summary.json").exists()
    assert (tmp_path / optimizer.__name__ / "round_diagnostics.jsonl").exists()
