from cognitune.evaluator import SyntheticEvaluator
from cognitune.optimizers import run_gp_bo
from cognitune.search_space.initial_design import build_initial_design


def test_gp_bo_writes_budgeted_trace(space, tmp_path):
    initial = build_initial_design(space, 0, 5)["configs"]
    summary = run_gp_bo(space, SyntheticEvaluator(space, 0), initial, 6, 0, 24, tmp_path)
    assert summary["trials"] == 6
    assert len((tmp_path / "trials.jsonl").read_text().splitlines()) == 6

