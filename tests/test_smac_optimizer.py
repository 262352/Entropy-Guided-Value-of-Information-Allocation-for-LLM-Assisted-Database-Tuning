import json

from cognitune.evaluator import SyntheticEvaluator
from cognitune.optimizers import run_smac
from cognitune.search_space.initial_design import build_initial_design


def test_smac_ask_tell_preserves_supplied_initial_order(space, tmp_path):
    initial = build_initial_design(space, 2, 5)["configs"]
    summary = run_smac(space, SyntheticEvaluator(space, 2), initial, 6, 2, tmp_path)
    records = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert summary["trials"] == 6
    assert [space.config_key(item["config"]) for item in records[:5]] == [space.config_key(item) for item in initial]

