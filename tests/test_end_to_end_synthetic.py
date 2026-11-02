import json
from pathlib import Path

import yaml

from cognitune.experiment.aggregate import aggregate_runs
from cognitune.experiment.runner import run_method


ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_twenty_trials_and_relation_ablation(tmp_path):
    data = yaml.safe_load((ROOT / "configs/experiments/synthetic.yaml").read_text())
    data["experiment"]["space"] = str(ROOT / "configs/knobs/synthetic.yaml")
    data["experiment"]["budget"] = 20
    data["experiment"]["candidate_pool_size"] = 32
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(yaml.safe_dump(data, sort_keys=False))
    summaries = {}
    for method in ("cognitune", "smac", "gp_bo"):
        summaries[method] = run_method(experiment, method, 0, tmp_path / "artifacts")
    summaries["cognitune_no_relation"] = run_method(
        experiment, "cognitune", 0, tmp_path / "artifacts", disable_relation_graph=True,
    )
    assert all(summary["trials"] == 20 for summary in summaries.values())
    run_root = tmp_path / "artifacts/runs/synthetic_smoke"
    output = tmp_path / "summary"
    aggregate_runs(run_root, output)
    assert (output / "best_so_far.png").exists()
    initial_sequences = []
    for method in ("cognitune", "smac", "gp_bo"):
        trace = run_root / method / "seed_0/trials.jsonl"
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        initial_sequences.append([(row["config"], row["utility"]) for row in rows[:5]])
    assert initial_sequences[0] == initial_sequences[1] == initial_sequences[2]

