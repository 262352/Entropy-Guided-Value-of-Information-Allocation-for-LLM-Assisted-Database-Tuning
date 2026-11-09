import json

from cognitune.evaluator import SyntheticEvaluator
from cognitune.optimizers import run_cognitune_v2
from cognitune.search_space.initial_design import build_initial_design


def test_verbose_trace_does_not_change_search_and_exposes_decision_before_outcome(space, tmp_path, capsys):
    initial = build_initial_design(space, 0, 3)["configs"]
    traces = []
    for verbose in (False, True):
        folder = tmp_path / str(verbose)
        run_cognitune_v2(
            space, SyntheticEvaluator(space, 0), initial, budget=6, seed=0,
            candidate_pool_size=32, run_dir=folder, router_interval=3,
            verbose_diagnostics=verbose,
        )
        traces.append([json.loads(line) for line in (folder / "trials.jsonl").read_text().splitlines()])
    assert [(r["config"], r["utility"], r["candidate_source"]) for r in traces[0]] == [
        (r["config"], r["utility"], r["candidate_source"]) for r in traces[1]
    ]
    events = [json.loads(line) for line in (tmp_path / "True/debug_events.jsonl").read_text().splitlines()]
    for trial in range(3, 6):
        proposal_index = next(i for i, e in enumerate(events) if e["event"] == "proposal" and e["trial"] == trial)
        outcome_index = next(i for i, e in enumerate(events) if e["event"] == "outcome" and e["trial"] == trial)
        assert proposal_index < outcome_index
        proposal = events[proposal_index]
        row = traces[1][trial]
        anchor = max(traces[1][:trial], key=lambda r: r["utility"])["config"]
        assert set(proposal["changed_from_incumbent"]) == {
            name for name in row["config"] if row["config"][name] != anchor.get(name)
        }
        assert events[outcome_index]["actual_utility"] == row["utility"]
    output = capsys.readouterr().out
    assert "router_refresh_start" in output
    assert "router_refresh_done" in output
    assert "posterior_std" in output
