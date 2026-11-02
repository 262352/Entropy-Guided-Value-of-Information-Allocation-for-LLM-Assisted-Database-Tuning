from __future__ import annotations

import json

from cognitune.experiment.recorder import RunRecorder, TrialRecord


def test_every_trial_is_written_with_knobs(tmp_path):
    recorder = RunRecorder(tmp_path)
    for trial, value in enumerate((100.0, 120.0)):
        recorder.append(TrialRecord(
            trial=trial, method="gp_bo", seed=0,
            config={"a": trial, "b": value}, raw_metric=value, utility=value / 100,
            best_utility=value / 100, best_so_far=value / 100, status="SUCCESS",
            candidate_source="test",
        ))
    recorder.finalize()
    lines = (tmp_path / "knobs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["config"] == {"a": 1, "b": 120.0}
    csv_lines = (tmp_path / "knobs.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].startswith("trial,status,raw_metric,utility,best_so_far,candidate_source")
    assert ",a,b" in csv_lines[0]
