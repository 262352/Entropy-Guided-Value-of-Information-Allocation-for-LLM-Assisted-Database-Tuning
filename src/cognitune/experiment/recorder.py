from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrialRecord:
    trial: int
    method: str
    seed: int
    config: dict[str, Any]
    raw_metric: float
    utility: float
    best_utility: float
    best_so_far: float
    status: str
    candidate_source: str
    elapsed_seconds: float = 0.0
    acquisition_value: float | None = None
    posterior_mean: float | None = None
    posterior_std: float | None = None
    state: dict | None = None
    transitions: list[dict] | None = None
    priority_score: dict | None = None
    selection_entropy: dict | None = None
    value_entropy: dict | None = None
    relation_sensitivity: list | None = None
    soft_edges: list | None = None
    coordination_blocks: list | None = None
    knob_states: dict | None = None
    challenge_target: str | None = None
    challenge_reason: str | None = None
    relation_calibration: list[dict] | None = None
    metadata: dict | None = None


class RunRecorder:
    def __init__(self, run_dir: str | Path, *, resume: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trials.jsonl"
        self.knobs_path = self.run_dir / "knobs.jsonl"
        self.knobs_csv_path = self.run_dir / "knobs.csv"
        if resume and not self.trace_path.exists() and self.knobs_path.exists():
            raise ValueError("knob trace exists without trial trace; refusing to overwrite evidence")
        if resume and self.trace_path.exists():
            # Validate before opening any output for writing.
            self.records = [TrialRecord(**json.loads(line)) for line in self.trace_path.read_text().splitlines() if line.strip()]
            if [r.trial for r in self.records] != list(range(len(self.records))):
                raise ValueError("cannot resume a non-contiguous trial trace")
            if not self.knobs_path.exists():
                raise ValueError("cannot resume without the matching knobs trace")
            knobs = [json.loads(line) for line in self.knobs_path.read_text().splitlines() if line.strip()]
            if len(knobs) != len(self.records) or any(
                k.get("trial") != r.trial or k.get("config") != r.config or k.get("status") != r.status
                or k.get("raw_metric") != r.raw_metric for k, r in zip(knobs, self.records)
            ):
                raise ValueError("trial and knob traces disagree; refusing to overwrite evidence")
            return
        self.trace_path.unlink(missing_ok=True)
        self.knobs_path.unlink(missing_ok=True)
        self.knobs_csv_path.unlink(missing_ok=True)
        self.records: list[TrialRecord] = []

    def append(self, record: TrialRecord) -> None:
        self.records.append(record)
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        knob_record = {
            "trial": record.trial,
            "status": record.status,
            "raw_metric": record.raw_metric,
            "utility": record.utility,
            "best_so_far": record.best_so_far,
            "candidate_source": record.candidate_source,
            "config": record.config,
        }
        with self.knobs_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(knob_record, sort_keys=True) + "\n")

    def rewrite(self) -> None:
        self.trace_path.write_text(
            "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in self.records),
            encoding="utf-8",
        )

    def finalize(self) -> dict:
        if not self.records:
            raise RuntimeError("cannot finalize an empty run")
        best = max(self.records, key=lambda item: item.utility)
        summary = {
            "method": best.method,
            "seed": best.seed,
            "trials": len(self.records),
            "best_utility": best.utility,
            "best_raw_metric": best.raw_metric,
            "best_config": best.config,
            "best_trial": best.trial,
        }
        self._write_knobs_csv()
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    def _write_knobs_csv(self) -> None:
        knob_names = sorted({name for record in self.records for name in (record.config or {})})
        with self.knobs_csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "trial", "status", "raw_metric", "utility", "best_so_far",
                "candidate_source", *knob_names,
            ])
            for record in self.records:
                config = record.config or {}
                writer.writerow([
                    record.trial, record.status, record.raw_metric, record.utility,
                    record.best_so_far, record.candidate_source,
                    *[config.get(name, "") for name in knob_names],
                ])
