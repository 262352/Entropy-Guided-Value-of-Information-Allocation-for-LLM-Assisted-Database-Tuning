#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--space", default="configs/knobs/postgresql_60_live.yaml")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    diagnostics = [json.loads(line) for line in (run_dir / "round_diagnostics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    space = yaml.safe_load(Path(args.space).read_text(encoding="utf-8"))["knobs"]
    memory = {knob["name"] for knob in space if knob.get("category") == "memory"}
    active_counts = [row["active_count"] for row in diagnostics]
    active_sets = {tuple(row["active_knobs"]) for row in diagnostics}
    knob_active = Counter(name for row in diagnostics for name in row["active_knobs"])
    memory_active = {name: knob_active.get(name, 0) / len(diagnostics) for name in sorted(memory)}
    expert_names = sorted(diagnostics[-1]["expert_weights"]) if diagnostics else []
    expert_weights = {
        expert: [round(row["expert_weights"].get(expert, 0.0), 4) for row in diagnostics]
        for expert in expert_names
    }
    gates = [value for row in diagnostics for value in row["gate"].values()]
    sources = Counter()
    for row in diagnostics:
        for source, count in (row.get("sources") or {}).items():
            sources[source] += count
    report = {
        "run_dir": str(run_dir),
        "rounds": len(diagnostics),
        "active_count_min": min(active_counts) if active_counts else 0,
        "active_count_max": max(active_counts) if active_counts else 0,
        "active_count_mean": round(sum(active_counts) / len(active_counts), 3) if active_counts else 0,
        "unique_active_sets": len(active_sets),
        "memory_active_fraction": memory_active,
        "missing_memory_knobs": [name for name, value in memory_active.items() if value == 0],
        "knob_active_fraction": {name: round(count / len(diagnostics), 3) for name, count in knob_active.most_common()},
        "expert_weights": expert_weights,
        "expert_weight_range": {
            expert: [min(values), max(values)] for expert, values in expert_weights.items()
        },
        "gate_min": min(gates) if gates else 0,
        "gate_max": max(gates) if gates else 0,
        "sources": dict(sources),
        "last_active": diagnostics[-1]["active_knobs"] if diagnostics else [],
    }
    output = Path(args.output) if args.output else run_dir / "component_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
