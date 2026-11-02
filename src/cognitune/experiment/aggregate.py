from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .metrics import best_so_far, first_target_trial
from .plots import plot_best_curves, plot_cognitive_diagnostics, plot_individual_curves


def aggregate_runs(input_dir: str | Path, output_dir: str | Path) -> dict:
    input_path, output_path = Path(input_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    curves: dict[str, list[list[float]]] = defaultdict(list)
    rows = []
    final_configs = []
    cognitive_records = None
    for trace in sorted(input_path.rglob("trials.jsonl")):
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
        if not records:
            continue
        method = records[0]["method"]
        curve = best_so_far([record["utility"] for record in records])
        curves[method].append(curve)
        final_configs.append({"method": method, "seed": records[0]["seed"], "config": max(records, key=lambda item: item["utility"])["config"]})
        if method == "cognitune" and cognitive_records is None:
            cognitive_records = records
        rows.append({
            "method": method, "seed": records[0]["seed"], "budget": len(records),
            "best_utility": curve[-1], "target_90_trial": "",
            **{f"best_at_{point}": curve[min(point, len(curve)) - 1] for point in (10, 20, 30, 50)},
        })
    if not rows:
        raise ValueError(f"no trials.jsonl found under {input_path}")
    reference = max(row["best_utility"] for row in rows)
    for row in rows:
        method_curves = curves[row["method"]]
        matching = next(curve for curve in method_curves if curve[-1] == row["best_utility"])
        row["target_90_trial"] = first_target_trial(matching, 0.9 * reference) or ""
    fields = list(rows[0])
    with (output_path / "fixed_budget.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"reference_best": reference, "runs": rows}
    (output_path / "aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_path / "final_configurations.json").write_text(json.dumps(final_configs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_best_curves(dict(curves), output_path / "best_so_far.png")
    plot_individual_curves(dict(curves), output_path / "per_seed_best_so_far.png")
    if cognitive_records:
        plot_cognitive_diagnostics(cognitive_records, output_path)
    return summary
