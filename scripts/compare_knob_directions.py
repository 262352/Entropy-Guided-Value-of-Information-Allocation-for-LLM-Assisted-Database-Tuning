#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def best_config(run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    best = next(record for record in records if record["trial"] == summary["best_trial"])
    return best["config"], summary["best_trial"], summary["best_raw_metric"]


def position(knob, value):
    if knob["type"] in {"integer", "float"}:
        lower, upper = float(knob["lower"]), float(knob["upper"])
        if knob.get("log"):
            import math
            return (math.log(float(value)) - math.log(lower)) / (math.log(upper) - math.log(lower))
        return (float(value) - lower) / (upper - lower)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True, help="run dir with summary.json/trials.jsonl")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--space", default="configs/knobs/postgresql_60_live.yaml")
    parser.add_argument("--output", default="artifacts/analysis/knob_direction_compare.csv")
    args = parser.parse_args()
    knobs = {knob["name"]: knob for knob in yaml.safe_load(Path(args.space).read_text(encoding="utf-8"))["knobs"]}
    ours, ours_trial, ours_tps = best_config(Path(args.ours))
    baseline, baseline_trial, baseline_tps = best_config(Path(args.baseline))
    rows = []
    for name, knob in knobs.items():
        default = knob["default"]
        ours_value = ours.get(name, default)
        baseline_value = baseline.get(name, default)
        ours_pos = position(knob, ours_value)
        baseline_pos = position(knob, baseline_value)
        rows.append({
            "knob": name,
            "default": default,
            "ours": ours_value,
            "baseline": baseline_value,
            "ours_position": "" if ours_pos is None else round(ours_pos, 6),
            "baseline_position": "" if baseline_pos is None else round(baseline_pos, 6),
            "ours_delta": "" if ours_pos is None else round(ours_pos - (position(knob, default) or 0), 6),
            "baseline_delta": "" if baseline_pos is None else round(baseline_pos - (position(knob, default) or 0), 6),
            "position_gap": "" if ours_pos is None or baseline_pos is None else round(ours_pos - baseline_pos, 6),
            "ours_changed": ours_value != default,
            "baseline_changed": baseline_value != default,
        })
    rows.sort(key=lambda row: -abs(float(row["position_gap"])) if row["position_gap"] != "" else 0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"ours: trial={ours_trial} tps={ours_tps:.1f}")
    print(f"baseline: trial={baseline_trial} tps={baseline_tps:.1f}")
    print("largest knob position gaps:")
    for row in rows[:15]:
        print(f"  {row['knob']:38s} ours={row['ours']} baseline={row['baseline']} gap={row['position_gap']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
