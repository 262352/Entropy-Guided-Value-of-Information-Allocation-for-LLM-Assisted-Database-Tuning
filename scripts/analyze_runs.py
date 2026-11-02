#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


def parse_retries(stdout_path: Path) -> dict:
    if not stdout_path.is_file():
        return {}
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    result = {}
    for label in ("Completed Transactions", "Aborted Transactions", "Rejected Transactions (Server Retry)"):
        marker = f"{label}:"
        if marker not in text:
            continue
        block = text.split(marker, 1)[1].split("\n\n", 1)[0]
        result[label] = sum(int(value) for value in re.findall(r"\[\s*(\d+)\]", block))
    return result


def best_config_diff(config: dict, defaults: dict) -> dict:
    diff = {}
    for name, value in config.items():
        default = defaults.get(name)
        if value == default:
            continue
        entry = {"default": default, "value": value}
        try:
            entry["relative"] = abs(float(value) - float(default)) / max(abs(float(default)), 1e-9)
        except (TypeError, ValueError):
            pass
        diff[name] = entry
    return diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="workload run directory containing <method>/seed_<n>")
    parser.add_argument("--space", default="configs/knobs/postgresql_60_live.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    defaults = {
        knob["name"]: knob["default"]
        for knob in yaml.safe_load(Path(args.space).read_text(encoding="utf-8"))["knobs"]
    }
    report = {}
    for seed_dir in sorted(Path(args.runs).glob("*/seed_*")):
        method, seed = seed_dir.parent.name, seed_dir.name
        trace = seed_dir / "trials.jsonl"
        if not trace.is_file():
            continue
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
        best_goodput = best_throughput = None
        for record in records:
            summary = (record.get("metadata") or {}).get("benchbase_summary")
            if not summary or not Path(summary).is_file():
                continue
            data = json.loads(Path(summary).read_text(encoding="utf-8"))
            goodput = data.get("Goodput (requests/second)")
            throughput = data.get("Throughput (requests/second)")
            candidate = {
                "trial": record["trial"], "status": record["status"],
                "goodput": goodput, "throughput": throughput,
                "config": record["config"], "summary": summary,
                "retries": parse_retries(Path(summary).parent / "benchbase.stdout.log"),
            }
            if isinstance(goodput, (int, float)) and (best_goodput is None or goodput > best_goodput["goodput"]):
                best_goodput = candidate
            if isinstance(throughput, (int, float)) and (best_throughput is None or throughput > best_throughput["throughput"]):
                best_throughput = candidate
        report[f"{method}/{seed}"] = {
            "best_goodput": best_goodput,
            "best_throughput": best_throughput,
            "best_goodput_config_diff": best_config_diff(best_goodput["config"], defaults) if best_goodput else {},
            "best_throughput_config_diff": best_config_diff(best_throughput["config"], defaults) if best_throughput else {},
        }
        print(
            f"{method:10s} {seed:7s} "
            f"best_goodput={best_goodput['goodput']:.1f}@{best_goodput['trial']} "
            f"({best_goodput['retries'].get('Rejected Transactions (Server Retry)', 0)} retries) | "
            f"best_throughput={best_throughput['throughput']:.1f}@{best_throughput['trial']} "
            f"(goodput={best_throughput['goodput']:.1f}, "
            f"{best_throughput['retries'].get('Rejected Transactions (Server Retry)', 0)} retries)"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
