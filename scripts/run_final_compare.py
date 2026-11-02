#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import statistics
from pathlib import Path

from cognitune.experiment.runner import resolve_project_path, run_method


def main():
    parser = argparse.ArgumentParser(description="Run only CogniTune-v2 and MCTuner against existing Goodput results")
    parser.add_argument("--workloads", nargs="+", default=["tpcc", "ycsb"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--lock", default="/root/GPTuner/.runtime/baseline-run.lock")
    args = parser.parse_args()
    lock_path = Path(args.lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another PostgreSQL tuning experiment owns the shared lock") from error
    artifacts = resolve_project_path(args.artifacts)
    output_dir = artifacts / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    for workload in args.workloads:
        runtime = artifacts / "suite" / f"{workload}.yaml"
        if not runtime.exists():
            raise FileNotFoundError(f"missing runtime config {runtime}")
        run_root = artifacts / "runs" / f"postgres_{workload}_compare"
        existing = _existing_results(run_root, exclude={"cognitune_v2", "mctuner"})
        print(f"=== workload={workload} existing_best={existing['best']:.3f} ({existing['best_method']}@{existing['best_trial']}) ===", flush=True)
        rows = []
        for method in ("cognitune_v2", "mctuner"):
            run_dir = run_root / method / f"seed_{args.seed}"
            summary_path = run_dir / "summary.json"
            if summary_path.exists() and (run_dir / "final_validation.json").exists() and not args.force:
                print(f"=== {workload}/{method} skipped (complete) ===", flush=True)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                print(f"=== {workload}/{method} start ===", flush=True)
                summary = run_method(runtime, method, args.seed, args.artifacts)
                print(f"=== {workload}/{method} done ===", flush=True)
            row = _row(workload, method, run_dir, summary)
            rows.append(row)
            print(
                f"[{workload}/{method}] best_trial={row['best_trial']:.3f}@{row['best_trial_index']} "
                f"validation_median={row['validation_median']:.3f} values={row['validation_values']}",
                flush=True,
            )
        best_before = max([existing["best"]] + [row["best_trial"] for row in rows if row["method"] == "cognitune_v2"])
        mctuner = next(row for row in rows if row["method"] == "mctuner")
        mctuner["new_best"] = mctuner["best_trial"] > best_before
        mctuner["true_value"] = mctuner["validation_median"] if mctuner["new_best"] else mctuner["best_trial"]
        for row in rows:
            if row["method"] == "cognitune_v2":
                row["true_value"] = row["validation_median"]
                row["new_best"] = row["best_trial"] > existing["best"]
        payload = {"workload": workload, "existing": existing, "methods": rows}
        (output_dir / f"final_compare_{workload}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        csv_path = output_dir / f"final_compare_{workload}.csv"
        with csv_path.open("w", encoding="utf-8") as stream:
            stream.write("method,best_trial,best_trial_index,validation_median,true_value,new_best,validation_values,llm_calls,llm_tokens\n")
            for row in rows:
                stream.write(
                    f"{row['method']},{row['best_trial']:.6f},{row['best_trial_index']},{row['validation_median']:.6f},"
                    f"{row['true_value']:.6f},{row['new_best']},\"{row['validation_values']}\","
                    f"{row['llm_calls']},{row['llm_tokens']}\n"
                )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def _existing_results(run_root: Path, exclude: set[str]) -> dict:
    best = -1.0
    best_method = None
    best_trial = None
    per_method = {}
    for trace in sorted(run_root.glob("*/seed_*/trials.jsonl")):
        method = trace.parent.parent.name
        if method in exclude:
            continue
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
        successful = [record for record in records if record["status"] == "SUCCESS"]
        if not successful:
            continue
        top = max(successful, key=lambda record: record["raw_metric"])
        per_method[method] = {"best": top["raw_metric"], "trial": top["trial"]}
        if top["raw_metric"] > best:
            best = top["raw_metric"]
            best_method = method
            best_trial = top["trial"]
    return {"best": best, "best_method": best_method, "best_trial": best_trial, "per_method": per_method}


def _row(workload: str, method: str, run_dir: Path, summary: dict) -> dict:
    validation = json.loads((run_dir / "final_validation.json").read_text(encoding="utf-8"))["results"]
    values = [float(item["raw_metric"]) for item in validation if item.get("status") == "SUCCESS"]
    median = statistics.median(values) if values else 0.0
    best_trial = float(summary.get("best_raw_metric", 0.0))
    return {
        "workload": workload,
        "method": method,
        "best_trial": best_trial,
        "best_trial_index": summary.get("best_trial"),
        "validation_median": median,
        "validation_values": [round(value, 3) for value in values],
        "llm_calls": summary.get("llm_calls"),
        "llm_tokens": summary.get("llm_tokens"),
    }


if __name__ == "__main__":
    main()
