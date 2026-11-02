#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import random
from pathlib import Path

from cognitune.config import load_experiment
from cognitune.experiment.aggregate import aggregate_runs
from cognitune.experiment.runner import PROJECT_ROOT, run_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="configs/experiments/postgres_tpcc_deepseek.yaml")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--lock", default="/root/GPTuner/.runtime/baseline-run.lock")
    args = parser.parse_args()
    experiment = PROJECT_ROOT / args.experiment
    config = load_experiment(experiment)
    lock_path = Path(args.lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another PostgreSQL tuning experiment owns the shared lock") from error
        for seed in config.seeds:
            methods = list(config.methods)
            random.Random(seed).shuffle(methods)
            for method in methods:
                trace = PROJECT_ROOT / args.artifacts / "runs" / config.name / method / f"seed_{seed}" / "trials.jsonl"
                if trace.exists() and len(trace.read_text(encoding="utf-8").splitlines()) == config.budget:
                    continue
                print(f"=== experiment={config.name} method={method} seed={seed} start ===", flush=True)
                summary = run_method(experiment, method, seed, args.artifacts)
                best = summary.get("best_raw_metric")
                best_text = f"{float(best):.3f}" if isinstance(best, (int, float)) else "n/a"
                print(f"=== experiment={config.name} method={method} seed={seed} done best_TPS={best_text} ===", flush=True)
        run_root = PROJECT_ROOT / args.artifacts / "runs" / config.name
        summary_root = PROJECT_ROOT / args.artifacts / "summaries" / config.name
        print(json.dumps(aggregate_runs(run_root, summary_root), indent=2), flush=True)


if __name__ == "__main__":
    main()
