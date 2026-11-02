#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from cognitune.experiment.aggregate import aggregate_runs
from cognitune.experiment.runner import PROJECT_ROOT, run_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["cognitune", "smac", "gp_bo"])
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--candidate-pool-size", type=int, default=256)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--include-no-relation", action="store_true")
    args = parser.parse_args()
    source = PROJECT_ROOT / "configs/experiments/synthetic.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["experiment"]["budget"] = args.budget
    data["experiment"]["candidate_pool_size"] = args.candidate_pool_size
    generated = PROJECT_ROOT / args.artifacts / "synthetic_runtime.yaml"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    for seed in args.seeds:
        methods = list(args.methods)
        random.Random(seed).shuffle(methods)
        for method in methods:
            run_method(generated, method, seed, args.artifacts)
        if args.include_no_relation:
            run_method(generated, "cognitune", seed, args.artifacts, disable_relation_graph=True)
    aggregate_runs(PROJECT_ROOT / args.artifacts / "runs/synthetic_smoke", PROJECT_ROOT / args.artifacts / "summaries/synthetic_smoke")


if __name__ == "__main__":
    main()
