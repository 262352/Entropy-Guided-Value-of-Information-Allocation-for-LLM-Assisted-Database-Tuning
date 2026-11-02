#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

from cognitune.config import load_yaml
from cognitune.environment import collect_environment_profile
from cognitune.experiment.runner import PROJECT_ROOT, resolve_project_path, run_method


def main():
    parser = argparse.ArgumentParser(description="Run the full workload-major comparison suite")
    parser.add_argument("--suite", default="configs/experiments/suite_postgres.yaml")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--workloads", nargs="+", default=None, help="override the workload list, e.g. tpcc ycsb")
    parser.add_argument("--force", action="store_true", help="rerun methods that already have a summary")
    parser.add_argument("--lock", default="/root/GPTuner/.runtime/baseline-run.lock")
    parser.add_argument("--dry-run", action="store_true", help="prepare configs and profiles without evaluating")
    args = parser.parse_args()

    lock_path = Path(args.lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another PostgreSQL tuning experiment owns the shared lock") from error

    suite_path = resolve_project_path(args.suite)
    suite = load_yaml(suite_path)["suite"]
    base = load_yaml(resolve_project_path(suite["base_experiment"]))["experiment"]
    artifacts = resolve_project_path(args.artifacts)
    profile_dir = resolve_project_path(suite.get("profile_dir", "artifacts/profiles"))
    meta_dir = resolve_project_path(suite.get("meta_dir", "artifacts/meta"))
    runtime_dir = artifacts / "suite"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if meta_dir.exists():
        shutil.rmtree(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    workloads = [str(item) for item in (args.workloads or suite["workloads"])]
    methods = [str(item) for item in suite["methods"]]
    seeds = [int(item) for item in suite["seeds"]]
    print(
        f"=== suite={suite.get('name', 'suite')} workloads={workloads} methods={methods} seeds={seeds} "
        f"budget={suite['budget']} ===",
        flush=True,
    )

    completed = []
    for workload in workloads:
        profile_path = _ensure_profile(workload, profile_dir, suite["legacy_config"])
        runtime_path = _write_runtime_config(
            runtime_dir, workload, suite, base, profile_path,
        )
        runtime_name = f"postgres_{workload}_compare"
        if args.dry_run:
            print(f"=== dry-run workload={workload} config={runtime_path} profile={profile_path} ===", flush=True)
            continue
        print(f"=== workload={workload} start ({len(methods)} methods x {len(seeds)} seed) ===", flush=True)
        for seed in seeds:
            order = list(methods)
            random.Random(seed).shuffle(order)
            print(f"=== workload={workload} seed={seed} method_order={order} ===", flush=True)
            for method in order:
                print(f"=== workload={workload} method={method} seed={seed} start ===", flush=True)
                run_path = artifacts / "runs" / runtime_name / method / f"seed_{seed}"
                summary_path = run_path / "summary.json"
                if summary_path.exists() and (run_path / "final_validation.json").exists() and not args.force:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    print(f"=== workload={workload} method={method} seed={seed} skipped (complete) ===", flush=True)
                else:
                    summary = run_method(runtime_path, method, seed, args.artifacts, meta_dir=meta_dir)
                completed.append({
                    "workload": workload, "method": method, "seed": seed,
                    "best_raw_metric": summary.get("best_raw_metric"),
                    "trials": summary.get("trials"),
                })
                print(f"=== workload={workload} method={method} seed={seed} done ===", flush=True)
        _write_meta(meta_dir, runtime_name, workload, seeds, artifacts)
        print(f"=== workload={workload} complete ===", flush=True)

    print(json.dumps(completed, indent=2, sort_keys=True), flush=True)


def _write_runtime_config(runtime_dir: Path, workload: str, suite: dict, base: dict, profile_path: Path) -> Path:
    try:
        profile_value = profile_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        profile_value = str(profile_path)
    runtime = {
        "experiment": {
            "name": f"postgres_{workload}_compare",
            "dbms": base["dbms"],
            "workload": workload,
            "objective": base.get("objective", "throughput"),
            "space": base["space"],
            "budget": int(suite["budget"]),
            "n_init": int(suite["n_init"]),
            "seeds": [int(seed) for seed in suite["seeds"]],
            "methods": [str(method) for method in suite["methods"]],
            "candidate_pool_size": int(suite.get("candidate_pool_size", base.get("candidate_pool_size", 10_000))),
            "final_validation_repeats": int(suite.get("final_validation_repeats", base.get("final_validation_repeats", 3))),
            "failure_penalty": float(base.get("failure_penalty", 1.0)),
            "initialization": {"method": "iid_random"},
            "cognitune": dict(base.get("cognitune", {})),
            "llm": dict(base.get("llm", {})) | {"environment_profile": profile_value},
            "evaluator": dict(base.get("evaluator", {})) | {"workload": workload},
        }
    }
    runtime_dir.mkdir(parents=True, exist_ok=True)
    destination = runtime_dir / f"{workload}.yaml"
    destination.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
    return destination


def _ensure_profile(workload: str, profile_dir: Path, legacy_config: str) -> Path:
    path = profile_dir / f"postgresql_{workload}.json"
    if path.exists():
        return path
    sys.path.insert(0, "/root/GPTuner/baselines_random/src")
    from db_bo_baselines.benchbase import BenchBaseRunner
    from db_bo_baselines.config import load_config

    cfg = load_config(resolve_project_path(legacy_config))
    runner = BenchBaseRunner(cfg["benchbase"], cfg["postgres"])
    benchbase_xml = runner.generate_config(workload)
    profile = collect_environment_profile(
        benchbase_xml,
        host=cfg["postgres"].get("host", "127.0.0.1"),
        port=int(cfg["postgres"].get("port", 55432)),
        user=cfg["postgres"]["user"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"=== profile written path={path} workload={workload} ===", flush=True)
    return path


def _write_meta(meta_dir: Path, runtime_name: str, workload: str, seeds: list[int], artifacts: Path) -> None:
    for seed in seeds:
        trace = artifacts / "runs" / runtime_name / "gp_bo" / f"seed_{seed}" / "trials.jsonl"
        if not trace.exists():
            continue
        records = []
        for line in trace.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            item = json.loads(line)
            if item.get("status") == "SUCCESS":
                records.append({
                    "config": item["config"], "utility": float(item["utility"]),
                    "status": item["status"],
                })
        destination = meta_dir / f"{workload}_seed{seed}.json"
        destination.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"=== meta written path={destination} records={len(records)} ===", flush=True)


if __name__ == "__main__":
    main()
