from __future__ import annotations

import argparse
import json
from pathlib import Path

from cognitune.config import load_experiment, load_search_space, load_yaml
from cognitune.evaluator import CommandEvaluator, PostgreSQLBenchBaseEvaluator, SyntheticEvaluator
from cognitune.llm import CachedLLMClient, DeepSeekLLMClient
from cognitune.optimizers import (
    run_cdbtune, run_cognitune, run_cognitune_v2, run_cognitune_v3, run_dbtune, run_gp_bo,
    run_mctuner, run_ottertune, run_restune, run_smac,
)
from cognitune.optimizers.common import set_progress_tag
from cognitune.search_space.initial_design import build_initial_design, load_initial_design, save_initial_design


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_evaluator(space, config, seed, artifact_root=None):
    settings = config.evaluator
    if settings.get("type", "synthetic") == "synthetic":
        return SyntheticEvaluator(
            space, seed, config.objective, float(settings.get("default_metric", 1000.0)),
            float(settings.get("noise_std", 0.01)),
        )
    if settings.get("type") == "postgres_benchbase":
        return PostgreSQLBenchBaseEvaluator(
            space=space,
            legacy_config=resolve_project_path(settings["legacy_config"]),
            workload=settings.get("workload", config.workload), seed=seed,
            artifact_root=artifact_root or PROJECT_ROOT / "artifacts/real_measurements",
            n_init=config.n_init, failure_penalty=config.failure_penalty,
        )
    default_metric = settings.get("default_metric")
    if not isinstance(default_metric, (int, float)):
        raise ValueError("replace evaluator.default_metric before running a real database experiment")
    command = settings.get("command")
    if not command or any(str(part).startswith("REPLACE_") for part in command):
        raise ValueError("replace evaluator.command before running a real database experiment")
    return CommandEvaluator(
        command=command, objective=config.objective, default_metric=float(default_metric),
        timeout_seconds=int(settings.get("timeout_seconds", 900)),
        metric_regex=settings.get("metric_regex"), result_json=settings.get("result_json"),
        apply_command=settings.get("apply_command"), restart_command=settings.get("restart_command"),
        warmup_command=settings.get("warmup_command"), reset_command=settings.get("reset_command"),
        health_command=settings.get("health_command"), restore_command=settings.get("restore_command"),
        failure_penalty=config.failure_penalty, workdir=settings.get("workdir"),
    )


def run_method(experiment_path: str | Path, method: str, seed: int, artifacts: str | Path | None = None, disable_relation_graph: bool = False, meta_dir: str | Path | None = None) -> dict:
    experiment_path = resolve_project_path(str(experiment_path))
    raw = load_yaml(experiment_path)["experiment"]
    config = load_experiment(experiment_path)
    space = load_search_space(resolve_project_path(raw["space"]))
    artifact_root = resolve_project_path(str(artifacts)) if artifacts else PROJECT_ROOT / "artifacts"
    design_path = artifact_root / "initial_design" / f"{config.dbms}_{config.workload}_seed_{seed}.json"
    if not design_path.exists():
        save_initial_design(design_path, build_initial_design(space, seed, config.n_init))
    initial = load_initial_design(design_path, space, config.n_init, seed)
    label = "cognitune_no_relation" if method == "cognitune" and disable_relation_graph else method
    run_dir = artifact_root / "runs" / config.name / label / f"seed_{seed}"
    meta_root = Path(meta_dir) if meta_dir is not None else artifact_root / "meta"
    evaluator = make_evaluator(space, config, seed, artifact_root / "real_measurements" / config.name)
    evaluator.prepare()
    set_progress_tag(config.workload)
    try:
        if method == "cognitune_v3":
            from cognitune.adaptive.proposals import FeedbackExpertProposer, StructuredMockProposer
            context = {}
            if config.llm.get("provider") == "deepseek":
                transport = DeepSeekLLMClient(
                    model=config.llm.get("model", "deepseek-flash"),
                    base_url=config.llm.get("base_url", "https://api.deepseek.com"),
                    timeout_seconds=int(config.llm.get("timeout_seconds", 300)),
                )
                proposer = FeedbackExpertProposer(transport, run_dir / "expert_requests")
                profile_path = resolve_project_path(config.llm["environment_profile"])
                context = json.loads(profile_path.read_text())
            elif config.evaluator.get("type", "synthetic") == "synthetic":
                proposer = StructuredMockProposer()
            else:
                raise ValueError("cognitune_v3 real experiments require a configured DeepSeek provider")
            settings = config.cognitune
            summary = run_cognitune_v3(
                space, evaluator, initial, config.budget, seed, config.candidate_pool_size, run_dir,
                proposer=proposer, cognitive_context=context,
                active_target=int(settings.get("active_target", 18)),
                audit_size=int(settings.get("audit_size", 4)), eta=float(settings.get("eta", .8)),
                posterior_samples=int(settings.get("posterior_samples", 256)),
                fantasies=int(settings.get("fantasies", 16)),
                probe_interval=int(settings.get("probe_interval", 8)),
                search_policy=settings.get("search_policy", "legacy"),
            )
        elif method in {"cognitune", "cognitune_v2"}:
            llm_client = None
            cognitive_context = None
            if config.llm.get("provider") == "deepseek":
                llm_client = CachedLLMClient(
                    DeepSeekLLMClient(
                        model=config.llm.get("model", "deepseek-flash"),
                        base_url=config.llm.get("base_url", "https://api.deepseek.com"),
                        timeout_seconds=int(config.llm.get("timeout_seconds", 300)),
                        max_tokens=int(config.llm.get("max_tokens", 16000)),
                    ),
                    artifact_root / "llm_cache",
                )
                profile_path = resolve_project_path(config.llm["environment_profile"])
                cognitive_context = json.loads(profile_path.read_text(encoding="utf-8"))
            if method == "cognitune_v2":
                summary = run_cognitune_v2(
                    space, evaluator, initial, config.budget, seed, config.candidate_pool_size, run_dir,
                    use_relations=not disable_relation_graph,
                    relation_threshold=float(config.cognitune.get("relation_threshold", 0.1)),
                    method_name=label,
                    llm_client=llm_client,
                    cognitive_context=cognitive_context,
                    verbose_diagnostics=bool(config.cognitune.get("verbose_diagnostics", False)),
                )
            else:
                summary = run_cognitune(
                    space, evaluator, initial, config.budget, seed, config.candidate_pool_size, run_dir,
                    use_relations=not disable_relation_graph,
                    relation_threshold=float(config.cognitune.get("relation_threshold", 0.01)),
                    method_name=label,
                    llm_client=llm_client,
                    cognitive_context=cognitive_context,
                    exploit_knob_budget=int(config.cognitune.get("exploit_knob_budget", 9)),
                    challenge_knob_budget=int(config.cognitune.get("challenge_knob_budget", 3)),
                    challenge_interval=int(config.cognitune.get("challenge_interval", 5)),
                )
            if disable_relation_graph:
                summary["method"] = label
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif method == "gp_bo":
            summary = run_gp_bo(space, evaluator, initial, config.budget, seed, config.candidate_pool_size, run_dir)
        elif method == "smac":
            summary = run_smac(space, evaluator, initial, config.budget, seed, run_dir)
        elif method == "ottertune":
            summary = run_ottertune(space, evaluator, initial, config.budget, seed, config.candidate_pool_size, run_dir)
        elif method == "cdbtune":
            summary = run_cdbtune(space, evaluator, initial, config.budget, seed, config.candidate_pool_size, run_dir)
        elif method == "dbtune":
            summary = run_dbtune(space, evaluator, initial, config.budget, seed, config.candidate_pool_size, run_dir)
        elif method == "mctuner":
            llm_client = None
            cognitive_context = None
            if config.llm.get("provider") == "deepseek":
                llm_client = CachedLLMClient(
                    DeepSeekLLMClient(
                        model=config.llm.get("model", "deepseek-flash"),
                        base_url=config.llm.get("base_url", "https://api.deepseek.com"),
                        timeout_seconds=int(config.llm.get("timeout_seconds", 300)),
                        max_tokens=int(config.llm.get("max_tokens", 16000)),
                    ),
                    artifact_root / "llm_cache",
                )
                profile_path = resolve_project_path(config.llm["environment_profile"])
                cognitive_context = json.loads(profile_path.read_text(encoding="utf-8"))
            summary = run_mctuner(
                space, evaluator, initial, config.budget, seed, config.candidate_pool_size, run_dir,
                method_name=label, llm_client=llm_client, cognitive_context=cognitive_context,
            )
        elif method == "restune":
            summary = run_restune(space, evaluator, initial, config.budget, seed, config.candidate_pool_size, run_dir, meta_dir=meta_root)
        else:
            raise ValueError(f"unsupported method {method!r}")
        validations = []
        for repeat in range(config.final_validation_repeats):
            validation = evaluator.evaluate(summary["best_config"], config.budget + repeat).as_dict()
            validations.append(validation)
            print(
                f"[{label} seed={seed}] final-validation {repeat + 1}/{config.final_validation_repeats} "
                f"TPS={validation['raw_metric']:.3f} status={validation['status']}",
                flush=True,
            )
        (run_dir / "final_validation.json").write_text(json.dumps({
            "outside_tuning_budget": True,
            "repeats": config.final_validation_repeats,
            "results": validations,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        set_progress_tag(None)
        evaluator.restore_default()
        evaluator.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run one CogniTune experiment method")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--method", required=True, choices=("cognitune", "cognitune_v2", "cognitune_v3", "mctuner", "smac", "gp_bo"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--artifacts")
    parser.add_argument("--disable-selection-entropy", action="store_true")
    parser.add_argument("--disable-value-entropy", action="store_true")
    parser.add_argument("--disable-relation-graph", action="store_true")
    parser.add_argument("--disable-coordination-blocks", action="store_true")
    parser.add_argument("--disable-challenge", action="store_true")
    parser.add_argument("--disable-llm-prior", action="store_true")
    args = parser.parse_args(argv)
    unsupported = [
        name for name in ("disable_selection_entropy", "disable_value_entropy", "disable_coordination_blocks", "disable_challenge", "disable_llm_prior")
        if getattr(args, name)
    ]
    if unsupported:
        raise ValueError(f"these reserved ablations are not enabled in the first-stage experiment: {unsupported}")
    print(json.dumps(run_method(args.experiment, args.method, args.seed, args.artifacts, args.disable_relation_graph), indent=2))
