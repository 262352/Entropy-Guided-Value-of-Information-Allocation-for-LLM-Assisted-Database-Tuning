from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cognitune.agents import AgentCoordinator
from cognitune.candidates import generate_cognitune_pool, one_knob_probes
from cognitune.evaluator import Evaluator
from cognitune.experiment.recorder import RunRecorder
from cognitune.graph import TypedRelationGraph, build_coordination_blocks, calibrate_block, save_graph
from cognitune.llm import LLMClient, MockLLMClient
from cognitune.search_space.space import SearchSpace
from cognitune.state import KnobState, StateManager
from cognitune.surrogate import GPPerformanceSurrogate
from cognitune.uncertainty import priority_statistics, relation_statistics, value_statistics
from .common import evaluate_and_record


def run_cognitune(
    space: SearchSpace,
    evaluator: Evaluator,
    initial_configs: list[dict],
    budget: int,
    seed: int,
    candidate_pool_size: int,
    run_dir: str | Path,
    use_relations: bool = True,
    relation_threshold: float = 0.01,
    method_name: str = "cognitune",
    llm_client: LLMClient | None = None,
    cognitive_context: dict | None = None,
    exploit_knob_budget: int | None = None,
    challenge_knob_budget: int = 0,
    challenge_interval: int = 5,
) -> dict:
    run_dir = Path(run_dir)
    recorder = RunRecorder(run_dir)
    coordinator = AgentCoordinator(space, llm_client or MockLLMClient(space), run_dir / "llm")
    responses = coordinator.collect(cognitive_context or {"seed": seed, "budget": budget})
    priority = priority_statistics(responses, list(space.by_name))
    values = value_statistics(responses, space.knobs)
    relation_items = relation_statistics(responses, space.by_name) if use_relations else []
    protected = {term["knob"] for rule in space.hard_constraints for term in rule.get("terms", [])}
    protected |= {knob.condition.parent for knob in space.knobs if knob.condition}
    graph = TypedRelationGraph.build(space, relation_items, relation_threshold, protected)
    blocks, decisions = build_coordination_blocks(graph)
    save_graph(graph, run_dir / "graph")
    (run_dir / "graph" / "coordination_blocks.json").write_text(json.dumps({
        "blocks": blocks, "decisions": [asdict(item) for item in decisions],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state = StateManager.initialize(
        priority, protected, exploit_budget=exploit_knob_budget,
        challenge_budget=challenge_knob_budget, challenge_interval=challenge_interval,
    )
    cognitive_log = {
        "priority_score": {name: item["priority_score"] for name, item in priority.items()},
        "selection_entropy": {name: item["selection_entropy"] for name, item in priority.items()},
        "value_entropy": {name: item["value_entropy"] for name, item in values.items()},
        "relation_sensitivity": relation_items,
        "soft_edges": [asdict(edge) for edge in graph.edges if edge.edge_type == "soft_relation"],
        "coordination_blocks": blocks,
    }

    configs, utilities = [], []
    best_utility = -float("inf")
    for trial, config in enumerate(initial_configs):
        best_utility, result = evaluate_and_record(
            evaluator, recorder, method_name, seed, trial, config, "shared_initial", best_utility,
            budget=budget, state=state.as_dict(), knob_states=state.as_dict(), **cognitive_log,
        )
        configs.append(config)
        utilities.append(result.utility)

    surrogate = GPPerformanceSurrogate(space)
    while len(configs) < budget:
        surrogate.fit(configs, utilities)
        trial = len(configs)
        incumbent = configs[int(np.argmax(utilities))]
        probes = one_knob_probes(space, incumbent, list(space.by_name), seed * 31_337 + trial)
        improvements = surrogate.condition_improvement(incumbent, probes)
        uncertainty = {
            name: float(np.clip(
                0.5 * priority[name]["selection_entropy"]
                + 0.5 * values.get(name, {}).get("value_entropy", 0.0),
                0.0, 1.0,
            ))
            for name in space.by_name
        }
        priority_scores = {name: priority[name]["priority_score"] for name in space.by_name}
        transitions = state.update(
            trial, improvements, uncertainty, {name: trial for name in space.by_name},
            priority_scores=priority_scores,
        )
        pool = generate_cognitune_pool(
            space, incumbent, {space.config_key(config) for config in configs}, state,
            blocks, values, seed * 100_003 + trial, candidate_pool_size,
        )
        scores = surrogate.log_expected_improvement(pool.configs)
        selected_index = int(np.argmax(scores))
        selected = pool.configs[selected_index]
        prediction = surrogate.posterior([selected])
        challenge_target = None
        challenge_reason = None
        if pool.sources[selected_index] == "challenge":
            challenge_target = next((name for name in state.challenge_names() if selected.get(name) != incumbent.get(name)), None)
            challenge_reason = "periodic uncertain-state validation"
        best_utility, result = evaluate_and_record(
            evaluator, recorder, method_name, seed, trial, selected, pool.sources[selected_index], best_utility,
            budget=budget, acquisition_value=float(scores[selected_index]), posterior_mean=float(prediction.mean[0]),
            posterior_std=float(prediction.std[0]), state=state.as_dict(), transitions=transitions,
            knob_states=state.as_dict(), challenge_target=challenge_target,
            challenge_reason=challenge_reason, relation_calibration=[], **cognitive_log,
        )
        if pool.sources[selected_index] in {"coordination_block", "challenge"}:
            changed = {name for name in space.by_name if selected.get(name) != incumbent.get(name)}
            calibration = calibrate_block(graph, changed, result.utility - max(utilities))
            recorder.records[-1].relation_calibration = calibration
            recorder.records[-1].soft_edges = [asdict(edge) for edge in graph.edges if edge.edge_type == "soft_relation"]
            recorder.rewrite()
        configs.append(selected)
        utilities.append(result.utility)
    save_graph(graph, run_dir / "graph")
    summary = recorder.finalize()
    llm_records = [json.loads(path.read_text()) for path in (run_dir / "llm").glob("*.json") if "_invalid_" not in path.name]
    summary.update({
        "llm_calls": len(responses),
        "llm_tokens": sum(item["input_tokens"] + item["output_tokens"] for item in llm_records),
        "llm_input_tokens": sum(item["input_tokens"] for item in llm_records),
        "llm_output_tokens": sum(item["output_tokens"] for item in llm_records),
        "llm_cache_hit_tokens": sum(item.get("cache_hit_tokens", 0) for item in llm_records),
        "llm_cache_miss_tokens": sum(item.get("cache_miss_tokens", 0) for item in llm_records),
        "coordination_block_count": len(blocks),
        "active_knob_count": len(state.names(KnobState.ACTIVE)),
        "reactivated_knobs": sum(transition["to"] == "ACTIVE" and transition["from"] != "ACTIVE" for record in recorder.records for transition in (record.transitions or [])),
        "challenge_successes": sum(record.candidate_source == "challenge" and record.utility >= 0 for record in recorder.records),
    })
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
