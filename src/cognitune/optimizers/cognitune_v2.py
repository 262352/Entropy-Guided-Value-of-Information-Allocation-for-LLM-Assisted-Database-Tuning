from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cognitune.agents import AgentCoordinator
from cognitune.agents.expert import V2_EXPERTS, V2_TEMPLATES
from cognitune.agents.ownership import build_ownership
from cognitune.candidates import one_knob_probes
from cognitune.candidates.common import deduplicate_legal, local_mutations
from cognitune.evaluator import Evaluator
from cognitune.experiment.recorder import RunRecorder
from cognitune.graph import TypedRelationGraph, build_coordination_blocks, save_graph
from cognitune.llm import LLMClient, MockLLMClient
from cognitune.search_space.space import SearchSpace
from cognitune.state import KnobState, StateManager
from cognitune.surrogate import GPPerformanceSurrogate
from cognitune.uncertainty import priority_statistics, relation_statistics, value_statistics
from cognitune.uncertainty.value_entropy import _entropy, recommendation_distribution, value_grid
from .common import evaluate_and_record


def run_cognitune_v2(
    space: SearchSpace,
    evaluator: Evaluator,
    initial_configs: list[dict],
    budget: int,
    seed: int,
    candidate_pool_size: int,
    run_dir: str | Path,
    use_relations: bool = True,
    relation_threshold: float = 0.1,
    method_name: str = "cognitune_v2",
    llm_client: LLMClient | None = None,
    cognitive_context: dict | None = None,
    active_target: int = 18,
    active_min: int = 15,
    active_max: int = 20,
    router_interval: int = 10,
    verbose_diagnostics: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if verbose_diagnostics:
        (run_dir / "debug_events.jsonl").unlink(missing_ok=True)
    recorder = RunRecorder(run_dir)
    coordinator = AgentCoordinator(space, llm_client or MockLLMClient(space), run_dir / "llm")
    context = cognitive_context or {"seed": seed, "budget": budget}
    responses = coordinator.collect(context, experts=V2_EXPERTS, templates=("role_template",))
    ownership = build_ownership(space)
    priority = priority_statistics(responses, list(space.by_name))
    values = value_statistics(responses, space.knobs)
    relation_items = relation_statistics(responses, space.by_name) if use_relations else []
    protected = {term["knob"] for rule in space.hard_constraints for term in rule.get("terms", [])}
    protected |= {knob.condition.parent for knob in space.knobs if knob.condition}
    graph = TypedRelationGraph.build(space, relation_items, relation_threshold, protected)
    blocks, _ = build_coordination_blocks(graph)
    save_graph(graph, run_dir / "graph")
    (run_dir / "graph" / "coordination_blocks.json").write_text(
        json.dumps({"blocks": [list(block) for block in blocks]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expert_weight = _uniform(V2_EXPERTS)
    grids, dists, per_expert = _aggregate_distributions(responses, space, ownership, expert_weight)
    base_scores, gate = _initial_gate(space, priority, values, blocks)
    active = _select_active(space, gate, base_scores, blocks, protected, set(), active_target, active_min, active_max, ownership, expert_weight, None, 0)
    state = StateManager.initialize(priority, protected, exploit_budget=active_target, challenge_budget=0, challenge_interval=5)
    _apply_active(state, active, 0)
    diagnostics_path = run_dir / "round_diagnostics.jsonl"
    diagnostics_path.unlink(missing_ok=True)
    cognitive_log = {
        "priority_score": {name: item["priority_score"] for name, item in priority.items()},
        "selection_entropy": {name: item["selection_entropy"] for name, item in priority.items()},
        "value_entropy": {name: item["value_entropy"] for name, item in values.items()},
        "relation_sensitivity": relation_items,
        "soft_edges": [edge.__dict__ for edge in graph.edges if edge.edge_type == "soft_relation"],
        "coordination_blocks": [list(block) for block in blocks],
    }
    if verbose_diagnostics:
        _debug_event(run_dir, "initialization", {
            "active_knobs": sorted(active), "protected_knobs": sorted(protected),
            "gate": gate, "base_scores": base_scores, "ownership": ownership,
            "expert_weights": expert_weight, **cognitive_log,
        })

    configs: list[dict] = []
    utilities: list[float] = []
    best_utility = -float("inf")
    raw_utilities: list[float] = []
    for trial, config in enumerate(initial_configs):
        best_before = best_utility
        best_utility, result = evaluate_and_record(
            evaluator, recorder, method_name, seed, trial, config, "shared_initial", best_utility,
            budget=budget, state=state.as_dict(), knob_states=state.as_dict(), **cognitive_log,
        )
        configs.append(config)
        utilities.append(result.utility)
        raw_utilities.append(result.utility)
        _write_diag(diagnostics_path, trial, config, result.utility, best_utility, "shared_initial",
                    active, gate, expert_weight, priority, values, {}, {})
        if verbose_diagnostics:
            _debug_event(run_dir, "shared_initial", {
                "trial": trial, "round": trial + 1, "config": config,
                "raw_metric": result.raw_metric, "utility": result.utility,
                "status": result.status, "best_utility": best_utility,
                "elapsed_seconds": result.elapsed_seconds,
            })

    surrogate = GPPerformanceSurrogate(space)
    llm_calls = len(responses)
    llm_tokens = _response_tokens(run_dir)
    while len(configs) < budget:
        trial = len(configs)
        incumbent = configs[int(np.argmax(utilities))]
        best_before = best_utility
        previous_config = configs[-2] if len(configs) >= 2 else configs[-1]
        old_active, old_gate, old_weights = set(active), dict(gate), dict(expert_weight)
        gate = _update_gate(space, gate, previous_config, configs[-1], utilities, active, priority, values, ownership, expert_weight)
        surrogate.fit(configs, utilities)
        probes = one_knob_probes(space, incumbent, list(space.by_name), seed * 31_337 + trial, count=2)
        improvements = surrogate.condition_improvement(incumbent, probes)
        active = _select_active(space, gate, base_scores, blocks, protected, active, active_target, active_min, active_max, ownership, expert_weight, improvements, trial)
        _apply_active(state, active, trial)
        pool, sources = _build_pool(
            space, incumbent, {space.config_key(config) for config in configs}, active,
            blocks, grid_map=grids, dist_map=dists, expert_map=per_expert,
            expert_weights=expert_weight, gate=gate, seed=seed * 100_003 + trial,
            size=candidate_pool_size,
        )
        scores, logei, gate_bonus, expert_bonus = _voi_scores(
            space, surrogate, pool, incumbent, active, gate, expert_weight, per_expert, grids, sources,
        )
        selected_index = int(np.argmax(scores))
        selected = pool[selected_index]
        prediction = surrogate.posterior([selected])
        if verbose_diagnostics:
            changed = {name: {"incumbent": incumbent.get(name), "proposed": selected.get(name)}
                       for name in space.by_name if incumbent.get(name) != selected.get(name)}
            versus_previous = {name for name in space.by_name if configs[-1].get(name) != selected.get(name)}
            top = np.argsort(scores)[-10:][::-1]
            _debug_event(run_dir, "proposal", {
                "trial": trial, "round": trial + 1, "incumbent_trial": int(np.argmax(utilities)),
                "incumbent_utility": best_before, "active_knobs": sorted(active),
                "active_added": sorted(active - old_active), "active_removed": sorted(old_active - active),
                "gate_before": old_gate, "gate_after": gate,
                "expert_weights_before": old_weights, "conditional_improvements": improvements,
                "pool_size": len(pool), "candidate_sources": dict(Counter(sources)),
                "selected_source": sources[selected_index], "changed_from_incumbent": changed,
                "extra_knobs_if_attributed_vs_previous_trial": sorted(versus_previous - changed.keys()),
                "posterior_mean": float(prediction.mean[0]), "posterior_std": float(prediction.std[0]),
                "selected_logei": float(logei[selected_index]), "selected_score": float(scores[selected_index]),
                "best_pool_logei": float(np.max(logei)),
                "selected_logei_rank": 1 + int(np.sum(logei > logei[selected_index])),
                "gate_bonus": float(gate_bonus[selected_index]), "expert_bonus": float(expert_bonus[selected_index]),
                "top_candidates": [{"source": sources[int(i)], "score": float(scores[i]),
                    "logei": float(logei[i]), "gate_bonus": float(gate_bonus[i]),
                    "expert_bonus": float(expert_bonus[i])} for i in top],
            })
        best_utility, result = evaluate_and_record(
            evaluator, recorder, method_name, seed, trial, selected, sources[selected_index], best_utility,
            budget=budget, acquisition_value=float(scores[selected_index]),
            posterior_mean=float(prediction.mean[0]), posterior_std=float(prediction.std[0]),
            state=state.as_dict(), knob_states=state.as_dict(), **cognitive_log,
        )
        expert_weight = _update_experts(space, expert_weight, per_expert, grids, configs[-1], selected,
                                        result.utility, best_before, active, ownership, sources[selected_index])
        diag = {
            "active_knobs": sorted(active),
            "active_count": len(active),
            "gate": {name: round(float(gate[name]), 6) for name in space.by_name},
            "expert_weights": {name: round(float(expert_weight[name]), 6) for name in V2_EXPERTS},
            "voi": float(scores[selected_index]),
            "logei": float(logei[selected_index]),
            "gate_bonus": float(gate_bonus[selected_index]),
            "expert_bonus": float(expert_bonus[selected_index]),
            "improvements": {name: round(float(improvements.get(name, 0.0)), 6) for name in space.by_name},
            "sources": dict(Counter(sources)),
        }
        _write_diag(diagnostics_path, trial, selected, result.utility, best_utility, sources[selected_index],
                    active, gate, expert_weight, priority, values, diag, {})
        if verbose_diagnostics:
            _debug_event(run_dir, "outcome", {
                "trial": trial, "round": trial + 1, "status": result.status,
                "raw_metric": result.raw_metric, "actual_utility": result.utility,
                "utility_gain_vs_incumbent": result.utility - best_before,
                "prediction_error": result.utility - float(prediction.mean[0]),
                "new_best": result.utility > best_before, "best_utility": best_utility,
                "expert_weights_before": old_weights, "expert_weights_after": expert_weight,
                "elapsed_seconds": result.elapsed_seconds, "metadata": result.metadata,
            })
        if router_interval > 0 and trial >= router_interval and trial % router_interval == 0:
            chosen = sorted(V2_EXPERTS, key=lambda name: expert_weight[name], reverse=True)[:2]
            if verbose_diagnostics:
                _debug_event(run_dir, "router_refresh_start", {"trial": trial, "experts": chosen})
            refreshed = coordinator.collect(
                context | {"round": trial, "incumbent": incumbent, "active_knobs": sorted(active),
                           "current_best_utility": best_utility},
                experts=tuple(chosen), templates=("role_template",), tag=f"router_{trial}",
            )
            responses.extend(refreshed)
            llm_calls += len(refreshed)
            llm_tokens = _response_tokens(run_dir)
            priority = priority_statistics(responses, list(space.by_name))
            values = value_statistics(responses, space.knobs)
            grids, dists, per_expert = _aggregate_distributions(responses, space, ownership, expert_weight)
            if verbose_diagnostics:
                _debug_event(run_dir, "router_refresh_done", {"trial": trial,
                    "llm_calls": llm_calls, "llm_tokens": llm_tokens,
                    "priority_score": {name: item["priority_score"] for name, item in priority.items()}})
        configs.append(selected)
        utilities.append(result.utility)
    summary = recorder.finalize()
    summary.update({
        "llm_calls": llm_calls,
        "llm_tokens": llm_tokens,
        "active_knob_count": len(active),
        "active_knobs": sorted(active),
        "expert_weights": expert_weight,
        "coordination_block_count": len(blocks),
        "router_refreshes": max(0, (budget - 1) // router_interval) if router_interval else 0,
    })
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _debug_event(run_dir, event, payload):
    record = {"event": event, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
    with (Path(run_dir) / "debug_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"[CogniTune DEBUG: {event}] {record['timestamp_utc']}\n"
          + json.dumps(payload, sort_keys=True, indent=2), flush=True)


def _uniform(experts):
    return {name: 1.0 / len(experts) for name in experts}


def _response_tokens(run_dir: Path) -> int:
    total = 0
    for path in (run_dir / "llm").glob("*.json"):
        if "_invalid_" in path.name:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        total += int(record.get("input_tokens", 0)) + int(record.get("output_tokens", 0))
    return total


def _aggregate_distributions(responses, space, ownership, expert_weights):
    grids, dists, per_expert = {}, {}, {}
    for knob in space.knobs:
        grid = value_grid(knob)
        grids[knob.name] = grid
        mixture = np.zeros(len(grid))
        total_weight = 0.0
        for expert in V2_EXPERTS:
            weight = expert_weights.get(expert, 0.0) * ownership[knob.name].get(expert, 0.0)
            if weight <= 0:
                continue
            distributions = [
                recommendation_distribution(knob, response.recommendations[knob.name], grid)
                for response in responses
                if response.expert == expert and knob.name in response.recommendations
            ]
            if not distributions:
                continue
            aggregate = np.mean(distributions, axis=0)
            per_expert.setdefault(expert, {})[knob.name] = aggregate
            mixture += weight * aggregate
            total_weight += weight
        dists[knob.name] = mixture / total_weight if total_weight > 0 else np.ones(len(grid)) / len(grid)
    return grids, dists, per_expert


def _initial_gate(space, priority, values, blocks):
    block_size = {name: 1 for name in space.by_name}
    for block in blocks:
        for name in block:
            block_size[name] = len(block)
    scores = {}
    for name in space.by_name:
        scores[name] = float(np.clip(
            0.55 * priority[name]["priority_score"]
            + 0.25 * priority[name]["selection_entropy"]
            + 0.15 * values.get(name, {}).get("value_entropy", 0.0)
            + 0.05 * min(block_size.get(name, 1), 4) / 4.0,
            0.0, 1.0,
        ))
    gate = {name: float(np.clip(0.15 + 0.75 * score, 0.05, 0.95)) for name, score in scores.items()}
    return scores, gate


def _select_active(space, gate, base_scores, blocks, protected, current, target, minimum, maximum, ownership=None, expert_weights=None, improvements=None, trial=0):
    target = max(minimum, min(maximum, target))
    improvement_scale = max([float(value) for value in (improvements or {}).values()] + [1e-9])
    scores = {}
    for name in space.by_name:
        affinity = 0.0
        if ownership and expert_weights:
            affinity = sum(ownership[name].get(expert, 0.0) * expert_weights.get(expert, 0.0) for expert in expert_weights)
        improvement = float((improvements or {}).get(name, 0.0)) / improvement_scale
        scores[name] = float(0.45 * gate[name] + 0.20 * base_scores[name] + 0.25 * improvement + 0.10 * affinity)
    for block in blocks:
        if len(block) > 1:
            for name in block:
                scores[name] += 0.04
    for name in current:
        scores[name] += 0.06
    mandatory = {"shared_buffers", "work_mem", "maintenance_work_mem"}
    selected = set(protected) | {name for name in mandatory if name in space.by_name}
    if trial % 10 == 0 and "effective_cache_size" in space.by_name:
        selected.add("effective_cache_size")
    if ownership and expert_weights:
        # Group coverage: every expert group contributes its strongest
        # knobs.  This prevents a high-priority memory knob such as
        # effective_cache_size from being crowded out by the global top-k.
        group_quotas = {"memory_expert": 3, "storage_wal_expert": 2}
        for expert, quota in group_quotas.items():
            if expert not in expert_weights:
                continue
            owned = [name for name in space.by_name if ownership[name].get(expert, 0.0) > 0]
            for name in sorted(owned, key=lambda item: (-scores[item], item))[:quota]:
                if len(selected) < target:
                    selected.add(name)
    for name in sorted(scores, key=lambda item: (-scores[item], item)):
        if len(selected) >= target:
            break
        selected.add(name)
    return selected


def _apply_active(state: StateManager, active: set[str], trial: int) -> None:
    for name, status in state.statuses.items():
        if name in active:
            status.state = KnobState.ACTIVE
            status.reason = "voi_gate"
        elif status.state == KnobState.ACTIVE:
            status.state = KnobState.WATCH
            status.reason = "voi_gate_evicted"
        status.last_changed_trial = trial


def _sample_value(rng, grid, distribution):
    probabilities = np.asarray(distribution, dtype=float)
    probabilities = probabilities / probabilities.sum()
    value = grid[int(rng.choice(len(grid), p=probabilities))]
    return value.item() if isinstance(value, np.generic) else value


def _sample_near(rng, grid, distribution, incumbent_value, knob):
    """Sample inside a trust region around the incumbent for numeric knobs."""
    if knob.kind in {"integer", "float"} and incumbent_value is not None:
        values = np.asarray(grid, dtype=float)
        center = float(incumbent_value)
        if knob.log:
            low, high = center / 2.0, center * 2.0
        else:
            span = float(knob.upper) - float(knob.lower)
            low, high = center - 0.25 * span, center + 0.25 * span
        mask = (values >= low) & (values <= high)
        if mask.any():
            indices = np.flatnonzero(mask)
            probabilities = np.asarray(distribution, dtype=float)[indices]
            if probabilities.sum() > 0:
                probabilities = probabilities / probabilities.sum()
                value = grid[indices[int(rng.choice(len(indices), p=probabilities))]]
                return value.item() if isinstance(value, np.generic) else value
    return _sample_value(rng, grid, distribution)


def _sample_config(space, incumbent, names, grids, dists, rng, max_changes=None):
    config = dict(incumbent)
    candidates = list(names)
    if max_changes is not None and len(candidates) > max_changes:
        candidates = list(rng.choice(candidates, size=max_changes, replace=False))
    for name in candidates:
        config[name] = _sample_value(rng, grids[name], dists[name])
    return space.repair(config)


def _build_pool(space, incumbent, seen, active, blocks, grid_map, dist_map, expert_map,
                expert_weights, gate, seed, size):
    rng = np.random.default_rng(seed)
    experts = list(expert_weights)
    weights = np.asarray([expert_weights[name] for name in experts], dtype=float)
    weights /= weights.sum()
    counts = _allocate(size, (0.35, 0.10, 0.25, 0.05, 0.10, 0.15))
    tagged = []
    active_list = sorted(active)
    for index in range(counts[0]):
        expert = experts[int(rng.choice(len(experts), p=weights))]
        config = dict(incumbent)
        if index < counts[0] // 2 and active_list:
            # Global expert proposal: broad coverage, as in MCTuner's pruned
            # space.  This is the mode that finds a new basin.
            for name in active_list:
                source = expert_map.get(expert, {}).get(name)
                config[name] = _sample_value(rng, grid_map[name], source if source is not None else dist_map[name])
            tagged.append((space.repair(config), f"expert_global:{expert}"))
        else:
            # Sparse expert proposal: refine within a trust region.
            change_count = int(rng.integers(1, min(4, len(active_list)) + 1)) if active_list else 0
            names = list(rng.choice(active_list, size=change_count, replace=False)) if change_count else []
            for name in names:
                source = expert_map.get(expert, {}).get(name)
                config[name] = _sample_near(
                    rng, grid_map[name], source if source is not None else dist_map[name],
                    incumbent.get(name), space.by_name[name],
                )
            tagged.append((space.repair(config), f"expert_local:{expert}"))
    for _ in range(counts[1]):
        block = blocks[int(rng.integers(len(blocks)))] if blocks else tuple(active_list)
        names = list(block)
        keep = min(2, len(names))
        if len(names) > keep:
            names = list(rng.choice(names, size=keep, replace=False))
        config = dict(incumbent)
        for name in names:
            config[name] = _sample_near(rng, grid_map[name], dist_map[name], incumbent.get(name), space.by_name[name])
        tagged.append((space.repair(config), "relation_block"))
    for _ in range(counts[2]):
        config = dict(incumbent)
        change_count = int(rng.integers(1, min(2, len(active_list)) + 1)) if active_list else 0
        names = list(rng.choice(active_list, size=change_count, replace=False)) if change_count else []
        for name in names:
            config[name] = _sample_near(rng, grid_map[name], dist_map[name], incumbent.get(name), space.by_name[name])
        tagged.append((space.repair(config), "local_perturbation"))
    for config in space.sample_iid(seed + 17, counts[3]):
        merged = dict(incumbent)
        for name in active_list:
            if name in config:
                merged[name] = config[name]
        tagged.append((space.repair(merged), "active_iid"))
    inactive = [name for name in space.by_name if name not in active]
    probes = sorted(inactive, key=lambda name: gate[name] * (1.0 - gate[name]), reverse=True)
    for _ in range(counts[4]):
        if not probes:
            break
        pool_size = max(1, min(len(probes), len(probes) // 2 + 1))
        name = probes[int(rng.integers(pool_size))]
        config = dict(incumbent)
        config[name] = _sample_near(rng, grid_map[name], dist_map[name], incumbent.get(name), space.by_name[name])
        tagged.append((space.repair(config), "gate_probe"))
    for _ in range(counts[5]):
        if not active_list:
            break
        name = active_list[int(rng.integers(len(active_list)))]
        config = dict(incumbent)
        config[name] = _sample_near(rng, grid_map[name], dist_map[name], incumbent.get(name), space.by_name[name])
        tagged.append((space.repair(config), "one_knob_expert"))
    for name in ("effective_cache_size", "wal_buffers", "huge_pages"):
        if name in space.by_name and name not in active:
            config = dict(incumbent)
            config[name] = _sample_near(rng, grid_map[name], dist_map[name], incumbent.get(name), space.by_name[name])
            tagged.append((space.repair(config), "coverage_probe"))
    legal = []
    local_seen = set()
    for config, source in tagged:
        key = space.config_key(config)
        if key not in seen and key not in local_seen and space.validate(config):
            legal.append(config)
            local_seen.add(key)
    if len(legal) < max(1, size // 5):
        fallback = deduplicate_legal(space, space.sample_iid(seed + 2_000_003, size), seen | local_seen)
        legal.extend(fallback)
        tagged.extend([(config, "iid_fallback") for config in fallback])
    return legal, [source for _, source in tagged][:len(legal)] if len(tagged) == len(legal) else _sources_for(legal, tagged, space, seen)


def _sources_for(legal, tagged, space, seen):
    # Keep source labels aligned after fallback; for the common case the
    # tagged list and legal list have the same order.
    labels = []
    tagged_keys = {space.config_key(config): source for config, source in tagged}
    for config in legal:
        labels.append(tagged_keys.get(space.config_key(config), "fallback"))
    return labels


def _allocate(total, ratios):
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    for index in np.argsort([value - int(value) for value in raw])[::-1][: total - sum(counts)]:
        counts[int(index)] += 1
    return counts


def _binary_entropy(probability):
    p = min(max(float(probability), 1e-6), 1 - 1e-6)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def _voi_scores(space, surrogate, pool, incumbent, active, gate, expert_weights, per_expert, grids, sources):
    logei = np.nan_to_num(surrogate.log_expected_improvement(pool), nan=-1e9, neginf=-1e9, posinf=1e9)
    gate_values = []
    expert_values = []
    expert_entropy = -sum(w * math.log(max(w, 1e-9)) for w in expert_weights.values())
    for config in pool:
        changed = [name for name in active if name in config and name in incumbent and config.get(name) != incumbent.get(name)]
        gate_values.append(sum(_binary_entropy(gate[name]) for name in changed) / max(1, len(changed)))
        matches = []
        for expert, probabilities in per_expert.items():
            values = []
            for name in changed:
                if name in probabilities:
                    grid = grids[name]
                    index = _index_of(grid, config[name])
                    values.append(float(probabilities[name][index]))
            if values:
                matches.append(float(np.mean(values)))
        match = max(matches) if matches else 0.0
        expert_values.append(expert_entropy * (1.0 - match))
    gate_bonus = np.asarray(gate_values)
    expert_bonus = np.asarray(expert_values)
    gate_bonus = gate_bonus / max(1e-9, float(gate_bonus.max())) if gate_bonus.size else gate_bonus
    expert_bonus = expert_bonus / max(1e-9, float(expert_bonus.max())) if expert_bonus.size else expert_bonus
    # VoI is used to route experts and activate knobs; the candidate choice is
    # driven by the surrogate's LogEI.  Structural bonuses only break ties
    # among candidates whose EI is already in the top quartile.
    threshold = float(np.quantile(logei, 0.75)) if logei.size else -float("inf")
    bonuses = 0.02 * gate_bonus + 0.02 * expert_bonus
    scores = logei + np.where(logei >= threshold, bonuses, 0.0)
    return scores, logei, gate_bonus, expert_bonus


def _index_of(grid, value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(np.argmin(np.abs(np.asarray(grid, dtype=float) - float(value))))
    matches = np.where(np.asarray(grid, dtype=object) == value)[0]
    return int(matches[0]) if len(matches) else 0


def _update_experts(space, weights, per_expert, grids, previous, selected, utility, best_before, active, ownership, source=""):
    changed = [name for name in active if name in selected and name in previous and selected.get(name) != previous.get(name)]
    reward = 1.0 if utility >= best_before else 0.0
    credited = source.split(":", 1)[1] if source.startswith("expert") and ":" in source else None
    updated = {}
    for expert in V2_EXPERTS:
        before = weights.get(expert, 1.0 / len(V2_EXPERTS))
        likelihoods = []
        for name in changed:
            probabilities = per_expert.get(expert, {}).get(name)
            if probabilities is None or ownership[name].get(expert, 0.0) <= 0 or name not in selected:
                continue
            likelihoods.append(float(probabilities[_index_of(grids[name], selected[name])]))
        likelihood = float(np.mean(likelihoods)) if likelihoods else 0.2
        delta = 0.02 * (reward * likelihood - (1 - reward) * (1 - likelihood) * 0.5)
        if credited == expert:
            delta += 0.08 if reward else -0.05
        updated[expert] = max(0.02, min(0.98, before + delta))
    total = sum(updated.values())
    return {name: value / total for name, value in updated.items()}


def _update_gate(space, gate, previous, selected, utilities, active, priority, values, ownership=None, expert_weights=None):
    improved = len(utilities) == 0 or utilities[-1] >= max(utilities[:-1] or [-float("inf")])
    updated = dict(gate)
    changed = [name for name in space.by_name if name in selected and name in previous and selected.get(name) != previous.get(name)]
    for name in changed:
        delta = 0.06 if improved else -0.03
        knob = space.by_name[name]
        value = selected.get(name)
        if knob.kind in {"integer", "float"} and value is not None:
            if float(value) <= float(knob.lower) or float(value) >= float(knob.upper):
                delta -= 0.02
        updated[name] = float(np.clip(updated[name] + delta, 0.02, 0.98))
    for name in active:
        if name not in changed:
            updated[name] = float(np.clip(updated[name] - 0.005, 0.02, 0.98))
    if ownership and expert_weights:
        for name in space.by_name:
            if name in active or name in changed:
                continue
            affinity = sum(ownership[name].get(expert, 0.0) * expert_weights.get(expert, 0.0) for expert in expert_weights)
            updated[name] = float(np.clip(updated[name] + 0.004 * affinity, 0.02, 0.98))
    return updated


def _write_diag(path, trial, config, utility, best, source, active, gate, weights, priority, values, extra, _):
    record = {
        "trial": trial, "source": source, "utility": utility, "best_utility": best,
        "active_count": len(active), "active_knobs": sorted(active),
        "gate": {name: float(gate[name]) for name in gate},
        "expert_weights": {name: float(weights[name]) for name in weights},
        "priority_score": {name: item["priority_score"] for name, item in priority.items()},
        "selection_entropy": {name: item["selection_entropy"] for name, item in priority.items()},
        "value_entropy": {name: item["value_entropy"] for name, item in values.items()},
        "config": config,
    }
    record.update(extra)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
