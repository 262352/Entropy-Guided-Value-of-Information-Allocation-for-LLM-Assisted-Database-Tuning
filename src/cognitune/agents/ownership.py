from __future__ import annotations

from cognitune.search_space.space import SearchSpace

# Explicit cross-group ownership: a knob can be shared by several experts.
_OVERLAP = {
    "work_mem": {"memory_expert": 0.7, "concurrency_execution_expert": 0.3},
    "temp_buffers": {"memory_expert": 0.8, "concurrency_execution_expert": 0.2},
    "wal_buffers": {"storage_wal_expert": 0.7, "memory_expert": 0.3},
    "max_connections": {"concurrency_execution_expert": 0.7, "memory_expert": 0.3},
    "effective_io_concurrency": {"storage_wal_expert": 0.6, "optimizer_workload_expert": 0.4},
    "maintenance_work_mem": {"memory_expert": 0.8, "storage_wal_expert": 0.2},
    "max_parallel_workers": {"concurrency_execution_expert": 0.8, "optimizer_workload_expert": 0.2},
    "max_parallel_workers_per_gather": {"concurrency_execution_expert": 0.7, "optimizer_workload_expert": 0.3},
    "max_worker_processes": {"concurrency_execution_expert": 0.8, "memory_expert": 0.2},
}

_CATEGORY_EXPERT = {
    "memory": "memory_expert",
    "concurrency": "concurrency_execution_expert",
    "execution": "concurrency_execution_expert",
    "storage": "storage_wal_expert",
    "optimizer": "optimizer_workload_expert",
}

EXPERTS = (
    "memory_expert",
    "concurrency_execution_expert",
    "storage_wal_expert",
    "optimizer_workload_expert",
)


def build_ownership(space: SearchSpace) -> dict[str, dict[str, float]]:
    ownership: dict[str, dict[str, float]] = {}
    for knob in space.knobs:
        if knob.name in _OVERLAP:
            weights = dict(_OVERLAP[knob.name])
        else:
            primary = _CATEGORY_EXPERT.get(knob.category, "optimizer_workload_expert")
            weights = {primary: 1.0}
        total = sum(weights.values())
        ownership[knob.name] = {expert: weight / total for expert, weight in weights.items()}
    return ownership


def expert_knobs(ownership: dict[str, dict[str, float]], expert: str) -> list[str]:
    return sorted(name for name, weights in ownership.items() if weights.get(expert, 0.0) > 0)
