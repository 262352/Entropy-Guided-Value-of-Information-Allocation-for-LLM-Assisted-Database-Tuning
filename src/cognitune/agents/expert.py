from __future__ import annotations

from dataclasses import dataclass


# Legacy five-expert ensemble, kept for backward compatibility and ablations.
EXPERTS = (
    "memory_expert", "concurrency_expert", "optimizer_expert",
    "execution_expert", "storage_wal_expert",
)
TEMPLATES = ("mechanism_template", "workload_template", "resource_template")

# Four-expert MoE used by the entropy-guided VoI method.  The groups cover the
# 60-knob PostgreSQL space with a small number of role prompts.
V2_EXPERTS = (
    "memory_expert",
    "concurrency_execution_expert",
    "storage_wal_expert",
    "optimizer_workload_expert",
)
V2_TEMPLATES = ("role_template", "interaction_template")

# MCTuner-style baseline: seven domain experts, one role prompt each.
MCTUNER_EXPERTS = EXPERTS + ("workload_expert", "resource_expert")
MCTUNER_TEMPLATES = ("role_template",)


@dataclass(frozen=True)
class ExpertPrompt:
    expert: str
    template: str
    prompt: str
