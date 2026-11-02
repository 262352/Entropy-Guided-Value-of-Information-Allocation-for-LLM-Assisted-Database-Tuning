from __future__ import annotations

import json
from typing import Any

import numpy as np

from cognitune.search_space.space import SearchSpace
from .base import LLMClient, LLMResult


class MockLLMClient(LLMClient):
    """Deterministic prompt ensemble used by tests and offline experiments."""

    def __init__(self, space: SearchSpace, model: str = "mock-cognitive-expert-v1"):
        self.space = space
        self.model = model

    def _values(self, name: str, shift: int) -> list[Any]:
        knob = self.space.by_name[name]
        if knob.kind in {"integer", "float"}:
            if knob.log:
                grid = np.geomspace(float(knob.lower), float(knob.upper), 7)
            else:
                grid = np.linspace(float(knob.lower), float(knob.upper), 7)
            values = grid[max(0, 2 + shift):min(7, 5 + shift)]
            return [int(np.rint(v)) if knob.kind == "integer" else float(v) for v in values]
        choices = [False, True] if knob.kind == "bool" else list(knob.choices)
        return [choices[(len(choices) // 2 + shift) % len(choices)]]

    def complete(self, prompt: str, *, expert: str, template: str) -> LLMResult:
        del prompt
        names = [knob.name for knob in self.space.knobs]
        expert_names = [
            "memory_expert", "concurrency_expert", "optimizer_expert", "execution_expert",
            "storage_wal_expert", "concurrency_execution_expert", "storage_wal_expert",
            "optimizer_workload_expert", "workload_expert", "resource_expert",
        ]
        expert_index = expert_names.index(expert)
        template_names = ["mechanism_template", "workload_template", "resource_template", "role_template", "interaction_template"]
        template_index = template_names.index(template)
        priority = {"high": [], "medium": [], "low": []}
        recommendations = {}
        for index, knob in enumerate(self.space.knobs):
            domain_match = {
                "memory_expert": {"memory"},
                "concurrency_expert": {"concurrency"},
                "optimizer_expert": {"optimizer"},
                "execution_expert": {"execution"},
                "storage_wal_expert": {"storage"},
                "concurrency_execution_expert": {"concurrency", "execution"},
                "optimizer_workload_expert": {"optimizer"},
                "workload_expert": set(),
                "resource_expert": set(),
            }[expert]
            rank = 2 if knob.category in domain_match else (index + expert_index + template_index) % 3
            level = ("low", "medium", "high")[rank]
            priority[level].append(knob.name)
            values = self._values(knob.name, template_index - 1)
            item = {"recommended_values": values, "confidence": 0.65 + .05 * (rank), "reason": f"{expert} {template} mechanism prior"}
            if knob.kind in {"integer", "float"}:
                item["recommended_interval"] = [min(values), max(values)]
            recommendations[knob.name] = item
        relations = []
        numeric = [knob for knob in self.space.knobs if knob.kind in {"integer", "float"}]
        if len(numeric) >= 2:
            source, target = numeric[0], numeric[1]
            source_mid = (float(source.lower) + float(source.upper)) / 2
            low_target = self._values(target.name, -1)
            high_target = self._values(target.name, 1)
            relations.append({
                "source": source.name, "target": target.name,
                "reason": "resource competition changes the joint recommendation",
                "branches": [
                    {"source_interval": [source.lower, source_mid], "target_values": low_target},
                    {"source_interval": [source_mid, source.upper], "target_values": high_target},
                ],
            })
        payload = {"expert": expert, "template": template, "priority": priority, "recommendations": recommendations, "conditional_relations": relations}
        text = json.dumps(payload, sort_keys=True)
        return LLMResult(text=text, model=self.model, input_tokens=0, output_tokens=len(text.split()))

