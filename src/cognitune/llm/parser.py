from __future__ import annotations

import json
from typing import Any

from cognitune.search_space.space import SearchSpace
from .schemas import (
    PRIORITIES, ConditionalRelation, ExpertResponse, Recommendation, RelationBranch,
)


class StructuredResponseParser:
    def __init__(self, space: SearchSpace):
        self.space = space

    def _valid_value(self, knob_name: str, value: Any) -> bool:
        knob = self.space.by_name[knob_name]
        if knob.kind in {"integer", "float"}:
            return not isinstance(value, bool) and knob.lower <= float(value) <= knob.upper
        if knob.kind == "bool":
            return isinstance(value, bool)
        return value in knob.choices

    def _interval(self, knob_name: str, raw: Any) -> tuple[float, float] | None:
        if raw is None:
            return None
        if len(raw) != 2 or float(raw[0]) > float(raw[1]):
            raise ValueError(f"{knob_name}: invalid interval")
        if not self._valid_value(knob_name, raw[0]) or not self._valid_value(knob_name, raw[1]):
            raise ValueError(f"{knob_name}: interval is outside legal bounds")
        return float(raw[0]), float(raw[1])

    def parse(self, text: str) -> ExpertResponse:
        raw = json.loads(text)
        priority = raw.get("priority", {})
        if set(priority) != set(PRIORITIES):
            raise ValueError("priority must contain exactly high, medium, and low")
        flattened = [name for level in PRIORITIES for name in priority[level]]
        expected = set(self.space.by_name)
        if set(flattened) != expected or len(flattened) != len(expected):
            missing = sorted(expected - set(flattened))
            unknown = sorted(set(flattened) - expected)
            duplicates = sorted({name for name in flattened if flattened.count(name) > 1})
            raise ValueError(f"priority partition is invalid; missing={missing}, unknown={unknown}, duplicates={duplicates}")
        recommendations = {}
        for name, item in raw.get("recommendations", {}).items():
            if name not in expected:
                raise ValueError(f"unknown recommendation knob {name}")
            values = tuple(item.get("recommended_values", ()))
            if any(not self._valid_value(name, value) for value in values):
                raise ValueError(f"{name}: illegal recommended value")
            interval = self._interval(name, item.get("recommended_interval"))
            confidence = float(item.get("confidence", 1.0))
            if not 0 <= confidence <= 1:
                raise ValueError(f"{name}: confidence must be in [0,1]")
            recommendations[name] = Recommendation(values, interval, confidence, str(item.get("reason", "")))
        relations, unverified = [], []
        for item in raw.get("conditional_relations", []):
            source, target = item.get("source"), item.get("target")
            if source not in expected or target not in expected or source == target:
                raise ValueError("conditional relation has invalid source or target")
            branches = item.get("branches") or []
            if len(branches) < 2:
                unverified.append(dict(item))
                continue
            parsed_branches = []
            for branch in branches:
                target_values = tuple(branch.get("target_values", ()))
                target_interval = self._interval(target, branch.get("target_interval"))
                if not target_values and target_interval is None:
                    raise ValueError("each branch needs target_values or target_interval")
                if any(not self._valid_value(target, value) for value in target_values):
                    raise ValueError(f"{target}: illegal branch value")
                source_interval = self._interval(source, branch.get("source_interval"))
                source_values = tuple(branch.get("source_values", ()))
                if source_interval is None and not source_values:
                    raise ValueError("each branch needs a source interval or values")
                if any(not self._valid_value(source, value) for value in source_values):
                    raise ValueError(f"{source}: illegal branch value")
                parsed_branches.append(RelationBranch(source_interval, source_values, target_values, target_interval))
            relations.append(ConditionalRelation(source, target, str(item.get("reason", "")), tuple(parsed_branches)))
        return ExpertResponse(
            expert=str(raw["expert"]), template=str(raw["template"]),
            priority={level: tuple(priority[level]) for level in PRIORITIES},
            recommendations=recommendations, conditional_relations=tuple(relations),
            unverified_relations=tuple(unverified),
        )
