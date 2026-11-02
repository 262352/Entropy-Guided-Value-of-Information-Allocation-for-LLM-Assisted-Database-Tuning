from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from cognitune.config.schema import KnobSpec
from .constraints import forbidden_matches, hard_constraints_satisfied
from .encoder import SpaceEncoder


class SearchSpace:
    def __init__(self, knobs: list[KnobSpec], forbidden=None, hard_constraints=None):
        if not knobs or len({knob.name for knob in knobs}) != len(knobs):
            raise ValueError("search space needs unique knobs")
        self.knobs = list(knobs)
        self.by_name = {knob.name: knob for knob in knobs}
        self.forbidden = list(forbidden or [])
        self.hard_constraints = list(hard_constraints or [])
        for knob in knobs:
            if knob.condition and knob.condition.parent not in self.by_name:
                raise ValueError(f"{knob.name}: unknown condition parent")
        self.encoder = SpaceEncoder(knobs)

    def get_default(self) -> dict[str, Any]:
        return self.repair({knob.name: knob.default for knob in self.knobs})

    def get_active_knobs(self, config: dict[str, Any]) -> set[str]:
        active = set()
        for knob in self.knobs:
            condition = knob.condition
            if condition is None or config.get(condition.parent) in condition.values:
                active.add(knob.name)
        return active

    def _sample_one(self, rng: np.random.Generator) -> dict[str, Any]:
        config: dict[str, Any] = {}
        ordered = [knob for knob in self.knobs if knob.condition is None] + [knob for knob in self.knobs if knob.condition is not None]
        for knob in ordered:
            if knob.condition and config.get(knob.condition.parent) not in knob.condition.values:
                continue
            if knob.kind == "integer":
                if knob.log:
                    raw = math.exp(rng.uniform(math.log(float(knob.lower)), math.log(float(knob.upper))))
                    value = int(np.clip(np.rint(raw), knob.lower, knob.upper))
                else:
                    value = int(rng.integers(int(knob.lower), int(knob.upper) + 1))
            elif knob.kind == "float":
                value = float(math.exp(rng.uniform(math.log(float(knob.lower)), math.log(float(knob.upper))))) if knob.log else float(rng.uniform(knob.lower, knob.upper))
            elif knob.kind == "bool":
                value = bool(rng.integers(0, 2))
            else:
                value = knob.choices[int(rng.integers(0, len(knob.choices)))]
            config[knob.name] = value
        return config

    def sample_iid(self, seed: int, n: int) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seed)
        result, seen = [], set()
        attempts = 0
        while len(result) < n:
            attempts += 1
            if attempts > max(10_000, n * 1_000):
                raise RuntimeError("unable to draw enough legal unique IID configurations")
            candidate = self._sample_one(rng)
            key = self.config_key(candidate)
            if key not in seen and self.validate(candidate):
                seen.add(key)
                result.append(candidate)
        return result

    def repair(self, config: dict[str, Any]) -> dict[str, Any]:
        fixed: dict[str, Any] = {}
        for knob in self.knobs:
            if knob.condition and fixed.get(knob.condition.parent) not in knob.condition.values:
                continue
            value = config.get(knob.name, knob.default)
            if knob.kind == "integer":
                value = int(np.clip(np.rint(float(value)), knob.lower, knob.upper))
            elif knob.kind == "float":
                value = float(np.clip(float(value), knob.lower, knob.upper))
            elif knob.kind == "bool":
                value = bool(value)
            elif value not in knob.choices:
                value = knob.default
            fixed[knob.name] = value
        for clause in self.forbidden:
            if forbidden_matches(fixed, clause):
                last = next(name for name in reversed(list(clause)) if name in self.by_name)
                fixed[last] = self.by_name[last].default
                if forbidden_matches(fixed, clause):
                    first = next(iter(clause))
                    fixed[first] = self.by_name[first].default
        for rule in self.hard_constraints:
            if rule.get("type") == "sum_le" and not hard_constraints_satisfied(fixed, [rule]):
                terms = rule["terms"]
                total = sum(float(fixed.get(t["knob"], 0)) * float(t.get("multiplier", 1)) for t in terms)
                if total > 0:
                    scale = float(rule["limit"]) / total
                    for term in terms:
                        name = term["knob"]
                        if name in fixed and self.by_name[name].kind in {"integer", "float"}:
                            fixed[name] = fixed[name] * scale
                    return self.repair(fixed) if not hard_constraints_satisfied(fixed, [rule]) else {
                        name: (int(np.rint(value)) if self.by_name[name].kind == "integer" else value)
                        for name, value in fixed.items()
                    }
        return fixed

    def validate(self, config: dict[str, Any]) -> bool:
        active = self.get_active_knobs(config)
        if set(config) != active:
            return False
        for name in active:
            knob, value = self.by_name[name], config[name]
            if knob.kind in {"integer", "float"} and not knob.lower <= value <= knob.upper:
                return False
            if knob.kind == "integer" and (isinstance(value, bool) or int(value) != value):
                return False
            if knob.kind == "bool" and not isinstance(value, bool):
                return False
            if knob.kind in {"categorical", "ordinal"} and value not in knob.choices:
                return False
        return not any(forbidden_matches(config, clause) for clause in self.forbidden) and hard_constraints_satisfied(config, self.hard_constraints)

    def encode(self, config: dict[str, Any]) -> np.ndarray:
        repaired = self.repair(config)
        return self.encoder.encode(repaired, self.get_active_knobs(repaired))

    def encode_many(self, configs: list[dict[str, Any]]) -> np.ndarray:
        return np.stack([self.encode(config) for config in configs], axis=0)

    def decode(self, vector) -> dict[str, Any]:
        return self.repair(self.encoder.decode(np.asarray(vector, dtype=np.float64)))

    def config_key(self, config: dict[str, Any]) -> str:
        return json.dumps(self.repair(config), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        payload = {
            "knobs": [knob.__dict__ | {"condition": knob.condition.__dict__ if knob.condition else None} for knob in self.knobs],
            "forbidden": self.forbidden, "hard_constraints": self.hard_constraints,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()

    def to_configspace(self, seed: int = 0):
        import ConfigSpace as CS
        from ConfigSpace.conditions import InCondition
        from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause

        cs = CS.ConfigurationSpace(seed=seed)
        hypers = {}
        for knob in self.knobs:
            if knob.kind == "integer":
                hp = CS.UniformIntegerHyperparameter(knob.name, int(knob.lower), int(knob.upper), default_value=int(knob.default), log=knob.log)
            elif knob.kind == "float":
                hp = CS.UniformFloatHyperparameter(knob.name, float(knob.lower), float(knob.upper), default_value=float(knob.default), log=knob.log)
            elif knob.kind == "ordinal":
                hp = CS.OrdinalHyperparameter(knob.name, list(knob.choices), default_value=knob.default)
            else:
                choices = [False, True] if knob.kind == "bool" else list(knob.choices)
                hp = CS.CategoricalHyperparameter(knob.name, choices, default_value=knob.default)
            cs.add_hyperparameter(hp)
            hypers[knob.name] = hp
        for knob in self.knobs:
            if knob.condition:
                cs.add_condition(InCondition(hypers[knob.name], hypers[knob.condition.parent], list(knob.condition.values)))
        for clause in self.forbidden:
            pieces = [ForbiddenEqualsClause(hypers[name], value) for name, value in clause.items()]
            cs.add_forbidden_clause(ForbiddenAndConjunction(*pieces))
        return cs
