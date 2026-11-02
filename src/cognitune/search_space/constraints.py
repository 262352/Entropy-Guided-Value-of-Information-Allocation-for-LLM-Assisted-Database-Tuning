from __future__ import annotations

from typing import Any


def forbidden_matches(config: dict[str, Any], clause: dict[str, Any]) -> bool:
    return all(config.get(name) == value for name, value in clause.items())


def sum_constraint_satisfied(config: dict[str, Any], rule: dict[str, Any]) -> bool:
    total = 0.0
    for term in rule["terms"]:
        total += float(config.get(term["knob"], 0.0)) * float(term.get("multiplier", 1.0))
    return total <= float(rule["limit"])


def hard_constraints_satisfied(config: dict[str, Any], rules: list[dict[str, Any]]) -> bool:
    for rule in rules:
        if rule.get("type") == "sum_le" and not sum_constraint_satisfied(config, rule):
            return False
    return True

