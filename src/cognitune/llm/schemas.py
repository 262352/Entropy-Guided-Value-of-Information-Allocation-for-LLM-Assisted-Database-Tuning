from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PRIORITIES = ("high", "medium", "low")


@dataclass(frozen=True)
class Recommendation:
    recommended_values: tuple[Any, ...]
    recommended_interval: tuple[float, float] | None
    confidence: float
    reason: str


@dataclass(frozen=True)
class RelationBranch:
    source_interval: tuple[float, float] | None
    source_values: tuple[Any, ...]
    target_values: tuple[Any, ...]
    target_interval: tuple[float, float] | None


@dataclass(frozen=True)
class ConditionalRelation:
    source: str
    target: str
    reason: str
    branches: tuple[RelationBranch, ...]


@dataclass(frozen=True)
class ExpertResponse:
    expert: str
    template: str
    priority: dict[str, tuple[str, ...]]
    recommendations: dict[str, Recommendation]
    conditional_relations: tuple[ConditionalRelation, ...]
    unverified_relations: tuple[dict[str, Any], ...] = ()

