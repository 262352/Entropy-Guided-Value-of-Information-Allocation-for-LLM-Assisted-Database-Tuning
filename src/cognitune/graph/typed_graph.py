from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitune.search_space.space import SearchSpace


@dataclass(frozen=True)
class GraphNode:
    name: str
    kind: str
    category: str
    protected: bool = False


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    lower_bound: float = 1.0
    support_count: int = 1
    status: str = "verified"
    metadata: dict[str, Any] = field(default_factory=dict)


class TypedRelationGraph:
    """Separates executable constraints from uncertain cognitive relations."""

    def __init__(self, nodes: list[GraphNode], edges: list[GraphEdge]):
        self.nodes = {node.name: node for node in nodes}
        self.edges = list(edges)

    @classmethod
    def build(
        cls,
        space: SearchSpace,
        relation_stats: list[dict[str, Any]],
        relation_threshold: float = 0.01,
        protected: set[str] | None = None,
    ) -> "TypedRelationGraph":
        protected = protected or set()
        nodes = [
            GraphNode(knob.name, knob.kind, knob.category, knob.name in protected)
            for knob in space.knobs
        ]
        edges: list[GraphEdge] = []
        for knob in space.knobs:
            if knob.condition:
                edges.append(GraphEdge(
                    knob.condition.parent,
                    knob.name,
                    "activation",
                    metadata={"values": list(knob.condition.values)},
                ))
        for rule_index, rule in enumerate(space.hard_constraints):
            names = [term["knob"] for term in rule.get("terms", [])]
            for index, source in enumerate(names):
                for target in names[index + 1:]:
                    edges.append(GraphEdge(
                        source,
                        target,
                        "hard_constraint",
                        metadata={"rule_index": rule_index, "rule": rule},
                    ))
        for item in relation_stats:
            if item["lower_bound"] < relation_threshold:
                continue
            edges.append(GraphEdge(
                item["source"], item["target"], "soft_relation",
                weight=float(item["weight"]), lower_bound=float(item["lower_bound"]),
                support_count=int(item["support_count"]), status=str(item.get("status", "unverified")),
                metadata={"mean": item["mean"], "std": item["std"]},
            ))
        return cls(nodes, edges)

    def incident(self, name: str, edge_types: set[str] | None = None) -> list[GraphEdge]:
        return [
            edge for edge in self.edges
            if (edge.source == name or edge.target == name)
            and (edge_types is None or edge.edge_type in edge_types)
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges],
        }

