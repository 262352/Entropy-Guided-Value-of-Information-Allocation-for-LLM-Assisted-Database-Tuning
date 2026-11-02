from __future__ import annotations

from dataclasses import dataclass

from .typed_graph import GraphEdge, TypedRelationGraph


@dataclass(frozen=True)
class BlockDecision:
    source: str
    target: str
    lower_bound: float
    accepted: bool
    reason: str


def build_coordination_blocks(
    graph: TypedRelationGraph,
    max_block_size: int = 4,
    complexity_penalty: float = 0.02,
) -> tuple[list[tuple[str, ...]], list[BlockDecision]]:
    """Greedily merge strong soft edges when their evidence pays for complexity."""
    blocks = [{name} for name in sorted(graph.nodes)]
    decisions: list[BlockDecision] = []
    soft_edges = sorted(
        (edge for edge in graph.edges if edge.edge_type == "soft_relation"),
        key=lambda edge: (-edge.lower_bound, edge.source, edge.target),
    )

    def locate(name: str) -> set[str]:
        return next(block for block in blocks if name in block)

    for edge in soft_edges:
        left, right = locate(edge.source), locate(edge.target)
        if left is right:
            decisions.append(_decision(edge, False, "already_connected"))
            continue
        merged = left | right
        incremental_cost = complexity_penalty * max(1, len(merged) - 1)
        if len(merged) > max_block_size:
            decisions.append(_decision(edge, False, "block_size_limit"))
        elif edge.lower_bound <= incremental_cost:
            decisions.append(_decision(edge, False, "complexity_penalty"))
        else:
            blocks.remove(left)
            blocks.remove(right)
            blocks.append(merged)
            decisions.append(_decision(edge, True, "evidence_exceeds_cost"))
    normalized = sorted((tuple(sorted(block)) for block in blocks), key=lambda item: (item[0], len(item)))
    return normalized, decisions


def _decision(edge: GraphEdge, accepted: bool, reason: str) -> BlockDecision:
    return BlockDecision(edge.source, edge.target, edge.lower_bound, accepted, reason)

