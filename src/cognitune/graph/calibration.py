from __future__ import annotations

from dataclasses import replace

from .typed_graph import TypedRelationGraph


def calibrate_block(graph: TypedRelationGraph, changed_knobs: set[str], joint_gain: float, tolerance: float = 0.01) -> list[dict]:
    """Update soft-edge reliability from a block challenge against its incumbent baseline."""
    updates = []
    revised = []
    for edge in graph.edges:
        involved = edge.edge_type == "soft_relation" and {edge.source, edge.target}.issubset(changed_knobs)
        if not involved:
            revised.append(edge)
            continue
        retained = joint_gain > tolerance
        factor = 1.1 if retained else 0.5
        status = "retained" if retained else "demoted"
        updated = replace(
            edge,
            weight=max(0.0, edge.weight * factor),
            lower_bound=max(0.0, edge.lower_bound * factor),
            status=status,
            support_count=edge.support_count + 1,
            metadata=edge.metadata | {"last_joint_gain": joint_gain},
        )
        revised.append(updated)
        updates.append({
            "source": edge.source, "target": edge.target, "joint_gain": joint_gain,
            "old_weight": edge.weight, "new_weight": updated.weight, "status": status,
        })
    graph.edges = revised
    return updates

