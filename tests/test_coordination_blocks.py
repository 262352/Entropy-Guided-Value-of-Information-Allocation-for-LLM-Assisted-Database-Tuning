from cognitune.graph import GraphEdge, GraphNode, TypedRelationGraph, build_coordination_blocks


def graph():
    nodes = [GraphNode(str(index), "float", "test") for index in range(6)]
    edges = [
        GraphEdge("0", "1", "hard_constraint", 10, 10),
        GraphEdge("1", "2", "activation", 10, 10),
        GraphEdge("0", "2", "soft_relation", .8, .8),
        GraphEdge("2", "3", "soft_relation", .7, .7),
        GraphEdge("3", "4", "soft_relation", .6, .6),
        GraphEdge("4", "5", "soft_relation", .5, .5),
    ]
    return TypedRelationGraph(nodes, edges)


def test_deterministic_strong_first_bounded_blocks():
    first, decisions = build_coordination_blocks(graph(), max_block_size=4, complexity_penalty=.01)
    second, _ = build_coordination_blocks(graph(), max_block_size=4, complexity_penalty=.01)
    assert first == second
    assert max(map(len, first)) <= 4
    assert decisions[0].source == "0" and decisions[0].target == "2"
    block = next(set(item) for item in first if "0" in item)
    assert "1" not in block

