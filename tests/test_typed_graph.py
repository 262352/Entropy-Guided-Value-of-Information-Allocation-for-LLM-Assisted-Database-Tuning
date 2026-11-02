from cognitune.graph import TypedRelationGraph, calibrate_block


def test_graph_keeps_edge_types_separate(space):
    relations = [{
        "source": "cache_mb", "target": "worker_threads", "weight": 0.3,
        "lower_bound": 0.25, "support_count": 3, "mean": 0.3, "std": 0.05,
        "status": "unverified",
    }]
    graph = TypedRelationGraph.build(space, relations, relation_threshold=0.1)
    types = {edge.edge_type for edge in graph.edges}
    assert "activation" in types
    assert "soft_relation" in types
    assert all(edge.edge_type == "soft_relation" for edge in graph.incident("cache_mb", {"soft_relation"}))


def test_low_confidence_relation_is_filtered(space):
    item = {"source": "cache_mb", "target": "worker_threads", "weight": 0.01, "lower_bound": 0.01, "support_count": 1, "mean": 0.02, "std": 0.01}
    graph = TypedRelationGraph.build(space, [item], relation_threshold=0.1)
    assert not any(edge.edge_type == "soft_relation" for edge in graph.edges)


def test_real_block_challenge_can_retain_or_demote_soft_edge(space):
    item = {"source": "cache_mb", "target": "worker_threads", "weight": 0.2, "lower_bound": 0.2, "support_count": 1, "mean": 0.2, "std": 0.0}
    graph = TypedRelationGraph.build(space, [item], relation_threshold=0.1)
    retained = calibrate_block(graph, {"cache_mb", "worker_threads"}, 0.05)
    assert retained[0]["status"] == "retained"
    demoted = calibrate_block(graph, {"cache_mb", "worker_threads"}, -0.01)
    assert demoted[0]["status"] == "demoted"
