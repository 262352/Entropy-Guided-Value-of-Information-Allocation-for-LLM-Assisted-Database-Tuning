from cognitune.state import KnobState, StateManager


def test_state_transitions_are_reversible_and_protected():
    stats = {name: {"priority_score": score} for name, score in {"a": .9, "b": .1, "protected": 0}.items()}
    manager = StateManager.initialize(stats, {"protected"}, active_fraction=.34)
    assert manager.statuses["protected"].state == KnobState.ACTIVE
    manager.update(3, {"b": 0}, {"b": 0}, {"b": 3})
    manager.update(4, {"b": 0}, {"b": 0}, {"b": 4})
    assert manager.statuses["b"].state == KnobState.DORMANT
    manager.update(5, {"b": .2}, {"b": 0}, {"b": 5})
    assert manager.statuses["b"].state == KnobState.ACTIVE
    assert manager.statuses["protected"].state == KnobState.ACTIVE



def test_active_budget_caps_the_main_search_space():
    stats = {
        f"k{index}": {"priority_score": (30 - index) / 30, "selection_entropy": (index % 5) / 4}
        for index in range(30)
    }
    manager = StateManager.initialize(stats, exploit_budget=5, challenge_budget=2, challenge_interval=5)
    active = manager.names(KnobState.ACTIVE)
    assert len(active) == 7
    assert len(manager.names(KnobState.WATCH)) == 23

    uncertainty = {name: 1.0 for name in stats}
    manager.update(5, {name: 0.0 for name in stats}, uncertainty, {name: 5 for name in stats},
                   priority_scores={name: item["priority_score"] for name, item in stats.items()})
    assert len(manager.names(KnobState.ACTIVE)) == 7
    assert manager.challenge_names()
