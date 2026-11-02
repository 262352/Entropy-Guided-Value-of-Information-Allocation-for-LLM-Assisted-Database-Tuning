import json

from cognitune.evaluator import SyntheticEvaluator


def test_three_methods_observe_identical_initial_values(space):
    configs = space.sample_iid(13, 4)
    configs.insert(0, space.get_default())
    observations = []
    for _method in ("cognitune", "smac", "gp_bo"):
        evaluator = SyntheticEvaluator(space, 13)
        observations.append([evaluator.evaluate(item, trial).utility for trial, item in enumerate(configs)])
    assert observations[0] == observations[1] == observations[2]

