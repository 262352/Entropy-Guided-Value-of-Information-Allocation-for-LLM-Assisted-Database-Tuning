import pytest

from cognitune.evaluator import SyntheticEvaluator


def test_measurement_is_reproducible_by_seed_and_config(space):
    config = space.sample_iid(9, 1)[0]
    first = SyntheticEvaluator(space, 3).evaluate(config, 0)
    second = SyntheticEvaluator(space, 3).evaluate(config, 99)
    assert first.raw_metric == pytest.approx(second.raw_metric)
    assert first.utility == pytest.approx(second.utility)


def test_default_utility_is_only_deterministic_noise(space):
    result = SyntheticEvaluator(space, 0, noise_std=0).evaluate(space.get_default(), 0)
    assert result.utility == pytest.approx(0.0)

