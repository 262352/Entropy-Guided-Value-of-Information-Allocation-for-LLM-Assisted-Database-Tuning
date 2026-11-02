from types import SimpleNamespace

import pytest

from cognitune.config.schema import KnobSpec
from cognitune.uncertainty.relation_sensitivity import relation_sensitivity


TARGET = KnobSpec("target", "integer", 5, 1, 10)


def relation(branch_values):
    branches = tuple(SimpleNamespace(target_values=tuple(values), target_interval=None) for values in branch_values)
    return SimpleNamespace(branches=branches)


def test_identical_branches_have_zero_sensitivity():
    assert relation_sensitivity(relation([(2,), (2,)]), TARGET) == pytest.approx(0.0)


def test_distinct_deterministic_branches_are_sensitive():
    assert relation_sensitivity(relation([(2,), (9,)]), TARGET) > 0.1


def test_missing_branch_is_not_computable():
    assert relation_sensitivity(relation([(2,)]), TARGET) is None

