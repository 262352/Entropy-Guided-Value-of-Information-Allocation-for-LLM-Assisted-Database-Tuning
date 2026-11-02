from types import SimpleNamespace

import pytest

from cognitune.uncertainty import priority_statistics


def responses(levels):
    return [SimpleNamespace(priority={name: (("k",) if name == level else ()) for name in ("high", "medium", "low")}) for level in levels]


def test_unanimous_priorities_have_equal_low_entropy_and_distinct_scores():
    high = priority_statistics(responses(["high"] * 30), ["k"])["k"]
    medium = priority_statistics(responses(["medium"] * 30), ["k"])["k"]
    low = priority_statistics(responses(["low"] * 30), ["k"])["k"]
    assert high["selection_entropy"] == pytest.approx(low["selection_entropy"])
    assert high["priority_score"] > medium["priority_score"] > low["priority_score"]
    assert high["selection_entropy"] < 0.25


def test_uniform_disagreement_is_near_maximum_entropy():
    item = priority_statistics(responses(["high", "medium", "low"] * 10), ["k"])["k"]
    assert item["selection_entropy"] == pytest.approx(1.0)

