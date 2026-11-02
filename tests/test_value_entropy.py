from types import SimpleNamespace

from cognitune.config.schema import KnobSpec
from cognitune.uncertainty import value_statistics


KNOB = KnobSpec("x", "float", 5.0, 0.0, 10.0)


def response(interval, values=()):
    recommendation = SimpleNamespace(recommended_values=tuple(values), recommended_interval=interval, confidence=1.0)
    return SimpleNamespace(recommendations={"x": recommendation})


def test_identical_ranges_have_no_disagreement():
    value = value_statistics([response((2, 4)), response((2, 4))], [KNOB])["x"]["value_entropy"]
    assert value < 1e-10


def test_conflicting_narrow_ranges_exceed_single_wide_opinion():
    conflict = value_statistics([response((1, 2)), response((8, 9))], [KNOB])["x"]["value_entropy"]
    wide = value_statistics([response((1, 9))], [KNOB])["x"]["value_entropy"]
    assert conflict > wide


def test_nonoverlapping_point_recommendations_disagree():
    value = value_statistics([response(None, (1,)), response(None, (9,))], [KNOB])["x"]["value_entropy"]
    assert value > 0

