"""Optional installed-official-core smoke, using a synthetic evaluator only."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from cognitune.config import load_search_space
from cognitune.evaluator import SyntheticEvaluator
from cognitune.search_space.initial_design import build_initial_design

ROOT = Path(__file__).resolve().parents[1]


def adapter():
    sys.path.insert(0, str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('twitter_adapter_smoke', ROOT/'scripts/compare_twitter.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sixty_knob_domains_and_shared_design_roundtrip():
    a = adapter()
    space = load_search_space(a.SPACE_PATH)
    assert len(space.knobs) == 60
    for k in space.knobs:
        values = [k.lower,k.default,k.upper] if k.kind in {'integer','float'} else [False,True] if k.kind == 'bool' else k.choices
        for value in values:
            actual = a.decode_one(k, a.encode_one(k, value))
            if isinstance(value, (int,float)):
                assert np.isclose(actual, value)
            else:
                assert actual == value
    design = build_initial_design(space, 0, 10)
    assert design['metadata']['lhs_used'] is False
    assert all(space.validate(c) for c in design['configs'])


def test_official_adapter_completes_budget_preserving_exact_shared_starts(space, tmp_path, monkeypatch):
    a = adapter()
    if not (a.OFFICIAL/'lamcts/MCTS.py').exists():
        pytest.skip('official MCTuner runtime not installed')
    monkeypatch.setattr(a, 'BUDGET', 5)
    initial = build_initial_design(space, 0, 3)['configs']
    chosen = ['cache_mb','worker_threads','join_strategy']
    selection = {'selected_knobs':chosen, 'knob_info':{
        n:{'knob_lb':space.by_name[n].lower,'knob_ub':space.by_name[n].upper} for n in chosen}}
    result = a.run_official(space, SyntheticEvaluator(space,0,'throughput',1000,.01),
                            initial, selection, tmp_path)
    rows = [json.loads(line) for line in (tmp_path/'trials.jsonl').read_text().splitlines()]
    assert result['trials'] == 5
    assert [r['config'] for r in rows[:3]] == initial
    assert all(space.validate(r['config']) for r in rows)
    assert rows[-1]['candidate_source'] == 'official_mcts_gp_ei'
