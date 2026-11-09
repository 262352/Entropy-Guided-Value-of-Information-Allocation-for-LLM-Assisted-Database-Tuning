import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from cognitune.config import load_search_space
from cognitune.search_space.initial_design import build_initial_design,save_initial_design

ROOT=Path(__file__).resolve().parents[1]


def adapter():
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('stable_protocol_adapter',ROOT/'scripts/compare_twitter.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def source_run(tmp_path,a):
    source=tmp_path/'old';source.mkdir()
    (source/'protocol.json').write_text(json.dumps({'model':a.MODEL,'metric':a.harness.METRIC,'n_init':10,'seed':0}))
    space=load_search_space(a.SPACE_PATH)
    save_initial_design(source/'initial_design/postgresql_twitter_seed_0.json',build_initial_design(space,0,10))
    (source/'profile.json').write_text(json.dumps({'workload':a.profile(space)['workload']}))
    for folder in ['priors/mctuner_official','llm_cache/mctuner_official','llm_cache/cognitune_v3','measurements/default_before']:
        p=source/folder;p.mkdir(parents=True);(p/'old.json').write_text('{"preserve":true}')
    return source


def test_preparation_reuse_preserves_exact_iid_but_never_reuses_tps_or_our_feedback(tmp_path):
    a=adapter();source=source_run(tmp_path,a);a.OUT=tmp_path/'new'
    a.reuse_preparation(source)
    assert (a.OUT/'priors/mctuner_official/old.json').exists()
    assert (a.OUT/'llm_cache/mctuner_official/old.json').exists()
    assert not (a.OUT/'measurements').exists()
    assert not (a.OUT/'default_before.json').exists()
    assert not (a.OUT/'llm_cache/cognitune_v3').exists()
    design='initial_design/postgresql_twitter_seed_0.json'
    assert (source/design).read_bytes()==(a.OUT/design).read_bytes()
    original=(source/'measurements/default_before/old.json').read_bytes()
    a.reuse_preparation(source)
    assert (source/'measurements/default_before/old.json').read_bytes()==original


def test_preparation_from_other_workload_is_rejected_before_copy(tmp_path):
    a=adapter();source=source_run(tmp_path,a);a.OUT=tmp_path/'new'
    (source/'profile.json').write_text('{"workload":"other workload"}')
    with pytest.raises(ValueError,match='another workload'):
        a.reuse_preparation(source)
    assert not a.OUT.exists()


def test_resume_completes_missing_default_postcheck_without_repeating_saved_checks(tmp_path,monkeypatch):
    from types import SimpleNamespace
    a=adapter();a.OUT=tmp_path;a.ALLOW_UNSTABLE=False;calls=[]
    class FakeEvaluator:
        def __init__(self,*args): pass
        def evaluate(self,config,trial):
            calls.append(trial)
            return SimpleNamespace(as_dict=lambda:{'status':'SUCCESS','raw_metric':101})
    monkeypatch.setattr(a,'LocalBenchBaseEvaluator',FakeEvaluator)
    a.finish_default_check(None,'cognitune_v3',{},100)
    assert calls==[0,1,2]
    path=tmp_path/'default_after_cognitune_v3.json';before=path.read_bytes()
    a.finish_default_check(None,'cognitune_v3',{},100)
    assert calls==[0,1,2] and path.read_bytes()==before
    with pytest.raises(RuntimeError,match='drift exceeds'):
        a.finish_default_check(None,'cognitune_v3',{},50)
