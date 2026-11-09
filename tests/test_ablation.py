import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from cognitune.adaptive.ablation import Ablation, VARIANTS, project_patch
from cognitune.adaptive.proposals import StructuredMockProposer
from cognitune.adaptive.stable_search import StableSearch
from cognitune.agents.ownership import build_ownership
from cognitune.evaluator import SyntheticEvaluator
from cognitune.optimizers.cognitune_v3 import run_cognitune_v3, unit, value_at
from cognitune.search_space.initial_design import build_initial_design


class FastSurrogate:
    """Deterministic PSD surrogate for exercising complete controller paths."""
    def __init__(self, space): self.space=space
    def fit(self, configs, utilities): pass
    def joint_posterior(self, configs):
        x=np.array([self.space.encode(c) for c in configs])
        mean=.1*x.mean(axis=1)
        covariance=.02*np.exp(-np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
        return mean,covariance, .001
    def log_noisy_expected_improvement(self, configs, seed=0):
        return self.joint_posterior(configs)[0]


class MovingProposer(StructuredMockProposer):
    def __call__(self,expert,context):
        result=super().__call__(expert,context)
        for k in context['allowed_knobs']:
            if k['kind'] in {'integer','float'}:
                from cognitune.config.schema import KnobSpec
                result['regions'][k['name']]['center']=value_at(KnobSpec(**k),.2 if context['trial']%2 else .8)
        return result


def execute(space,path,variant,proposer=None,**kwargs):
    return run_cognitune_v3(space,SyntheticEvaluator(space,0,noise_std=0),
        build_initial_design(space,0,3)['configs'],15,0,40,path,
        proposer=proposer or MovingProposer(),search_policy='stable',ablation=variant,
        posterior_samples=32,fantasies=4,audit_size=2,active_target=4,
        surrogate_factory=FastSurrogate,**kwargs)


@pytest.mark.parametrize('variant',VARIANTS)
def test_variants_execute_same_budget_and_enforce_intervention(space,tmp_path,variant):
    summary=execute(space,tmp_path,variant)
    assert summary['trials']==15 and summary['ablation']['name']==variant
    events=[json.loads(x) for x in (tmp_path/'round_diagnostics.jsonl').read_text().splitlines()]
    proposals=[x for x in events if x['event']=='proposal']
    assert all(sum(x['candidate_quota'].values())==20 for x in proposals)
    assert all(sum(x['hypothesis_quotas'].values())==20 for x in proposals)
    if variant=='fixed_knobs':
        assert len({tuple(x['active']) for x in proposals})==1
        assert not any(x['selection_reason']=='reserved_reentry' for x in proposals)
    if variant=='fixed_regions':
        updates=[r for x in proposals[1:] for expert in x['region_changes'].values() for r in expert.values()]
        assert updates and all(r['before']==r['after'] for r in updates)
        assert any(r['requested']['center']!=value_at(space.by_name[n],r['before']['center'])
                   for x in proposals[1:] for expert in x['region_changes'].values() for n,r in expert.items())
        assert not any(x['range_updates'] for x in events if x['event']=='outcome')
    if variant=='uniform_budget':
        assert all(max(x['hypothesis_quotas'].values())-min(x['hypothesis_quotas'].values())<=1 for x in proposals)
        assert all(len(set(x['candidate_quota'].values()))==1 for x in proposals)
    if variant=='nei_only':
        assert not any(x['selection_reason']=='decision_information' for x in proposals)


@pytest.mark.parametrize('variant',VARIANTS[1:])
def test_variant_checkpoint_resume_is_exact_and_mismatch_is_rejected(space,tmp_path,variant):
    class Interrupt(MovingProposer):
        def __call__(self,expert,context):
            if context['trial']==6: raise RuntimeError('stop here')
            return super().__call__(expert,context)
    expected=tmp_path/'expected';actual=tmp_path/'actual'
    execute(space,expected,variant)
    with pytest.raises(RuntimeError,match='stop here'): execute(space,actual,variant,Interrupt())
    before=(actual/'trials.jsonl').read_bytes()
    with pytest.raises(ValueError,match='protocol differs'):
        execute(space,actual,'full',resume=True)
    assert before==(actual/'trials.jsonl').read_bytes()
    execute(space,actual,variant,resume=True)
    rows=lambda p:[json.loads(x) for x in (p/'trials.jsonl').read_text().splitlines()]
    assert [r['config'] for r in rows(expected)]==[r['config'] for r in rows(actual)]
    assert (actual/'trials.jsonl').read_bytes().startswith(before)


def test_fixed_region_projection_cannot_be_bypassed_by_explicit_patch(space):
    regions={'random_page_cost':{'center':.5,'halfwidth':.1},
             'join_strategy':{'center':0,'halfwidth':.1,'values':['nested']}}
    raw={'random_page_cost':8.,'join_strategy':'merge'}
    projected=project_patch(raw,regions,space,unit,value_at)
    assert unit(space.by_name['random_page_cost'],projected['random_page_cost'])==pytest.approx(.6)
    assert projected['join_strategy']=='nested' and raw['join_strategy']=='merge'


def test_fixed_knobs_retains_tested_low_potential_direction(space):
    c=StableSearch(space,build_ownership(space),ablation='fixed_knobs')
    c.select({'cache_mb'},{},[],3,[],1,.01)
    active,report=c.select({'worker_threads'},{},[],100,[],1,.01)
    assert active=={'cache_mb'} and report['reason']=='ablation_fixed_knobs'


def test_fixed_regions_keeps_real_coverage_but_disables_an_actual_width_trigger(space):
    states=[]
    for variant in ['full','fixed_regions']:
        c=StableSearch(space,build_ownership(space),ablation=variant)
        c.state['active']=['random_page_cost'];anchor=space.get_default();changes=[]
        for trial,position in enumerate([-.11,.11,-.1]):
            cfg={**anchor,'random_page_cost':value_at(space.by_name['random_page_cost'],
                 unit(space.by_name['random_page_cost'],anchor['random_page_cost'])+position)}
            changes+=c.observe(anchor,cfg,0,0,trial,.01,unit,source='test')
        states.append((c.state['nodes']['random_page_cost'],changes))
    assert states[0][0]['width']==pytest.approx(.12) and states[0][1]
    assert states[1][0]['width']==.15 and not states[1][1]
    assert states[0][0]['tests']==states[1][0]['tests']==3


def test_nei_ablation_blocks_information_selection_even_when_trigger_is_satisfied(space,tmp_path,monkeypatch):
    import cognitune.optimizers.cognitune_v3 as optimizer
    class CalibratedSurrogate(FastSurrogate):
        def joint_posterior(self,configs):
            m,c,_=super().joint_posterior(configs)
            return m,c,10.0  # Prequential residuals fit the predicted interval.
    def significant_information(mean,covariance,noise,audit_count,sources,indices,*args):
        return np.ones(len(indices)),np.zeros(len(indices)),{():1.0}
    monkeypatch.setattr(optimizer,'information_gains',significant_information)
    observed={}
    for variant in ['full','nei_only']:
        p=tmp_path/variant
        run_cognitune_v3(space,SyntheticEvaluator(space,0,noise_std=0),
            build_initial_design(space,0,3)['configs'],22,0,40,p,proposer=MovingProposer(),
            search_policy='stable',ablation=variant,posterior_samples=32,fantasies=4,
            audit_size=2,active_target=4,surrogate_factory=CalibratedSurrogate)
        rows=[json.loads(x) for x in (p/'round_diagnostics.jsonl').read_text().splitlines()]
        proposals=[r for r in rows if r['event']=='proposal']
        assert any(r['calibrated'] for r in proposals)
        observed[variant]=sum(r['selection_reason']=='decision_information' for r in proposals)
    assert observed['full']>0 and observed['nei_only']==0


def test_variant_names_and_policy_combinations_fail_before_creating_trace(space,tmp_path):
    with pytest.raises(ValueError,match='unknown ablation'): Ablation('typo')
    with pytest.raises(ValueError,match='require the stable'):
        run_cognitune_v3(space,None,[{},{}],3,0,40,tmp_path,proposer=None,ablation='fixed_knobs')
    assert not (tmp_path/'trials.jsonl').exists()


def test_shell_runner_routes_five_separate_runs_without_starting_database(tmp_path):
    scripts=tmp_path/'scripts';scripts.mkdir()
    root=Path(__file__).resolve().parents[1]
    shutil.copy2(root/'scripts/run_twitter_ablation.sh',scripts)
    (scripts/'run_twitter_compare.sh').write_text(
        '#!/usr/bin/env bash\nprintf "%s %s\\n" "$TWITTER_RUN_DIR" "$*" >> calls.txt\n')
    subprocess.run(['bash',str(scripts/'run_twitter_ablation.sh'),'all','--resume'],check=True,capture_output=True)
    calls=(tmp_path/'calls.txt').read_text().splitlines()
    assert len(calls)==5
    for variant,line in zip(VARIANTS,calls):
        assert f'twitter_ablation_50_v2/{variant} cognitune_v3' in line
        assert f'--ablation {variant}' in line and '--resume' in line
    invalid=subprocess.run(['bash',str(scripts/'run_twitter_ablation.sh'),'all','--ablation=full'],capture_output=True)
    assert invalid.returncode==2 and len((tmp_path/'calls.txt').read_text().splitlines())==5


def test_full_history_reaches_bootstrap_proposals_and_joint_reviews_including_failures(space,tmp_path):
    calls=[];reviews=[]
    class Capture(MovingProposer):
        def __call__(self,expert,context):
            calls.append(context)
            result=super().__call__(expert,context)
            if expert=='memory_expert' and context.get('phase')!='structure_bootstrap':
                result['joint_hypotheses']=[{'patch':{'cache_mb':2048,'worker_threads':4}}]
            return result
        def review(self,owners,hypothesis,context):
            reviews.append(context)
            return {e:{'patches':[hypothesis['patch']]} for e in owners}
    class OneFailure(SyntheticEvaluator):
        def evaluate(self,config,trial):
            result=super().evaluate(config,trial)
            if trial==7:
                result.status='CRASH';result.raw_metric=0.;result.utility=-1.
            return result
    initial=build_initial_design(space,0,3)['configs']
    run_cognitune_v3(space,OneFailure(space,0,noise_std=0),initial,15,0,40,tmp_path,
        proposer=Capture(),search_policy='stable',posterior_samples=32,fantasies=4,audit_size=2,
        active_target=4,surrogate_factory=FastSurrogate)
    assert reviews and any(c.get('phase')=='structure_bootstrap' for c in calls)
    for context in calls+reviews:
        assert len(context['feedback'])==context['trial']
        assert [r['trial'] for r in context['feedback']]==list(range(context['trial']))
        assert len(context['global_knob_domains'])==len(space.knobs)
        if context['trial']>7: assert context['feedback'][7]['status']=='CRASH'
    assert len(calls[-1]['feedback'])==14
