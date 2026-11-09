import json
import numpy as np
import pytest

from cognitune.adaptive.stable_search import StableSearch, build_mode_pool
from cognitune.agents.ownership import build_ownership, EXPERTS
from cognitune.config.schema import KnobSpec
from cognitune.search_space.space import SearchSpace
from cognitune.optimizers.cognitune_v3 import unit,value_at,bounded_patch,run_cognitune_v3
from cognitune.adaptive.proposals import StructuredMockProposer
from cognitune.evaluator import SyntheticEvaluator
from cognitune.search_space.initial_design import build_initial_design


def tiny():
    return SearchSpace([KnobSpec(n,'float',.5,0,1,category='memory') for n in 'abcd'])


def potentials():
    return {n:{'p_improve':.8 if n=='c' else .3,'expected_gain':.4 if n=='c' else .05,
               'interval':[.01,.6],'state':'verify'} for n in 'abcd'}


def test_predicted_ranking_cannot_evict_an_untested_direction():
    sp=tiny();c=StableSearch(sp,build_ownership(sp));p=potentials()
    c.select({'a','b'},p,[('c',)],0,[],2,.01)
    for t in range(1,30):
        active,report=c.select({'a','c'},p,[('c',)],t,[],2,.01)
        assert active=={'a','b'} and report['swaps']==[]


def test_review_swaps_are_bounded_and_need_real_evidence():
    sp=tiny();c=StableSearch(sp,build_ownership(sp));p=potentials()
    c.select({'a','b'},p,[('c',)],0,['a'],2,.01)
    for t,pos in [(1,.3),(2,.7)]:
        c.observe(sp.get_default(),{'a':.5,'b':pos,'c':.5,'d':.5},0,0,t,.01,unit,source='group:b')
    p['b']={'p_improve':0,'expected_gain':0,'interval':[-.02,0],'state':'pause'}
    active,report=c.select({'a','c'},p,[('c',)],6,['a'],2,.01)
    assert active=={'a','c'}
    assert report['swaps'][0]['exit_tests']=={'b':2}
    assert report['swaps'][0]['reason']=='measured_low_potential'
    assert c.state['nodes']['c']['tests']==0


def test_region_width_changes_on_covered_measured_evidence_without_llm_permission():
    sp=tiny();c=StableSearch(sp,build_ownership(sp));c.state['active']=['a']
    anchor=sp.get_default()
    # Both sides of the local region have been evaluated, without a gain.
    updates=[]
    for t,pos in enumerate([.39,.61,.4]):
        updates+=c.observe(anchor,{**anchor,'a':pos},0,0,t,.01,unit,source='group:a')
    assert updates and c.state['nodes']['a']['width']==pytest.approx(.12)
    # Repeated improving boundary evaluations expand the same search region.
    for t,pos in enumerate([.4,.6,.4],start=3):
        updates+=c.observe(anchor,{**anchor,'a':pos},.1,0,t,.01,unit,source='group:a')
    assert c.state['nodes']['a']['width']>.12
    assert {u['reason'] for u in updates}>={'covered_local_plateau_refine','repeated_boundary_improvement'}


def test_global_reference_runs_do_not_certify_local_coverage_or_saturation():
    sp=tiny();c=StableSearch(sp,build_ownership(sp));c.state['active']=['a']
    c.observe(sp.get_default(),{n:.9 for n in 'abcd'},0,0,1,.01,unit,source='reference',reference=True)
    assert c.state['nodes']['a']['tests']==0


def test_generated_group_candidates_preserve_interaction_and_audit_size():
    sp=tiny();ownership=build_ownership(sp);c=StableSearch(sp,ownership)
    c.state['active']=['a','b']
    expert='memory_expert';proposals={e:{'patches':[]} for e in EXPERTS}
    proposals[expert]={'patches':[{'a':.65,'b':.65}]}
    scopes={e:(['a','b'] if e==expert else []) for e in EXPERTS}
    regions={e:{} for e in EXPERTS};regions[expert]={n:{'center':.5,'halfwidth':.15} for n in ['a','b']}
    captured=[]
    audit,m,modes,q,channels=build_mode_pool(c,proposals,scopes,set('ab'),regions,sp.get_default(),sp,set(),
        np.random.default_rng(0),80,4,lambda cfg,key,audit=False:captured.append((cfg,key,audit)),unit,value_at,bounded_patch)
    assert sum(q.values())==80 and len({len(v) for v in audit.values()})==1
    assert all('a' in modes[key]['knobs'] and 'b' in modes[key]['knobs'] for _,key,_ in captured)
    assert any('explicit_patch' in kinds for kinds in channels.values())
    assert all(sp.validate(cfg) for cfg,_,_ in captured)


@pytest.mark.parametrize('kind,choices,default,recommended',[
    ('categorical',('a','b','c'),'a','c'),('bool',(),False,True)])
def test_discrete_group_sampling_uses_expert_values_instead_of_numeric_neighborhood(kind,choices,default,recommended):
    sp=SearchSpace([KnobSpec('choice',kind,default,choices=choices,category='memory'),
                   KnobSpec('x','float',.5,0,1,category='memory')])
    c=StableSearch(sp,build_ownership(sp));c.state['active']=['choice','x']
    expert='memory_expert'
    proposals={e:{'patches':[]} for e in EXPERTS}
    scopes={e:(['choice','x'] if e==expert else []) for e in EXPERTS}
    regions={e:{} for e in EXPERTS}
    regions[expert]={'choice':{'center':0,'halfwidth':.01,'values':[recommended]},
                     'x':{'center':.5,'halfwidth':.15}}
    class NoExploration:
        def __init__(self): self.base=np.random.default_rng(0)
        def random(self): return .5
        def integers(self,*args,**kwargs): return self.base.integers(*args,**kwargs)
        def uniform(self,*args,**kwargs): return self.base.uniform(*args,**kwargs)
    captured=[]
    build_mode_pool(c,proposals,scopes,{'choice','x'},regions,sp.get_default(),sp,set(),
        NoExploration(),40,4,lambda cfg,key,audit=False:captured.append(cfg),unit,value_at,bounded_patch)
    assert len(captured)>=4
    assert all(cfg['choice']==recommended for cfg in captured)


class Interrupt:
    def __call__(self,expert,context):
        if context['trial']==5: raise RuntimeError('interrupt stable')
        return StructuredMockProposer()(expert,context)


def test_stable_optimizer_executes_coverage_preserves_state_and_resumes(space,tmp_path):
    initial=build_initial_design(space,0,3)['configs']
    args=(space,SyntheticEvaluator(space,0,'throughput',1000,0),initial,14,0,60,tmp_path)
    with pytest.raises(RuntimeError,match='interrupt stable'):
        run_cognitune_v3(*args,proposer=Interrupt(),search_policy='stable',
            audit_size=2,posterior_samples=32,fantasies=4,active_target=4)
    before=(tmp_path/'trials.jsonl').read_bytes()
    cp=json.loads((tmp_path/'controller_checkpoint.json').read_text())
    assert cp['completed']==5 and cp['extra_state']['active']
    summary=run_cognitune_v3(*args,proposer=StructuredMockProposer(),search_policy='stable',resume=True,
        audit_size=2,posterior_samples=32,fantasies=4,active_target=4)
    assert summary['trials']==14 and summary['method_version']=='stable-space-v1'
    assert (tmp_path/'trials.jsonl').read_bytes().startswith(before)
    events=[json.loads(x) for x in (tmp_path/'round_diagnostics.jsonl').read_text().splitlines()]
    ps=[e for e in events if e['event']=='proposal']
    assert any(e['selection_reason']=='reserved_local_coverage' for e in ps)
    assert ps[-1]['selection_reason']=='qLogNEI'
    assert all(sum(e['hypothesis_quotas'].values())==40 for e in ps)
    assert all(len(e['active'])==4 for e in ps)
    assert len({tuple(e['active']) for e in ps[:6]})==1
    assert any(sum(e['actual_execution_coverage'].values())>0 for e in ps)


def test_low_single_knob_potential_cannot_break_a_supported_interaction():
    sp=tiny();c=StableSearch(sp,build_ownership(sp));p=potentials()
    c.select({'a','b'},p,[('c',)],0,[],2,.01)
    for n in 'ab':
        c.state['nodes'][n]['tests']=3
        c.state['nodes'][n]['tested_values']=[.2,.8];c.state['nodes'][n]['fresh_tests']=3
        p[n]={'p_improve':0,'expected_gain':0,'interval':[-.02,0],'state':'pause'}
    p['a+b']={'p_improve':.95,'expected_gain':.2,'interval':[.1,.3],'state':'exploit'}
    active,report=c.select({'a','c'},p,[('a','b'),('c',)],6,[],2,.01)
    assert active=={'a','b'} and report['swaps']==[]


def test_same_value_repetitions_do_not_satisfy_retirement_coverage():
    sp=tiny();c=StableSearch(sp,build_ownership(sp));p=potentials()
    c.select({'a','b'},p,[('c',)],0,['a'],2,.01)
    for t in range(1,5):
        c.observe(sp.get_default(),{**sp.get_default(),'b':.7},0,0,t,.01,unit,source='group:b')
    p['b']={'p_improve':0,'expected_gain':0,'interval':[-.01,0],'state':'pause'}
    active,report=c.select({'a','c'},p,[('c',)],12,['a'],2,.01)
    assert active=={'a','b'} and report['swaps']==[]


def test_anchor_change_keeps_history_but_requires_fresh_evidence_before_retirement():
    sp=tiny();c=StableSearch(sp,build_ownership(sp));p=potentials()
    c.select({'a','b'},p,[('c',)],0,['a'],2,.01)
    c.anchor_context(sp.get_default(),0,unit)
    for t,pos in [(1,.3),(2,.7)]:
        c.observe(sp.get_default(),{**sp.get_default(),'b':pos},0,0,t,.01,unit,source='group:b')
    assert c.anchor_context({**sp.get_default(),'a':.95},3,unit)
    assert c.state['nodes']['b']['tests']==2 and c.state['nodes']['b']['fresh_tests']==0
    p['b']={'p_improve':0,'expected_gain':0,'interval':[-.01,0],'state':'pause'}
    active,_=c.select({'a','c'},p,[('c',)],12,['a'],2,.01)
    assert active=={'a','b'}


def test_binary_knob_can_finish_coverage_without_two_impossible_alternative_values():
    sp=SearchSpace([KnobSpec('a','float',.5,0,1,category='memory'),
                    KnobSpec('b','bool',False,category='memory'),
                    KnobSpec('c','float',.5,0,1,category='memory')])
    c=StableSearch(sp,build_ownership(sp));p=potentials()
    c.select({'a','b'},p,[('c',)],0,['a'],2,.01)
    for t in [1,2]:
        c.observe(sp.get_default(),{**sp.get_default(),'b':True},0,0,t,.01,unit,source='group:b')
    p['b']={'p_improve':0,'expected_gain':0,'interval':[-.01,0],'state':'pause'}
    active,report=c.select({'a','c'},p,[('c',)],6,['a'],2,.01)
    assert active=={'a','c'} and report['swaps'][0]['exit_distinct_values']=={'b':1}
