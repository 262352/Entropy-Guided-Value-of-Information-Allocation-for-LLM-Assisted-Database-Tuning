import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from cognitune.adaptive.prompt_context import full_execution_history, shared_history_prompt, token_usage
from cognitune.adaptive.proposals import FeedbackExpertProposer
from cognitune.experiment.recorder import TrialRecord


def context():
    domain={'name':'x','kind':'integer','lower':1,'upper':10,'default':3}
    return {'allowed_knobs':[domain],'global_knob_domains':[domain],
            'workload':{'name':'test','clients':16},'anchor':{'x':3},'trial':10,
            'feedback':[{'trial':i,'config':{'x':i%10+1},'raw_metric':100+i,
                         'status':'SUCCESS'} for i in range(10)]}


def test_all_experts_share_history_prefix_even_when_scope_anchor_and_quota_change():
    ctx=context();a,layout=shared_history_prompt('a',ctx)
    other={**ctx,'anchor':{'x':9},'candidate_quota':42,'allowed_knobs':[],'trial':11}
    b,lb=shared_history_prompt('b',other)
    assert a[:layout['shared_prefix_characters']]==b[:lb['shared_prefix_characters']]
    assert layout['shared_prefix_sha256']==lb['shared_prefix_sha256']
    assert json.loads(a)['feedback']==ctx['feedback'] and json.loads(b)['expert']=='b'
    assert json.loads(b)['allowed_knobs']==[]  # Full knowledge never widens writable scope.


def test_appended_history_preserves_previous_rows_without_volatile_prefix():
    ctx=context();a,layout=shared_history_prompt('a',ctx)
    next_context={**ctx,'trial':11,'anchor':{'x':7},'feedback':ctx['feedback']+[
        {'trial':10,'config':{'x':6},'raw_metric':120,'status':'SUCCESS'}]}
    b,_=shared_history_prompt('a',next_context)
    # The old closing history bracket changes to a comma, after all old rows.
    assert os.path.commonprefix([a,b])==a[:layout['shared_prefix_characters']-1]
    assert len(json.loads(b)['feedback'])==11


def test_serialization_is_stable_under_dictionary_and_domain_input_order():
    ctx=context();other={k:v for k,v in reversed(list(ctx.items()))}
    other['workload']={'clients':16,'name':'test'}
    assert shared_history_prompt('a',ctx)==shared_history_prompt('a',other)


def test_full_history_contains_all_rows_failures_and_inactive_knobs():
    records=[TrialRecord(i,'cognitune_v3',0,{'inactive':i},100+i,0.,0.,100.,
        'CRASH' if i==4 else 'SUCCESS','test',state={'changed_knobs':['z','a']}) for i in range(49)]
    history=full_execution_history(records)
    assert len(history)==49 and [r['trial'] for r in history]==list(range(49))
    assert history[4]['status']=='CRASH' and history[0]['config']=={'inactive':0}
    assert full_execution_history(records[:10])==history[:10]
    assert history[0]['changed_knobs']==['a','z']


@pytest.mark.parametrize('usage,expected',[
    ({'prompt_tokens':100,'completion_tokens':10,'prompt_cache_hit_tokens':80,'prompt_cache_miss_tokens':20},(80,20,.8)),
    ({'prompt_tokens':100,'prompt_tokens_details':{'cached_tokens':70}},(70,30,.7)),
    ({'input_tokens':100,'input_tokens_details':{'cached_tokens':60}},(60,40,.6)),
    ({'prompt_cache_hit_tokens':10,'prompt_cache_miss_tokens':90},(10,90,.1)),
    ({'prompt_tokens':100},(None,None,None)),
    ({},(None,None,None)),
])
def test_usage_uses_actual_provider_counts_and_leaves_missing_telemetry_unknown(usage,expected):
    stats=token_usage(usage)
    assert (stats['cache_hit_tokens'],stats['cache_miss_tokens'],stats['cache_hit_rate'])==expected


def test_invalid_reply_retry_usage_is_counted_and_local_reuse_is_separate(tmp_path):
    calls=[]
    def post(payload):
        calls.append(json.loads(json.dumps(payload)))
        value=11 if len(calls)==1 else 4
        return {'id':str(len(calls)),'choices':[{'finish_reason':'stop','message':{
            'content':json.dumps({'regions':{'x':{'center':value}}})}}],
            'usage':{'prompt_tokens':100,'prompt_cache_hit_tokens':80,'prompt_cache_miss_tokens':20}}
    p=FeedbackExpertProposer(SimpleNamespace(model='fake',_post=post),tmp_path)
    answer=p('a',context());assert answer['regions']['x']['center']==4
    p('a',context());assert len(calls)==2
    assert calls[0]['messages']==calls[1]['messages'][:2]
    rows=[json.loads(l) for l in (tmp_path/'request_metrics.jsonl').read_text().splitlines()]
    assert [r['origin'] for r in rows]==['provider','provider','local_response']
    spec=importlib.util.spec_from_file_location('cache_report',Path(__file__).resolve().parents[1]/'scripts/report_llm_cache.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    report=m.report(tmp_path)['total']
    assert report['provider_responses']==2 and report['local_response_reuses']==1
    assert report['cache_hit_tokens']==160 and report['cache_hit_rate_for_reported_responses']==.8
