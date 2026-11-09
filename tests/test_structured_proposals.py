import json
from types import SimpleNamespace
import pytest

from cognitune.adaptive.proposals import FeedbackExpertProposer, validate_proposal


CONTEXT = {'allowed_knobs':[{'name':'x','kind':'integer','lower':1,'upper':10,'default':3}]}


@pytest.mark.parametrize('center,width', [('open_datasync',0),(None,None),(0,.1)])
def test_discrete_hold_accepts_actual_choice_and_never_decodes_numeric_index(center,width):
    from cognitune.adaptive.proposals import normalize_proposal
    context={'allowed_knobs':[{'name':'wal_sync_method','kind':'categorical',
        'choices':['fsync','fdatasync','open_sync','open_datasync']}],
        'anchor':{'wal_sync_method':'open_datasync'}}
    raw={'regions':{'wal_sync_method':{'center':center,'halfwidth':width,'action':'hold'}}}
    answer,repairs=normalize_proposal(raw,context)
    validate_proposal(answer,context)
    assert answer['regions']['wal_sync_method']['center']=='open_datasync'
    assert 'halfwidth' not in answer['regions']['wal_sync_method']
    assert repairs and raw['regions']['wal_sync_method']['halfwidth']==width


@pytest.mark.parametrize('region', [
    {'center':0,'action':'shift'}, {'center':'invented','action':'hold'},
    {'values':'fsync'}, {'values':[],'action':'shift'}])
def test_discrete_invalid_changes_are_not_silently_replaced(region):
    from cognitune.adaptive.proposals import normalize_proposal
    context={'allowed_knobs':[{'name':'wal_sync_method','kind':'categorical',
        'choices':['fsync','open_datasync']}], 'anchor':{'wal_sync_method':'open_datasync'}}
    with pytest.raises(ValueError):
        answer,_=normalize_proposal({'regions':{'wal_sync_method':region}},context)
        validate_proposal(answer,context)


def test_discrete_values_and_boolean_types_are_validated():
    context={'allowed_knobs':[{'name':'flag','kind':'bool'}]}
    validate_proposal({'regions':{'flag':{'values':[True,False]}}},context)
    with pytest.raises(ValueError,match='boolean required'):
        validate_proposal({'regions':{'flag':{'values':[1]}}},context)


def test_validation_handles_domains_and_rejects_dense_or_invalid_recommendations():
    validate_proposal({'regions':{'x':{'center':4,'halfwidth':.1}}, 'patches':[{'x':5}]}, CONTEXT)
    with pytest.raises(ValueError, match='out-of-domain'):
        validate_proposal({'regions':{'x':{'center':11}}}, CONTEXT)
    with pytest.raises(ValueError, match='halfwidth'):
        validate_proposal({'regions':{'x':{'halfwidth':float('nan')}}}, CONTEXT)


def test_feedback_specific_cache_and_all_attempt_usage_are_retained(tmp_path):
    calls=[]
    answers=iter([{'regions':{'x':{'center':11}}}, {'regions':{'x':{'center':5}}},
                  {'regions':{'x':{'center':6}}}])
    def post(payload):
        calls.append(json.loads(json.dumps(payload)))
        return {'choices':[{'message':{'content':json.dumps(next(answers))}}],
                'model':'fake', 'usage':{'total_tokens':3}}
    proposer=FeedbackExpertProposer(SimpleNamespace(model='fake', _post=post), tmp_path)
    a=proposer('test', {**CONTEXT, 'trial':1})
    assert a['regions']['x']['center'] == 5 and len(calls) == 2
    assert 'validation error' in calls[1]['messages'][-1]['content']
    assert proposer('test', {**CONTEXT, 'trial':1}) == a and len(calls) == 2
    proposer('test', {**CONTEXT, 'trial':2})
    assert len(calls) == 3
    records=[json.loads(path.read_text()) for path in tmp_path.glob('*.json')]
    assert sum(len(r['attempts']) for r in records) == 3


def test_known_inactive_history_is_deferred_without_trimming_interacting_patch(tmp_path):
    from cognitune.adaptive.proposals import normalize_proposal
    domain = {**CONTEXT['allowed_knobs'][0], 'name':'inactive'}
    context = {**CONTEXT, 'global_knob_domains':CONTEXT['allowed_knobs'] + [domain]}
    raw = {'regions':{'x':{'center':5}, 'inactive':{'center':4}},
           'patches':[{'x':5}, {'x':6,'inactive':4}],
           'joint_hypotheses':[{'reason':'interesting but no executable patch'}]}
    answer, changes = normalize_proposal(raw,context)
    validate_proposal(answer,context)
    assert answer['regions'] == {'x':{'center':5}}
    assert answer['patches'] == [{'x':5}]
    assert answer['joint_hypotheses'] == []
    assert len(changes) == 3
    assert 'inactive' in raw['regions']  # original response is preserved
    assert changes[1]['patch'] == {'x':6,'inactive':4}


def test_unknown_name_and_illegal_active_value_still_fail():
    from cognitune.adaptive.proposals import normalize_proposal
    answer,_ = normalize_proposal({'regions':{'typo':{'center':5},'x':{'center':11}}},CONTEXT)
    with pytest.raises(ValueError):
        validate_proposal(answer,CONTEXT)
    with pytest.raises(ValueError,match='no executable scoped'):
        normalize_proposal({'regions':{}},CONTEXT)


def test_prompt_filters_stale_history_and_accepts_valid_scoped_part_without_retry(tmp_path):
    domain={**CONTEXT['allowed_knobs'][0], 'name':'inactive'}
    calls=[]
    def post(payload):
        calls.append(json.loads(json.dumps(payload)))
        return {'choices':[{'message':{'content':json.dumps({'regions':{'x':{'center':4},'inactive':{'center':3}}})}}]}
    proposer=FeedbackExpertProposer(SimpleNamespace(model='fake',_post=post),tmp_path)
    context={**CONTEXT,'global_knob_domains':CONTEXT['allowed_knobs']+[domain],
             'previous_regions':{'x':{'center':3},'inactive':{'center':4}}}
    answer=proposer('test',context)
    assert len(calls)==1
    sent=json.loads(calls[0]['messages'][1]['content'])
    assert set(sent['previous_regions']) == {'x'}
    assert sent['current_scope_names'] == ['x']
    assert answer['regions'] == {'x':{'center':4}}
    assert answer['interface_repairs'][0]['knob'] == 'inactive'


def joint_context():
    return {**CONTEXT,'global_knob_domains':[
        {**CONTEXT['allowed_knobs'][0], 'name':n} for n in ['x','y','z','w','v']]}


def test_joint_patch_array_is_merged_without_changing_recommended_values():
    from cognitune.adaptive.proposals import normalize_proposal
    raw={'regions':{'x':{'center':4}},'joint_hypotheses':[
        {'patch':[{'x':4},{'y':6}], 'mechanism':'interaction'}]}
    answer,repairs=normalize_proposal(raw,joint_context())
    validate_proposal(answer,joint_context())
    assert answer['joint_hypotheses'] == [{'patch':{'x':4,'y':6},'mechanism':'interaction'}]
    assert repairs[0]['action']=='merge_joint_patch_objects'
    assert isinstance(raw['joint_hypotheses'][0]['patch'],list)


@pytest.mark.parametrize('hypothesis',[
    {'patch':[{'x':4},{'x':5}]},  # Never choose one side of a conflict.
    {'patch':{'x':4,'y':5,'z':6,'w':7,'v':8}},  # Never truncate a joint patch.
    {'patch':{'x':100}},
    {'patch':{'unknown':3}},
    {'patch':'x=4'},
    {'patch':[{'x':4},'y=6']},
    {'patch':[{}]},
    'a textual interaction hypothesis',
])
def test_invalid_optional_joint_does_not_destroy_valid_scoped_proposal(hypothesis):
    from cognitune.adaptive.proposals import normalize_proposal
    raw={'regions':{'x':{'center':4}},'patches':[{'x':5}],
         'joint_hypotheses':[hypothesis,{'patch':{'x':4,'y':6}}]}
    answer,repairs=normalize_proposal(raw,joint_context())
    validate_proposal(answer,joint_context())
    assert answer['regions']==raw['regions'] and answer['patches']==raw['patches']
    assert answer['joint_hypotheses']==[{'patch':{'x':4,'y':6}}]
    assert any(r['action']=='reject_optional_joint_hypothesis' for r in repairs)


def test_knobs_mapping_alias_preserves_values_but_names_only_are_not_invented():
    from cognitune.adaptive.proposals import normalize_proposal
    raw={'regions':{'x':{'center':4}},'joint_hypotheses':[
        {'knobs':{'x':4,'y':6}}, {'knobs':['x','y']}]}
    answer,repairs=normalize_proposal(raw,joint_context())
    validate_proposal(answer,joint_context())
    assert len(answer['joint_hypotheses'])==1
    assert answer['joint_hypotheses'][0]['patch']=={'x':4,'y':6}
    assert len(repairs)==2


def test_joint_review_still_requires_owner_agreement_for_normalized_patch(tmp_path):
    calls=[]
    def post(payload):
        sent=json.loads(payload['messages'][1]['content']);calls.append(sent)
        return {'choices':[{'message':{'content':json.dumps({
            'regions':{},'patches':[{'x':4,'y':6}],'joint_hypotheses':[]})}}]}
    proposer=FeedbackExpertProposer(SimpleNamespace(model='fake',_post=post),tmp_path)
    context={'allowed_knobs':joint_context()['global_knob_domains'][:2]}
    reviews=proposer.review({'a','b'},{'patch':{'x':4,'y':6}},context)
    assert set(reviews)=={'a','b'} and len(calls)==2
    assert all(r['patches']==[{'x':4,'y':6}] for r in reviews.values())



def test_optional_bootstrap_ranking_does_not_abort_usable_regions():
    from cognitune.adaptive.proposals import normalize_proposal
    answer,repairs=normalize_proposal({'regions':{'x':{'center':4}},
        'focus_knobs':['x','unknown','x',{'name':'x'}]},CONTEXT)
    validate_proposal(answer,CONTEXT)
    assert answer['focus_knobs']==['x']
    assert repairs[0]['action']=='normalize_optional_focus_ranking'
