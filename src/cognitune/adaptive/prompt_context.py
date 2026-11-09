"""Stable shared prefixes and complete, append-only execution evidence."""
from __future__ import annotations

import hashlib
import json


PROMPT_LAYOUT = 'shared_history_v1'
HISTORY_POLICY = 'all_completed_trials_v1'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def full_execution_history(records):
    """Keep every completed evaluation, including failures and IID starts.

    Historical rows depend only on that recorded evaluation, never the current
    expert, active scope, anchor or remaining budget, so earlier rows stay fixed.
    """
    history=[]
    for r in records:
        state=r.state or {}
        history.append({'trial':r.trial,'config':r.config,'raw_metric':r.raw_metric,
            'utility':r.utility,'status':r.status,'source':r.candidate_source,
            'elapsed_seconds':r.elapsed_seconds,'anchor_trial':state.get('anchor_trial'),
            'changed_knobs':sorted(state.get('changed_knobs',[]))})
    return history


def shared_history_prompt(expert,context):
    """One valid JSON user message: shared facts/history BEFORE volatile fields.

    Each section is serialized independently; sorting the entire outer object
    would put the changing anchor/allowed scope before the reusable history.
    """
    domains=context.get('global_knob_domains',context['allowed_knobs'])
    domains=sorted(domains,key=lambda k:k['name'])
    history=context.get('feedback',[])
    sections=[('workload',context.get('workload',{})),('global_knob_domains',domains),('feedback',history)]
    shared='{'+','.join(canonical(k)+':'+canonical(v) for k,v in sections)
    dynamic={k:v for k,v in context.items() if k not in {'workload','global_knob_domains','feedback'}}
    dynamic['expert']=expert
    content=shared+''.join(','+canonical(k)+':'+canonical(dynamic[k]) for k in sorted(dynamic))+'}'
    return content,{'version':PROMPT_LAYOUT,'history_policy':HISTORY_POLICY,'history_count':len(history),
        'shared_prefix_characters':len(shared),
        'shared_prefix_sha256':hashlib.sha256(shared.encode()).hexdigest()}


def token_usage(usage):
    """Use returned token counts; unknown telemetry stays unknown, never 0%."""
    usage=usage or {}
    def count(value):
        return value if type(value) is int and value>=0 else None
    prompt=count(usage.get('prompt_tokens',usage.get('input_tokens')))
    details=usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
    hit=count(usage.get('prompt_cache_hit_tokens',details.get('cached_tokens')))
    miss=count(usage.get('prompt_cache_miss_tokens'))
    if prompt is not None and hit is not None and miss is None and prompt>=hit:
        miss=prompt-hit
    if prompt is None and hit is not None and miss is not None:
        prompt=hit+miss
    if hit is None and prompt is not None and miss is not None and prompt>=miss:
        hit=prompt-miss
    rate=hit/(hit+miss) if hit is not None and miss is not None and hit+miss>0 else None
    return {'input_tokens':prompt,'output_tokens':count(usage.get('completion_tokens',usage.get('output_tokens'))),
            'cache_hit_tokens':hit,'cache_miss_tokens':miss,'cache_hit_rate':rate}
