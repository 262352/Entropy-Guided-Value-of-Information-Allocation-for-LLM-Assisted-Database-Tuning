"""Structured per-round LLM proposals with explicit prompt/usage provenance."""
from __future__ import annotations

import hashlib
import copy
import json
import math
import time
from pathlib import Path

from cognitune.llm.deepseek import _parse_json_object
from .prompt_context import shared_history_prompt, token_usage


class FeedbackExpertProposer:
    def __init__(self, transport, directory):
        self.transport = transport
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, expert, context):
        context = dict(context)
        allowed_names = {k['name'] for k in context['allowed_knobs']}
        context['previous_regions'] = {n:r for n,r in context.get('previous_regions', {}).items()
                                       if n in allowed_names}
        context['current_scope_names'] = sorted(allowed_names)
        user_content, prompt_layout = shared_history_prompt(expert,context)
        payload = {
            'model': self.transport.model,
            'messages': [
                {'role': 'system', 'content': (
                    'You are a PostgreSQL tuning specialist. Return one JSON object. '
                    'Use only measured feedback, never invent TPS. Propose changes relative to anchor. '
                    'Only use allowed_knobs and their actual PostgreSQL units and legal values. '
                    'current_scope_names is the exact writable scope this round. Global domains '
                    'and full historical configurations are reference information, not permission '
                    'to restore inactive knobs. Omit inactive regions and patches. '
                    'feedback contains ALL completed evaluations in chronological order, including '
                    'IID starts and failures. Use the full history to compare contexts and trends; '
                    'a failed/non-SUCCESS evaluation is not evidence of measured throughput. '
                    'related_trial_ids points to relevant rows without repeating configurations. '
                    'Do not reproduce the history in your answer; return only structured '
                    'recommendations with concise evidence-based reasons. '
                    'For numeric regions provide center (actual value), halfwidth (0.01..0.5 of '
                    'the normalized global domain; log domain when log=true), action '
                    '(hold/refine/shift/expand/pause), and reason. For categorical, ordinal and '
                    'boolean regions use {"values":[legal_actual_value],"action":"hold","reason":"..."}; '
                    'an optional center must also be a legal actual choice, never a numeric index '
                    'or null. Do not provide halfwidth for discrete regions. hold/pause without '
                    'a center or values means retain the anchor choice. '
                    'Provide up to 4 sparse patches (at most 4 knobs each). '
                    'For interactions with knobs outside your scope, list joint_hypotheses '
                    'with a sparse patch (at most 4 knobs) and mechanism. Each element must have '
                    'shape {"patch":{"knob_a":1,"knob_b":2},"mechanism":"reason"}. '
                    'patch is ONE JSON object, never a list; these require review '
                    'by the owning specialists. Do not claim a soft performance hypothesis '
                    'is a hard constraint. Shape: {"regions":{"knob":{"center":1,'
                    '"halfwidth":0.1,"action":"hold","reason":"..."}},'
                    '"patches":[{"knob":1}],"joint_hypotheses":[]}.'
                )},
                {'role': 'user', 'content': user_content},
            ],
            'response_format': {'type': 'json_object'}, 'thinking': {'type': 'disabled'},
            'temperature': 0.2, 'max_tokens': 6000,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.directory / (digest + '.json')
        if path.exists():
            cached = json.loads(path.read_text())
            answer, scope_repairs = normalize_proposal(cached['answer'], context)
            validate_proposal(answer, context)
            self._record_metric({'origin':'local_response','request_key':digest,'expert':expert,
                'trial':context.get('trial'),'prompt_layout':prompt_layout})
            return answer
        attempts = []
        original_messages = list(payload['messages'])
        for attempt in range(3):
            started=time.monotonic()
            response = self.transport._post(payload)
            latency=time.monotonic()-started
            raw = response['choices'][0]['message'].get('content') or ''
            attempt_record = {'content':raw, 'model':response.get('model'), 'usage':response.get('usage')}
            attempts.append(attempt_record)
            stats=token_usage(response.get('usage'))
            attempt_record['token_usage']=stats
            attempt_record['latency_seconds']=latency
            self._record_metric({'origin':'provider','request_key':digest,'attempt':attempt+1,
                'expert':expert,'trial':context.get('trial'),'phase':context.get('phase','proposal'),
                'joint_review':bool(context.get('joint_review')),'prompt_layout':prompt_layout,
                'model':response.get('model'),'request_id':response.get('id'),
                'finish_reason':response['choices'][0].get('finish_reason'),
                'latency_seconds':latency,**stats})
            try:
                answer, repairs = _parse_json_object(raw)
                answer, scope_repairs = normalize_proposal(answer, context)
                validate_proposal(answer, context)
            except (ValueError, TypeError, KeyError) as error:
                attempt_record['validation_error'] = str(error)
                payload['messages'] = original_messages + [{'role':'user', 'content':
                    'Regenerate the complete JSON. regions and patches may use ONLY '+json.dumps(sorted(allowed_names))+'. joint_hypotheses may use global domains but each patch must be one JSON object. Correct this validation error: '+str(error)}]
                if attempt == 2:
                    path.with_suffix('.failed.json').write_text(json.dumps({'attempts':attempts, 'payload':payload,
                        'prompt_layout':prompt_layout}, indent=2))
                    raise ValueError(f'invalid structured proposal from {expert}: {error}') from error
                continue
            # Include invalid attempts in token accounting, never just the last.
            path.write_text(json.dumps({'payload':payload, 'answer':answer, 'repairs':repairs,
                'model':response.get('model'), 'usage':response.get('usage'), 'attempts':attempts,
                'scope_repairs':scope_repairs,'prompt_layout':prompt_layout}, indent=2))
            return answer
        raise RuntimeError('unreachable proposal retry state')

    def _record_metric(self,record):
        record={'unix_time':time.time(),**record}
        with (self.directory/'request_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record,sort_keys=True)+'\n')
        if record['origin']=='local_response':
            print(f"[LLM CACHE] expert={record['expert']} trial={record['trial']} source=local_response (no API request)",flush=True)
        else:
            rate=record['cache_hit_rate']
            rate_text=f'{rate:.1%}' if rate is not None else 'unknown'
            print(f"[LLM CACHE] expert={record['expert']} trial={record['trial']} "
                  f"history={record['prompt_layout']['history_count']} "
                  f"hit={record['cache_hit_tokens']} miss={record['cache_miss_tokens']} "
                  f"hit_rate={rate_text} latency={record['latency_seconds']:.2f}s source=provider",flush=True)

    def review(self, experts, hypothesis, context):
        """Anonymized proposer, visible knob semantics; no scalar TPS voting."""
        reviews = {}
        for expert in sorted(experts):
            reviews[expert] = self(expert, {**context, 'joint_review': {
                'hypothesis': hypothesis,
                'instruction': 'Review this cross-domain patch; put an agreed legal patch in patches. '
                               'Explain revisions in regions.reason. Empty patches means no agreement.'}})
        return reviews


class StructuredMockProposer:
    """Offline plumbing only; never substituted for an unavailable real LLM."""
    def __call__(self, expert, context):
        return {'regions': {k['name']: {
            'center': context['anchor'].get(k['name'], k['default']),
            'halfwidth': .15, 'action': 'hold', 'reason': 'offline synthetic smoke'}
            for k in context['allowed_knobs']}, 'patches': [], 'joint_hypotheses': []}


def validate_proposal(answer, context):
    if not isinstance(answer.get('regions'), dict) or not isinstance(answer.get('patches', []), list):
        raise ValueError('regions must be an object and patches a list')
    allowed = {k['name']:k for k in context['allowed_knobs']}
    global_knobs = {k['name']:k for k in context.get('global_knob_domains', context['allowed_knobs'])}

    def legal(name, value, domains):
        if name not in domains:
            raise ValueError('unknown or out-of-scope knob: '+name)
        knob = domains[name]
        kind = knob['kind']
        if kind in {'integer','float'}:
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
                raise ValueError('numeric value required for '+name)
            if not knob['lower'] <= value <= knob['upper']:
                raise ValueError('out-of-domain value for '+name)
            if kind == 'integer' and value != round(value):
                raise ValueError('integer value required for '+name)
        elif kind == 'bool':
            if not isinstance(value, bool):
                raise ValueError('boolean required for '+name)
        elif not any(type(value) is type(v) and value == v for v in knob['choices']):
            raise ValueError('illegal choice for '+name)

    if 'focus_knobs' in answer:
        focus=answer['focus_knobs']
        if not isinstance(focus,list) or len(focus)>6 or any(not isinstance(n,str) or n not in allowed for n in focus):
            raise ValueError('focus_knobs must contain at most six allowed knob names')
    for name, region in answer['regions'].items():
        if name not in allowed or not isinstance(region, dict):
            raise ValueError('invalid region or scope: '+name)
        if 'center' in region:
            legal(name, region['center'], allowed)
        if 'halfwidth' in region and allowed[name]['kind'] in {'integer','float'}:
            width = region['halfwidth']
            if not isinstance(width, (int,float)) or not .01 <= width <= .5:
                raise ValueError('halfwidth must be normalized and within [.01,.5]')
        if region.get('action','hold') not in {'hold','refine','shift','expand','pause'}:
            raise ValueError('unknown region action')
        if 'values' in region and (not isinstance(region['values'], list) or not region['values']):
            raise ValueError('values must be a nonempty list of legal choices for '+name)
        for value in region.get('values', []):
            legal(name, value, allowed)
    for patch in answer.get('patches', []):
        if not isinstance(patch, dict) or len(patch) > 4:
            raise ValueError('patch must be an object with at most four knobs')
        for name, value in patch.items():
            legal(name, value, allowed)
    if not isinstance(answer.get('joint_hypotheses', []), list):
        raise ValueError('joint_hypotheses must be a list')
    for hypothesis in answer.get('joint_hypotheses', []):
        patch = hypothesis.get('patch')
        if not isinstance(patch, dict) or not 1 <= len(patch) <= 4:
            raise ValueError('joint hypothesis needs a sparse patch')
        for name, value in patch.items():
            legal(name, value, global_knobs)


def normalize_proposal(answer, context):
    """Project known inactive suggestions out of the writable interface.

    Preserve every dropped item for diagnostics. Never clip an illegal value,
    guess a parameter name, or turn a partial joint patch into a new proposal.
    """
    answer = copy.deepcopy(answer)
    allowed = {k['name'] for k in context['allowed_knobs']}
    known = {k['name'] for k in context.get('global_knob_domains', context['allowed_knobs'])}
    repairs = []
    if 'focus_knobs' in answer:
        original = answer['focus_knobs']
        focus = list(dict.fromkeys(n for n in original if isinstance(n,str) and n in allowed))[:6] if isinstance(original,list) else []
        if original != focus:
            repairs.append({'action':'normalize_optional_focus_ranking','original':original,'retained':focus})
        answer['focus_knobs'] = focus
    regions = answer.get('regions')
    if isinstance(regions, dict):
        for name in list(regions):
            if name in known and name not in allowed:
                repairs.append({'action':'defer_inactive_region', 'knob':name, 'region':regions.pop(name)})
        domains = {k['name']: k for k in context['allowed_knobs']}
        for name, region in regions.items():
            if name not in domains or not isinstance(region, dict):
                continue
            knob = domains[name]
            if knob['kind'] in {'integer', 'float'}:
                continue
            # A nominal choice has no numeric trust-region width. Keep the
            # original field in the audit instead of rejecting a legal choice.
            if 'halfwidth' in region:
                repairs.append({'action':'remove_discrete_halfwidth', 'knob':name,
                                'original':region.pop('halfwidth')})
            choices = [False, True] if knob['kind'] == 'bool' else knob['choices']
            def is_choice(value):
                return any(type(value) is type(v) and value == v for v in choices)
            anchor = context.get('anchor', {}).get(name)
            # Only an explicit hold/pause authorizes retaining the current
            # choice. Never decode an invented numeric index as a category.
            if (region.get('action') in {'hold', 'pause'} and is_choice(anchor)
                    and not is_choice(region.get('center')) and not region.get('values')
                    and (region.get('center') is None or
                         type(region.get('center')) in {int, float})):
                repairs.append({'action':'resolve_discrete_hold_to_anchor', 'knob':name,
                                'original':region.get('center'), 'retained':anchor})
                region['center'] = anchor
            if 'center' not in region and not region.get('values'):
                raise ValueError('discrete region needs legal center or values for '+name)
    patches = answer.get('patches', [])
    if isinstance(patches, list):
        keep = []
        for patch in patches:
            if isinstance(patch, dict) and set(patch) <= known and not set(patch) <= allowed:
                repairs.append({'action':'defer_entire_out_of_scope_patch', 'patch':patch})
            else:
                keep.append(patch)
        answer['patches'] = keep
    hypotheses = answer.get('joint_hypotheses', [])
    keep = []
    if not isinstance(hypotheses, list):
        repairs.append({'action':'reject_optional_joint_field', 'reason':'joint_hypotheses must be a list',
                        'hypotheses':hypotheses})
        hypotheses = []
    for original in hypotheses:
        hypothesis = copy.deepcopy(original)
        if not isinstance(hypothesis, dict):
            repairs.append({'action':'reject_optional_joint_hypothesis', 'reason':'hypothesis must be an object',
                            'hypothesis':original})
            continue
        if not hypothesis.get('patch'):
            # A known alternate field is usable only when it supplies exact
            # values. A list of knob names does not define an executable patch.
            if 'patch' not in hypothesis and isinstance(hypothesis.get('knobs'), dict) and hypothesis['knobs']:
                hypothesis['patch'] = copy.deepcopy(hypothesis['knobs'])
                repairs.append({'action':'normalize_joint_knobs_mapping', 'hypothesis':original})
            else:
                repairs.append({'action':'record_non_executable_hypothesis', 'hypothesis':original})
                continue
        patch = hypothesis['patch']
        if isinstance(patch, list):
            merged = {}
            error = None
            for part in patch:
                if not isinstance(part, dict) or not part:
                    error = 'patch list must contain nonempty knob-value objects'
                    break
                for name, value in part.items():
                    if name in merged:
                        # JSON booleans and numbers must not compare as equal.
                        if json.dumps(merged[name], sort_keys=True) != json.dumps(value, sort_keys=True):
                            error = 'conflicting values for '+name
                            break
                    merged[name] = value
                if error:
                    break
            if error:
                repairs.append({'action':'reject_optional_joint_hypothesis', 'reason':error,
                                'hypothesis':original})
                continue
            hypothesis['patch'] = merged
        try:
            # The SAME strict domain/size validation as direct joint input.
            # Rejection is local to the optional hypothesis, never a partial
            # projection that changes the meaning of a joint configuration.
            validate_proposal({'regions':{}, 'patches':[], 'joint_hypotheses':[hypothesis]}, context)
        except (ValueError, TypeError, KeyError) as error:
            repairs.append({'action':'reject_optional_joint_hypothesis', 'reason':str(error),
                            'hypothesis':original})
            continue
        if isinstance(patch, list):
            repairs.append({'action':'merge_joint_patch_objects', 'hypothesis':original,
                            'normalized_patch':hypothesis['patch']})
        keep.append(hypothesis)
    answer['joint_hypotheses'] = keep
    if repairs:
        answer['interface_repairs'] = list(answer.get('interface_repairs', [])) + repairs
    # An ordinary expert must still provide executable scoped information.
    # An empty review means disagreement and is allowed.
    if (allowed and not context.get('joint_review') and not answer.get('regions')
            and not answer.get('patches') and not answer.get('joint_hypotheses')):
        raise ValueError('no executable scoped recommendation; expected knobs: '+', '.join(sorted(allowed)))
    return answer, repairs
