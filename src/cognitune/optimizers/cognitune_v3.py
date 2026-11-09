"""Decision-entropy guided search adaptation (new method; v2 is preserved).

Finite candidate/audit approximations and all controller actions are logged.
The default empirical calibration gate requires >=8 out-of-sample residuals.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from cognitune.adaptive.stable_search import StableSearch, build_mode_pool, coverage_candidates
from cognitune.adaptive.ablation import Ablation, project_patch
from cognitune.adaptive.prompt_context import full_execution_history, HISTORY_POLICY, PROMPT_LAYOUT
from cognitune.adaptive.checkpoint import recover_controller, save_checkpoint
from cognitune.adaptive.decision import allocate, condition, factor, information_gains, label_probabilities
from cognitune.agents.ownership import EXPERTS, build_ownership
from cognitune.experiment.recorder import RunRecorder
from cognitune.optimizers.common import evaluate_and_record
from cognitune.surrogate.gp import GPPerformanceSurrogate


def unit(knob, value):
    if knob.kind not in {'integer', 'float'}:
        choices = [False, True] if knob.kind == 'bool' else list(knob.choices)
        return choices.index(value) / max(1, len(choices) - 1)
    lo, hi, v = float(knob.lower), float(knob.upper), float(value)
    if knob.log:
        lo, hi, v = math.log(lo), math.log(hi), math.log(v)
    return (v - lo) / max(hi - lo, 1e-12)


def value_at(knob, value):
    value = float(np.clip(value, 0, 1))
    if knob.kind not in {'integer', 'float'}:
        choices = [False, True] if knob.kind == 'bool' else list(knob.choices)
        return choices[int(round(value * (len(choices) - 1)))]
    lo, hi = float(knob.lower), float(knob.upper)
    result = math.exp(math.log(lo) + value * (math.log(hi) - math.log(lo))) if knob.log else lo + value * (hi - lo)
    return int(round(result)) if knob.kind == 'integer' else result


def bounded_patch(space, anchor, patch, allowed, max_changes=4, distance=1.5):
    if not isinstance(patch, dict) or not set(patch) <= set(allowed):
        return None
    try:
        # Do not coerce strings such as "off" into True.
        for name, value in patch.items():
            knob = space.by_name[name]
            if knob.kind == 'bool' and not isinstance(value, bool):
                return None
            if knob.kind not in {'integer', 'float', 'bool'} and value not in knob.choices:
                return None
            if knob.kind in {'integer', 'float'} and not np.isfinite(float(value)):
                return None
        result = space.repair({**anchor, **patch})
        changed = [n for n in set(anchor) | set(result) if anchor.get(n) != result.get(n)]
        if len(changed) > max_changes or not space.validate(result):
            return None
        # Include changes introduced by constraint repair in both bounds.
        shift = sum(abs(unit(space.by_name[n], result[n]) - unit(space.by_name[n], anchor[n]))
                    for n in changed if n in anchor and n in result)
        return result if shift <= distance + 1e-12 else None
    except (TypeError, ValueError, OverflowError):
        return None


def run_cognitune_v3(space, evaluator, initial_configs, budget, seed, candidate_pool_size,
                     run_dir, *, proposer, cognitive_context=None, active_target=18,
                     audit_size=4, eta=0.8, delta=0.01, shortlist_size=16,
                     posterior_samples=256, fantasies=16, probe_interval=8,
                     surrogate_factory=GPPerformanceSurrogate, resume=False, search_policy='legacy', ablation='full'):
    variant = Ablation(ablation)
    if ablation != 'full' and search_policy != 'stable':
        raise ValueError('ablations require the stable search policy')
    if search_policy not in {'legacy','stable'}:
        raise ValueError('unknown search policy')
    if not 0 < eta <= 1 or len(initial_configs) < 2 or budget < len(initial_configs):
        raise ValueError('invalid budget, shared initialization, or eta')
    if candidate_pool_size < 40 or audit_size < 1:
        raise ValueError('need >=40 candidates and a positive audit size')
    directory = Path(run_dir)
    if (directory / 'trials.jsonl').exists() and not resume:
        raise FileExistsError('refusing to overwrite an existing tuning trace')
    recorder = RunRecorder(directory, resume=resume)
    rng = np.random.default_rng(seed)
    ownership = build_ownership(space)
    regions = {e: {} for e in EXPERTS}
    age = {k.name: 0 for k in space.knobs}
    quotas, _ = allocate({}, EXPERTS, candidate_pool_size - 20)
    residuals, groups = [], []
    best = -float('inf')
    diagnostics = directory / 'round_diagnostics.jsonl'
    parameters = {'space_fingerprint':space.fingerprint(), 'budget':budget, 'seed':seed,
        'candidate_pool_size':candidate_pool_size, 'active_target':active_target,
        'audit_size':audit_size, 'eta':eta, 'delta':delta, 'shortlist_size':shortlist_size,
        'posterior_samples':posterior_samples, 'fantasies':fantasies, 'probe_interval':probe_interval}
    parameters['expert_history_policy']=HISTORY_POLICY
    parameters['expert_prompt_layout']=PROMPT_LAYOUT
    global_domains=[{**k.__dict__,'condition':str(k.condition)} for k in space.knobs]
    stable = StableSearch(space,ownership,ablation=ablation) if search_policy=='stable' else None
    if ablation != 'full':
        parameters['ablation']=variant.describe()
    if stable:
        parameters['search_policy']='stable-v1'
        parameters['structure_episode']=6
        parameters['minimum_executed_tests']=2
        parameters['max_swaps']=2
        parameters['coverage_interval']=3
        parameters['effective_active_target']=min(active_target,max(6,(budget-len(initial_configs))//4))
        parameters['final_polish_evaluations']=6
    recovery_mode = None
    preserved_observations = len(recorder.records)
    if recorder.records or (resume and (directory / 'controller_checkpoint.json').exists()):
        regions, age, quotas, residuals, groups, rng, recovery_mode = recover_controller(
            directory, recorder.records, parameters, space, initial_configs)
        best = max((r.utility for r in recorder.records),default=-float('inf'))
        if stable:
            saved=json.loads((directory/'controller_checkpoint.json').read_text())
            if saved['completed']!=len(recorder.records) or not saved.get('extra_state'):
                raise ValueError('stable policy requires a complete matching controller checkpoint')
            stable=StableSearch(space,ownership,state=saved['extra_state'],ablation=ablation)

    def checkpoint():
        save_checkpoint(directory, recorder.records, parameters, regions, age, quotas, residuals, groups, rng,
                        extra_state=stable.snapshot() if stable else None)


    def event(kind, **fields):
        row = {'event': kind, **fields}
        with diagnostics.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        print('[CogniTune-v3] ' + json.dumps(row, sort_keys=True), flush=True)

    if recovery_mode:
        event('resume', completed=len(recorder.records), remaining=budget - len(recorder.records),
              recovery_mode=recovery_mode, note='observations preserved; scoped proposal interface corrected')
        checkpoint()
    for trial in range(len(recorder.records), len(initial_configs)):
        config = initial_configs[trial]
        best, _ = evaluate_and_record(evaluator, recorder, 'cognitune_v3', seed,
                                     trial, config, 'shared_initial', best, budget=budget)
        checkpoint()
    while len(recorder.records) < budget:
        trial = len(recorder.records)
        valid = [r for r in recorder.records if r.status == 'SUCCESS']
        if len(valid) < 2:
            raise RuntimeError('fewer than two successful measurements')
        anchor_record = max(valid, key=lambda r: r.utility)
        anchor = dict(anchor_record.config)  # fixed for this entire round
        history=full_execution_history(recorder.records)
        anchor_context_changed=stable.anchor_context(anchor,trial,unit) if stable else False
        gp = surrogate_factory(space)
        gp.fit([r.config for r in valid], [r.utility for r in valid])
        seen = {space.config_key(r.config) for r in recorder.records}

        # Re-audit every knob, including paused knobs. Joint groups discovered
        # in previous rounds are tested as groups rather than forced active.
        probe_groups = [(k.name,) for k in space.knobs] + groups
        probe_configs, slices = [anchor], []
        for group in probe_groups:
            start = len(probe_configs)
            for direction in (-1, 1):
                patch = {}
                for n in group:
                    knob = space.by_name[n]
                    current = anchor.get(n, knob.default)
                    if knob.kind in {'integer', 'float'}:
                        patch[n] = value_at(knob, unit(knob, current) + direction * .15)
                    else:
                        choices = [False, True] if knob.kind == 'bool' else list(knob.choices)
                        patch[n] = choices[(choices.index(current) + direction) % len(choices)]
                    if knob.condition and n not in anchor:
                        patch[knob.condition.parent] = knob.condition.values[0]
                candidate = bounded_patch(space, anchor, patch, space.by_name)
                if candidate is not None:
                    probe_configs.append(candidate)
            slices.append((start, len(probe_configs)))
        pm, pc, _ = gp.joint_posterior(probe_configs)
        draws = pm + rng.standard_normal((posterior_samples, len(pm))) @ factor(pc).T
        potentials = {}
        for group, (start, end) in zip(probe_groups, slices):
            gains = np.max(draws[:, start:end] - draws[:, :1], axis=1) if end > start else np.zeros(posterior_samples)
            p = float(np.mean(gains > delta))
            low, high = map(float, np.quantile(gains, [0.1, 0.9]))
            potentials['+'.join(group)] = {'p_improve': p, 'interval': [low, high],
                                          'expected_gain':float(np.mean(np.maximum(gains,0))),
                                          'state': 'exploit' if p >= .9 else 'pause' if p <= .1 and high - low < 2 * delta else 'verify'}
        ranked = sorted(probe_groups, key=lambda g: (
            potentials['+'.join(g)]['state'] != 'pause',
            potentials['+'.join(g)]['p_improve'] + .01 * min(max(age[n] for n in g), 20)), reverse=True)
        protected = [n for n in ('shared_buffers', 'work_mem', 'maintenance_work_mem') if n in space.by_name]
        active = set(protected)
        limit = min(parameters.get('effective_active_target',active_target),len(space.knobs))
        if limit < len(active):
            raise ValueError('active budget is smaller than protected knob count')
        for expert in EXPERTS:
            if any(expert in ownership[n] for n in active):
                continue
            owned = [n for n in space.by_name if expert in ownership[n]]
            if owned and len(active) < limit:
                active.add(max(owned, key=lambda n: potentials[n]['p_improve']))
        for group in ranked:
            closure = set(group)
            for name in group:
                if space.by_name[name].condition:
                    closure.add(space.by_name[name].condition.parent)
            if len(active | closure) <= limit:
                active.update(closure)
        structure_report = None
        if stable and not stable.state['active']:
            # Choose the first subspace using workload knowledge and actual IID
            # feedback, rather than freezing an uncertain GP-only ranking.
            bootstrap={};support={n:0.0 for n in space.by_name}
            for expert in EXPERTS:
                names=[n for n in space.by_name if expert in ownership[n]]
                if not names: continue
                bootstrap[expert]=proposer(expert, {
                    'phase':'structure_bootstrap','trial':trial,'remaining_trials':budget-trial,
                    'workload':cognitive_context or {},'anchor':anchor,
                    'anchor_raw_metric':anchor_record.raw_metric,
                    'allowed_knobs':[{**space.by_name[n].__dict__,'condition':str(space.by_name[n].condition)} for n in names],
                    'global_knob_domains':global_domains,
                    'candidate_quota':quotas[expert],
                    'instruction':'Choose up to six knobs most promising for this measured workload. '
                                  'Return focus_knobs in descending importance, and regions/patches for these knobs. '
                                  'This chooses the initial search subspace; avoid generic logging knobs unless execution evidence supports them.',
                    'feedback':history})
                focus=bootstrap[expert].get('focus_knobs') or list(bootstrap[expert].get('regions',{}))[:6]
                for index,name in enumerate(focus):
                    if name in names: support[name]+=float(len(focus)-index)/max(1,len(focus))
            ordered=sorted(space.by_name,key=lambda n:(support[n],potentials[n]['expected_gain']),reverse=True)
            active=set(protected)
            for expert in EXPERTS:
                if any(expert in ownership[n] for n in active): continue
                choice=next((n for n in ordered if expert in ownership[n]),None)
                if choice and len(active)<limit: active.add(choice)
            for name in ordered:
                if len(active)>=limit: break
                active.add(name)
            stable.state['bootstrap']=bootstrap
            event('structure_bootstrap',trial=trial,expert_priors=bootstrap,support=support,selected=sorted(active))
        if stable:
            active,structure_report=stable.select(active,potentials,ranked,trial,protected,limit,delta)
        for n in age:
            age[n] = 0 if n in active else age[n] + 1

        proposals, scopes, region_changes, patch_projection = {}, {}, {}, {}
        for expert in EXPERTS:
            owned = [n for n in space.by_name if expert in ownership[n]]
            scope = sorted(set(owned) & active)
            if not scope:
                scope = sorted(owned, key=lambda n: -potentials[n]['p_improve'])[:1]
            scopes[expert] = scope
            context = {
                'workload': cognitive_context or {}, 'trial': trial, 'remaining_trials': budget - trial,
                'anchor': anchor, 'anchor_raw_metric': anchor_record.raw_metric,
                'allowed_knobs': [{**space.by_name[n].__dict__, 'condition': str(space.by_name[n].condition)} for n in scope],
                'global_knob_domains':global_domains,
                'conditional_potential': {n: potentials[n] for n in scope},
                'previous_regions': {n:{'center':value_at(space.by_name[n], r['center']),
                    **({'halfwidth':r['halfwidth']} if space.by_name[n].kind in {'integer','float'}
                       else {'values':r.get('values',[value_at(space.by_name[n],r['center'])])}),
                    'center_units':'actual_knob_value'}
                    for n,r in regions[expert].items() if n in scope}, 'candidate_quota': quotas[expert],
                'local_execution_coverage': {n:stable.state['nodes'][n]['tests'] for n in scope} if stable else {},
                'related_trial_ids':[r['trial'] for r in history if set(r['changed_knobs']) & set(scope)],
                'feedback':history,
            }
            print(f'[EXPERT] trial={trial} expert={expert} quota={quotas[expert]} knobs={scope}', flush=True)
            proposal = proposer(expert, context)
            proposals[expert] = proposal
            changes = {}
            for name in scope:
                knob = space.by_name[name]
                old = regions[expert].get(name, {'center': unit(knob, anchor.get(name, knob.default)), 'halfwidth': .15})
                if anchor_context_changed and variant.dynamic_regions:
                    old={**old,'center':unit(knob,anchor.get(name,knob.default))}
                request = proposal.get('regions', {}).get(name, {})
                target = unit(knob, request.get('center', anchor.get(name, knob.default)))
                if not np.isfinite(target):
                    raise ValueError('non-finite expert region center')
                # Large expansions require an observed outside improvement.
                radius = old['halfwidth']
                previous = valid[-1]
                outside_success = (previous.trial == trial - 1 and previous.utility > anchor_record.utility - 1e-12
                                   and name in previous.config
                                   and abs(unit(knob, previous.config[name]) - old['center']) > radius)
                if outside_success and request.get('action') == 'expand':
                    radius = min(.5, radius * 1.5, max(radius, float(request.get('halfwidth', radius * 1.5))))
                # Conservative refinement requires concentrated posterior
                # maximizer among equal-count in-region probes, below.
                numeric = knob.kind in {'integer','float'}
                center = float(np.clip(target, old['center'] - .1, old['center'] + .1)) if numeric else target
                if stable:
                    node=stable.state['nodes'][name]
                    if not node.get('region_initialized'):
                        if knob.kind in {'integer','float'} and 'halfwidth' in request:
                            node['width']=float(np.clip(request['halfwidth'],.05,.25))
                        node['region_initialized']=True
                    radius=node['width']
                regions[expert][name] = {'center': float(np.clip(center, 0, 1)), 'halfwidth': radius}
                if not numeric:
                    regions[expert][name]['values'] = list(request.get('values') or
                        [request.get('center',anchor.get(name,knob.default))])
                if not variant.dynamic_regions and name in context['previous_regions']:
                    regions[expert][name] = dict(old)
                changes[name] = {'before': old, 'requested': request, 'after': dict(regions[expert][name])}
            region_changes[expert] = changes
            if not variant.dynamic_regions:
                projected=[project_patch(p,regions[expert],space,unit,value_at)
                           for p in proposal.get('patches',[])]
                patch_projection[expert]={'original':proposal.get('patches',[]),'projected':projected}
                proposals[expert]={**proposal,'patches':projected}

        def sample(expert, count, outside=False, inside=False):
            scope = scopes[expert]
            result, keys = [], set()
            for _ in range(max(100, count * 40)):
                if len(result) >= count or not scope:
                    break
                names = rng.choice(scope, size=int(rng.integers(1, min(4, len(scope)) + 1)), replace=False)
                patch = {}
                for name in names:
                    knob, region = space.by_name[name], regions[expert][name]
                    if outside or (not inside and rng.random() < .1):
                        u = rng.random()  # nonzero mass beyond current support
                    else:
                        u = rng.uniform(max(0, region['center'] - region['halfwidth']), min(1, region['center'] + region['halfwidth']))
                    prefs = region.get('values')
                    if knob.kind not in {'integer', 'float'}:
                        choices = [False, True] if knob.kind == 'bool' else list(knob.choices)
                        values = choices if outside or rng.random() < .1 else prefs or choices
                        patch[name] = values[int(rng.integers(len(values)))]
                        continue
                    patch[name] = value_at(knob, u)
                config = bounded_patch(space, anchor, patch, space.by_name)
                if config is None:
                    continue
                is_outside = any(abs(unit(space.by_name[n], config.get(n, space.by_name[n].default)) - regions[expert][n]['center']) > regions[expert][n]['halfwidth'] + 1e-12 for n in scope)
                if (outside and not is_outside) or (inside and is_outside):
                    continue
                key = space.config_key(config)
                if key not in keys and key not in seen:
                    keys.add(key)
                    result.append(config)
            return result

        pool, provenance, audit_keys, audit_provenance = {}, {}, set(), {}

        def add(config, source, audit=False):
            if config is None or not space.validate(config):
                return
            key = space.config_key(config)
            if key in seen:
                return
            pool[key] = config
            provenance.setdefault(key, set()).add(source)
            if source in {'reference','joint','coverage','reentry'}:
                channels.setdefault(key,set()).add(source)
            if audit:
                audit_keys.add(key)
                audit_provenance.setdefault(key, set()).add(source)

        channels={}
        hypotheses={}
        hypothesis_quotas={}
        if stable:
            audits,m,hypotheses,hypothesis_quotas,channels=build_mode_pool(
                stable,proposals,scopes,active,regions,anchor,space,seen,rng,
                candidate_pool_size-20,audit_size,add,unit,value_at,bounded_patch,groups)
            outside_probabilities={}
        else:
            # Independent equal-size audit draws: quotas cannot buy more audit votes.
            audits = {e: sample(e, audit_size) for e in EXPERTS}
            m = min(map(len, audits.values()))
            audits = {e: values[:m] for e, values in audits.items()}
            outside_probabilities = {}
            for expert in EXPERTS:
                for config in audits[expert]:
                    add(config, expert, True)
                explicit = []
                for patch in proposals[expert].get('patches', [])[:4]:
                    config = bounded_patch(space, anchor, patch, scopes[expert])
                    if config is not None:
                        explicit.append(config)
                # Audits and explicit patches are included in this expert's quota.
                for config in explicit[:max(0, quotas[expert] - m)]:
                    add(config, expert)
                for config in sample(expert, max(0, quotas[expert] - m - len(explicit))):
                    add(config, expert)
                inside, outside = sample(expert, 4, inside=True), sample(expert, 4, outside=True)
                if len(inside) == len(outside) == 4:
                    rm, rc, _ = gp.joint_posterior(inside + outside)
                    rd = rm + rng.standard_normal((posterior_samples, 8)) @ factor(rc).T
                    pout = float(np.mean(rd[:, 4:].max(axis=1) > rd[:, :4].max(axis=1) + delta))
                    outside_probabilities[expert] = pout
                    # Only equal-sized finite probes support this probability.
                    if pout > .8:
                        for config in outside:
                            add(config, 'reference')
                    optima = np.argmax(rd[:, :4], axis=1)
                    if pout < .1 and np.max(np.bincount(optima, minlength=4)) / posterior_samples > .8:
                        for name, region in regions[expert].items():
                            request = proposals[expert].get('regions', {}).get(name, {})
                            if request.get('action') == 'refine':
                                region['halfwidth'] = max(.02, region['halfwidth'] * .8, min(region['halfwidth'], float(request.get('halfwidth', region['halfwidth'] * .8))))
                                region_changes[expert][name]['after'] = dict(region)


        # Explicit sparse joint hypotheses are accepted only after agreement
        # from affected owners. Review never sees the proposing identity.
        if not stable:
            groups = []
        joint_count = 0
        for expert in EXPERTS:
            for hypothesis in proposals[expert].get('joint_hypotheses', [])[:1]:
                if joint_count >= 4:
                    break
                patch = hypothesis.get('patch', {})
                config = bounded_patch(space, anchor, patch, space.by_name)
                if config is None or not patch:
                    continue
                owners = {e for n in patch for e in ownership[n]}
                if not variant.dynamic_regions and any(
                    project_patch(patch,regions[e],space,unit,value_at)!=patch for e in owners):
                    event('joint_review',trial=trial,hypothesis=hypothesis,reviews={},agreed=False,
                          reason='outside_frozen_recommended_region')
                    continue
                if len(owners) < 2 or not hasattr(proposer, 'review'):
                    continue
                reviews = proposer.review(owners, hypothesis, {'workload': cognitive_context or {}, 'anchor': anchor,
                    'trial':trial,'remaining_trials':budget-trial,'anchor_raw_metric':anchor_record.raw_metric,
                    'global_knob_domains':global_domains,'feedback':history,
                    'allowed_knobs': [{**space.by_name[n].__dict__, 'condition': str(space.by_name[n].condition)} for n in patch]})
                agreed = all(any(bounded_patch(space, anchor, p, space.by_name) == config
                                 for p in review.get('patches', [])) for review in reviews.values())
                event('joint_review', trial=trial, hypothesis=hypothesis, reviews=reviews, agreed=agreed)
                if agreed:
                    if not stable or set(patch)<=active:
                        add(config, 'joint', True)
                    group=tuple(sorted(patch))
                    if group not in groups: groups.append(group)
                    groups=groups[-8:] if stable else groups
                    joint_count += 1
        # Independent full-domain reference and sparse boundary proposals.
        references = space.sample_iid(seed + 100003 * (trial + 1),
            6 if stable and (trial-len(initial_configs)+1)%3==0 else 12)
        for index, config in enumerate(references):
            add(config, 'reference', index < m)
        coverage_keys=[];coverage_targets=[]
        coverage_due=bool(stable and (trial-len(initial_configs)+1)%3==0 and trial<budget-6)
        if coverage_due:
            configs_for_coverage,coverage_targets=coverage_candidates(stable,anchor,space,seen,unit,value_at,bounded_patch)
            for cfg in configs_for_coverage:
                add(cfg,'coverage')
                coverage_keys.append(space.config_key(cfg))
                channels.setdefault(space.config_key(cfg),set()).add('reserved_local_coverage')
        if not pool:
            raise RuntimeError('no legal unseen proposals')
        keys = sorted(pool)
        configs = [pool[k] for k in keys]
        scores = gp.log_noisy_expected_improvement(configs, seed=seed + trial)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError('non-finite qLogNEI scores')
        best_index = int(np.argmax(scores))
        prospect = np.flatnonzero(scores >= scores[best_index] + math.log(eta))
        # Include source coverage within the return constraint, then fill by NEI.
        shortlist = []
        for source in (*(hypotheses if stable else EXPERTS), 'joint', 'reference'):
            choices = [i for i in prospect if source in provenance[keys[i]]]
            if choices:
                shortlist.append(max(choices, key=lambda i: scores[i]))
        shortlist = list(dict.fromkeys(shortlist))
        for index in sorted(prospect, key=lambda i: -scores[i]):
            if index not in shortlist:
                shortlist.append(int(index))
            if len(shortlist) >= shortlist_size:
                break
        if not audit_keys:
            fallback_key = keys[best_index]
            audit_keys.add(fallback_key)
            audit_provenance[fallback_key] = {'reference'}
        ref_keys = sorted(audit_keys)
        joint_keys = ref_keys + [keys[i] for i in shortlist if keys[i] not in audit_keys]
        joint_configs = [anchor] + [pool[k] for k in joint_keys]
        mean, covariance, noise = gp.joint_posterior(joint_configs)
        sources = [set()] + [audit_provenance[k] for k in ref_keys]
        audit_count = 1 + len(ref_keys)
        indices = [1 + joint_keys.index(keys[i]) for i in shortlist]
        gains, errors, prior = information_gains(mean, covariance, noise, audit_count, sources,
            indices, rng, delta, fantasies, posterior_samples)
        selected, reason = best_index, 'qLogNEI'
        # Empirical prequential coverage guards against trusting miscalibrated
        # posterior information estimates. High label entropy alone is no veto.
        calibration = float(np.mean(np.asarray(residuals[-20:]) <= 1.96)) if residuals else 0.0
        calibrated = len(residuals) >= 8 and calibration >= .7
        winner = int(np.argmax(gains))
        if variant.information_selection and trial < budget - (6 if stable else 1) and m > 0 and calibrated and gains[winner] > 2 * errors[winner]:
            selected, reason = shortlist[winner], 'decision_information'
        # Explicit real re-entry evaluations are part of the total budget;
        # proxy re-audits alone never count as experimental coverage.
        if variant.dynamic_knobs and probe_interval and (trial - len(initial_configs) + 1) % probe_interval == 0 and trial < budget - (6 if stable else 1):
            inactive = sorted((n for n in age if n not in active), key=lambda n: -age[n])
            for name in inactive:
                start, end = slices[list(probe_groups).index((name,))]
                if start == end:
                    continue
                candidate = probe_configs[start]
                if space.config_key(candidate) not in seen:
                    add(candidate, 'reentry')
                    selected_key, reason = space.config_key(candidate), 'reserved_reentry'
                    break
            else:
                selected_key = keys[selected]
        else:
            selected_key = keys[selected]
        if coverage_due and coverage_keys:
            selected_key=max(coverage_keys,key=lambda k:scores[keys.index(k)])
            reason='reserved_local_coverage'
        selected_config = pool[selected_key]
        # Selected real probes may sit outside both audit and shortlist.
        if selected_key in joint_keys:
            chosen_index = 1 + joint_keys.index(selected_key)
            sm, sc, sn = mean, covariance, noise
        else:
            sm, sc, sn = gp.joint_posterior(joint_configs[:audit_count] + [selected_config])
            chosen_index = len(sm) - 1
        event('proposal', trial=trial, anchor_trial=anchor_record.trial, active=sorted(active),
              expert_history_policy=HISTORY_POLICY,history_count=len(history),
              ablation=variant.describe(), frozen_region_patch_projection=patch_projection,
              conditional_potential=potentials, proposals=proposals, region_changes=region_changes,
              outside_potential=outside_probabilities, candidate_quota=quotas,
              anchor_context_reset=anchor_context_changed, structure_control=structure_report, hypothesis_quotas=hypothesis_quotas, hypotheses=hypotheses,
              actual_execution_coverage={n:stable.state['nodes'][n]['tests'] for n in active} if stable else {},
              generation_kinds=sorted(channels.get(selected_key,set())), coverage_targets=coverage_targets,
              audit_size_per_expert=m if not stable else None, audit_size_per_hypothesis=m if stable else None,
              audit_configs=audits,
              audit_sources={k: sorted(audit_provenance[k]) for k in ref_keys},
              pool_unique=len(pool), eta=eta, calibration=calibration, calibrated=calibrated,
              realized_candidate_counts={h:sum(h in sources for sources in provenance.values()) for h in hypotheses},
              best_log_nei=float(scores[best_index]),
              selected_log_nei=float(scores[keys.index(selected_key)]) if selected_key in keys else None,
              selection_reason=reason, selected_config=selected_config,
              shortlist=[{'config': configs[i], 'log_nei': float(scores[i]), 'IG': float(g), 'IG_se': float(se)}
                         for i, g, se in zip(shortlist, gains, errors)],
              label_probabilities={','.join(k) or 'no_gain': v for k, v in prior.items()})
        source = '+'.join(sorted(provenance[selected_key]))
        predicted_mean = float(sm[chosen_index])
        predicted_std = float(np.sqrt(max(sc[chosen_index, chosen_index] + sn, 1e-12)))
        best, result = evaluate_and_record(evaluator, recorder, 'cognitune_v3', seed, trial,
            selected_config, source, best, budget=budget, posterior_mean=predicted_mean,
            posterior_std=predicted_std, state={'active': sorted(active), 'reason': reason, 'quota': quotas,
                'changed_knobs':[n for n in set(anchor)|set(selected_config) if anchor.get(n)!=selected_config.get(n)],
                'anchor_trial':anchor_record.trial})
        before = dict(quotas)
        posterior_labels = prior
        if result.status == 'SUCCESS':
            residuals.append(abs(result.utility - predicted_mean) / predicted_std)
            cm, cc = condition(sm, sc, chosen_index, result.utility, sn)
            cd = cm[:audit_count] + rng.standard_normal((posterior_samples, audit_count)) @ factor(cc[:audit_count, :audit_count]).T
            posterior_labels = label_probabilities(cd, sources, delta)
            if stable:
                credit=stable.update_credit(posterior_labels)
                quotas=stable.expert_quotas(candidate_pool_size-20)
            else:
                quotas, credit = allocate(posterior_labels, EXPERTS, candidate_pool_size - 20)
        else:
            credit = {}  # Infrastructure failures never become expert blame.
        range_updates=[]
        if stable and result.status=='SUCCESS':
            range_updates=stable.observe(anchor,selected_config,result.utility,anchor_record.utility,trial,delta,unit,
                source=source,reference='reference' in provenance[selected_key])
        event('outcome', trial=trial, range_updates=range_updates, status=result.status, raw_metric=result.raw_metric,
              gain_vs_anchor=result.utility - anchor_record.utility, posterior_credit=credit,
              quota_before=before, quota_after=quotas, quota_changed=before != quotas,
              label_probabilities={','.join(k) or 'no_gain': v for k, v in posterior_labels.items()})
        checkpoint()
    summary = recorder.finalize()
    summary['method_version'] = 'stable-space-v1' if stable else 'decision-entropy-v3'
    summary['ablation'] = variant.describe()
    summary['preserved_observations'] = preserved_observations
    summary['resume_recovery_mode'] = recovery_mode
    summary['protocol'] = {'eta': eta, 'delta_utility': delta, 'audit_size': audit_size,
        'expert_history_policy':HISTORY_POLICY,'expert_prompt_layout':PROMPT_LAYOUT,
        'posterior_samples': posterior_samples, 'fantasies': fantasies, 'candidate_pool_size': candidate_pool_size,
        'probe_interval': probe_interval, 'search_policy':search_policy, 'final_trial': 'pure_qLogNEI',
        'information_scope': 'finite equal-size audit sets, not the full proposal distributions'}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
