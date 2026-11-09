"""Evidence-driven search structure and region control.

Execution counts are coverage of a joint experiment, not per-knob causal credit.
Predictive probes never increment real execution coverage.
"""
from __future__ import annotations
import copy
import numpy as np

from .decision import allocate
from .ablation import Ablation
from cognitune.agents.ownership import EXPERTS


class StableSearch:
    def __init__(self, space, ownership, *, episode=6, minimum_tests=2, max_swaps=2, state=None, ablation='full'):
        self.space, self.ownership = space, ownership
        self.ablation = Ablation(ablation)
        self.episode, self.minimum_tests, self.max_swaps = episode, minimum_tests, max_swaps
        self.state = state or {'active':[], 'last_review':None, 'nodes':{}, 'challengers':{},
                               'hypothesis_credit':{}, 'hypotheses':{}, 'hypothesis_quota':{}}
        for k in space.knobs:
            self.state['nodes'].setdefault(k.name, {'entered':None, 'tests':0, 'lifetime_tests':0,
                'last_gain':None, 'fresh_tests':0, 'tested_values':[], 'width':.15, 'window':[], 'range_changes':[]})

    def snapshot(self):
        return copy.deepcopy(self.state)

    def anchor_context(self, anchor, trial, unit):
        previous=self.state.get('anchor')
        self.state['anchor']=dict(anchor)
        if previous is None: return False
        changed=[n for n in set(previous)|set(anchor) if previous.get(n)!=anchor.get(n)]
        large=len(changed)>4 or any(abs(unit(self.space.by_name[n],anchor[n])-unit(self.space.by_name[n],previous[n]))>.3
                                     for n in changed if n in anchor and n in previous)
        if large:
            for n in self.state['active']:
                node=self.state['nodes'][n]
                node['window']=[];node['fresh_tests']=0
            # A new anchor invalidates local region comparisons, not the fact
            # that these directions were actually evaluated. Require a fresh
            # local test before retirement rather than resetting their lifetime.
        return large

    def distinct_tests_required(self, name):
        knob=self.space.by_name[name]
        if knob.kind=='bool': return 1
        if knob.kind in {'categorical','ordinal'}: return max(1,min(2,len(knob.choices)-1))
        if knob.kind=='integer': return max(1,min(2,int(knob.upper-knob.lower)))
        return 2

    def select(self, proposed, potentials, ranked_groups, trial, protected, limit, delta):
        s=self.state; nodes=s['nodes']; previous=set(s['active'])
        if not previous:
            s['active']=sorted(proposed);s['last_review']=trial
            for n in proposed: nodes[n]['entered']=trial
            return set(proposed), {'review':True,'reason':'initialize','swaps':[]}
        active=set(previous)
        if not self.ablation.dynamic_knobs:
            return active, {'review':False,'reason':'ablation_fixed_knobs','swaps':[]}
        report={'review':False,'reason':'stable_episode','swaps':[]}
        if trial-s['last_review'] < self.episode:
            return active,report
        s['last_review']=trial;report.update(review=True,reason='evidence_review')
        def score(name):
            return potentials[name].get('expected_gain',potentials[name]['p_improve'] * delta)
        incoming=[]
        for group in ranked_groups:
            closure=set(group)
            for n in group:
                if self.space.by_name[n].condition:
                    closure.add(self.space.by_name[n].condition.parent)
            new=closure-active
            if new and len(new)<=self.max_swaps:
                incoming.append((closure,new))
        incoming.sort(key=lambda item: -np.mean([score(n) for n in item[1]]))
        used=0
        for closure,new in incoming:
            new=closure-active
            if not new or used+len(new)>self.max_swaps: continue
            linked=set()
            for group in ranked_groups:
                info=potentials.get('+'.join(group),{})
                if len(group)>1 and set(group)<=active and info.get('p_improve',0)>=.7 and info.get('expected_gain',0)>delta:
                    linked.update(group)
            eligible=[n for n in active if n not in protected and n not in closure and n not in linked
                      and trial-nodes[n]['entered']>=self.episode
                      and nodes[n]['tests']>=self.minimum_tests
                      and len(nodes[n]['tested_values'])>=self.distinct_tests_required(n) and nodes[n]['fresh_tests']>=1
                      and (nodes[n]['last_gain'] is None or trial-nodes[n]['last_gain']>=self.episode)]
            eligible.sort(key=lambda n:score(n))
            if len(eligible)<len(new): continue
            old=eligible[:len(new)]
            sat=all(potentials[n]['state']=='pause' for n in old)
            good=min(new,key=score);bad=max(old,key=score)
            advantage=(score(good)>max(delta,1.25*score(bad))
                       and potentials[good]['p_improve']-potentials[bad]['p_improve']>.15)
            key=','.join(sorted(new))
            s['challengers'][key]=s['challengers'].get(key,0)+1 if advantage else 0
            credible=potentials[good]['interval'][0]>potentials[bad]['interval'][1]+delta
            scheduled=(trial-nodes[bad]['entered']>=2*self.episode and
                potentials[good]['state']=='verify' and potentials[bad]['state']=='verify' and
                score(good)>=.9*score(bad))
            if not (sat or credible or (advantage and s['challengers'][key]>=2) or scheduled):
                continue
            candidate=(active-set(old))|closure
            if len(candidate)>limit: continue
            # Preserve coverage of every previously represented specialist.
            if any(any(e in self.ownership[n] for n in active) and
                   not any(e in self.ownership[n] for n in candidate) for e in EXPERTS): continue
            reason='measured_low_potential' if sat else 'credible_alternative' if credible else 'persistent_alternative_advantage' if advantage and s['challengers'][key]>=2 else 'scheduled_alternative_verification'
            report['swaps'].append({'out':old,'in':sorted(new),'reason':reason,
                'exit_tests':{n:nodes[n]['tests'] for n in old},
                'exit_distinct_values':{n:len(nodes[n]['tested_values']) for n in old},
                'fresh_context_tests':{n:nodes[n]['fresh_tests'] for n in old},
                'exit_residence':{n:trial-nodes[n]['entered'] for n in old}})
            active=candidate;used+=len(new)
            for n in new:
                nodes[n]['entered']=trial;nodes[n]['tests']=0;nodes[n]['window']=[];nodes[n]['fresh_tests']=0;nodes[n]['tested_values']=[]
        s['active']=sorted(active)
        report['unverified_retained']=sorted(n for n in active if nodes[n]['tests']<self.minimum_tests)
        return active,report

    def observe(self, anchor, config, utility, anchor_utility, trial, delta, unit, *, source, reference=False):
        if reference:
            return []  # Full-space reference runs cannot certify local saturation.
        changes=[]
        active=set(self.state['active'])
        changed=[n for n in set(anchor)&set(config) if anchor[n]!=config[n]]
        if len(changed)>4: return []
        for n in changed:
            if n not in active: continue
            k=self.space.by_name[n];node=self.state['nodes'][n]
            distance=abs(unit(k,config[n])-unit(k,anchor[n]))
            if k.kind in {'integer','float'} and distance<min(.01,node['width']*.1): continue
            node['tests']+=1;node['lifetime_tests']+=1;node['fresh_tests']+=1
            coordinate=unit(k,config[n])
            if not any(abs(coordinate-v)<.005 for v in node['tested_values']): node['tested_values'].append(coordinate)
            gain=utility-anchor_utility
            if gain>delta: node['last_gain']=trial
            if k.kind not in {'integer','float'}: continue
            relative=(unit(k,config[n])-unit(k,anchor[n]))/node['width']
            node['window'].append({'position':relative,'gain':gain,'trial':trial,'source':source})
            node['window']=node['window'][-6:]
            if not self.ablation.dynamic_regions:
                continue  # Keep real coverage evidence, freeze the width only.
            window=node['window']
            if len(window)<3: continue
            positions=[x['position'] for x in window]
            gains=[x['gain'] for x in window]
            boundary_success=any(abs(x['position'])>=.7 and x['gain']>delta for x in window)
            covered=min(positions)<-.25 and max(positions)>.25
            width=node['width'];new=width;reason=None
            if boundary_success and sum(g>delta for g in gains)>=2:
                new=min(.5,width*1.5);reason='repeated_boundary_improvement'
            elif covered and all(g<=delta for g in gains[-3:]):
                new=max(.025,width*.8);reason='covered_local_plateau_refine'
            elif covered and any(abs(x['position'])<.7 and x['gain']>delta for x in window):
                new=max(.025,width*.8);reason='supported_interior_refine'
            if abs(new-width)>1e-12:
                change={'knob':n,'before':width,'after':new,'reason':reason,
                        'evidence':copy.deepcopy(window),'trial':trial}
                node['width']=new;node['window']=[];node['range_changes'].append(change)
                changes.append(change)
        return changes

    def hypotheses(self, proposals, scopes, active, total, audit_size, confirmed_groups=()):
        """Construct at most 12 sparse, explicit search modes, including partners.

        Modes with the same parameter group share credit regardless of proposer.
        Explicit patches preserve the entire group when expanded into candidates.
        """
        modes={}
        for expert in EXPERTS:
            scope=scopes[expert]
            groups=[]
            for patch in proposals[expert].get('patches',[])[:4]:
                if patch and set(patch)<=set(scope) and len(patch)<=4:
                    groups.append(tuple(sorted(patch)))
            # Coverage direction is coherent and prioritized by real tests.
            ordered=sorted(scope,key=lambda n:(self.state['nodes'][n]['tests'],n))
            if ordered: groups.append(tuple(sorted(ordered[:min(2,len(ordered))])))
            if not groups and scope: groups=[tuple(scope[:1])]
            for group in list(dict.fromkeys(groups))[:2]:
                key='group:'+','.join(group)
                if key not in modes: modes[key]={'knobs':list(group),'experts':[],'patches':[],'kind':'local'}
                modes[key]['experts'].append(expert)
                modes[key]['patches'].extend(p for p in proposals[expert].get('patches',[]) if set(p)==set(group))
        # Retain acknowledged joint groups only when all partners are active.
        # Otherwise admission is handled at a structure review; no hidden widening.
        for group in confirmed_groups:
            if set(group)<=set(active) and len(group)<=4:
                key='group:'+','.join(sorted(group))
                if key not in modes and len(modes)<12:
                    modes[key]={'knobs':list(group),'experts':sorted({e for n in group for e in self.ownership[n]}),
                                'patches':[],'kind':'joint'}
        if not modes: raise RuntimeError('no executable search hypotheses')
        # More groups cannot steal all candidate slots; audit size is fixed by
        # the protocol and equalized after feasibility sampling.
        keys=list(modes)
        probabilities={(k,):self.state['hypothesis_credit'].get(k,0) for k in keys}
        quotas,_=allocate(probabilities if self.ablation.adaptive_allocation else {},keys,total,minimum=min(audit_size+1,total//len(keys)))
        self.state['hypotheses']=copy.deepcopy(modes)
        self.state['hypothesis_quota']=quotas
        return modes,quotas

    def update_credit(self, probabilities):
        credit={k:0.0 for k in self.state['hypotheses']}
        for subset,p in probabilities.items():
            for key in subset:
                if key in credit: credit[key]+=p/len(subset)
        self.state['hypothesis_credit']={k:v*.8 for k,v in self.state['hypothesis_credit'].items()}
        self.state['hypothesis_credit'].update(credit)
        total=sum(self.state['hypothesis_quota'].values())
        probabilities={(k,):v for k,v in credit.items()} if self.ablation.adaptive_allocation else {}
        quotas,_=allocate(probabilities,list(credit),total,minimum=min(5,total//len(credit)))
        self.state['hypothesis_quota']=quotas
        return credit

    def expert_quotas(self, total):
        if not self.ablation.adaptive_allocation:
            return allocate({},EXPERTS,total)[0]
        weights={e:0.0 for e in EXPERTS}
        for key,q in self.state['hypothesis_quota'].items():
            experts=self.state['hypotheses'][key]['experts']
            for expert in experts: weights[expert]+=q/len(experts)
        result,_=allocate({(e,):w/max(1,sum(weights.values())) for e,w in weights.items()},EXPERTS,total)
        return result


def build_mode_pool(controller, proposals, scopes, active, regions, anchor, space, seen,
                    rng, count, audit_size, add, unit, value_at, bounded_patch, confirmed_groups=()):
    modes,quotas=controller.hypotheses(proposals,scopes,active,count,audit_size,confirmed_groups)
    channels={}
    def register(config,key,kind,audit=False):
        if config is None: return
        add(config,key,audit)
        ck=space.config_key(config)
        channels.setdefault(ck,set()).add(kind)

    def sample(key,n):
        mode=modes[key];names=mode['knobs'];result=[];local=set()
        for _ in range(max(100,n*60)):
            if len(result)>=n: break
            explicit=mode['patches'][int(rng.integers(len(mode['patches'])))] if mode['patches'] else {}
            patch={}
            for name in names:
                k=space.by_name[name]
                owners=[e for e in mode['experts'] if name in regions[e]]
                if k.kind not in {'integer','float'}:
                    choices=[False,True] if k.kind=='bool' else list(k.choices)
                    if name in explicit and rng.random()<.75:
                        patch[name]=explicit[name]
                    else:
                        prefs=[v for e in owners for v in regions[e][name].get('values',[])]
                        values=choices if rng.random()<.1 else prefs or [anchor.get(name,k.default)]
                        patch[name]=values[int(rng.integers(len(values)))]
                    continue
                center=np.mean([regions[e][name]['center'] for e in owners]) if owners else unit(k,anchor.get(name,k.default))
                width=controller.state['nodes'][name]['width']
                if name in explicit and controller.ablation.dynamic_regions: center=unit(k,explicit[name])
                if rng.random()<.1:
                    u=rng.random()
                else:
                    lo,hi=max(0,center-width),min(1,center+width)
                    if name in explicit and controller.ablation.dynamic_regions:
                        base=unit(k,anchor.get(name,k.default))
                        if center>base: lo=max(lo,base)
                        if center<base: hi=min(hi,base)
                    u=rng.uniform(lo,hi)
                patch[name]=value_at(k,u)
            cfg=bounded_patch(space,anchor,patch,space.by_name)
            if cfg is None: continue
            ck=space.config_key(cfg)
            if ck in seen or ck in local: continue
            local.add(ck);result.append(cfg)
        return result

    audits={k:sample(k,min(audit_size,quotas[k])) for k in modes}
    m=min(map(len,audits.values()))
    audits={k:v[:m] for k,v in audits.items()}
    for key,mode in modes.items():
        for cfg in audits[key]: register(cfg,key,'audit_group_sample',True)
        explicit=[];keys=set()
        for patch in mode['patches']:
            cfg=bounded_patch(space,anchor,patch,space.by_name)
            if cfg is not None and space.config_key(cfg) not in seen and space.config_key(cfg) not in keys:
                explicit.append(cfg);keys.add(space.config_key(cfg))
        explicit=explicit[:max(0,quotas[key]-m)]
        for cfg in explicit: register(cfg,key,'explicit_patch')
        for cfg in sample(key,max(0,quotas[key]-m-len(explicit))): register(cfg,key,'structured_group_sample')
    return audits,m,modes,quotas,channels


def coverage_candidates(controller, anchor, space, seen, unit, value_at, bounded_patch):
    active=controller.state['active']
    if not active: return [],[]
    targets=sorted(active,key=lambda n:(controller.state['nodes'][n]['tests'],n))[:2]
    # Include joint and single directions: a constraint or categorical encoding
    # must not make an under-tested parameter unreachable.
    groups=[targets]+[[n] for n in targets]
    result=[];keys=set()
    for names in groups:
        for direction in (-1,1):
            patch={}
            for name in names:
                k=space.by_name[name];current=anchor.get(name,k.default)
                history=controller.state['nodes'][name]['window']
                preferred=-1 if history and all(x['position']>=0 for x in history) else 1 if history and all(x['position']<=0 for x in history) else direction
                if k.kind in {'integer','float'}:
                    width=controller.state['nodes'][name]['width']
                    patch[name]=value_at(k,unit(k,current)+preferred*width*.75)
                else:
                    choices=[False,True] if k.kind=='bool' else list(k.choices)
                    patch[name]=choices[(choices.index(current)+direction)%len(choices)]
                if k.condition and name not in anchor: patch[k.condition.parent]=k.condition.values[0]
            cfg=bounded_patch(space,anchor,patch,space.by_name)
            if cfg is None: continue
            key=space.config_key(cfg)
            if key in keys or key in seen: continue
            if not any(cfg.get(n)!=anchor.get(n) for n in names): continue
            keys.add(key);result.append(cfg)
    return result,targets
