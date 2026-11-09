#!/usr/bin/env python3
"""Read-only analysis of real tuning traces; write separate reviewable artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def read_rows(path):
    if not path.exists(): return []
    content=path.read_bytes()
    result=[]
    lines=content.splitlines()
    for i,line in enumerate(lines):
        if not line.strip(): continue
        try: result.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1 and not content.endswith(b'\n'): break
            raise
    return result


def measurement_stats(rows):
    # Never silently remove failed final validations to inflate a median.
    valid=[r for r in rows if r.get('status')=='SUCCESS' and math.isfinite(float(r['raw_metric']))
           and r['raw_metric']>0]
    values=[float(r['raw_metric']) for r in valid]
    complete=bool(rows) and len(valid)==len(rows)
    return {'total':len(rows),'successes':len(valid),'values':values,
            'median':statistics.median(values) if complete else None,
            'cv':statistics.stdev(values)/statistics.mean(values) if complete and len(values)>1 else None}


def collect(label, directory):
    directory=Path(directory)
    trace=directory/'trials.jsonl'
    rows=read_rows(trace)
    if not rows: raise ValueError('No complete trial records: '+str(trace))
    if [r['trial'] for r in rows]!=list(range(len(rows))):
        raise ValueError('Non-contiguous trial trace: '+str(trace))
    base=directory.parents[2]
    protocol=json.loads((base/'protocol.json').read_text())
    n_init=protocol['n_init']
    curve=[];best=-math.inf;first_best=None
    for r in rows:
        value=float(r['raw_metric'])
        if r['status']=='SUCCESS' and math.isfinite(value) and value>0 and value>best:
            best=value;first_best=r['trial']+1
        curve.append(best if math.isfinite(best) else None)
    initial_best=max((float(r['raw_metric']) for r in rows[:n_init] if r['status']=='SUCCESS'),default=None)
    validation_path=directory/'final_validation.json'
    validations=json.loads(validation_path.read_text())['results'] if validation_path.exists() else []
    validation=measurement_stats(validations)
    # The declared protocol uses three independent best-config rechecks.
    if validation['total']!=protocol.get('final_validation_repeats',3):
        validation['median']=None
    before=measurement_stats(json.loads((base/'default_before.json').read_text()))
    after_path=base/('default_after_'+rows[0]['method']+'.json')
    after=measurement_stats(json.loads(after_path.read_text())) if after_path.exists() else measurement_stats([])
    events=read_rows(directory/'round_diagnostics.jsonl')
    proposals=[e for e in events if e.get('event')=='proposal']
    outcomes=[e for e in events if e.get('event')=='outcome']
    swaps=[s for e in proposals for s in (e.get('structure_control') or {}).get('swaps',[])]
    ranges=[r for e in outcomes for r in e.get('range_updates',[])]
    improvements=[]
    best_initial=initial_best
    for r in rows[n_init:]:
        if r['status']=='SUCCESS' and (best_initial is None or r['raw_metric']>best_initial):
            improvements.append({'round':r['trial']+1,'TPS':r['raw_metric'],
                'source':r['candidate_source'],'reason':(r.get('state') or {}).get('reason')})
            best_initial=r['raw_metric']
    width_changes=sum(c['before']['halfwidth']!=c['after']['halfwidth']
                      for e in proposals for expert in e.get('region_changes',{}).values() for c in expert.values())
    centre_changes=sum(c['before']['center']!=c['after']['center']
                       for e in proposals for expert in e.get('region_changes',{}).values() for c in expert.values())
    row={'label':label,'completed':len(rows),'budget':protocol['budget'],
         'successes':sum(r['status']=='SUCCESS' for r in rows),
         'best_TPS':best if math.isfinite(best) else None,'first_best_round':first_best,
         'initial_best_TPS':initial_best,
         'search_gain_over_initial_percent':100*(best/initial_best-1) if initial_best else None,
         'validation_median_TPS':validation['median'],'validation_cv':validation['cv'],
         'default_median_TPS':before['median'],'default_cv':before['cv'],
         'default_drift_percent':100*(after['median']/before['median']-1) if after['median'] and before['median'] else None,
         'validated_gain_over_default_percent':100*(validation['median']/before['median']-1) if validation['median'] and before['median'] else None,
         'active_sets':len({tuple(e['active']) for e in proposals}),
         'knobs_admitted':sum(len(s['in']) for s in swaps) if any(e.get('structure_control') for e in proposals) else None,
         'width_update_events':len(ranges),'region_width_changes_in_proposals':width_changes,
         'region_center_changes':centre_changes,
         'information_selections':sum(e['selection_reason']=='decision_information' for e in proposals),
         'coverage_selections':sum(e['selection_reason']=='reserved_local_coverage' for e in proposals),
         'expert_quota_changes':sum(e['quota_changed'] for e in outcomes)}
    details={'source':str(trace.resolve()),'source_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
        'protocol':protocol,'validation':validation,'default_before':before,'default_after':after,
        'selection_reasons':dict(Counter(e['selection_reason'] for e in proposals)),
        'swap_reasons':dict(Counter(s['reason'] for s in swaps)),
        'range_update_reasons':dict(Counter(r['reason'] for r in ranges)),
        'improvements':improvements,'curve':curve,
        'reference_selections':sum('reference' in r['candidate_source'] for r in rows[n_init:]),
        'calibrated_rounds':sum(e['calibrated'] for e in proposals)}
    return row,details


def write_report(runs, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    collected=[collect(label,path) for label,path in runs]
    rows=[r for r,_ in collected]
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'diagnostics.json').write_text(json.dumps({r['label']:{'summary':r,**d} for r,d in collected},indent=2)+'\n')
    lines=['# Twitter experiment audit','',
        'Peak = maximum successful tuning observation. Validation = median of all three successful outside-budget rechecks.',
        'Each run uses its own default baseline. Cross-day/single-seed differences are descriptive, not causal evidence.',
        'Old stable-v1 includes a discrete-region code fix after evaluation 28; use a fresh full run for ablations.','',
        '| Method | Completed | Peak TPS | Best round | Validation median TPS | Default median TPS | Active sets | Admissions | Width updates |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v): return f'{v:,.2f}' if v is not None else 'pending/invalid'
    for r,d in collected:
        admissions=r['knobs_admitted'] if r['knobs_admitted'] is not None else 'N/A'
        lines.append(f"| {r['label']} | {r['completed']}/{r['budget']} | {fmt(r['best_TPS'])} | {r['first_best_round']} | {fmt(r['validation_median_TPS'])} | {fmt(r['default_median_TPS'])} | {r['active_sets']} | {admissions} | {r['width_update_events']} |")
    for r,d in collected:
        lines+=['',f"## {r['label']}",'',f"- Selection reasons: {json.dumps(d['selection_reasons'])}",
            f"- Swap reasons: {json.dumps(d['swap_reasons'])}",
            f"- Range updates: {json.dumps(d['range_update_reasons'])}",
            f"- Improvements: {json.dumps(d['improvements'])}",
            f"- Validation observations: {d['validation']['values']}"]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'font.size':10})
    fig,ax=plt.subplots(figsize=(9,4.8))
    for r,d in collected:
        ax.plot(range(1,len(d['curve'])+1),d['curve'],label=r['label'],lw=1.8)
    ax.set(xlabel='Configuration evaluations (including IID starts)',ylabel='Best-so-far throughput (TPS)',
           title='Twitter tuning: observed best throughput')
    ax.grid(ls='--',alpha=.3);ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(output/f'best_tps.{ext}',dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8))
    for i,(r,d) in enumerate(collected):
        ax.scatter([i-.13],[r['best_TPS']],marker='^',color='#2474b5',label='Observed peak' if i==0 else None)
        if d['validation']['values']:
            ax.scatter([i+.13]*len(d['validation']['values']),d['validation']['values'],s=18,alpha=.5,color='#e1812c')
        if r['validation_median_TPS'] is not None:
            ax.scatter([i+.13],[r['validation_median_TPS']],marker='_',s=180,color='#a84a08',label='Recheck median' if i==0 else None)
    ax.set_xticks(range(len(rows)),[r['label'] for r in rows],rotation=15,ha='right')
    ax.set(ylabel='Throughput (TPS)',title='Observed peaks and outside-budget rechecks')
    ax.grid(axis='y',ls='--',alpha=.3);ax.legend();fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(output/f'validation_tps.{ext}',dpi=220)
    plt.close(fig)
    print(output.resolve())
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',default=[],metavar='LABEL=RUN_DIRECTORY')
    parser.add_argument('--root',type=Path,default=ROOT/'artifacts/twitter_ablation_50_v2')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    runs=[tuple(value.split('=',1)) for value in args.run]
    if not runs:
        runs=[(d.name,d/'runs/cognitune_v3/seed_0') for d in sorted(args.root.iterdir())
              if (d/'runs/cognitune_v3/seed_0/trials.jsonl').exists()]
    if not runs: parser.error('No runs found; specify --run LABEL=RUN_DIRECTORY')
    write_report(runs,args.output or args.root/'analysis')
