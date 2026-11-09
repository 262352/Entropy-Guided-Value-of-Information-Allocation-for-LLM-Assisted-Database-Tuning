#!/usr/bin/env python3
"""Plot supplied Twitter/TPC-C series and measured YCSB/CH throughput deltas."""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/paper_revision/figures'
LABELS = {'ours':'Ours', 'mctuner_official':'MCTuner',
          'gp_bo':'GP-BO','smac':'SMAC','ottertune':'OtterTune',
          'cdbtune':'CDBTune','restune':'ResTune','dot_official':'DOT'}
COLORS = dict(zip(LABELS, ['#d62728','#ff7f0e','#2ca02c','#9467bd',
                          '#8c564b','#e377c2','#7f7f7f','#1f77b4']))


def supplied_series():
    """Load archived illustrative input so another checkout needs no chat attachment."""
    path=ROOT/'docs/paper_revision/figures/supplied_throughput_series.json'
    return json.loads(path.read_text())['series'],path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.labelsize':12,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5))
    supplied,attachment=supplied_series()
    report = {'created_utc':datetime.now(timezone.utc).isoformat(),
              'formula':'delta = (max successful throughput through round r - default) / default',
              'default_definition':'median of three pre-run default measurements',
              'notes':['Twitter/TPC-C use user-provided illustrative arrays, including their padding/truncation to 100; they are not measured traces.',
                       'Their denominator is the corresponding historical pre-run measured default median, not the first initialization point.',
                       'YCSB/CH are baseline-only; CH objective also includes analytical latency, absent from this throughput-only figure.',
                       'TPC-H omitted because its main metric is latency.'], 'workloads':{}}
    rows=[];handles={}
    specs=[('twitter','Twitter',ROOT/'artifacts/twitter_pair_50_v3',100),
           ('tpcc','TPC-C',ROOT/'artifacts/real_pair_50',100),
           ('ycsb','YCSB',ROOT/'artifacts/remaining_baselines_100_v1/ycsb',100),
           ('chbenchmark','CH-benCHmark',ROOT/'artifacts/remaining_baselines_100_v1/chbenchmark',100)]
    for ax,(workload,title,source,budget) in zip(axs.flat,specs):
        legacy = workload in ('twitter','tpcc')
        if legacy:
            default_path=source/'default_before.json'
            default=statistics.median(r['raw_metric'] for r in json.loads(default_path.read_text()))
            paths={m:attachment for m in supplied[workload]}
        else:
            default_path=source/'default.json'
            default=json.loads(default_path.read_text())['raw_metric']
            protocol=json.loads((source.parent/'protocol.json').read_text())
            paths={m:source/m/'ledger' for m in protocol['methods']}
        assert math.isfinite(default) and default>0
        meta={'default':default,'default_source':str(default_path.relative_to(ROOT)),
              'unit':'legacy Goodput requests/s' if workload=='tpcc' else 'transactional TPS' if workload=='chbenchmark' else 'successful requests/s',
              'data_origin':'user_supplied_arrays' if legacy else 'measured_trial_ledger',
              'series':{}}
        for method in LABELS:
            if method not in paths: continue
            path=paths[method]
            if legacy:
                records=[{'trial':i,'raw_metric':v,'status':'SUPPLIED'} for i,v in enumerate(supplied[workload][method])]
                indices=[r['trial'] for r in records]
            else:
                files=sorted(path.glob('trial_*/result.json'))
                indices=[int(p.parent.name[6:]) for p in files]
                records=[json.loads(p.read_text()) for p in files]
            assert indices==list(range(len(records))) and 0<len(records)<=budget
            x=[];y=[];best=-math.inf
            for trial,r in zip(indices,records):
                value=float(r['raw_metric'])
                if r['status'] in ('SUCCESS','SUPPLIED') and math.isfinite(value) and value>0:
                    best=max(best,value)
                delta=(best-default)/default if math.isfinite(best) else math.nan
                x.append(trial+1);y.append(delta)
                rows.append({'workload':workload,'method':method,'round':trial+1,'data_origin':meta['data_origin'],
                             'raw_throughput':value,'default_throughput':default,
                             'best_throughput':best,'delta':delta,'status':r['status']})
            assert all(a<=b for a,b in zip(y,y[1:]) if math.isfinite(a) and math.isfinite(b))
            line,=ax.plot(x,y,label=LABELS[method],color=COLORS[method],marker='o',
                         markersize=4,markevery=10,linewidth=2.6 if method in ('ours','mctuner_official') else 1.9,
                         drawstyle='steps-post',alpha=.95)
            handles.setdefault(method,line)
            meta['series'][method]={'rounds':len(records),'source':str(path) if legacy else str(path.relative_to(ROOT)),
                                    'best_throughput':best,'final_delta':y[-1]}
        ax.set_xlim(1,budget)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(which='major',linestyle='--',linewidth=.7,alpha=.55)
        ax.grid(which='minor',linestyle=':',linewidth=.5,alpha=.3)
        ax.axhline(0,color='gray',linewidth=.8,alpha=.6)
        ax.set_ylabel(r'Best throughput improvement $\Delta$')
        ax.set_xlabel(f'{title} — configuration evaluations',labelpad=10)
        report['workloads'][workload]=meta
    order=[m for m in LABELS if m in handles]
    fig.legend([handles[m] for m in order],[LABELS[m] for m in order],loc='lower center',
               bbox_to_anchor=(.5,.06),ncol=4,frameon=False,fontsize=12,columnspacing=1.5)
    fig.text(.5,.025,'Best-so-far throughput improvement over default',ha='center',fontsize=14)
    fig.subplots_adjust(bottom=.23,top=.97,wspace=.22,hspace=.36,left=.08,right=.98)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'throughput_delta_2x2.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    (OUT/'throughput_delta_2x2.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'throughput_delta_2x2.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
