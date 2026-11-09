#!/usr/bin/env python3
"""Persistent workload-major baseline queue, with no execution path for our method."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid
import statistics
import signal
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cognitune.config import load_search_space
from cognitune.experiment.resilience import DurableEvaluator,InfrastructureUnavailable,ReplayDivergence,atomic_json
from cognitune.search_space.initial_design import build_initial_design,load_initial_design,save_initial_design
from baseline_workloads import WorkloadBackend,SPECS,control_variation,acceptable_drift
from baseline_methods import METHODS,run_method
import real_benchbase as h


def root_path(args): return ROOT/args.output


def protocol(args):
    return {'version':'other-baselines-v1','workloads':args.workloads,'methods':args.methods,
        'budget':args.budget,'n_init':10,'initialization':'shared IID including default',
        'seed':0,'warmup':15,'duration':60,'space':'configs/knobs/postgresql_60_diagnostic.yaml',
        'hybrid_utility':'0.5 log(transactional_tps/default) + 0.5 log(default_analytical_p95/analytical_p95)',
        'candidate_attempts':2,'recovery_controls':'default configuration executes same workload',
        'real_attempt_accounting':'100 optimizer configuration slots; failed attempts/retries/controls/preflight/rechecks counted separately in raw ledgers',
        'adapters':{'gp_bo':'common GP LogEI','smac':'SMAC library',
          'ottertune':'ARD importance GP adaptation; not original factor-analysis/workload-map pipeline',
          'cdbtune':'online DDPG adaptation, not a pretrained official CDBTune model',
          'restune':'weighted source-GP adaptation; cold start if no historical source tasks',
          'mctuner_official':'upstream expert selection/tree/classifier/GP with common measured utility',
          'dot_official':'upstream DOT GP epoch/RFECV/LRT; PostgreSQL, shared initial and measurement adapters'}}


def prepare(args):
    from build_baseline_benchbase import build
    directory=root_path(args);directory.mkdir(parents=True,exist_ok=True)
    p=protocol(args)
    target=directory/'protocol.json'
    if target.exists() and json.loads(target.read_text())!=p:
        raise ValueError('Protocol changed; use a new output directory instead of mixing runs')
    atomic_json(target,p)
    jar=build();space=load_search_space(ROOT/p['space'])
    design=directory/'initial_design.json'
    if not design.exists():
        # Reuse the already persisted shared IID configs from the paired Twitter run.
        source=ROOT/'artifacts/twitter_pair_50_stable_v1/initial_design/postgresql_twitter_seed_0.json'
        if source.exists():
            data=json.loads(source.read_text());load_initial_design(source,space,10,0);save_initial_design(design,data)
        else: save_initial_design(design,build_initial_design(space,0,10))
    load_initial_design(design,space,10,0)
    atomic_json(directory/'source_provenance.json',{'jar_sha256':hashlib.sha256(jar.read_bytes()).hexdigest(),
        'official_dot_commit':subprocess.check_output(['git','-C',str(ROOT/'.runtime/DOT-official'),'rev-parse','HEAD'],text=True).strip(),
        'official_mctuner_commit':subprocess.check_output(['git','-C',str(ROOT/'.runtime/MCTuner-official'),'rev-parse','HEAD'],text=True).strip()})
    return p,jar,space


def ensure_workload(args,workload,jar,space):
    directory=root_path(args)/workload
    backend=WorkloadBackend(workload,space,directory/'measurements/preflight',jar)
    backend.prepare()
    baseline=directory/'default.json'
    if not baseline.exists():
        check=DurableEvaluator(backend,directory/'default_ledger')
        records=[check.evaluate(space.get_default(),i).as_dict() for i in range(3)]
        if any(r['status']!='SUCCESS' for r in records): raise InfrastructureUnavailable('default baseline failed')
        value={'raw_metric':statistics.median(r['raw_metric'] for r in records)}
        if workload=='chbenchmark':
            value.update({k:statistics.median(r['metadata'][k] for r in records) for k in ['transactional_tps','analytical_p95_ms']})
        value['cv_by_metric']=control_variation(records,workload)
        value['cv']=value['cv_by_metric']['raw_metric']
        value['stable']=max(value['cv_by_metric'].values())<=.15
        value['records']=records;atomic_json(baseline,value)
    values=json.loads(baseline.read_text());backend.baseline=values
    if not values['stable']: raise InfrastructureUnavailable('default CV exceeds 15%; evidence preserved, comparison not launched')
    # Required health gate on this dataset; successful preflight remains resumable.
    gate=DurableEvaluator(backend,directory/'health_ledger')
    for i in range(12):
        result=gate.evaluate(space.get_default(),i)
        if result.status!='SUCCESS': raise InfrastructureUnavailable('12-round health gate failed')
    initial=load_initial_design(root_path(args)/'initial_design.json',space,10,0)
    shared=DurableEvaluator(backend,directory/'shared_initial')
    results={i:shared.evaluate(config,i).as_dict() for i,config in enumerate(initial)}
    atomic_json(directory/'preparation.json',{'baseline':values,'shared_initial':results})
    return values,results,initial


def worker(args):
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Worker interrupted; measurement ledger retained')
    signal.signal(signal.SIGTERM,interrupted)
    import numpy as np
    import torch
    p,jar,space=prepare(args)
    workload,method=args.worker
    values,shared,initial=ensure_workload(args,workload,jar,space)
    job=root_path(args)/workload/method
    session=job/'sessions'/uuid.uuid4().hex;session.mkdir(parents=True)
    backend=WorkloadBackend(workload,space,job/'measurements',jar,baseline=values)
    evaluator=DurableEvaluator(backend,job/'ledger',initial=shared)
    profile={'workload':workload,'objective':SPECS[workload]['objective'],**SPECS[workload],
             'warmup_seconds':15,'measurement_seconds':60,'knobs':len(space.knobs)}
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(2)
    sources=job/'source_tasks'
    source_manifest=job/'source_tasks.json'
    if not source_manifest.exists():
        sources.mkdir(parents=True,exist_ok=True)
        chosen={}
        for source in sorted((root_path(args)/'meta').glob('*.json')):
            if source.stem!=workload:
                value=json.loads(source.read_text());atomic_json(sources/source.name,value)
                chosen[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
        atomic_json(source_manifest,chosen)
    atomic_json(session/'implementation.json',{'source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),ROOT/'scripts/baseline_methods.py',ROOT/'scripts/baseline_workloads.py',ROOT/'src/cognitune/experiment/resilience.py']}})
    summary=run_method(method,space,evaluator,initial,args.budget,0,session,job,profile,sources)
    controls=DurableEvaluator(backend,job/'final_validation')
    validation=[controls.evaluate(summary['best_config'],i).as_dict() for i in range(3)]
    drift=DurableEvaluator(backend,job/'default_after')
    drift_records=[drift.evaluate(space.get_default(),i).as_dict() for i in range(3)]
    summary['default_after']=drift_records
    summary['comparison_valid']=acceptable_drift(drift_records,values,workload)
    summary['final_validation']=validation
    summary['validated']=all(v['status']=='SUCCESS' for v in validation)
    summary['session_directory']=str(session)
    atomic_json(job/'summary.json',summary)
    if method=='gp_bo':
        records=[{'config':json.loads(x.read_text())['config'],'utility':json.loads(x.read_text())['utility'],'status':json.loads(x.read_text())['status']} for x in sorted((job/'ledger').glob('trial_*/result.json'))]
        atomic_json(root_path(args)/'meta'/f'{workload}.json',records)
    print('[COMPLETE]',workload,method,'validated=',summary['validated'],flush=True)


def supervise(args):
    directory=root_path(args);directory.mkdir(parents=True,exist_ok=True)
    lock=(h.RUNTIME/'real-comparison.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(directory/'supervisor.json',{'pid':os.getpid(),'started':time.time()})
    prepare(args)
    # Workload-major queue. Errors defer a job, never start our optimizer.
    for cycle in range(args.cycles):
        pending=0
        for workload in args.workloads:
            for method in args.methods:
                job=directory/workload/method;done=job/'summary.json'
                if done.exists() and json.loads(done.read_text()).get('validated'): continue
                pending+=1;job.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,str(Path(__file__).resolve()),'--output',args.output,
                         '--budget',str(args.budget),'--workloads',*args.workloads,
                         '--methods',*args.methods,'--worker',workload,method]
                log_path=job/f'console_cycle_{cycle}_{int(time.time())}.log'
                env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                with log_path.open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True)
                    atomic_json(job/'status.json',{'state':'RUNNING','pid':child.pid,'cycle':cycle,'log':str(log_path)})
                    began=time.time()
                    while child.poll() is None:
                        progress=[p.stat().st_mtime for pattern in ['**/result.json','**/recovery_events.jsonl'] for p in (directory/workload).glob(pattern)]
                        if time.time()-max([began,*progress])>args.stall_seconds:
                            print('[WATCHDOG] no completed measurement/replay progress; terminating worker, preserving ledger',flush=True)
                            child.terminate()
                            try: child.wait(timeout=30)
                            except subprocess.TimeoutExpired: child.kill();child.wait(timeout=10)
                            break
                        atomic_json(directory/'heartbeat.json',{'time':time.time(),'workload':workload,'method':method,'pid':child.pid,'log':str(log_path)})
                        time.sleep(10)
                status='COMPLETE' if child.returncode==0 else 'DEFERRED'
                atomic_json(job/'status.json',{'state':status,'returncode':child.returncode,'cycle':cycle,'log':str(log_path)})
                print('[QUEUE]',workload,method,status,'log='+str(log_path),flush=True)
                if child.returncode==75 and not (directory/workload/'preparation.json').exists():
                    print('[QUEUE] workload preparation unhealthy; defer remaining methods instead of repeating failed measurements',flush=True)
                    break
        if pending==0: break
        if cycle+1<args.cycles:
            print('[QUEUE] cooling down before deferred jobs',flush=True);time.sleep(args.cooldown)
    atomic_json(directory/'queue_finished.json',{'time':time.time(),'jobs':{w+'/'+m:('complete' if (directory/w/m/'summary.json').exists() and json.loads((directory/w/m/'summary.json').read_text()).get('validated') else 'pending') for w in args.workloads for m in args.methods}})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='artifacts/remaining_baselines_100_v1')
    parser.add_argument('--budget',type=int,default=100)
    parser.add_argument('--workloads',nargs='+',choices=SPECS,default=['ycsb','tpch','chbenchmark'])
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--worker',nargs=2)
    parser.add_argument('--cycles',type=int,default=3)
    parser.add_argument('--cooldown',type=int,default=300)
    parser.add_argument('--stall-seconds',type=int,default=3600)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.budget<10: parser.error('Budget includes at least ten shared initial configurations')
    if args.worker and (args.worker[0] not in args.workloads or args.worker[1] not in args.methods): parser.error('worker not in declared baseline queue')
    try:
        if args.prepare_only: prepare(args)
        elif args.worker: worker(args)
        else: supervise(args)
    except InfrastructureUnavailable:
        traceback.print_exc();sys.exit(75)
    except ReplayDivergence:
        traceback.print_exc();sys.exit(76)

if __name__=='__main__':main()
