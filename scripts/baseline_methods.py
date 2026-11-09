"""Baseline adapters. No CogniTune/JointTune optimizer is registered here."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import numpy as np
from cognitune.experiment.recorder import RunRecorder
from cognitune.optimizers.common import evaluate_and_record
from cognitune.optimizers import run_gp_bo, run_smac, run_ottertune, run_cdbtune, run_restune

METHODS=('smac','gp_bo','cdbtune','restune','ottertune','mctuner_official','dot_official')


def run_method(method,space,evaluator,initial,budget,seed,run_dir,root,profile,meta_dir):
    if method not in METHODS: raise ValueError('Only declared comparison baselines may run')
    args=(space,evaluator,initial,budget,seed)
    if method=='smac': return run_smac(*args,run_dir)
    if method=='gp_bo': return run_gp_bo(*args,1000,run_dir,duplicate_new_best=False)
    if method=='ottertune': return run_ottertune(*args,1000,run_dir,duplicate_new_best=False)
    if method=='cdbtune': return run_cdbtune(*args,1000,run_dir,duplicate_new_best=False)
    if method=='restune': return run_restune(*args,1000,run_dir,meta_dir=meta_dir,duplicate_new_best=False)
    if method=='mctuner_official':
        import compare_twitter as pair
        pair.OUT=Path(root);pair.BUDGET=budget;pair.N_INIT=len(initial);pair.SEED=seed
        goal={'throughput':'Maximize successful measured throughput.',
              'latency':'Minimize measured successful analytical-query p95 latency.',
              'hybrid':'Maximize equal-weight log improvement of transactional throughput and analytical p95 latency.'}[profile['objective']]
        selection=pair.official_selection(space,context=profile,goal=goal+' Respect durability and authoritative legal domains.')
        return pair.run_official(space,evaluator,initial,selection,run_dir)
    return run_dot(space,evaluator,initial,budget,seed,run_dir)


def run_dot(space,evaluator,initial,budget,seed,run_dir):
    """Use upstream GP epochs, RFECV and LRT; replace only DB/encoding interface."""
    base=Path(__file__).resolve().parents[1]/'.runtime/DOT-official'
    sys.path[:0]=[str(base/'tuner'),str(base/'Drivers')]
    spec=importlib.util.spec_from_file_location('upstream_dot_main',base/'tuner/main.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    from TwoActionLRT import TwoActionLRT
    from compare_twitter import encode_one,decode_one
    folder=Path(run_dir);folder.mkdir(parents=True,exist_ok=True)
    recorder=RunRecorder(folder);best=-float('inf')
    full={k.name:[k.kind,[k.lower,k.upper,k.default]] for k in space.knobs}
    selected=random.sample(list(full),min(20,len(full)))
    frozen={}; path=folder/'upstream_points.csv'
    with path.open('w') as f: csv.writer(f).writerow(['iteration','timestamp','TPS',*full])
    def csv_append(record):
        row=[record.trial,record.trial,math.exp(record.utility)]
        row += [int(record.config[n]) if isinstance(record.config[n],bool) else record.config[n] for n in full]
        with path.open('a') as f: csv.writer(f).writerow(row)
    class Normalizer:
        def __init__(self,mapping): self.knob_names=list(mapping)
        def normalize(self,config):
            return [encode_one(space.by_name[n],bool(config[n]) if space.by_name[n].kind=='bool' else config[n]) for n in self.knob_names]
        def denormalize(self,values):
            return {n:decode_one(space.by_name[n],v) for n,v in zip(self.knob_names,values)}
    upstream.Normalizer=Normalizer
    for trial,cfg in enumerate(initial):
        best,_=evaluate_and_record(evaluator,recorder,'dot_official',seed,trial,cfg,'shared_initial',best,budget=budget)
        csv_append(recorder.records[-1])
    def objective(norm_values,driver,normalizer,full_knob_dict,frozen_values,**kwargs):
        nonlocal best
        config=space.get_default()|frozen_values|normalizer.denormalize(norm_values)
        config=space.repair(config)
        best,result=evaluate_and_record(evaluator,recorder,'dot_official',seed,len(recorder.records),config,'official_dot_gp',best,budget=budget)
        return -math.exp(result.utility)
    class Callback:
        def __init__(self,**kwargs): pass
        def __call__(self,result): csv_append(recorder.records[-1]);return False
    upstream.objective_func=objective;upstream.LoggingCallback=Callback
    x0=[Normalizer({n:full[n] for n in selected}).normalize(r.config) for r in recorder.records]
    y0=[-math.exp(r.utility) for r in recorder.records]
    flags=(0,0,0,0,0,0,1,0,0);bandit=TwoActionLRT();updated=0
    while len(recorder.records)<budget:
        initial_pts,n_calls=upstream.decide_calls(x0,updated,len(selected),flags,budget-len(recorder.records))
        x,y=upstream.run_optimization_iteration(None,Normalizer({n:full[n] for n in selected}),full,frozen,selected,
            'common_metric',x0,y0,initial_pts,n_calls,seed,str(folder/'models'),'common',str(path))
        if len(recorder.records)<budget:
            selected,updated,x0,y0=upstream.feature_selection_cycle(x,y,selected,full,frozen,flags,None,True,str(path),bandit=bandit,n_calls=n_calls)
    summary=recorder.finalize()
    summary['adapter_notes']=['official GP/RFECV/LRT functions; PostgreSQL evaluator replaces MySQL/SSH',
        'shared IID observations supplied as initial history; official decide_calls controls epoch sizes',
        'common global normalized/log coordinates and rounded categories; positive exp(log-utility) score',
        'fixed-duration full workload; upstream adaptive/partial benchmarking disabled for equal measurement windows',
        'no-prior variant: seeded random initial knob order; adaptation decisions otherwise upstream']
    return summary
