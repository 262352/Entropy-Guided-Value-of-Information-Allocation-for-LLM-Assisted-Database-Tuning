"""Paired Twitter comparison for decision-entropy CogniTune v3; no legacy baseline or synthetic results."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import random
import statistics
import shutil
import sys
import time
import textwrap

import numpy as np
import torch

from real_benchbase import ROOT, RUNTIME, LocalBenchBaseEvaluator, private_env, sql, collect_default_measurements
from cognitune.config import load_search_space
from cognitune.experiment.recorder import RunRecorder
from cognitune.llm import DeepSeekLLMClient
from cognitune.llm.json_request import request_json_object
from cognitune.optimizers.cognitune_v3 import run_cognitune_v3
from cognitune.adaptive.proposals import FeedbackExpertProposer
from cognitune.adaptive.ablation import Ablation, VARIANTS
from cognitune.adaptive.prompt_context import HISTORY_POLICY, PROMPT_LAYOUT
from build_measured_benchbase import build as build_benchbase
import real_benchbase as harness
from cognitune.optimizers.common import evaluate_and_record
from cognitune.search_space.initial_design import build_initial_design, load_initial_design, save_initial_design

OUT = ROOT/'artifacts/twitter_pair_50_v3'
SPACE_PATH = ROOT/'configs/knobs/postgresql_60_diagnostic.yaml'
OFFICIAL = RUNTIME/'MCTuner-official'
MODEL = 'deepseek-flash'
BUDGET = 50
N_INIT = 10
SEED = 0
METHODS = ['cognitune_v3','mctuner_official']
ALLOW_UNSTABLE = False
VERBOSE = False
RESUME = False
SEARCH_POLICY = 'legacy'
ABLATION = 'full'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def profile(space):
    return {
        'dbms': 'PostgreSQL 14.9', 'workload': 'BenchBase Twitter', 'scalefactor': 10, 'users': 5000, 'tweets': 200000,
        'terminals': 16, 'warmup_seconds': 15, 'measurement_seconds': 60,
        'transaction_mix': {'GetTweet':1, 'GetTweetsFromFollowing':1, 'GetFollowers':7, 'GetUserTweets':90, 'InsertTweet':1},
        'objective': 'maximize successful operations wholly within the measurement window per second',
        'hardware': {'logical_cpus': os.cpu_count(), 'available_cpus': len(os.sched_getaffinity(0)),
                     'mem_total_kb': int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:'))),
                     'storage': 'persistent ext4-backed Docker overlay, fsync enabled'},
        'knob_count':len(space.knobs),
        'policy': 'No changing fsync/synchronous_commit/full_page_writes; one database template per trial; no outcome hints from another method.',
    }


def proposer():
    transport = DeepSeekLLMClient(api_key=private_env()['DEEPSEEK_API_KEY'], model=MODEL)
    return FeedbackExpertProposer(transport, OUT/'llm_cache/cognitune_v3')


def official_selection(space, context=None, goal=None):
    destination=OUT/'priors/mctuner_official/selection.json'
    if destination.exists():
        return json.loads(destination.read_text())
    sys.path.insert(0,str(OFFICIAL))
    from knob_selection import constants, knowledge
    from knob_selection.gpt import GPT
    import knob_selection.knob_selection as upstream_selection
    from knob_selection.knob_selection import KnowledgePreprocessor
    base=OFFICIAL/'knob_selection/knowledge_collection/postgres'
    constants.KNOWLEDGE_SOURCES_PATH=str(base/'knowledge_sources')
    constants.OFFICIAL_DOC_PATH=str(base/'knob_info/official_document.json')
    constants.SYSTEM_VIEW_PATH=str(base/'knob_info/system_view.json')
    upstream_selection.KNOB_INFO_PATH=str(base/'knob_info')
    upstream_selection.KNOWLEDGE_SOURCES_PATH=str(base/'knowledge_sources')
    original_doc=knowledge.get_official_doc_description
    def bounded_doc(name,gpt):
        k=space.by_name[name]
        domain={'type':k.kind,'unit':k.unit,'default':k.default,'lower':k.lower,'upper':k.upper,'choices':list(k.choices)}
        return original_doc(name,gpt)+'\nAUTHORITATIVE HOST DOMAIN: '+json.dumps(domain)+'. All numeric range endpoints must be JSON numbers expressed in the stated unit, inside these bounds. For boolean/categorical knobs use legal choice values.'
    knowledge.get_official_doc_description=bounded_doc
    KnowledgePreprocessor._get_official_doc_description=lambda self,name: bounded_doc(name,self)
    KnowledgePreprocessor._get_knob_metadata=lambda self,name: (space.by_name[name].unit,space.by_name[name].kind)
    # Upstream formats an already interpolated f-string twice, interpreting
    # literal JSON/context braces as format fields. Replace only its three
    # intended placeholders; the actual expert wording is unchanged.
    source=textwrap.dedent(inspect.getsource(KnowledgePreprocessor._create_expert_prompts))
    source=source.replace('base_prompt.format(expert_name=f"\'{name}\'", expert_domain=domain, expert_impact=impact)',
        'base_prompt.replace("{expert_name}", repr(name)).replace("{expert_domain}", domain).replace("{expert_impact}", impact)')
    namespace=dict(upstream_selection.__dict__)
    exec(compile(source,'<upstream-prompt-brace-fix>','exec'),namespace)
    KnowledgePreprocessor._create_expert_prompts=namespace['_create_expert_prompts']
    # Transport-only replacement: keep the upstream router/expert prompts and ranking.
    transport=DeepSeekLLMClient(api_key=private_env()['DEEPSEEK_API_KEY'],model=MODEL,max_tokens=4096)
    cache=OUT/'llm_cache/mctuner_official'
    cache.mkdir(parents=True,exist_ok=True)
    def request(gpt,prompt,max_retries=3):
        return request_json_object(transport, prompt, cache, max_retries=max_retries)
    GPT.get_gpt_response_json=request
    ctx=context if context is not None else profile(space)
    processor=KnowledgePreprocessor(str(destination.parent),
        goal or ('Maximize committed-transaction throughput only. Throughput weight=1, latency weight=0. '
        'Respect durability and the provided host-specific numeric units/bounds.'),
        json.dumps(ctx), 'https://api.deepseek.com', private_env()['DEEPSEEK_API_KEY'],model=MODEL)
    infos={}
    for index,k in enumerate(space.knobs):
        item=destination.parent/(k.name+'_final.json')
        if item.exists():
            info=json.loads(item.read_text())
        else:
            info=processor.process_knob(k.name)
        if not isinstance(info.get('score'),(int,float)):
            raise ValueError('Invalid official expert score: '+k.name)
        infos[k.name]=info
        print(f'Official MCTuner selection {index+1}/60: {k.name}',flush=True)
    selected=sorted(infos,key=lambda n:(-infos[n]['score'],n))[:20]
    selection={'selected_knobs':selected,'knob_info':infos,'model':MODEL,
               'upstream_commit':os.popen('git -C '+str(OFFICIAL)+' rev-parse HEAD').read().strip()}
    dump(destination,selection)
    return selection


def encode_one(k,value):
    if k.kind in {'integer','float'}:
        if k.log:
            return (math.log(float(value))-math.log(k.lower))/(math.log(k.upper)-math.log(k.lower))
        return (float(value)-k.lower)/(k.upper-k.lower)
    choices=[False,True] if k.kind=='bool' else list(k.choices)
    return choices.index(value)/max(1,len(choices)-1)


def decode_one(k,value):
    value=float(np.clip(value,0,1))
    if k.kind in {'integer','float'}:
        result=math.exp(math.log(k.lower)+value*(math.log(k.upper)-math.log(k.lower))) if k.log else k.lower+value*(k.upper-k.lower)
        return int(round(result)) if k.kind=='integer' else result
    choices=[False,True] if k.kind=='bool' else list(k.choices)
    return choices[int(round(value*(len(choices)-1)))]


def run_official(space,evaluator,initial,selection,run_dir):
    sys.path.insert(0,str(OFFICIAL))
    from lamcts.MCTS import MCTuner
    class SharedStart(MCTuner):
        def _initial_sampling(self,n_inits):
            # The shared persisted IID configurations are evaluated below.
            pass
    chosen=selection['selected_knobs']
    knobs=[space.by_name[n] for n in chosen]
    lows,highs=[],[]
    for k in knobs:
        info=selection['knob_info'][k.name]
        if k.kind in {'integer','float'}:
            lo,hi=info.get('knob_lb'),info.get('knob_ub')
            if isinstance(lo,bool) or isinstance(hi,bool) or not isinstance(lo,(int,float)) or not isinstance(hi,(int,float)):
                raise ValueError('Invalid numeric range: '+k.name+': '+str(info))
            lo,hi=max(k.lower,lo),min(k.upper,hi)
            if lo>=hi:
                raise ValueError('Degenerate pruned range: '+k.name)
            lows.append(encode_one(k,lo)); highs.append(encode_one(k,hi))
        else:
            lows.append(0.0); highs.append(1.0)
    recorder=RunRecorder(run_dir)
    agent=SharedStart(np.array(lows),np.array(highs),len(knobs),0,lambda x:None,
                      Cp=1.0,leaf_size=10,kernel_type='rbf',gamma_type='auto')
    best=-float('inf')
    diagnostics=[]
    for trial,cfg in enumerate(initial):
        best,result=evaluate_and_record(evaluator,recorder,'mctuner_official',SEED,trial,cfg,'shared_initial',best,budget=BUDGET)
        agent.samples.append((np.array([encode_one(k,cfg[k.name]) for k in knobs]),result.utility))
    while len(recorder.records)<BUDGET:
        trial=len(recorder.records)
        # Upstream tree splitting, UCT traversal and GP/EI proposal are unchanged.
        # Its classifier expects MAXIMIZATION; the upstream outer minimization
        # bookkeeping is replaced by this common maximization recorder.
        agent._rebuild_tree()
        leaf,path=agent.select_leaf()
        candidates=leaf.classifier.propose_samples_bo(1,path,agent.lb,agent.ub,agent.samples)
        point=candidates[0]
        incumbent=max(recorder.records,key=lambda r:r.utility).config
        cfg=dict(incumbent)
        cfg.update({k.name:decode_one(k,v) for k,v in zip(knobs,point)})
        cfg=space.repair(cfg)
        vector=np.array([encode_one(k,cfg[k.name]) for k in knobs])
        best,result=evaluate_and_record(evaluator,recorder,'mctuner_official',SEED,trial,cfg,'official_mcts_gp_ei',best,budget=BUDGET)
        agent.samples.append((vector,result.utility))
        agent.backpropagate(leaf,result.utility)
        diagnostics.append({'trial':trial,'nodes':len(agent.nodes),'depth':len(path),'selected_knobs':chosen,'source':'official_mcts_gp_ei','utility':result.utility})
        (Path(run_dir)/'round_diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in diagnostics))
    summary=recorder.finalize()
    summary['selected_knobs']=chosen
    summary['adapter_notes']=['shared full-space IID initial observations; 20 selected knobs thereafter',
        'coordinates normalized by shared global domains (log transform for log knobs)',
        'official outer minimization bookkeeping bypassed; all rewards use common maximization utility',
        'categorical knobs encoded as rounded ordinal indices; no categorical range pruning',
        'upstream tree/classifier/GP/EI modules imported without source modifications']
    dump(Path(run_dir)/'summary.json',summary)
    return summary


class GuardedEvaluator(LocalBenchBaseEvaluator):
    def evaluate(self,config,trial):
        result=super().evaluate(config,trial)
        if trial<12 and result.status!='SUCCESS':
            raise RuntimeError('12-trial health gate failed; stopping before full diagnostic: '+str(result.metadata))
        return result


def finish_default_check(space,method,default,default_metric):
    """Complete a missing post-check on resume without repeating tuning."""
    path=OUT/('default_after_'+method+'.json')
    if path.exists():
        drift=json.loads(path.read_text())
    else:
        check=LocalBenchBaseEvaluator(space,OUT/'measurements'/('default_after_'+method),default_metric)
        drift=[check.evaluate(default,i).as_dict() for i in range(3)]
        dump(path,drift)
    if len(drift)!=3 or not all(r['status']=='SUCCESS' for r in drift):
        raise RuntimeError('Default health check failed after '+method)
    current=statistics.median(r['raw_metric'] for r in drift)
    print(f'[DRIFT] method={method} default_values={[r["raw_metric"] for r in drift]} median_ratio={current/default_metric:.4f}',flush=True)
    if abs(current/default_metric-1)>0.20 and not ALLOW_UNSTABLE:
        raise RuntimeError('Baseline drift exceeds 20%; comparison must not be interpreted')
    if abs(current/default_metric-1)>0.20:
        print('[EXPLORATORY WARNING] Default drift exceeded 20%; retained for diagnosis only.',flush=True)


def run():
    lock=(RUNTIME/'real-comparison.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    space=load_search_space(SPACE_PATH)
    initial_path=OUT/'initial_design/postgresql_twitter_seed_0.json'
    if not initial_path.exists():
        save_initial_design(initial_path,build_initial_design(space,SEED,N_INIT))
    initial=load_initial_design(initial_path,space,N_INIT,SEED)
    dump(OUT/'profile.json',profile(space))
    protocol={'budget':BUDGET,'n_init':N_INIT,'seed':SEED,'model':MODEL,
        'expert_history_policy':HISTORY_POLICY,'expert_prompt_layout':PROMPT_LAYOUT,
        'methods':['cognitune_v3','mctuner_official'],'candidate_pool_size':100, 'expert_audit_size':4, 'eta':0.8,
        'warmup_seconds':15,'measurement_seconds':60,'metric':harness.METRIC, 'metric_patch':json.loads((RUNTIME/'benchbase-measured/provenance.json').read_text()), 'dataset':json.loads((RUNTIME/'twitter-template.json').read_text()),
        'mode':'single-seed real diagnostic, not full paper reproduction',
        'final_validation_repeats':3,'baseline_drift_checks':3,
        'default_preflight_max_cv':0.15,'baseline_drift_stop_fraction':0.20,
        'allow_unstable_baseline':ALLOW_UNSTABLE,
        'database_shutdown_timeout_seconds':int(os.environ.get('COGNITUNE_PG_STOP_TIMEOUT','600')),
        'template_restore_timeout_seconds':int(os.environ.get('COGNITUNE_PG_TEMPLATE_TIMEOUT','900')),
        'verbose_diagnostics':VERBOSE,
        'protocol_override':f'User selected a new {BUDGET}-trial exploratory diagnostic. Both methods get 3 final validations; this independent pair is not merged with legacy baselines.',
        'baseline_adapter':'Official prompts, ranking, spatial decomposition, UCT and GP/EI imported. Prompt brace formatting, hardcoded paths and minimization/maximization interface inconsistencies repaired in adapter. Shared normalized coordinates and ordinal categorical mapping are explicit deviations.',
        'initial_design_sha256':hashlib.sha256(initial_path.read_bytes()).hexdigest()}
    if SEARCH_POLICY=='stable':
        protocol['search_policy']='stable-v1'
        protocol['structure_episode']=6
        protocol['minimum_executed_tests']=2
        protocol['max_structure_swaps']=2
        protocol['coverage_interval']=3
        protocol['effective_active_target']=min(18,max(6,(BUDGET-N_INIT)//4))
        protocol['final_polish_evaluations']=6
        protocol['allocation_unit']='parameter-group search hypothesis'
    if ABLATION!='full':
        protocol['ablation']=Ablation(ABLATION).describe()
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        if previous.get('search_policy','legacy') != protocol.get('search_policy','legacy'):
            raise RuntimeError('Search policy differs; use a separate artifact directory')
        for key in ['expert_history_policy','expert_prompt_layout','ablation','budget','n_init','seed','model','methods','candidate_pool_size','expert_audit_size','eta','metric','metric_patch','dataset','initial_design_sha256','structure_episode','minimum_executed_tests','max_structure_swaps','coverage_interval','effective_active_target','final_polish_evaluations']:
            if previous.get(key) != protocol.get(key):
                raise RuntimeError('Existing run has incompatible protocol: '+key+'; use a new artifact directory')
    dump(protocol_path,protocol)
    code_paths=['src/cognitune/optimizers/cognitune_v3.py','src/cognitune/adaptive/stable_search.py',
                'src/cognitune/adaptive/proposals.py','src/cognitune/adaptive/ablation.py',
                'src/cognitune/adaptive/decision.py','src/cognitune/adaptive/prompt_context.py','src/cognitune/surrogate/gp.py',
                'configs/knobs/postgresql_60_diagnostic.yaml','scripts/compare_twitter.py','scripts/real_benchbase.py']
    with (OUT/'implementation_sessions.jsonl').open('a') as stream:
        stream.write(json.dumps({'unix_time':time.time(),'ablation':Ablation(ABLATION).describe(),
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code_paths}},sort_keys=True)+'\n')
    # MCTuner selection is workload-specific; ours queries experts each round.
    selection=official_selection(space) if 'mctuner_official' in METHODS else None
    default=space.get_default()
    baseline_path=OUT/'default_before.json'
    if baseline_path.exists():
        defaults=json.loads(baseline_path.read_text())
    else:
        evaluator=GuardedEvaluator(space,OUT/'measurements/default_before',1.0)
        defaults=collect_default_measurements(evaluator,default,3)
        dump(baseline_path,defaults)
    default_metric=statistics.median(r['raw_metric'] for r in defaults)
    if not all(r['status']=='SUCCESS' for r in defaults):
        raise RuntimeError('Default baseline failed')
    cv=statistics.stdev(r['raw_metric'] for r in defaults)/statistics.mean(r['raw_metric'] for r in defaults)
    if cv>0.15 and not ALLOW_UNSTABLE:
        raise RuntimeError('Default baseline unstable (CV>15%); inspect I/O before tuning')
    print(f'[BASELINE] values={[r["raw_metric"] for r in defaults]} median={default_metric:.3f} CV={cv:.3%}',flush=True)
    if cv>0.15:
        print('[EXPLORATORY WARNING] Continuing with unstable baseline at explicit user request; performance differences are not reliable causal evidence.',flush=True)
    for method in METHODS:
        run_dir=OUT/'runs'/method/'seed_0'
        if (run_dir/'final_validation.json').exists():
            finish_default_check(space,method,default,default_metric)
            continue
        if (run_dir/'trials.jsonl').exists() and not (RESUME and method=='cognitune_v3'):
            raise RuntimeError('Partial method exists: inspect rather than silently rerun '+method)
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        evaluator=GuardedEvaluator(space,OUT/'measurements'/method,default_metric)
        started=time.monotonic()
        try:
            if method=='cognitune_v3':
                summary=run_cognitune_v3(space,evaluator,initial,BUDGET,SEED,100,run_dir,
                    proposer=proposer(),cognitive_context=profile(space),resume=RESUME,search_policy=SEARCH_POLICY,
                    ablation=ABLATION)
            else:
                summary=run_official(space,evaluator,initial,selection,run_dir)
            summary['total_seconds']=time.monotonic()-started
            summary['total_seconds_scope']='current_resume_session' if summary.get('preserved_observations') else 'entire_method_run'
            dump(run_dir/'summary.json',summary)
            validations=[evaluator.evaluate(summary['best_config'],BUDGET+i).as_dict() for i in range(3)]
            dump(run_dir/'final_validation.json',{'outside_tuning_budget':True,'results':validations})
        finally:
            evaluator.restore_default()
        finish_default_check(space,method,default,default_metric)
    print('Requested diagnostic method(s) complete',flush=True)



def prepare_dataset():
    """Load once; each trial clones this immutable workload template."""
    build_benchbase()
    traces = RUNTIME/'twitter-traces'
    traces.mkdir(exist_ok=True)
    records = {}
    # BenchBase's stock trace IDs exceed its generated table IDs. Deterministic
    # modulo remapping gives valid lookups; retain both hashes and exact rule.
    for source_name, output_name, bound in [
        ('twitter_tweetids.txt', 'tweetids.txt', 200000),
        ('twitter_user_ids.txt', 'userids.txt', 5000),
    ]:
        source = harness.BENCH/'data/twitter'/source_name
        data = source.read_bytes()
        mapped = ''.join(str((int(line)-1) % bound + 1)+'\n' for line in data.decode().splitlines())
        path = traces/output_name
        if path.exists() and path.read_text() != mapped:
            raise RuntimeError('Existing Twitter trace differs; use a new experiment location')
        path.write_text(mapped)
        records[source_name] = {'upstream_sha256':hashlib.sha256(data).hexdigest(),
            'mapped_sha256':hashlib.sha256(mapped.encode()).hexdigest(), 'rows':len(mapped.splitlines()),
            'rule':f'(id-1) % {bound} + 1'}
    stamp = RUNTIME/'twitter-template.json'
    exists = sql("SELECT count(*) FROM pg_database WHERE datname='cognitune_twitter_template'").strip() == '1'
    if exists:
        if not stamp.exists():
            raise RuntimeError('Template without provenance exists; inspect it instead of overwriting')
        previous=json.loads(stamp.read_text())
        if previous['traces'] != records:
            raise RuntimeError('Template trace provenance mismatch')
        print('[DATA] Reusing prepared Twitter template', flush=True)
        return
    if stamp.exists():
        raise RuntimeError('Twitter template missing but provenance exists; inspect before reloading')
    sql('CREATE DATABASE cognitune_twitter_template')
    xml = RUNTIME/'load-twitter.xml'
    harness.xml_config('cognitune_twitter_template', xml)
    harness.benchbase(xml, RUNTIME/'twitter-load', load=True)
    sql('ANALYZE', database='cognitune_twitter_template', timeout=600)
    sql('CHECKPOINT', timeout=600)
    counts = {table:int(sql(f'SELECT count(*) FROM {table}', database='cognitune_twitter_template'))
              for table in ['user_profiles','tweets','follows','followers']}
    if counts['user_profiles'] != 5000 or counts['tweets'] != 200000:
        raise RuntimeError('Twitter load counts differ from the declared scale')
    dump(stamp, {'workload':'BenchBase generated Twitter (not a raw Twitter archive)',
        'scale':10, 'table_counts':counts, 'traces':records,
        'benchbase_commit':json.loads((RUNTIME/'benchbase-measured/provenance.json').read_text())['upstream_commit']})
    print('[DATA] Twitter template ready: '+str(counts), flush=True)


def check():
    missing = [str(p) for p in [SPACE_PATH, RUNTIME/'cognitune.env',
        harness.PG/'bin/pg_ctl', harness.BENCH/'benchbase.jar',
        OFFICIAL/'lamcts/MCTS.py'] if not p.exists()]
    if missing:
        raise RuntimeError('Missing runtime prerequisites: '+', '.join(missing))
    build_benchbase()
    durability=json.loads(sql("SELECT json_object_agg(name,setting) FROM pg_settings WHERE name IN ('fsync','synchronous_commit','full_page_writes')"))
    if any(value!='on' for value in durability.values()):
        raise RuntimeError('Durability check failed: '+str(durability))
    print('[CHECK] PostgreSQL: '+sql('SELECT version()').splitlines()[0], flush=True)
    print('[CHECK] 60-knob space: '+load_search_space(SPACE_PATH).fingerprint(), flush=True)
    print('[CHECK] No database load, tuning run, or LLM request was performed', flush=True)


def reuse_preparation(previous):
    previous=Path(previous).resolve()
    if previous==OUT:
        raise ValueError('reuse source must differ from the new artifact directory')
    old=json.loads((previous/'protocol.json').read_text())
    if old['model']!=MODEL or old['metric']!=harness.METRIC or old['n_init']!=N_INIT or old['seed']!=SEED:
        raise ValueError('source preparation has incompatible model, metric or initialization')
    space=load_search_space(SPACE_PATH)
    design=previous/'initial_design/postgresql_twitter_seed_0.json'
    load_initial_design(design,space,N_INIT,SEED)
    if json.loads((previous/'profile.json').read_text())['workload'] != profile(space)['workload']:
        raise ValueError('cannot reuse recommendations from another workload')
    for relative in ['priors/mctuner_official','llm_cache/mctuner_official']:
        source=previous/relative;destination=OUT/relative
        if source.is_dir() and not destination.exists():
            shutil.copytree(source,destination)
    destination=OUT/'initial_design/postgresql_twitter_seed_0.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and destination.read_bytes()!=design.read_bytes():
        raise ValueError('existing initial design differs')
    if not destination.exists(): shutil.copy2(design,destination)
    print('[PREPARATION] Reused official selection/cache and exact shared IID configs; database measurements are independent of the source run',flush=True)
    dump(OUT/'reused_preparation.json',{'source':str(previous),'initial_design_sha256':hashlib.sha256(design.read_bytes()).hexdigest(),
        'reused':['official workload-specific MCTuner selection and request caches','shared initial configuration design'],
        'fresh':['database measurements','default performance baseline','CogniTune feedback requests','search controller state']})


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    os.environ.update(private_env())
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','prepare','run'])
    parser.add_argument('--budget',type=int,default=50)
    parser.add_argument('--method',choices=['both','cognitune_v3','mctuner_official'],default='both')
    parser.add_argument('--artifacts',type=Path,default=OUT)
    parser.add_argument('--allow-unstable-baseline',action='store_true')
    parser.add_argument('--search-policy',choices=['legacy','stable'],default='legacy')
    parser.add_argument('--ablation',choices=VARIANTS,default='full')
    parser.add_argument('--reuse-preparation-from',type=Path)
    parser.add_argument('--resume',action='store_true',help='resume CogniTune from saved observations/controller state')
    args=parser.parse_args()
    if args.ablation!='full' and (args.search_policy!='stable' or args.method!='cognitune_v3'):
        parser.error('ablations require --search-policy stable --method cognitune_v3')
    if args.budget<N_INIT:
        parser.error('budget must include all 10 shared IID initial configurations')
    OUT=args.artifacts.resolve()
    BUDGET=args.budget
    METHODS=['cognitune_v3','mctuner_official'] if args.method=='both' else [args.method]
    ALLOW_UNSTABLE=args.allow_unstable_baseline
    RESUME=args.resume
    SEARCH_POLICY=args.search_policy
    ABLATION=args.ablation
    VERBOSE=True
    OUT.mkdir(parents=True,exist_ok=True)
    if args.reuse_preparation_from:
        reuse_preparation(args.reuse_preparation_from)
    # Dataset loading and real experiments share the same instance lock as
    # the earlier TPC-C runner. Never load alongside another tuning process.
    if args.action=='check':
        check()
    elif args.action=='prepare':
        lock=(RUNTIME/'real-comparison.lock').open('w')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare_dataset()
    else:
        if not (RUNTIME/'twitter-template.json').exists():
            raise RuntimeError('Run prepare first (or use run_twitter_compare.sh)')
        build_benchbase()
        run()
