import json
from pathlib import Path
import importlib.util
import sys
import pytest
from cognitune.evaluator.base import EvaluationResult
from cognitune.experiment.resilience import DurableEvaluator, InfrastructureUnavailable, ReplayDivergence


def result(value=10,status='SUCCESS',config=None):
    return EvaluationResult(value,0.,status,config=config or {'x':1})


class Backend:
    def __init__(self, outcomes, controls=None):
        self.outcomes=list(outcomes);self.controls=list(controls or [result()]*3);self.calls=0;self.recoveries=0
    def measure(self,config,token):
        self.calls+=1;value=self.outcomes.pop(0)
        if isinstance(value,Exception): raise value
        value.config=config
        return value
    def recover(self,token):
        self.recoveries+=1;return self.controls.pop(0)


def test_replay_never_reexecutes_and_checks_config(tmp_path):
    backend=Backend([result()]);first=DurableEvaluator(backend,tmp_path)
    assert first.evaluate({'x':1},0).status=='SUCCESS'
    resumed=DurableEvaluator(backend,tmp_path)
    resumed.evaluate({'x':1},0)
    assert backend.calls==1
    with pytest.raises(ReplayDivergence):resumed.evaluate({'x':2},0)
    assert backend.calls==1


def test_transient_failure_needs_real_default_control(tmp_path):
    backend=Backend([ConnectionError('disconnected'),result(12)])
    value=DurableEvaluator(backend,tmp_path).evaluate({'x':1},0)
    assert value.raw_metric==12 and backend.recoveries==1 and backend.calls==2
    events=[json.loads(x) for x in (tmp_path/'recovery_events.jsonl').read_text().splitlines()]
    assert [e['kind'] for e in events]==['candidate_attempt','default_control','candidate_attempt']


def test_reproduced_zero_not_promoted_to_success(tmp_path):
    backend=Backend([result(0),result(0)])
    value=DurableEvaluator(backend,tmp_path).evaluate({'x':1},0)
    assert value.status=='CRASH' and value.raw_metric==0
    assert backend.recoveries==2
    assert value.metadata['classification']=='reproduced_candidate_failure_with_healthy_default'


def test_bad_control_suspends_without_poisoning_optimizer(tmp_path):
    backend=Backend([result(0)],controls=[result(0)])
    with pytest.raises(InfrastructureUnavailable):DurableEvaluator(backend,tmp_path).evaluate({'x':1},0)
    assert not (tmp_path/'trial_0000/result.json').exists()
    assert (tmp_path/'trial_0000/request.json').exists()


def test_shared_initial_is_config_checked(tmp_path):
    backend=Backend([])
    value=DurableEvaluator(backend,tmp_path,initial={0:result().as_dict()}).evaluate({'x':1},0)
    assert value.metadata['shared_initial_reused'] and backend.calls==0


def test_nonfinite_utility_is_rejected(tmp_path):
    bad=result();bad.utility=float('nan')
    # Logging rejects NaN as a final ledger entry; malformed values must not train a GP.
    backend=Backend([bad,result()])
    assert DurableEvaluator(backend,tmp_path).evaluate({'x':1},0).status=='SUCCESS'
    assert backend.recoveries==1


def load_workloads():
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    import baseline_workloads
    return baseline_workloads


def test_hybrid_metrics_keep_transactional_and_analytical_windows_separate():
    module=load_workloads()
    summary={'Final State':'EXIT','Measured Goodput (requests/second)':110,
             'Measured Type Statistics':{'NewOrder':{'throughput_per_second':80},
                'Payment':{'throughput_per_second':20},'Q1':{'p95_latency_ms':30},
                '_analytical':{'p95_latency_ms':25,'count':10}}}
    raw,meta=module.metrics('chbenchmark',summary)
    assert raw==100 and meta['analytical_p95_ms']==25


def test_missing_analytical_successes_is_not_valid_htap():
    module=load_workloads()
    with pytest.raises(ValueError):module.metrics('chbenchmark',{'Final State':'EXIT',
        'Measured Goodput (requests/second)':10,'Measured Type Statistics':{'NewOrder':{'throughput_per_second':10}}})


def test_ours_is_not_a_baseline_queue_choice():
    load_workloads()
    from baseline_methods import METHODS
    assert len(METHODS)==7
    assert not any('cognitune' in m or 'jointtune' in m for m in METHODS)


def test_prepare_recovers_a_disconnected_previous_worker(tmp_path,monkeypatch):
    import subprocess
    module=load_workloads()
    monkeypatch.setattr(module.h,'RUNTIME',tmp_path)
    (tmp_path/'baselines-ycsb-dataset.json').write_text(json.dumps({'spec':module.SPECS['ycsb']}))
    calls=[]
    def sql(statement):
        calls.append(statement)
        if len(calls)==1:raise subprocess.CalledProcessError(2,'psql')
        return '1'
    monkeypatch.setattr(module.h,'sql',sql)
    backend=module.WorkloadBackend('ycsb',object(),tmp_path/'out',tmp_path/'jar')
    resets=[]
    monkeypatch.setattr(backend,'reset',lambda:resets.append(True))
    backend.prepare()
    assert len(calls)==2 and resets==[True]


def test_htap_stability_checks_latency_even_when_tps_is_stable():
    module=load_workloads()
    records=[{'status':'SUCCESS','raw_metric':100,'metadata':{'analytical_p95_ms':x}} for x in [10,30,90]]
    variation=module.control_variation(records,'chbenchmark')
    assert variation['raw_metric']==0 and variation['analytical_p95_ms']>.15
    assert not module.acceptable_drift(records,{'raw_metric':100,'analytical_p95_ms':10},'chbenchmark')
    assert module.acceptable_drift(records,{'raw_metric':100},'ycsb')
