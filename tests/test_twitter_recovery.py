from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import real_benchbase as h
from cognitune.evaluator.base import EvaluationResult


def stop_failure(monkeypatch,alive,times,stderr='pg_ctl: server does not shut down\n'):
    calls=[]
    def command(*args,**kwargs):
        calls.append(args[0])
        return SimpleNamespace(returncode=1,stderr=stderr)
    monkeypatch.setattr(h.subprocess,'run',command)
    monkeypatch.setattr(h,'_postmaster_pid',lambda:12345)
    monkeypatch.setattr(h,'_pid_alive',alive)
    tick=iter(times)
    monkeypatch.setattr(h.time,'monotonic',lambda:next(tick))
    monkeypatch.setattr(h.time,'sleep',lambda seconds:None)
    monkeypatch.setenv('COGNITUNE_PG_STOP_TIMEOUT','600')
    return calls


def test_slow_normal_shutdown_waits_for_original_process_without_reissuing_stop(monkeypatch,capsys):
    alive=iter([True,False])
    calls=stop_failure(monkeypatch,lambda pid:next(alive),[0,91,93])
    h.pgctl('stop')
    assert len(calls)==1
    assert 'shutdown confirmed after 93.0s' in capsys.readouterr().out


def test_shutdown_deadline_does_not_attempt_start_or_force_kill(monkeypatch):
    calls=stop_failure(monkeypatch,lambda pid:True,[0,601])
    with pytest.raises(h.DatabaseLifecycleError,match='Normal shutdown still pending'):
        h.pgctl('stop')
    assert len(calls)==1
    assert calls[0][-1]=='stop'


def test_unrelated_pgctl_failure_is_not_retried(monkeypatch):
    calls=stop_failure(monkeypatch,lambda pid:True,[0],stderr='permission denied')
    with pytest.raises(h.DatabaseLifecycleError,match='permission denied'):
        h.pgctl('stop')
    assert len(calls)==1


def success(config):
    return EvaluationResult(174.6,5.16,'SUCCESS',config=config,metadata={
        'warmup_seconds':15,'measurement_seconds':60,
        'metric':h.METRIC,'workload':h.WORKLOAD,'template_reset':True}).as_dict()


def test_preflight_resumes_success_archives_failed_attempt_and_measures_missing_trials(tmp_path):
    config={'shared_buffers':16384}
    for trial,record in [(0,success(config)),(1,{'status':'CRASH','metadata':{'error':'shutdown timeout'}})]:
        folder=tmp_path/f'trial_{trial:04d}'
        folder.mkdir()
        (folder/'result.json').write_text(json.dumps(record))
    original=(tmp_path/'trial_0000/result.json').read_bytes()
    seen=[]
    def evaluate(cfg,trial):
        seen.append(trial)
        return EvaluationResult(**success(cfg))
    evaluator=SimpleNamespace(output=tmp_path,warmup=15,duration=60,evaluate=evaluate)
    rows=h.collect_default_measurements(evaluator,config)
    assert len(rows)==3 and seen==[1,2]
    assert (tmp_path/'trial_0000/result.json').read_bytes()==original
    archives=list(tmp_path.glob('trial_0001.failed-attempt-*'))
    assert len(archives)==1
    assert json.loads((archives[0]/'result.json').read_text())['status']=='CRASH'


def test_preflight_refuses_incompatible_success_instead_of_reusing_or_overwriting(tmp_path):
    folder=tmp_path/'trial_0000'
    folder.mkdir()
    path=folder/'result.json'
    path.write_text(json.dumps(success({'shared_buffers':100})))
    evaluator=SimpleNamespace(output=tmp_path,warmup=15,duration=60)
    with pytest.raises(ValueError,match='incompatible configuration/protocol'):
        h.collect_default_measurements(evaluator,{'shared_buffers':16384})
    assert path.exists()


def test_original_error_and_recovery_error_both_persist(tmp_path,monkeypatch):
    calls=[]
    def fail(action):
        calls.append(action)
        if len(calls)==1:
            raise RuntimeError('original operation failed')
        raise h.DatabaseLifecycleError('recovery storage failure')
    monkeypatch.setattr(h,'pgctl',fail)
    evaluator=h.LocalBenchBaseEvaluator(SimpleNamespace(get_default=lambda:{}),tmp_path,1)
    with pytest.raises(h.DatabaseLifecycleError,match='original measurement error saved'):
        evaluator.evaluate({},0)
    record=json.loads((tmp_path/'trial_0000/result.json').read_text())
    assert record['metadata']['error']=='original operation failed'
    assert record['metadata']['recovery_error']=='recovery storage failure'


def test_template_timeout_is_infrastructure_failure_without_shutdown_retry(tmp_path,monkeypatch):
    lifecycle=[]
    monkeypatch.setattr(h,'pgctl',lambda action:lifecycle.append(action))
    monkeypatch.setattr(h,'configure',lambda config:None)
    monkeypatch.setattr(h,'sql',lambda statement,**kwargs: json.dumps({'fsync':'on','synchronous_commit':'on','full_page_writes':'on'}) if 'json_object_agg' in statement else '')
    def stalled(*args):
        raise h.DatabaseLifecycleError('template copy storage timeout')
    monkeypatch.setattr(h,'maintenance_sql',stalled)
    evaluator=h.LocalBenchBaseEvaluator(SimpleNamespace(get_default=lambda:{}),tmp_path,1)
    with pytest.raises(h.DatabaseLifecycleError,match='storage timeout'):
        evaluator.evaluate({},0)
    evaluator.restore_default()
    assert lifecycle==['stop','start']
    record=json.loads((tmp_path/'trial_0000/result.json').read_text())
    assert record['status']=='INFRA_ERROR'
    assert record['metadata']['phase']=='template_restore'


def test_maintenance_sql_failure_is_not_retried_or_reclassified_as_optimizer_crash(monkeypatch):
    class Failed:
        returncode=1
        calls=0
        def communicate(self,timeout):
            self.calls+=1
            return ('','copy I/O failure')
        def poll(self):
            return 1
    process=Failed()
    monkeypatch.setattr(h,'private_env',lambda:{})
    monkeypatch.setattr(h.subprocess,'Popen',lambda *args,**kwargs:process)
    with pytest.raises(h.DatabaseLifecycleError,match='template_copy failed: copy I/O failure'):
        h.maintenance_sql('CREATE DATABASE cognitune_tpcc TEMPLATE cognitune_tpcc_template','template_copy',900)
    assert process.calls==1
