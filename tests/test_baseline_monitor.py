import importlib.util
import json
import os
from pathlib import Path
import time
import pytest

spec=importlib.util.spec_from_file_location('baseline_monitor',Path(__file__).resolve().parents[1]/'scripts/watch_baseline_queue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


@pytest.fixture
def queue(tmp_path):
    save(tmp_path/'protocol.json',{'workloads':['ycsb'],'methods':['smac'],'budget':100,'n_init':10})
    save(tmp_path/'supervisor.json',{'pid':11,'started':time.time()})
    save(tmp_path/'heartbeat.json',{'pid':22,'workload':'ycsb','method':'smac','time':time.time()})
    save(tmp_path/'ycsb/smac/status.json',{'state':'RUNNING','pid':22})
    return tmp_path


def test_healthy_check_has_progress_and_no_llm(queue):
    value=m.inspect_queue(queue,alive=lambda pid:pid in [11,22])
    assert value['state']=='NORMAL'
    assert value['completed']==0 and value['total']==1


def test_completed_queue_stops_monitoring(queue):
    save(queue/'ycsb/smac/summary.json',{'validated':True,'trials':100,'comparison_valid':False})
    value=m.inspect_queue(queue,alive=lambda pid:False)
    assert value['state']=='FINISHED' and value['drift_flags']==['ycsb/smac']


def test_dead_supervisor_and_deferred_jobs_need_repair(queue):
    save(queue/'ycsb/smac/status.json',{'state':'DEFERRED','returncode':1})
    value=m.inspect_queue(queue,alive=lambda pid:False)
    assert value['reason']=='queue_failed'


def test_repeated_failed_trials_are_abnormal_even_with_live_process(queue):
    for i in range(3):save(queue/f'ycsb/smac/ledger/trial_{i:04d}/result.json',{'status':'CRASH','raw_metric':0})
    assert m.inspect_queue(queue,alive=lambda pid:True)['reason']=='three_consecutive_failed_trials'


def test_stale_heartbeat_is_not_treated_as_healthy(queue):
    value=m.inspect_queue(queue,now=time.time()+200,alive=lambda pid:True)
    assert value['reason']=='supervisor_heartbeat_stale'


def test_live_orphan_is_not_killed(queue):
    value=m.inspect_queue(queue,alive=lambda pid:pid==22)
    assert value['state']=='WAITING'


def test_stalled_orphan_does_not_wait_forever(queue):
    value=m.inspect_queue(queue,now=time.time()+5000,alive=lambda pid:pid==22)
    assert value['reason']=='orphan_worker_stalled'


def test_normal_path_never_calls_repair_or_restart(queue,tmp_path,monkeypatch):
    monitor=m.Monitor(queue,tmp_path/'state')
    monkeypatch.setattr(m,'inspect_queue',lambda output:{'state':'NORMAL','reason':'running'})
    def forbidden(*args):raise AssertionError('healthy check called model/restart')
    monkeypatch.setattr(monitor,'repair',forbidden);monkeypatch.setattr(monitor,'start_queue',forbidden)
    began=time.time();event=monitor.check()
    assert event['model_invocations']==0 and event['action']=='none'
    assert 17999<event['next_check_at']-began<18002


def test_code_repair_only_runs_after_pause_then_resumes(queue,tmp_path,monkeypatch):
    monitor=m.Monitor(queue,tmp_path/'state');order=[]
    monkeypatch.setattr(m,'inspect_queue',lambda output:{'state':'ABNORMAL','reason':'queue_failed'})
    monkeypatch.setattr(monitor,'pause_owned_queue',lambda value:order.append('pause') or True)
    monkeypatch.setattr(monitor,'repair',lambda value:order.append('repair') or {'ready_to_resume':True})
    monkeypatch.setattr(monitor,'start_queue',lambda:order.append('resume') or True)
    event=monitor.check()
    assert order==['pause','repair','resume'] and event['model_invocations']==1


def test_unauthorized_own_method_never_restarts(queue,tmp_path,monkeypatch):
    save(queue/'protocol.json',{'workloads':['ycsb'],'methods':['cognitune_v3'],'budget':100,'n_init':10})
    monitor=m.Monitor(queue,tmp_path/'state')
    monkeypatch.setattr(monitor,'repair',lambda _:pytest.fail('unauthorized repair'))
    assert monitor.check()['action']=='refused_unauthorized_scope'


def test_missing_controller_restarts_without_model(queue,tmp_path,monkeypatch):
    monitor=m.Monitor(queue,tmp_path/'state')
    monkeypatch.setattr(m,'inspect_queue',lambda output:{'state':'ABNORMAL','reason':'supervisor_missing'})
    monkeypatch.setattr(monitor,'start_queue',lambda:True)
    monkeypatch.setattr(monitor,'repair',lambda _:pytest.fail('restart did not require model'))
    event=monitor.check()
    assert event['action']=='queue_restarted_without_model' and event['model_invocations']==0


def test_foreign_database_lock_prevents_model_repair(queue,tmp_path,monkeypatch):
    monitor=m.Monitor(queue,tmp_path/'state')
    monkeypatch.setattr(m,'inspect_queue',lambda output:{'state':'ABNORMAL','reason':'queue_failed'})
    monkeypatch.setattr(monitor,'pause_owned_queue',lambda _:False)
    monkeypatch.setattr(monitor,'repair',lambda _:pytest.fail('must not repair while another experiment holds lock'))
    assert monitor.check()['action']=='wait_for_owned_worker_or_database_lock'


def test_a_deferred_baseline_is_repaired_even_while_other_jobs_progress(queue):
    save(queue/'protocol.json',{'workloads':['ycsb'],'methods':['smac','gp_bo'],'budget':100,'n_init':10})
    save(queue/'ycsb/gp_bo/status.json',{'state':'DEFERRED','returncode':1})
    assert m.inspect_queue(queue,alive=lambda pid:True)['reason']=='deferred_baseline_failure'
