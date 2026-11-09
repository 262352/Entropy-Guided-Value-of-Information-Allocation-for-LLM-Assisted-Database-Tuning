"""Isolated PostgreSQL 14.9 / BenchBase harness for the paired diagnostic."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
import uuid
import xml.etree.ElementTree as ET

from cognitune.evaluator.base import Evaluator, EvaluationResult, metric_utility

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'
PG = Path('/opt/cognitune-experiment/postgresql-14.9')
DATA = Path('/var/lib/cognitune-experiment/data')
BENCH = RUNTIME / 'benchbase/target/benchbase-postgres'
PORT = 55432
WORKLOAD = 'twitter'
SCALE = 10
TERMINALS = 16
METRIC = 'Measured Goodput (requests/second)'
DATABASE = 'cognitune_twitter'
TEMPLATE = DATABASE + '_template'
BENCH_JAR = RUNTIME/'benchbase-measured/benchbase-measured.jar'


class DatabaseLifecycleError(RuntimeError):
    """Infrastructure failed to finish a database lifecycle operation."""


def _postmaster_pid():
    try:
        return int((DATA/'postmaster.pid').read_text().splitlines()[0])
    except FileNotFoundError:
        return None


def _pid_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid,0)
        state=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
        return state!='Z'
    except ProcessLookupError:
        return False
    except FileNotFoundError:
        return False


def collect_default_measurements(evaluator,config,count=3):
    """Resume preflight only; never overwrite successful or failed evidence."""
    records=[]
    for trial in range(count):
        folder=evaluator.output/f'trial_{trial:04d}'
        path=folder/'result.json'
        if path.exists():
            previous=json.loads(path.read_text())
            if previous.get('status')=='SUCCESS':
                metadata=previous.get('metadata') or {}
                if (previous.get('config')!=config or
                    metadata.get('warmup_seconds')!=evaluator.warmup or
                    metadata.get('measurement_seconds')!=evaluator.duration or
                    metadata.get('metric')!=METRIC or metadata.get('workload')!=WORKLOAD or
                    not metadata.get('template_reset') or previous.get('raw_metric',0)<=0):
                    raise ValueError('Saved default measurement has incompatible configuration/protocol: '+str(path))
                print(f'[RESUME] default trial={trial} reusing successful measurement Goodput={previous["raw_metric"]:.3f}',flush=True)
                records.append(previous)
                continue
        if folder.exists():
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            archived=folder.with_name(folder.name+'.failed-attempt-'+stamp+'-'+uuid.uuid4().hex[:6])
            folder.rename(archived)
            print(f'[RESUME] preserved incomplete/failed attempt at {archived}',flush=True)
        result=evaluator.evaluate(config,trial).as_dict()
        records.append(result)
        print(f'[PREFLIGHT] completed={trial+1}/{count} status={result["status"]} Goodput={result["raw_metric"]:.3f}',flush=True)
    return records


def private_env():
    env = dict(os.environ)
    for line in (RUNTIME / 'cognitune.env').read_text().splitlines():
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            env[key] = value.strip().strip('"').strip("'")
    env['LD_LIBRARY_PATH'] = str(PG / 'lib')
    return env


def sql(statement, database='postgres', timeout=180):
    result = subprocess.run(
        [str(PG/'bin/psql'), '-X', '-A', '-t', '-v', 'ON_ERROR_STOP=1',
         '-h', '127.0.0.1', '-p', str(PORT), '-U', 'benchadmin', '-d', database,
         '-c', statement], env=private_env(), cwd='/tmp', capture_output=True,
        text=True, check=True, timeout=timeout,
    )
    return result.stdout.strip()


def maintenance_sql(statement,phase,timeout):
    """Keep the maintenance client alive while reporting the tagged backend."""
    application='cognitune-control-'+uuid.uuid4().hex
    env=private_env()
    env['PGAPPNAME']=application
    command=[str(PG/'bin/psql'),'-X','-A','-t','-v','ON_ERROR_STOP=1',
             '-h','127.0.0.1','-p',str(PORT),'-U','benchadmin','-d','postgres','-c',statement]
    process=subprocess.Popen(command,env=env,cwd='/tmp',stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,text=True)
    started=time.monotonic()
    latest=[]
    try:
        while True:
            remaining=timeout-(time.monotonic()-started)
            if remaining<=0:
                raise subprocess.TimeoutExpired(command,timeout)
            try:
                stdout,stderr=process.communicate(timeout=min(15,remaining))
                if process.returncode:
                    raise DatabaseLifecycleError(f'{phase} failed: '+stderr[-1500:])
                return stdout.strip()
            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-started
                try:
                    latest=json.loads(sql("SELECT coalesce(json_agg(row_to_json(s)),'[]'::json) FROM "
                        "(SELECT pid,state,wait_event_type,wait_event FROM pg_stat_activity WHERE application_name='"+application+"') s",timeout=3))
                except (subprocess.SubprocessError,OSError,ValueError) as monitor_error:
                    latest=[{'monitor_error':type(monitor_error).__name__}]
                event={'phase':phase,'elapsed_seconds':round(elapsed,2),
                       'timeout_seconds':timeout,'application_name':application,'backend':latest}
                print('[DB WAIT] '+json.dumps(event,sort_keys=True),flush=True)
                with (RUNTIME/'db_wait_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(event)+'\n')
                if elapsed>=timeout:
                    raise subprocess.TimeoutExpired(command,timeout)
    except subprocess.TimeoutExpired as error:
        try:
            cancellation=sql("SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE application_name='"+application+"'",timeout=3)
        except (subprocess.SubprocessError,OSError) as cancel_error:
            cancellation='cancel request failed: '+type(cancel_error).__name__
        raise DatabaseLifecycleError(f'{phase} exceeded {timeout}s; backend={latest}; cancellation={cancellation!r}. '
            'A cancel request can remain pending during storage I/O. Automatic database restart/retry is disabled; inspect before rerunning.') from error
    finally:
        if process.poll() is None:
            # Close only our psql client after requesting cancellation. Never
            # stop or force-kill PostgreSQL to recover a stalled template copy.
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                print(f'[DB] maintenance client PID={process.pid} has not exited; backend tag={application}',flush=True)


def pgctl(action):
    started=time.monotonic()
    original_pid=_postmaster_pid() if action=='stop' else None
    stop_timeout=int(os.environ.get('COGNITUNE_PG_STOP_TIMEOUT','600'))
    if stop_timeout<90:
        raise ValueError('COGNITUNE_PG_STOP_TIMEOUT must be at least 90 seconds')
    command_timeout=15 if action=='stop' else 180
    command = ['runuser', '-u', 'cognitune-pg', '--', 'env',
               'LD_LIBRARY_PATH='+str(PG/'lib'), str(PG/'bin/pg_ctl'),
               '-D', str(DATA), '-l', str(DATA.parent/'server.log'),
               '-w', '-t', str(command_timeout)]
    if action == 'stop':
        command += ['-m', 'fast']
    try:
        process = subprocess.run(command+[action], cwd='/tmp', capture_output=True,
                                 text=True, timeout=command_timeout+15)
    except subprocess.TimeoutExpired as error:
        raise DatabaseLifecycleError(f'pg_ctl {action} did not return within {command_timeout+15}s; inspect PostgreSQL/storage before retrying') from error
    if process.returncode:
        if action=='stop':
            if original_pid is None:
                if _postmaster_pid() is None:
                    return
                raise DatabaseLifecycleError('Postmaster identity changed during stop')
            # pg_ctl can time out while the server is still flushing its shutdown
            # checkpoint. Wait for the same process; never restart over it.
            if 'server does not shut down' in process.stderr or not _pid_alive(original_pid):
                last_report=-float('inf')
                while _pid_alive(original_pid):
                    elapsed=time.monotonic()-started
                    if elapsed>=stop_timeout:
                        raise DatabaseLifecycleError(f'Normal shutdown still pending after {elapsed:.1f}s (PID {original_pid}); inspect storage I/O. No force kill performed.')
                    current=_postmaster_pid()
                    if current is not None and current!=original_pid:
                        raise DatabaseLifecycleError('Postmaster identity changed while waiting for shutdown')
                    if elapsed-last_report>=15:
                        print(f'[DB WAIT] original postmaster PID={original_pid} flushing shutdown; elapsed={elapsed:.1f}s limit={stop_timeout}s',flush=True)
                        last_report=elapsed
                    time.sleep(2)
                print(f'[DB] shutdown confirmed after {time.monotonic()-started:.1f}s',flush=True)
                return
            raise DatabaseLifecycleError('pg_ctl stop failed: '+process.stderr[-1500:])
        raise RuntimeError('pg_ctl '+action+' failed: '+process.stderr[-1500:])


def configure(config):
    lines = []
    for name, value in config.items():
        if not name.replace('_', '').isalnum():
            raise ValueError('Invalid setting name')
        if isinstance(value, bool):
            literal = 'on' if value else 'off'
        elif isinstance(value, (int, float)):
            literal = str(value)
        else:
            literal = "'"+str(value).replace("'", "''")+"'"
        lines.append(f'{name} = {literal}')
    path = DATA/'tuning.conf'
    path.write_text('\n'.join(lines)+'\n')
    os.chown(path, DATA.stat().st_uid, DATA.stat().st_gid)


def xml_config(database, output, warmup=15, duration=60):
    tree = ET.parse(BENCH/f'config/postgres/sample_{WORKLOAD}_config.xml')
    root = tree.getroot()
    settings = {
        'url': f'jdbc:postgresql://127.0.0.1:{PORT}/{database}?sslmode=disable&ApplicationName=cognitune_{WORKLOAD}&reWriteBatchedInserts=true',
        'username': 'benchadmin', 'password': private_env()['PGPASSWORD'],
        'scalefactor': str(SCALE), 'terminals': str(TERMINALS), 'isolation': 'TRANSACTION_SERIALIZABLE',
    }
    for name, value in settings.items():
        root.find(name).text = value
    for name, value in {'tracefile': str(RUNTIME/'twitter-traces/tweetids.txt'), 'tracefile2': str(RUNTIME/'twitter-traces/userids.txt'), 'randomSeed': '0', 'loaderThreads': '4'}.items():
        element = root.find(name)
        if element is None:
            element = ET.SubElement(root, name)
        element.text = value
    work = root.find('works/work')
    work.find('time').text = str(duration)
    work.find('rate').text = 'unlimited'
    element = work.find('warmup')
    if element is None:
        element = ET.SubElement(work, 'warmup')
    element.text = str(warmup)
    tree.write(output, encoding='unicode', xml_declaration=True)
    Path(output).chmod(0o600)


def benchbase(xml, output, load=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    command = ['java', '-Xmx4g', '-jar', str(BENCH_JAR), '-b', WORKLOAD,
               '-c', str(xml), '-d', str(output),
               '--create='+str(load).lower(), '--load='+str(load).lower(),
               '--execute='+str(not load).lower()]
    with (output/'benchbase.log').open('w') as log:
        subprocess.run(command, cwd=BENCH, stdout=log, stderr=subprocess.STDOUT,
                       timeout=3600 if load else 240, check=True)
    if load:
        return None
    summaries = list(output.glob('*.summary.json'))
    if len(summaries) != 1:
        raise RuntimeError('Expected one BenchBase summary: '+str(output))
    return json.loads(summaries[0].read_text())


class LocalBenchBaseEvaluator(Evaluator):
    def __init__(self, space, output, default_metric, warmup=15, duration=60):
        self.space, self.output = space, Path(output)
        self.default_metric = default_metric
        self.warmup, self.duration = warmup, duration
        self.minimum = 0.0
        self.lifecycle_failed = False
        self.output.mkdir(parents=True, exist_ok=True)

    def evaluate(self, config, trial):
        started = time.monotonic()
        folder = self.output/f'trial_{trial:04d}'
        if folder.exists():
            raise FileExistsError('Refusing to overwrite measurement '+str(folder))
        folder.mkdir()
        (folder/'config.json').write_text(json.dumps(config, indent=2))
        restart_seconds = 0.0
        phase_name='database_restart'
        try:
            print(f'[DB] trial={trial} stop/reconfigure/restart',flush=True)
            phase = time.monotonic()
            pgctl('stop')
            configure(config)
            pgctl('start')
            restart_seconds = time.monotonic()-phase
            durability=json.loads(sql("SELECT json_object_agg(name,setting) FROM pg_settings WHERE name IN ('fsync','synchronous_commit','full_page_writes')"))
            if any(value!='on' for value in durability.values()):
                raise RuntimeError('Durability check failed: '+str(durability))
            print(f'[DB] trial={trial} restore template (restart={restart_seconds:.2f}s)',flush=True)
            phase_name='template_restore'
            restore_timeout=int(os.environ.get('COGNITUNE_PG_TEMPLATE_TIMEOUT','900'))
            if restore_timeout<=0:
                raise ValueError('COGNITUNE_PG_TEMPLATE_TIMEOUT must be positive')
            sql(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='{DATABASE}' AND pid<>pg_backend_pid()")
            maintenance_sql(f'DROP DATABASE IF EXISTS {DATABASE}','template_drop',restore_timeout)
            maintenance_sql(f'CREATE DATABASE {DATABASE} TEMPLATE {TEMPLATE}','template_copy',restore_timeout)
            phase_name='benchmark'
            config_path = RUNTIME/'current-benchbase.xml'
            xml_config(DATABASE, config_path, self.warmup, self.duration)
            print(f'[DB] trial={trial} run BenchBase warmup={self.warmup}s measure={self.duration}s',flush=True)
            summary = benchbase(config_path, folder)
            if summary.get('Final State') != 'EXIT':
                raise RuntimeError('benchmark did not finish cleanly')
            measured = float(summary['Measured Successful Requests'])
            requests = float(summary['Measured Requests'])
            if not 0 < measured <= requests:
                raise RuntimeError(f'invalid measured success counts: {measured}/{requests}')
            metric = float(summary[METRIC])
            utility = metric_utility(metric, self.default_metric, 'throughput')
            self.minimum = min(self.minimum, utility)
            result = EvaluationResult(metric, utility, 'SUCCESS',
                time.monotonic()-started,
                metadata={'summary_path': str(folder), 'benchbase_summary': summary,
                          'warmup_seconds': self.warmup, 'measurement_seconds': self.duration,
                          'metric': METRIC, 'workload': WORKLOAD, 'measured_success_fraction': measured / requests, 'template_reset': True},
                config=config, restart_seconds=restart_seconds,
                warmup_seconds=self.warmup, benchmark_seconds=self.duration)
            print(f'[DB] trial={trial} Goodput={metric:.3f} measured_Throughput={summary["Throughput (requests/second)"]:.3f} elapsed={result.elapsed_seconds:.2f}s',flush=True)
        except DatabaseLifecycleError as error:
            self.lifecycle_failed=True
            result=EvaluationResult(0.0,self.minimum-1.0,'INFRA_ERROR',time.monotonic()-started,
                metadata={'error':str(error),'phase':phase_name},config=config)
            (folder/'result.json').write_text(json.dumps(result.as_dict(),indent=2))
            print(f'[DB INFRASTRUCTURE ERROR] {error}',flush=True)
            raise
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, RuntimeError) as error:
            print(f'[DB ERROR] trial={trial}: {error}',flush=True)
            result = EvaluationResult(0.0, self.minimum-1.0, 'CRASH',
                time.monotonic()-started, metadata={'error': str(error)}, config=config)
            # Restore a bootable instance before another trial or a diagnostic.
            (folder/'result.json').write_text(json.dumps(result.as_dict(),indent=2))
            try:
                pgctl('stop')
                configure(self.space.get_default())
                pgctl('start')
            except (subprocess.SubprocessError,OSError,RuntimeError) as recovery_error:
                self.lifecycle_failed=True
                result.metadata['recovery_error']=str(recovery_error)
                (folder/'result.json').write_text(json.dumps(result.as_dict(),indent=2))
                raise DatabaseLifecycleError('Database recovery failed; original measurement error saved at '+str(folder/'result.json')) from recovery_error
        (folder/'result.json').write_text(json.dumps(result.as_dict(), indent=2))
        return result

    def restore_default(self):
        if self.lifecycle_failed:
            print('[DB] automatic configuration restore skipped after lifecycle failure; inspect shutdown/storage state before retrying',flush=True)
            return
        pgctl('stop')
        configure(self.space.get_default())
        pgctl('start')
