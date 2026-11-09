"""Isolated YCSB/TPC-H/CH-benCHmark datasets and recoverable measurements."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from cognitune.evaluator.base import EvaluationResult, metric_utility
from cognitune.experiment.resilience import atomic_json
import real_benchbase as h

SPECS = {
    'ycsb': {'scale':100, 'terminals':16, 'benchmarks':'ycsb', 'objective':'throughput'},
    'tpch': {'scale':.1, 'terminals':1, 'benchmarks':'tpch', 'objective':'latency'},
    'chbenchmark': {'scale':1, 'terminals':16, 'benchmarks':'tpcc,chbenchmark', 'objective':'hybrid'},
}
OLTP = {'NewOrder','Payment','OrderStatus','Delivery','StockLevel'}


def xml_config(workload, database, destination, warmup=15, duration=60):
    spec=SPECS[workload]
    tree=ET.parse(h.BENCH/f'config/postgres/sample_{workload}_config.xml')
    root=tree.getroot()
    def put(name, value, parent=root, attrs=None):
        node=parent.find(name) if not attrs else next((x for x in parent.findall(name) if x.attrib==attrs),None)
        if node is None: node=ET.SubElement(parent,name,attrs or {})
        node.text=str(value)
    url=f'jdbc:postgresql://127.0.0.1:{h.PORT}/{database}?sslmode=disable&ApplicationName=baseline_{workload}&reWriteBatchedInserts=true&connectTimeout=10&socketTimeout=240'
    for name,value in {'url':url,'username':'benchadmin','password':h.private_env()['PGPASSWORD'],
                       'scalefactor':spec['scale'],'terminals':spec['terminals'],
                       'isolation':'TRANSACTION_SERIALIZABLE','randomSeed':0,'loaderThreads':4}.items(): put(name,value)
    if workload=='chbenchmark':
        put('terminals',12,attrs={'bench':'tpcc'});put('terminals',4,attrs={'bench':'chbenchmark'})
    for work in root.findall('works/work'):
        put('time',duration,work);put('warmup',warmup,work);put('rate','unlimited',work)
    tree.write(destination,encoding='unicode',xml_declaration=True)
    Path(destination).chmod(0o600)


def run_java(workload, xml, output, jar, *, load=False, timeout=360):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    command=['java','-Xmx4g','-jar',str(jar),'-b',SPECS[workload]['benchmarks'],
             '-c',str(xml),'-d',str(output),'--create='+str(load).lower(),
             '--load='+str(load).lower(),'--execute='+str(not load).lower()]
    with (output/'benchbase.log').open('w') as log:
        child=subprocess.Popen(command,cwd=h.BENCH,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            status=child.wait(timeout=timeout)
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            raise
    if status: raise RuntimeError(f'BenchBase exit {status}; see {output}/benchbase.log')
    if load: return None
    summaries=list(output.glob('*.summary.json'))
    if len(summaries)!=1: raise RuntimeError(f'Expected one summary, got {len(summaries)}')
    return json.loads(summaries[0].read_text())


def metrics(workload, summary):
    if summary.get('Final State')!='EXIT': raise ValueError('Benchmark incomplete')
    goodput=float(summary['Measured Goodput (requests/second)'])
    if not math.isfinite(goodput) or goodput<=0: raise ValueError('zero/nonfinite measured successful throughput')
    details={'measured_goodput':goodput,'measured_type_statistics':summary.get('Measured Type Statistics',{})}
    if workload=='ycsb': return goodput, details
    types=summary['Measured Type Statistics']
    analytic=[v for k,v in types.items() if k.startswith('Q')]
    if not analytic: raise ValueError('No successful measured analytical queries')
    if workload=='tpch':
        latency=float(types['_analytical']['p95_latency_ms'])
        if not math.isfinite(latency) or latency<=0: raise ValueError('Invalid measured p95 latency')
        details['p95_latency_ms']=latency
        return latency,details
    tp=sum(float(v['throughput_per_second']) for k,v in types.items() if k in OLTP)
    p95=float(types['_analytical']['p95_latency_ms'])
    if tp<=0 or p95<=0: raise ValueError('HTAP needs successful transactional and analytical measurements')
    details.update(transactional_tps=tp, analytical_p95_ms=p95)
    return tp,details


def control_variation(records, workload):
    values={'raw_metric':[r['raw_metric'] for r in records]}
    if workload=='chbenchmark':
        values['analytical_p95_ms']=[r['metadata']['analytical_p95_ms'] for r in records]
    return {key:statistics.stdev(xs)/statistics.mean(xs) for key,xs in values.items()}


def acceptable_drift(records, baseline, workload):
    if not all(r['status']=='SUCCESS' for r in records): return False
    if abs(statistics.median(r['raw_metric'] for r in records)/baseline['raw_metric']-1)>.20: return False
    return workload!='chbenchmark' or abs(statistics.median(r['metadata']['analytical_p95_ms'] for r in records)/baseline['analytical_p95_ms']-1)<=.20


class WorkloadBackend:
    def __init__(self, workload, space, output, jar, *, baseline=None, warmup=15, duration=60):
        self.workload,self.space,self.output,self.jar=workload,space,Path(output),Path(jar)
        self.baseline=baseline;self.warmup=warmup;self.duration=duration
        self.database='cognitune_rest_'+workload
        self.template=self.database+'_template'
        self.output.mkdir(parents=True,exist_ok=True)

    def reset(self):
        h.pgctl('stop');h.configure(self.space.get_default());h.pgctl('start')
        h.sql('SELECT 1',timeout=30)

    def prepare(self):
        stamp=h.RUNTIME/f'baselines-{self.workload}-dataset.json'
        try:
            exists=h.sql("SELECT count(*) FROM pg_database WHERE datname='"+self.template+"'")=='1'
        except (OSError,subprocess.SubprocessError,RuntimeError):
            # A prior worker may have stopped at an unavailable instance.
            self.reset()
            exists=h.sql("SELECT count(*) FROM pg_database WHERE datname='"+self.template+"'")=='1'
        if exists and stamp.exists():
            if json.loads(stamp.read_text())['spec']!=SPECS[self.workload]:
                raise ValueError('Existing template has incompatible workload settings')
            return
        self.reset()
        if exists: h.maintenance_sql('DROP DATABASE '+self.template,'reload_incomplete_template',900)
        h.sql('CREATE DATABASE '+self.template)
        directory=h.RUNTIME/'baseline_dataset_load'/self.workload/uuid.uuid4().hex
        directory.mkdir(parents=True)
        xml=directory/'load.xml';xml_config(self.workload,self.template,xml,self.warmup,self.duration)
        run_java(self.workload,xml,directory,self.jar,load=True,timeout=3600)
        h.sql('ANALYZE',database=self.template,timeout=600)
        tables=h.sql("SELECT count(*) FROM pg_tables WHERE schemaname='public'",database=self.template)
        atomic_json(stamp,{'workload':self.workload,'spec':SPECS[self.workload],'tables':int(tables),'load_log':str(directory),'jar':str(self.jar)})

    def measure(self, config, token):
        folder=self.output/token;folder.mkdir(parents=True,exist_ok=False)
        atomic_json(folder/'config.json',config)
        started=time.monotonic()
        try:
            h.pgctl('stop');h.configure(config);h.pgctl('start')
            durability=json.loads(h.sql("SELECT json_object_agg(name,setting) FROM pg_settings WHERE name IN ('fsync','synchronous_commit','full_page_writes')"))
            if set(durability.values())!={'on'}: raise ValueError('Durability settings changed')
            h.sql(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='{self.database}' AND pid<>pg_backend_pid()")
            h.maintenance_sql('DROP DATABASE IF EXISTS '+self.database,'baseline_drop',900)
            h.maintenance_sql('CREATE DATABASE '+self.database+' TEMPLATE '+self.template,'baseline_copy',900)
            xml=folder/'workload.xml';xml_config(self.workload,self.database,xml,self.warmup,self.duration)
            summary=run_java(self.workload,xml,folder,self.jar,timeout=self.warmup+self.duration+240)
            raw,detail=metrics(self.workload,summary)
            if self.baseline is None: utility=0.
            elif self.workload=='chbenchmark':
                utility=.5*math.log(detail['transactional_tps']/self.baseline['transactional_tps'])+.5*math.log(self.baseline['analytical_p95_ms']/detail['analytical_p95_ms'])
            else:
                utility=metric_utility(raw,self.baseline['raw_metric'],SPECS[self.workload]['objective'])
            result=EvaluationResult(raw,utility,'SUCCESS',time.monotonic()-started,config=config,
                metadata={**detail,'workload':self.workload,'summary':summary,'measurement_folder':str(folder),
                    'warmup_seconds':self.warmup,'measurement_seconds':self.duration})
        except Exception as error:
            result=EvaluationResult(0.,-10.,'CRASH',time.monotonic()-started,config=config,
                metadata={'error':str(error),'exception':type(error).__name__,'measurement_folder':str(folder)})
        atomic_json(folder/'result.json',result.as_dict())
        print(f'[MEASURE] {self.workload} {token} {result.status} metric={result.raw_metric:.4f}',flush=True)
        return result

    def recover(self, token):
        self.reset()
        return self.measure(self.space.get_default(),token)
