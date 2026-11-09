#!/usr/bin/env python3
"""Archive completed baseline results without copying runtime logs or secrets."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'artifacts/remaining_baselines_100_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/paper_revision')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_bytes = (args.input/'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),
                'source_root':str(args.input), 'protocol_sha256':hashlib.sha256(protocol_bytes).hexdigest(),
                'rounds':protocol['budget'], 'shared_IID_rounds':protocol['n_init'],
                'methods':protocol['methods'], 'protocol':protocol, 'datasets':{}}
    long_rows, summaries = [], []
    for workload in protocol['workloads']:
        default = json.loads((args.input/workload/'default.json').read_text())
        metric = 'p95_latency_ms' if workload=='tpch' else 'transactional_tps' if workload=='chbenchmark' else 'TPS_requests_per_second'
        filename = {'ycsb':'ycsb_100_rounds_tps.csv', 'tpch':'tpch_100_rounds_latency_ms.csv',
                    'chbenchmark':'chbenchmark_100_rounds_tps.csv'}[workload]
        columns, analytics, methods = {}, {}, {}
        for method in protocol['methods']:
            job = args.input/workload/method
            summary = json.loads((job/'summary.json').read_text())
            assert summary['validated'] and summary['trials']==protocol['budget'], job
            files = sorted((job/'ledger').glob('trial_*/result.json'))
            assert [int(f.parent.name[6:]) for f in files]==list(range(protocol['budget'])), job
            values, analytical_values, hashes, statuses = [], [], {}, {}
            for trial, path in enumerate(files):
                contents = path.read_bytes()
                record = json.loads(contents)
                value = record['raw_metric']
                status = record['status']
                statuses[status] = statuses.get(status, 0)+1
                if status=='SUCCESS':
                    assert math.isfinite(value) and value>0, path
                detail = record.get('metadata', {})
                relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
                hashes[relative] = hashlib.sha256(contents).hexdigest()
                values.append(value if status=='SUCCESS' else '')
                analytical_values.append(detail.get('analytical_p95_ms', ''))
                long_rows.append({'workload':workload, 'method':method, 'round':trial+1,
                                  'trial_index':trial, 'phase':'shared_IID' if trial<protocol['n_init'] else 'adaptive',
                                  'metric':metric, 'value':values[-1], 'status':status,
                                  'comparison_valid':summary['comparison_valid'],
                                  'timestamp':record.get('timestamp',''), 'source':relative,
                                  'analytical_p95_ms':detail.get('analytical_p95_ms','')})
            columns[method], analytics[method] = values, analytical_values
            methods[method] = {'comparison_valid':summary['comparison_valid'],
                               'status_counts':statuses, 'sources_sha256':hashes}
            repeats = summary['final_validation']
            assert len(repeats)==3
            summaries.append({'workload':workload,'method':method, 'rounds':summary['trials'],
                              'metric':metric,'default_metric':default['raw_metric'],
                              'best_metric':summary['best_raw_metric'], 'best_round':summary['best_trial']+1,
                              'best_utility':summary['best_utility'],
                              'final_repeat_values':json.dumps([r['raw_metric'] for r in repeats]),
                              'final_repeat_statuses':json.dumps([r['status'] for r in repeats]),
                              'final_repeat_median':statistics.median(r['raw_metric'] for r in repeats) if all(r['status']=='SUCCESS' for r in repeats) else '',
                              'comparison_valid':summary['comparison_valid'],
                              'default_after_values':json.dumps([r['raw_metric'] for r in summary['default_after']])})
        write_csv(args.output/filename, [{'round':i+1, **{m:columns[m][i] for m in protocol['methods']}} for i in range(protocol['budget'])])
        if workload=='chbenchmark':
            write_csv(args.output/'chbenchmark_100_rounds_analytical_p95_ms.csv',
                      [{'round':i+1, **{m:analytics[m][i] for m in protocol['methods']}} for i in range(protocol['budget'])])
        manifest['datasets'][workload] = {'file':filename, 'metric':metric,
                                          'direction':'minimize' if workload=='tpch' else 'maximize',
                                          'default_metric':default['raw_metric'], 'methods':methods}
    write_csv(args.output/'baseline_100_rounds_long.csv', long_rows)
    write_csv(args.output/'baseline_validation_summary.csv', summaries)
    names = [v['file'] for v in manifest['datasets'].values()]+['chbenchmark_100_rounds_analytical_p95_ms.csv',
             'baseline_100_rounds_long.csv', 'baseline_validation_summary.csv']
    manifest['exports_sha256'] = {n:hashlib.sha256((args.output/n).read_bytes()).hexdigest() for n in names}
    (args.output/'baseline_100_rounds_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(f'Archived {len(long_rows)} observations and {len(summaries)} validation summaries.')


if __name__=='__main__':
    main()
