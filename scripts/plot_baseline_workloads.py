#!/usr/bin/env python3
"""Snapshot durable baseline ledgers and plot observed best throughput."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'smac': 'SMAC', 'gp_bo': 'GP-BO', 'cdbtune': 'CDBTune',
          'restune': 'ResTune', 'ottertune': 'OtterTune',
          'mctuner_official': 'MCTuner', 'dot_official': 'DOT'}
TITLES = {'ycsb': 'YCSB', 'tpch': 'TPC-H', 'chbenchmark': 'CH-benCHmark'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'artifacts/remaining_baselines_100_v1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.input/'analysis/best_throughput_current'
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.input/'protocol.json').read_text())
    report = {'snapshot_utc': datetime.now(timezone.utc).isoformat(),
              'budget': protocol['budget'], 'initial_rounds': protocol['n_init'],
              'definition': 'Cumulative maximum successful measured throughput; no extrapolation or smoothing.',
              'note': 'TPC-H optimizes latency; its QPS is a secondary observed metric. Peaks are not final-repeat medians.',
              'series': {}}
    rows = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    combined, axes = plt.subplots(1, 3, figsize=(17, 4.7), dpi=180)
    colors = plt.get_cmap('tab10').colors
    for workload, axis in zip(protocol['workloads'], axes):
        single, standalone = plt.subplots(figsize=(8, 4.8), dpi=180)
        plotted = False
        for i, method in enumerate(protocol['methods']):
            job = args.input/workload/method
            # Snapshot filenames first: an atomic result arriving later belongs to the next plot.
            paths = sorted((job/'ledger').glob('trial_*/result.json'))
            if not paths:
                continue
            indices = [int(p.parent.name.split('_')[-1]) for p in paths]
            if indices != list(range(len(indices))):
                raise ValueError(f'Noncontiguous ledger: {job}')
            summary_path = job/'summary.json'
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            x, y, best, best_round = [], [], -math.inf, None
            for trial, path in zip(indices, paths):
                record = json.loads(path.read_text())
                metadata = record.get('metadata', {})
                value = metadata.get('transactional_tps' if workload == 'chbenchmark' else 'measured_goodput')
                valid = record['status'] == 'SUCCESS' and value is not None and math.isfinite(float(value)) and float(value) > 0
                if record['status'] == 'SUCCESS' and not valid:
                    raise ValueError(f'Missing/invalid throughput: {path}')
                if valid and float(value) > best:
                    best, best_round = float(value), trial + 1
                x.append(trial + 1)
                y.append(best if math.isfinite(best) else math.nan)
                rows.append({'workload': workload, 'method': method, 'round': trial+1,
                             'status': record['status'], 'throughput': value if valid else '',
                             'best_throughput': y[-1]})
            label = LABELS[method]
            if not summary.get('validated'):
                label += f' ({len(paths)}/{protocol["budget"]})'
            if summary.get('comparison_valid') is False:
                label += ' [drift]'
            for ax in (axis, standalone):
                ax.step(x, y, where='post', label=label, color=colors[i], linewidth=1.7,
                        linestyle='--' if summary.get('comparison_valid') is False else '-')
            report['series'][workload+'/'+method] = {
                'completed_rounds': len(paths), 'best_throughput': best if math.isfinite(best) else None,
                'first_best_round': best_round, 'validated': summary.get('validated', False),
                'comparison_valid': summary.get('comparison_valid'), 'source': str((job/'ledger').resolve())}
            plotted = True
        unit = 'queries/s (QPS)' if workload == 'tpch' else 'transactions/s (TPS)' if workload == 'chbenchmark' else 'requests/s'
        for ax in (axis, standalone):
            ax.set_title(TITLES[workload])
            ax.set_xlabel('Configuration evaluations (including 10 shared IID starts)')
            ax.set_ylabel('Best-so-far throughput ('+unit+')')
            ax.set_xlim(1, protocol['budget'])
            ax.grid(alpha=.25, linestyle='--')
            if plotted:
                ax.axvspan(1, protocol['n_init'], color='gray', alpha=.07)
                ax.legend(fontsize=8, loc='lower right', framealpha=.9)
            else:
                ax.set_yticks([])
                ax.text(.5, .5, 'Pending in serial queue\nNo completed tuning evaluations',
                        transform=ax.transAxes, ha='center', va='center', color='#555555')
        if plotted:
            single.tight_layout()
            for ext in ('png', 'pdf'):
                single.savefig(output/f'{workload}_best_throughput.{ext}', bbox_inches='tight')
        plt.close(single)
    combined.suptitle('Observed best throughput — current experiment snapshot', fontsize=13)
    combined.tight_layout()
    for ext in ('png', 'pdf'):
        combined.savefig(output/f'baseline_best_throughput.{ext}', bbox_inches='tight')
    plt.close(combined)
    (output/'snapshot.json').write_text(json.dumps(report, indent=2)+'\n')
    with (output/'curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['workload', 'method', 'round', 'status', 'throughput', 'best_throughput'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))
    print(output.resolve())


if __name__ == '__main__':
    main()
