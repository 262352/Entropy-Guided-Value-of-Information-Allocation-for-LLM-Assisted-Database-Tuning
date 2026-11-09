#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_output="${COGNITUNE_BASELINE_OUT:-artifacts/remaining_baselines_100_v1}"
case "${1:-start}" in
  start)
    if ! flock -n .runtime/real-comparison.lock true; then
      echo "An experiment already holds the database lock; use the status command."
      exit 0
    fi
    mkdir -p "$run_output/console"
    run_log="$run_output/console/supervisor_$(date -u +%Y%m%dT%H%M%SZ).log"
    export PYTHONPATH="$PWD/src"
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    nohup setsid "$PWD/.venv/bin/python" -u scripts/run_remaining_baselines.py --output "$run_output" --budget 100 > "$run_log" 2>&1 < /dev/null &
    echo "Supervisor PID: $!"
    echo "Console log: $PWD/$run_log"
    ;;
  status)
    .venv/bin/python - "$run_output" <<'PY'
import json,sys,os
from pathlib import Path
root=Path(sys.argv[1])
for name in ['supervisor.json','heartbeat.json','queue_finished.json']:
 p=root/name
 if p.exists():
  item=json.loads(p.read_text())
  if 'pid' in item:
   try:os.kill(item['pid'],0);item['process_alive']=True
   except ProcessLookupError:item['process_alive']=False
  print(name,json.dumps(item,ensure_ascii=False))
for p in sorted(root.glob('*/*/status.json')):
 completed=len(list((p.parent/'ledger').glob('trial_*/result.json')))
 info=json.loads(p.read_text())
 work=p.parent.parent
 if not (work/'preparation.json').exists():
  info['phase']='PREPARATION'
  info['default_measurements']=len(list((work/'default_ledger').glob('trial_*/result.json')))
  info['health_probes']=len(list((work/'health_ledger').glob('trial_*/result.json')))
  info['shared_initial_measurements']=len(list((work/'shared_initial').glob('trial_*/result.json')))
 print(p.parent.relative_to(root),str(completed)+'/100',json.dumps(info,ensure_ascii=False))
PY
    ;;
  *) echo 'Usage: bash scripts/run_remaining_baselines.sh [start|status]' >&2; exit 2 ;;
esac
