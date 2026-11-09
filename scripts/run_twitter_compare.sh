#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
METHOD="${1:-both}"
if [[ $# -gt 0 ]]; then shift; fi
case "$METHOD" in
  both|cognitune_v3|mctuner_official) ;;
  *) echo 'Usage: bash scripts/run_twitter_compare.sh [both|cognitune_v3|mctuner_official] [--resume] [--allow-unstable-baseline]' >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR="${TWITTER_RUN_DIR:-$ROOT/artifacts/twitter_pair_50_v3}"
mkdir -p "$RUN_DIR/console"
LOG="$RUN_DIR/console/${METHOD}_$(date -u +%Y%m%dT%H%M%SZ).log"
{
  echo "Twitter: $METHOD; 50 evaluations per method INCLUDING 10 shared IID starts."
  echo "Metric: measurement-window successful requests/second; warmup is excluded."
  echo "Workload: BenchBase generated Twitter, scale 10, 16 clients, mix 1/1/7/90/1, 15s + 60s."
  echo "Console log: $LOG"
  "$ROOT/.venv/bin/python" -u scripts/compare_twitter.py prepare --artifacts "$RUN_DIR" || exit $?
  "$ROOT/.venv/bin/python" -u scripts/compare_twitter.py run --budget 50 --method "$METHOD" --artifacts "$RUN_DIR" "$@"
} 2>&1 | tee "$LOG"
