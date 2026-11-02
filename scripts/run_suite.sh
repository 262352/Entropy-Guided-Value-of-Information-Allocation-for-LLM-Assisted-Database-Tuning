#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a
source "$ROOT/.runtime/cognitune.env"
set +a

if [[ ! -f /root/GPTuner/benchbase/target/benchbase-postgres/benchbase.jar ]]; then
  echo "BenchBase package is missing; run scripts/setup_real_environment.sh first." >&2
  exit 1
fi

exec "$ROOT/.venv/bin/python" scripts/run_suite.py \
  --suite configs/experiments/suite_postgres.yaml \
  --artifacts artifacts "$@"
