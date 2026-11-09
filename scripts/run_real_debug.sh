#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
METHOD="${1:-both}"
case "$METHOD" in
  both|cognitune_v2|mctuner_official) ;;
  *) echo "Usage: bash scripts/run_real_debug.sh [both|cognitune_v2|mctuner_official]" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
RUN_DIR="$ROOT/artifacts/real_pair_50"
mkdir -p "$RUN_DIR/console"
LOG="$RUN_DIR/console/${METHOD}_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "Running $METHOD; 50 evaluations each, including 10 shared IID starts."
echo "Console log: $LOG"
echo "Explicit exploratory mode: baseline instability is recorded, not used to stop."
echo "Metric: upstream Goodput includes warmup successes; measured Throughput is also logged."
echo "Normal shutdown wait limit: ${COGNITUNE_PG_STOP_TIMEOUT:-600}s; pending shutdown prints DB WAIT progress."
echo "Template restore wait limit: ${COGNITUNE_PG_TEMPLATE_TIMEOUT:-900}s; restore failures are infrastructure errors, not optimizer crashes."
"$ROOT/.venv/bin/python" -u "$ROOT/.runtime/compare_real.py" run \
  --budget 50 --method "$METHOD" --artifacts "$RUN_DIR" \
  --reuse-priors-from "$ROOT/artifacts/real_pair_20261102" \
  --allow-unstable-baseline --verbose 2>&1 | tee "$LOG"

if [[ -f "$RUN_DIR/runs/cognitune_v2/seed_0/final_validation.json" && \
      -f "$RUN_DIR/runs/mctuner_official/seed_0/final_validation.json" ]]; then
  "$ROOT/.venv/bin/python" "$ROOT/.runtime/analyze_real_pair.py" --artifacts "$RUN_DIR"
fi
