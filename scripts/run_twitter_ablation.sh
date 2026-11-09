#!/usr/bin/env bash
# No launch occurs until this script is explicitly invoked by the operator.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VARIANT="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$VARIANT" in
  all) VARIANTS=(full fixed_knobs fixed_regions uniform_budget nei_only) ;;
  full|fixed_knobs|fixed_regions|uniform_budget|nei_only) VARIANTS=("$VARIANT") ;;
  *) echo 'Usage: bash scripts/run_twitter_ablation.sh [all|full|fixed_knobs|fixed_regions|uniform_budget|nei_only] [--resume]' >&2; exit 2 ;;
esac
# Keep experimental identities and the 50-evaluation protocol unambiguous.
for OPT in "$@"; do
  case "$OPT" in
    --resume|--allow-unstable-baseline) ;;
    *) echo "Unsupported option: $OPT" >&2; exit 2 ;;
  esac
done
BASE="${TWITTER_ABLATION_DIR:-$ROOT/artifacts/twitter_ablation_50_v2}"
SOURCE="${TWITTER_PREPARATION_SOURCE:-$ROOT/artifacts/twitter_pair_50_v3}"
for ITEM in "${VARIANTS[@]}"; do
  echo "[ABLATION] variant=$ITEM; independent output=$BASE/$ITEM; 50 evaluations including 10 IID starts"
  TWITTER_RUN_DIR="$BASE/$ITEM" bash scripts/run_twitter_compare.sh cognitune_v3 \
    --search-policy stable --ablation "$ITEM" --reuse-preparation-from "$SOURCE" "$@"
done
