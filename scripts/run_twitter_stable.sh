#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TWITTER_RUN_DIR="${TWITTER_RUN_DIR:-$ROOT/artifacts/twitter_pair_50_stable_v1}"
METHOD="${1:-both}"
if [[ $# -gt 0 ]]; then shift; fi
exec bash "$ROOT/scripts/run_twitter_compare.sh" "$METHOD" \
  --search-policy stable \
  --reuse-preparation-from "$ROOT/artifacts/twitter_pair_50_v3" "$@"
