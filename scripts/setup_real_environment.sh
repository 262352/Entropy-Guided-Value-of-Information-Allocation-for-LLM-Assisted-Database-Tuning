#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHBASE=/root/GPTuner/benchbase
cd "$ROOT"
set -a
source "$ROOT/.runtime/cognitune.env"
set +a

if [[ ! -f "$BENCHBASE/target/benchbase-postgres/benchbase.jar" ]]; then
  (cd "$BENCHBASE" && JAVA_HOME=/opt/jdk-23-openjdk ./mvnw clean package -P postgres -DskipTests)
  tar -xzf "$BENCHBASE/target/benchbase-postgres.tgz" -C "$BENCHBASE/target"
fi

"$ROOT/.venv/bin/python" scripts/profile_environment.py
"$ROOT/.venv/bin/python" scripts/collect_llm_prior.py \
  --space configs/knobs/postgresql_60_live.yaml \
  --profile artifacts/profiles/postgresql_tpcc.json \
  --output artifacts/llm/deepseek_tpcc_v3 \
  --model deepseek-flash
