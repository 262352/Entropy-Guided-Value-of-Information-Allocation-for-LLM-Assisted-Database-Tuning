# HANDOFF — CogniTune v2

Read this before changing code. It is the state transfer for the next agent.

## 0. TL;DR

- The repository now contains **CogniTune v2**: four LLM experts propose
  configurations, an entropy/VoI controller routes experts and gates knobs,
  and a shared BoTorch GP with LogEI selects the next database experiment.
- `pytest -q`: **45 passed**.
- Synthetic TPC-C-like 60-knob smoke (30 trials, seed 0, MockLLM):
  `cognitune_v2` best **2,645,111** vs `mctuner` **2,106,289**.
  This is one synthetic seed, not paper evidence.
- **Real experiments are blocked by host storage I/O**: PostgreSQL startup /
  checkpointer get stuck in uninterruptible `D` state, disk is 82% full and
  `fsync` throughput is ~1.6 MB/s. `pg_isready` currently rejects connections.
  No valid real v2-vs-MCTuner result exists yet.
- Existing real TPC-C/YCSB results are snapshotted in `docs/results/`.
  They used the old protocol/shared initial points; do not mix them with a new
  independent-start protocol without rerunning all baselines.

## 1. Research thesis

The intended contribution is **entropy-guided value-of-information (VoI)
allocation for LLM-assisted database tuning**, not "multi-agent tuning" by
itself. Multi-agent LLM tuning is already covered by MCTuner / AgentTune /
FasterTune / QueryMind. The novelty must be:

1. **outcome-calibrated expert routing** (who to ask, not self-scored confidence);
2. **entropy-gated knob selection** (which 15–20 knobs receive trials, with
   boundary/uncertainty-driven eviction and admission);
3. **VoI-aware experiment selection** (current code uses LogEI as primary and
   uses gate/expert entropy only as a tie-break among top-EI candidates;
   full one-step lookahead is still future work).

The current implemented story is: **experts propose, GP critiques, entropy/VoI
allocates expert tokens and knob slots.**

## 2. Method as implemented (v2)

### 2.1 Four experts and ownership

`src/cognitune/agents/expert.py` defines:

- `memory_expert`
- `concurrency_execution_expert`
- `storage_wal_expert`
- `optimizer_workload_expert`

`src/cognitune/agents/ownership.py` gives each knob a normalized ownership
weight. Some knobs are shared across groups (`work_mem`, `wal_buffers`,
`max_connections`, `effective_io_concurrency`, etc.). This is the answer to
"related knobs under different agents": they are not hard-partitioned; the
ownership matrix and relation blocks coordinate them.

Default prompt ensemble is **4 experts × 1 `role_template`**, not the legacy
5×3=15 prompts. The router can re-query the top-2 experts every 10 rounds with
updated incumbent/active context.

### 2.2 ACTIVE knob selection (15–20, target 18)

`_select_active()` in `optimizers/cognitune_v2.py` scores each knob with:

```
0.45 * gate_probability
+ 0.20 * priority_score
+ 0.25 * normalized_GP_conditional_improvement
+ 0.10 * expert_affinity
```

Then:

- `shared_buffers`, `work_mem`, `maintenance_work_mem` are mandatory;
- `effective_cache_size` is forced into ACTIVE every **10 rounds** (plus
  `coverage_probe` candidates every round);
- memory group keeps at least 3 knobs, storage/WAL group at least 2;
- total is capped at 18 (allowed 15–20), with dynamic swaps.

Component check on the latest synthetic run:
18 ACTIVE every round, 24 unique ACTIVE sets over 30 rounds, core memory knobs
100%, `effective_cache_size` 23.3%, no missing core memory knob.

### 2.3 Candidate pool (multi-agent recommender)

`_build_pool()` uses:

| share | source | behavior |
|---:|---|---|
| 35% | `expert_global:{expert}` / `expert_local:{expert}` | half broad expert proposal over ACTIVE; half sparse 1–4 knobs in a trust region |
| 10% | `relation_block` | 1–2 knobs from a coordination block, trust-region bounded |
| 25% | `local_perturbation` | 1–2 ACTIVE knobs, trust-region bounded |
| 5% | `active_iid` | IID over all ACTIVE |
| 10% | `gate_probe` | one inactive knob, trust-region bounded |
| 15% | `one_knob_expert` | one ACTIVE knob from expert mixture |
| extra | `coverage_probe` | `effective_cache_size`, `wal_buffers`, `huge_pages` |

Trust region: numeric values within ±25% of the incumbent range (log knobs
×/÷2). This fix was essential: unbounded local/gate candidates dropped real
TPC-C from 381 to 44 TPS in a probe.

### 2.4 GP critic and acquisition

`GPPerformanceSurrogate` (BoTorch `SingleTaskGP`, Matérn 5/2, ARD,
Standardize, LogEI) is shared by CogniTune and GP-BO. Candidate selection is
primarily **LogEI**. Structural entropy bonuses (gate/expert) are added only
for candidates already in the top EI quartile (weight 0.02). Do not let
relation/gate bonuses override EI again; that was a real failure mode.

### 2.5 Expert routing / credit

Every selected candidate has a source tag. `expert_global:{name}` and
`expert_local:{name}` credit that expert with +0.08 on success / -0.05 on
failure; all experts also get a small likelihood-based update. Weights are
normalized. Router re-queries the top-2 experts every 10 rounds.

## 3. Code map

| Path | Purpose |
|---|---|
| `src/cognitune/optimizers/cognitune_v2.py` | **current method** |
| `src/cognitune/optimizers/mctuner.py` | MCTuner-style baseline |
| `src/cognitune/agents/{expert,ownership,ensemble,coordinator}.py` | four-expert prompt/ownership/routing |
| `src/cognitune/evaluator/postgres_benchbase.py` | BenchBase + Goodput + restart recovery |
| `src/cognitune/experiment/runner.py` | method dispatch, final validation |
| `scripts/run_final_compare.py` | run only v2 and MCTuner against existing results |
| `scripts/check_components.py` | per-round ACTIVE/gate/expert/source report |
| `scripts/compare_knob_directions.py` | knob-direction/range comparison |
| `tests/test_v2_optimizer.py` | v2/MCTuner regression tests |
| `docs/cognitune_paper_draft.md` | paper draft |
| `docs/investigation_2026-10-25.md` | prior incident report |

## 4. Current results and data

### 4.1 Real TPC-C (existing baselines, Goodput, 50 trials, seed 0)

See `docs/results/README.md`. Best: cognitune legacy 504.3, dbtune 504.3,
gp_bo 474.5, ottertune 412.4, cdbtune 391.4, smac 381.4, restune 381.4
(restune only 18/50). **These are legacy shared-initial runs, not v2.**

### 4.2 Real YCSB

Best trial vs final validation collapsed for several methods:
cognitune 3554.8 → 491/364/812; smac 3730.7 → 1596/1491/1787;
dbtune 3816.5 → 568/675/3419. Restune/cdbtune were stable.
This is why the next protocol must use 3 final validations/median and
measurement-drift checks.

### 4.3 Synthetic final smoke (not paper evidence)

- `cognitune_v2` final code: 2,645,111 (30 trials, seed 0)
- `mctuner`: 2,106,289 (30 trials, seed 0)
- Earlier final-candidate variants reached 4.03M on seed 0 but lost robustness
  when `effective_cache_size` was forced every round; current code uses
  periodic coverage and is the committed version.

## 5. Environment

Real harness (not in git):

- PostgreSQL 14.9, `127.0.0.1:55432`, user `benchadmin`;
- data dir `/home/pgbaseline/pgdata-14.9`;
- BenchBase `/root/GPTuner/benchbase/target/benchbase-postgres/benchbase.jar`;
- legacy controller `/root/GPTuner/baselines_random`;
- credentials in `.runtime/cognitune.env` (never commit);
- profiles in `artifacts/profiles/` (tracked).

### Current host blocker

- disk 82% full; `dd ... conv=fsync` ~1.6 MB/s;
- after immediate shutdown, `postgres: startup` enters `D` for >5 min;
- `kill -9` cannot kill `D`; shared memory remains;
- `pg_isready` rejects connections.

**Required fix before any real run:** reboot the host or restore the storage /
free disk space. A code change cannot fix a kernel `D` state.

## 6. Resume in a new container

The chat session itself is not portable. The repository plus this file is the
handoff. To continue with a fresh agent:

```bash
git clone https://github.com/262352/Entropy-Guided-Value-of-Information-Allocation-for-LLM-Assisted-Database-Tuning.git
cd Entropy-Guided-Value-of-Information-Allocation-for-LLM-Assisted-Database-Tuning
python3.10 -m venv .venv
.venv/bin/pip install -e '.[test]'
cp configs/env.example .runtime/cognitune.env   # fill API key + PG password
.venv/bin/pytest -q
```

Then open the workspace in DeepSeek Harness / your coding agent and say:

> Read AGENTS.md and HANDOFF.md. Check PostgreSQL health and run pytest. Continue
> from "Next actions". Do not rerun old baselines, do not commit artifacts or
> secrets, and do not start a 50-round real run before a clean 12-round probe.

If the new container lacks `/root/GPTuner` and the PostgreSQL templates, only
the synthetic evaluator works. Re-provision the external BenchBase/PostgreSQL
harness or copy the old host assets manually; do not commit them.

## 7. Next actions (priority order)

### P0 — recover the host

Reboot or free/restore storage. Verify:

```bash
pg_isready -h 127.0.0.1 -p 55432
```

### P1 — 12-round TPC-C probe (v2 first)

```bash
cd <repo>
rm -rf artifacts/runs/postgres_tpcc_compare/cognitune_v2
setsid nohup bash -c '
  set -a; source .runtime/cognitune.env; set +a
  export PYTHONPATH=src
  exec .venv/bin/python -c "
from cognitune.experiment.runner import run_method
run_method(\"/tmp/tpcc_probe_v2.yaml\", \"cognitune_v2\", 0, \"artifacts\")
"' > artifacts/v2_probe.log 2>&1 &
```

`/tmp/tpcc_probe_v2.yaml` may not exist in a new container; generate from
`artifacts/suite/tpcc.yaml` with budget 12, n_init 5, final_validation_repeats 1.
Then:

```bash
.venv/bin/python scripts/check_components.py \
  --run-dir artifacts/runs/postgres_tpcc_compare/cognitune_v2/seed_0 \
  --output artifacts/runs/postgres_tpcc_compare/cognitune_v2/seed_0/component_report.json
```

Pass criteria: 18 ACTIVE, core memory present, no CRASH, and best > 381.4
(the old shared-initial TPC-C best) within 12 rounds. If not, do not run 50.

### P2 — 12-round MCTuner probe and knob comparison

```bash
.venv/bin/python -c '
from cognitune.experiment.runner import run_method
run_method("/tmp/tpcc_probe_mctuner.yaml", "mctuner", 0, "artifacts")
'
.venv/bin/python scripts/compare_knob_directions.py \
  --ours artifacts/runs/postgres_tpcc_compare/cognitune_v2/seed_0 \
  --baseline artifacts/runs/postgres_tpcc_compare/mctuner/seed_0 \
  --output artifacts/analysis/v2_vs_mctuner_knobs.csv
```

### P3 — full comparison only for v2 and MCTuner

The agreed protocol: existing six baselines are **not** rerun; only
`cognitune_v2` and `mctuner`, each once. If MCTuner sets a new best, use its
3 final-validation median as its true value. `run_final_compare.py` implements
this and writes `artifacts/analysis/final_compare_*.json/csv`.

### P4 — method tuning, only if P1 fails

- The main known sensitivity is ACTIVE set and candidate trust region.
- Do not re-enable unbounded local/gate proposals.
- Do not let relation/gate bonuses outweigh LogEI.
- If v2 still loses on real TPC-C, compare active sets and top changed knobs
  against MCTuner with `compare_knob_directions.py`; tune group quotas and
  candidate ratios, not the GP class.
- Full one-step lookahead / expected decision-entropy reduction is **not**
  implemented yet; it is the next research increment after the probe is clean.

## 8. Open decisions / risks

- **Comparability**: existing baselines used shared initial points; running v2
  and MCTuner with a new independent-start protocol would make old results
  incomparable. Current decision is shared starts, only v2/MCTuner rerun.
- **Goodput**: TPC-C must use Goodput, not request Throughput. Goodput includes
  warm-up in its numerator; report relative improvement, not absolute TPS.
- **MCTuner port**: `mctuner.py` is a 7-expert + top-20 + GP-BO reimplementation,
  not the official code; document this in the paper.
- **Synthetic evidence is one seed**; do not claim superiority from it.
- **Effective cache size**: forcing it every round reduced synthetic best from
  ~4.0M to ~1.25M; current code forces it every 10 rounds plus probes. This is
  a deliberate accuracy/coverage tradeoff.

## 9. Git

Remote:

```text
https://github.com/262352/Entropy-Guided-Value-of-Information-Allocation-for-LLM-Assisted-Database-Tuning.git
```

Commit only source, configs, prompts, tests, docs, profiles and small result
snapshots. `.gitignore` excludes `.venv/`, `.runtime/`, `artifacts/`,
`results/`, logs and secrets. Large raw measurements and the Twitter archive
remain on the old host under `artifacts/real_measurements/` and
`results/archive/`; they are not in git.
