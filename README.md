# Entropy-Guided Value-of-Information Allocation for LLM-Assisted Database Tuning

This repository contains **CogniTune v2**, a database knob-tuning system that uses
four domain LLM experts as a proposal policy, a Gaussian-process critic, and an
entropy/value-of-information (VoI) controller for expert routing and knob gating.

The central idea is:

> LLM experts propose configurations; the multi-agent router decides which
> experts and knobs deserve budget; the GP/LogEI critic chooses the next
> experiment; entropy/VoI allocates LLM tokens and database trials.

## Status (2026-11-02)

- v2 method implemented; **45 unit tests pass**.
- Synthetic 60-knob smoke test works; component diagnostics and knob-direction
  diagnostics are included.
- Real PostgreSQL/TPC-C experiments are **blocked by storage I/O**: the
  PostgreSQL startup/checkpointer gets stuck in uninterruptible `D` state and
  `pg_isready` rejects connections. See [`HANDOFF.md`](HANDOFF.md).
- Existing TPC-C/YCSB/TPC-C result CSVs are in [`docs/results/`](docs/results/).

## Quick start (synthetic, no database)

```bash
cd <repo>
python3.10 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest -q
.venv/bin/python scripts/run_synthetic.py \
  --methods cognitune_v2 mctuner --budget 30 --seeds 0 \
  --artifacts artifacts/smoke
```

## Real PostgreSQL / BenchBase

The real evaluator reuses the external harness at `/root/GPTuner`:

- PostgreSQL 14.9 on `127.0.0.1:55432`, user `benchadmin`;
- BenchBase jar at `/root/GPTuner/benchbase/target/benchbase-postgres/benchbase.jar`;
- templates `benchbase_tpcc_template`, `benchbase_ycsb_template`,
  `benchbase_twitter_template`;
- credentials in `.runtime/cognitune.env` (never committed; see
  `configs/env.example`).

If the external harness is absent, only the synthetic evaluator is available.

## Repository map

| Path | Purpose |
|---|---|
| `src/cognitune/optimizers/cognitune_v2.py` | four-expert MoE + VoI + GP/LogEI method |
| `src/cognitune/optimizers/mctuner.py` | MCTuner-style baseline |
| `src/cognitune/agents/` | four experts, prompt ensemble, coordinator, ownership |
| `src/cognitune/state/` | ACTIVE/WATCH/DORMANT manager |
| `src/cognitune/surrogate/` | shared BoTorch GP |
| `src/cognitune/evaluator/postgres_benchbase.py` | BenchBase evaluator, Goodput, restart recovery |
| `scripts/run_final_compare.py` | run only v2 vs MCTuner against existing results |
| `scripts/check_components.py` | per-round ACTIVE/gate/expert diagnostics |
| `scripts/compare_knob_directions.py` | compare knob directions and ranges |
| `docs/HANDOFF` | full handoff for the next agent |

## Handoff

**Read [`HANDOFF.md`](HANDOFF.md) first**, then [`AGENTS.md`](AGENTS.md).
The GitHub repository is:

<https://github.com/262352/Entropy-Guided-Value-of-Information-Allocation-for-LLM-Assisted-Database-Tuning>


## 最新实验结果与运行入口（2026-11-09）

- [三负载七基线100轮结果与复测说明](docs/paper_revision/baseline_100_rounds_README.md)：YCSB/TPC-H/CH-benCHmark共2100条观测；TPC-H使用延迟，保留OtterTune漂移标记。
- [其余负载基线运行与恢复](docs/RUN_REMAINING_BASELINES.md)、[五小时巡检](docs/BASELINE_MONITOR.md)。
- [Twitter稳定实验](docs/RUN_TWITTER_STABLE.md)、[消融入口](docs/RUN_TWITTER_ABLATION.md)。
- [四图及来源说明](docs/paper_revision/figures/throughput_delta_2x2_README.md)、[独立绘图源码](docs/paper_revision/figures/plot_throughput_delta_2x2.py)。Twitter/TPC-C图线按用户提供的示意数列绘制，与真实基线观测分别标记，不是其完整100轮实测证据。
- [论文方法草稿](docs/paper_revision/cognitune_method.tex)、[方法与当前实现边界](docs/paper_revision/method_notes_zh.md)。草稿不表示全部新增机制已实现或验证。

原始大型测量、数据库、密钥及本机环境不入库，提交的是可检查的结果快照、来源哈希、源代码和说明。
