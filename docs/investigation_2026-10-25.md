# 2026-10-25 TPC-C 异常与 DBTune 崩溃排查

## 1. 结论摘要

1. **SMAC 的 1000+ 不是真实吞吐提升。** BenchBase 的
   `Throughput (requests/second)` 统计所有 measured requests，包括
   SERIALIZABLE 下被拒绝/重试的 TPC-C 事务。SMAC 的高分配置把
   `max_connections`、锁表、`commit_delay`、`work_mem` 等推到极端，
   制造了 52 万次 server retry；同一配置的 `Goodput` 只有 211.6。
2. **按 Goodput 重排后所有方法同一量级**：CDBTune 407.2、GP-BO 390.2、
   SMAC 390.3、CogniTune 380.6、OtterTune 376.4、DBTune 376.4（默认配置
   152.3）。CogniTune 并不存在“比 SMAC 差 3 倍”的问题；它此前被错误的
   Throughput 指标掩盖了真实提升（Goodput 152→380）。
3. **DBTune 的全零不是 DBTune 算法本身。** 第 10 轮候选触发的 PostgreSQL
   fast shutdown 卡住超过 60 秒，legacy restart 脚本报 `server does not
   shut down`，recovery 与仍在关闭的 postmaster 竞争，随后所有轮次都在
   `write_candidate` 阶段收到 `Connection refused`。前 10 轮 TPS 为
   20–356，并非全零。
4. 已修复指标、状态管理、候选池、DB 恢复、knob 落盘、方法顺序等问题；
   43 个单元测试和 7 方法 synthetic smoke 均通过。

## 2. 关键数据（原 50 轮 TPC-C，seed 0）

| 方法 | 最佳 Goodput | 对应轮次 | 最佳请求 Throughput | 该轮 Goodput | 该轮 server retry |
|---|---:|---:|---:|---:|---:|
| CDBTune | 407.2 | 25 | 515.5 | 133.0 | 185,881 |
| GP-BO | 390.2 | 13 | 1862.0 | 310.5 | 710,775 |
| SMAC | 390.3 | 9 | 1166.9 | 211.6 | 523,660 |
| CogniTune | 380.6 | 35 | 356.8 | 152.3 | 229,900 |
| OtterTune | 376.4 | 3 | 364.7 | 144.6 | 232,487 |
| DBTune | 376.4 | 3 | 356.8 | 152.3 | 229,900 |
| default | 152.3 | 0 | 356.8 | 152.3 | 229,900 |

修复后现场验证（默认配置 vs SMAC trial 30 配置）：

| 配置 | 请求 Throughput | Goodput |
|---|---:|---:|
| default | 88.1 | 160.2 |
| SMAC best-throughput config | 789.1 | 193.9 |

即请求吞吐 9 倍，但成功事务吞吐仅 +21%。

### SMAC 最佳 Throughput 配置的主要改动

`max_connections 100→504`、`max_locks_per_transaction 64→531`、
`max_pred_locks_per_transaction 64→705`、`commit_delay 0→6168µs`、
`commit_siblings 5→44`、`work_mem 4MB→610MB`、
`maintenance_work_mem 64MB→616MB`、`checkpoint_timeout 300→3182s`、
`max_wal_size 1GB→15.7GB`、`effective_io_concurrency 1→203`、
`autovacuum_vacuum_cost_delay 2→94ms`、`wal_compression off→on`。
这些改动没有带来 3 倍 Goodput；它们主要增加了可并发的冲突/重试事务。

## 3. CogniTune 的问题与修改

### 3.1 三态管理退化为“全部 ACTIVE”

- 现象：TPC-C 50 轮结束时 60 个旋钮全部 ACTIVE，`active_knob_budget=12`
  完全失效。
- 原因：`uncertainty = selection_entropy + value_entropy`（0–2）与阈值
  0.85 比较，几乎所有旋钮都被提升为 ACTIVE；同时所有 memory 类别旋钮被
  当作 protected，绕过 active cap。
- 修改：
  - uncertainty 归一化为 `0.5*H_sel + 0.5*H_val`；
  - 高不确定性只把 DORMANT 提升为 WATCH，不再直接提升 ACTIVE；
  - protected 仅包含 hard-constraint 参与旋钮与 activation parent；
  - active cap 为 `exploit_budget + challenge_budget`，并计入 protected；
  - challenge 旋钮在前 3 个 challenge interval 内保留，避免立即被 cap 降级；
  - 高不确定性旋钮不会被直接降为 DORMANT。

### 3.2 `active_global_random` 不是 IID

- 原实现是 `local_mutations(..., max_changes=len(active))`，即随机子集
  变异，不是规范要求的 ACTIVE 空间 IID 采样。
- 已替换为 `_active_iid_samples()`：`sample_iid` 后在 ACTIVE 集合上覆盖，
  WATCH/DORMANT 固定为 incumbent。

### 3.3 Challenge 从未真正发生

- 原 `challenge_due` 只对 DORMANT 生效，而状态机从未产生 DORMANT，
  所以 challenge 池为空。
- 现在 WATCH/DORMANT 在 `trial % challenge_interval == 0` 时均标记 due；
  challenge reserve 保证高 H_sel 旋钮早期进入主搜索空间。

### 3.4 指标错误

- `PostgreSQLBenchBaseEvaluator` 原来取 `Throughput (requests/second)`。
- 现在优先取 `Goodput (requests/second)`，同时把 `Throughput`、
  `Measured Requests` 写入 metadata，便于诊断重试放大。
- 旧 shared-initial 缓存没有 `metric` 字段，会自动失效并重新测量，
  避免 default metric 混用两种指标。

## 4. DBTune 全零的根因与修复

时间线（trial 10）：

1. 候选配置写入成功，开始 `pg_ctl restart -m fast -w`；
2. fast shutdown 卡住约 4 分钟，legacy 脚本 60 秒后报
   `pg_ctl: server does not shut down`；
3. `restart_or_recover` 触发 recovery，脚本移走 `postgresql.auto.conf` 后
   重启，但旧 postmaster 仍在关闭，启动失败；
4. 此后每个 trial 在 `write_candidate` 连接时得到 `Connection refused`，
   全部记为 CRASH/TPS=0；最后 `restore_default` 也失败，导致整个 suite 中断。

修复（`evaluator/postgres_benchbase.py`）：

- `_ensure_ready()`：连接失败时先走正常 recover，再走 `pg_ctl stop -m
  immediate -w -t 30` 强制停库 + 空 auto.conf recover；
- 每次 `prepare/evaluate` 前确保数据库可达；
- `restore_default()` 不再向外抛异常，避免单个方法失败中断整个 suite；
- DBTune 候选池改为 70% incumbent 局部扰动 + 30% 全局随机，贴合其
  `local_random` acquisition optimizer，避免早期 RF 外推导致的极端配置。

## 5. 每轮旋钮持久化

每个 run 目录新增：

- `knobs.jsonl`：每轮 `trial/status/raw_metric/utility/best_so_far/
  candidate_source/config`；
- `knobs.csv`：每轮一行，每个旋钮一列，可直接做特征/相关性分析。

即使进程中途崩溃，`knobs.jsonl` 也已逐轮回写。

## 6. 与原始方案文档的冲突（需要作者确认）

1. **方法数量冲突**：方案第 2 节要求第一阶段只做 CogniTune/SMAC/GP-BO，
   并明确禁止 OtterTune、强化学习；但当前仓库按后续要求实现了 7 方法，
   CDBTune 正是 DDPG（RL）。建议论文主表保留 3 核心方法，7 方法作为
   扩展实验，或正式修订方案。
2. **seed 数冲突**：方案默认 5 seeds；当前 suite 为 1 seed（此前的选择）。
   最终论文结果建议至少 3 seeds。
3. **指标定义冲突**：方案写 `utility=log(throughput/default)`；TPC-C
   SERIALIZABLE 必须用 Goodput（或 tpmC），否则会复现本次 SMAC 假高分。
4. **Goodput 的 warm-up 偏差**：BenchBase Goodput 的分子包含 warm-up
   完成事务、分母是 measured 时间，绝对值会偏高约 warmup/measure 比例；
   各方法同协议可比，但不能把绝对值当作稳态 TPS 写进论文。
5. **基线复测规则不公平**：只对基线“刷新最优”复测两次并取均值，会给
   基线额外 1 次数据库观测信息，而 CogniTune 没有。建议三种处理：
   所有方法同规则；或明确这是“测量降噪协议”，不计入 50 轮且单独报告；
   或只对最终 best config 做统一 final validation。
6. **schema 命名**：方案写 `priority`，DeepSeek 客户端输出
   `priority_by_knob` 后在 `_normalize_response` 转换为 `priority`；
   文档需注明这一层规范化。

## 7. 修改文件

- `src/cognitune/evaluator/postgres_benchbase.py`：Goodput 指标、硬恢复、
  旧缓存失效。
- `src/cognitune/state/state_manager.py`：active cap、protected、challenge
  reserve、不确定性降级策略。
- `src/cognitune/optimizers/cognitune.py`：归一化 uncertainty、传入
  priority、protected 来源。
- `src/cognitune/candidates/cognitune.py`：ACTIVE IID 候选。
- `src/cognitune/optimizers/dbtune.py`：local-first 候选池。
- `src/cognitune/experiment/recorder.py`：knobs.jsonl / knobs.csv。
- `scripts/run_suite.py`：按 seed 打乱方法顺序、支持 `--force`。
- `scripts/run_suite.sh`：透传参数。
- `scripts/analyze_runs.py`：Goodput/重试/配置差异分析。
- `tests/test_postgres_metrics.py`、`tests/test_recorder_knobs.py`、
  `tests/test_state_manager.py`：新增回归测试。

## 8. 测试与 smoke

- `pytest -q`：43 passed。
- `run_synthetic.py --methods` 7 方法、budget 10/12、seed 0：全部完成，
  生成 `knobs.jsonl`/`knobs.csv`。
- TPC-C 现场验证：default Goodput 160.2；SMAC 高吞吐配置 Goodput 193.9。

## 9. 复跑命令

旧 TPC-C/Twitter 结果已分别归档到：

- `results/archive/twitter_compare_20261024/`
- `results/archive/tpcc_partial_20261025/`

因为旧 run 目录仍有 `summary.json` 与 `final_validation.json`，新 suite 会
跳过它们；必须加 `--force`（旧的 shared-initial 会因缺少 `metric` 字段
自动重新测量）：

```bash
cd /root/CogniTune && nohup bash scripts/run_suite.sh --force > artifacts/compare_goodput.log 2>&1 &
```

建议先用 3 个核心方法 + 1 seed + 较小 budget 做一次 TPC-C 校验，再启动
完整 7 方法 × 3 负载 × 50 轮（预计 2–3 天）。
