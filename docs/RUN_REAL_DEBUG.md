# 在当前机器手动运行 50 轮真实对比

在仓库目录执行以下命令，两个方法会串行运行，每方法 50 次数据库评测，包含 10 个共享 IID 初始配置：

```bash
bash scripts/run_real_debug.sh both
```

也可以分别执行，等第一条命令结束后再运行第二条：

```bash
bash scripts/run_real_debug.sh cognitune_v2
bash scripts/run_real_debug.sh mctuner_official
```

当前环境已安装 PostgreSQL 14.9、BenchBase、私有凭据和适配器。启动脚本依赖本机 `.runtime/` 资产，不是新机器的自动部署脚本；凭据不会出现在命令行。数据库监听本机 55432 端口，使用独立的 `cognitune-pg` 系统账户。两个方法共用数据库实例，运行锁会拒绝同时调优。

## 每轮过程和输出

- PostgreSQL 停止、应用配置、启动、持久化检查、从模板恢复数据库、BenchBase 开始/完成。
- 15 秒预热和 60 秒测量，TPC-C 10 仓库、16 客户端，每次恢复同一模板。
- 我们方法在数据库试验**开始前**输出 proposal：incumbent、ACTIVE 增减、60 参数 gate 更新、专家权重、GP 条件改进、候选来源数量、实际改动参数、GP 均值/标准差、LogEI 排名、结构熵加分及候选前十名。
- 试验完成后输出 outcome：实际指标、utility、相对 incumbent 的增益、预测误差、新最佳标记、专家权重更新、耗时。
- 额外输出 `extra_knobs_if_attributed_vs_previous_trial`，检查当前代码用“上一次试验”归因与实际“从 incumbent 生成候选”之间的差别。只记录，不改归因算法。
- 专家重新询问时输出 router_refresh_start/done，包括专家名单、累计调用和 token 数。
- 前 10 次为共享初始化，尚无 GP 候选排名；后 40 次才是优化决策。每种方法另有 3 次最佳配置复测及 3 次默认漂移复测，不计入 50 次预算。首次运行还测 3 次默认配置。

## 输出位置

```text
artifacts/real_pair_50/
  console/                         # tee 保存的终端完整日志
  protocol.json                    # 预算、模型、控制条件、初始化指纹
  initial_design/                  # 两边共享的 10 个初始配置
  runs/cognitune_v2/seed_0/
    debug_events.jsonl             # 逐事件完整变量，适合脚本解析
    round_diagnostics.jsonl        # 每轮 gate、ACTIVE、专家、候选等
    trials.jsonl                   # 逐轮参数、指标、预测和来源
    summary.json
    final_validation.json
  runs/mctuner_official/seed_0/     # 官方核心适配基线的轨迹和树诊断
  measurements/                    # BenchBase 原始日志、CSV、summary
  analysis/                        # 两边完成后自动生成比较图和报告
```

查看进展：

```bash
tail -f artifacts/real_pair_50/console/*.log
tail -n 1 artifacts/real_pair_50/runs/cognitune_v2/seed_0/round_diagnostics.jsonl
```

## 本次探索记录的限制

用户明确选择在默认配置不稳定的机器上继续，因此脚本显式设置 `--allow-unstable-baseline`：默认配置变异或漂移过大时打印警告并记录，性能结果只能用于探索。原来的三次默认 Goodput 为 167.83、158.23、111.87（CV=20.5%）。新的 50 轮目录会重新测默认配置，不覆盖旧检查。前 12 轮若发生 CRASH，仍按仓库健康门槛停止；修复后应检查已保存轨迹再决定如何继续。

当前上游 BenchBase `Goodput` 把预热阶段成功事务计入分子，分母却只有测量时长，所以它不等于严格测量窗口的成功 TPS，可能高于测量窗口 Throughput。为保持当前仓库实验口径，优化仍使用这个原始 Goodput；每轮同时打印并保存原始 Throughput、延迟及所有 summary 字段。该计量问题没有在本次偷偷改为另一个目标。

正常关闭会等待 PostgreSQL 完成 shutdown checkpoint。`pg_ctl` 初次等待 15 秒后，脚本继续等待同一个 postmaster，默认总上限 600 秒，每 15 秒打印 `[DB WAIT]`。超过总上限会保存 `INFRA_ERROR` 并停止，不会强杀或在旧进程仍存活时修改配置。可以在启动命令前设置 `COGNITUNE_PG_STOP_TIMEOUT=900` 调整总上限；这不能修复持续的底层 I/O 阻塞。

如果在三次默认配置检查期间中断，再执行原命令即可续测：匹配配置和评测协议的成功测量会复用，失败或不完整的目录会改名为 `trial_XXXX.failed-attempt-时间戳-唯一后缀` 后重新测量，原始证据保留。这只恢复默认检查；优化方法的部分轨迹仍拒绝覆盖。

模板恢复的 DROP/CREATE 操作会使用独立的 `application_name` 标记其后端，每 15 秒输出 `[DB WAIT]`，包含阶段、耗时、后端 PID 和 `wait_event`，并写入 `.runtime/db_wait_events.jsonl`。默认等待上限为 900 秒，可通过 `COGNITUNE_PG_TEMPLATE_TIMEOUT` 调整。超时会请求取消该维护查询、保存 `INFRA_ERROR` 后停止；不会把它算作算法 CRASH，也不会立即再次停启整库。底层 I/O 阻塞可能让取消请求延迟生效，重跑前应确认维护查询已退出。

2026-11-02 的失败日志显示正常停库曾耗时约 383 秒，模板复制超过 240 秒；PostgreSQL 14.9 的 CREATE DATABASE 包含强制检查点及文件复制。当前数据位于共享磁盘 `/dev/sdb1` 上的 Docker overlay，可用空间约 206 GB。较长超时仅提供等待与诊断能力；若持续出现数分钟的关闭或恢复，应将数据目录迁至独立、稳定的持久化磁盘后再评估算法。直接改为内存盘、关闭 fsync 或跳过模板恢复会改变实验条件，当前脚本没有实施这些变更。

`mctuner_official` 是官方专家选参和 MCTS/UCT/GP-EI 核心的适配版本，修复硬编码路径、提示词花括号及外层最小化/内层最大化的接口冲突。共享 IID 初始化、归一化坐标和类别索引是明确的适配差异，不应声称完全复现论文原版。它不是仓库原来的简化 `mctuner` 基线。

已生成的初始 LLM 建议会从 `artifacts/real_pair_20261102` 复用，动态 router 仍按原算法调用同一 `deepseek-flash`。本轮不修改选参、奖励归因或候选排序算法。中途退出会保留结果；已有完整方法会跳过，已有部分轨迹会拒绝覆盖，不支持从中断轮自动恢复。需要重跑时先保留旧目录，再修改脚本的 RUN_DIR 使用新目录。
