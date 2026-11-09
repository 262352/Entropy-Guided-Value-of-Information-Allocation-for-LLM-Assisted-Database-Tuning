# 三负载七基线：各100轮原始结果

已完成YCSB、TPC-H、CH-benCHmark各七基线，共21组、2100条优化观测，全部SUCCESS；每组另有3次最佳配置复测及3次默认漂移检查。

| 文件 | 内容 |
| --- | --- |
| ycsb_100_rounds_tps.csv | YCSB每轮成功请求吞吐量（requests/s，按TPS展示），100行×7方法 |
| tpch_100_rounds_latency_ms.csv | TPC-H每轮合并成功分析查询p95延迟（ms），越低越好 |
| chbenchmark_100_rounds_tps.csv | CH每轮事务吞吐量（TPS），越高越好 |
| chbenchmark_100_rounds_analytical_p95_ms.csv | CH每轮合并成功分析查询p95延迟（ms） |
| baseline_100_rounds_long.csv | 全部2100条观测、状态、阶段、时间戳、来源及有效性标记 |
| baseline_validation_summary.csv | 21组默认值、最佳配置指标及轮次、3次复测值/状态/中位数、漂移检查 |
| baseline_100_rounds_manifest.json | 实验协议、单位、方向、原始源文件与导出文件SHA256 |

round为1–100，对应原日志trial_index 0–99。前10轮为共享IID初始化，后90轮为自适应调优。每轮表保留原始观测，没有累计最佳替换、平滑或插值，不混入恢复对照、健康探针与最终复测。CH的优化效用同时考虑TPS和分析p95；summary的best_metric对应效用最佳配置，不一定等于最高TPS配置。

方法标识：smac/gp_bo/cdbtune/restune/ottertune/mctuner_official/dot_official。经典基线存在仓库适配，具体边界见[运行说明](../RUN_REMAINING_BASELINES.md)，不能等同于全部原论文实现。

TPC-H OtterTune的默认性能漂移检查未通过（comparison_valid=false），虽然其100轮均成功，不能不加说明地用于公平排名；其他20组通过。最佳观测值与最终复测中位数须分开使用。

图目录中的Twitter/TPC-C数列是按用户要求导入的绘图示意输入，不属于此处2100条真实基线观测；图的CSV与README分别记录数据来源。不能把它们当作真实实验结果。

完整数据库镜像、日志、密钥、环境和检查点不提交。原始记录位于本机artifacts/remaining_baselines_100_v1；有原始数据的环境可运行`.venv/bin/python scripts/export_baseline_results.py`重建快照，其他checkout可直接读取本目录CSV和运行内嵌数据的独立绘图脚本。
