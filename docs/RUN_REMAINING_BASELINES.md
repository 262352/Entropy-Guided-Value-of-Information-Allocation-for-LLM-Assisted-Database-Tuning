# 其余负载的基线队列（2026-11-06）

用户授权实际启动其余负载的全部对比方法，暂不运行我们的方法。新预算为100次，包含10个
共享IID初始点和90次优化；不将初始化隐藏后再额外运行100次。TPCC/Twitter旧结果不覆盖。

## 范围与入口

工作负载：YCSB、TPC-H、CH-benCHmark。
方法：SMAC、GP-BO、CDBTune适配版、ResTune适配版、OtterTune适配版、官方MCTuner核心适配、
官方DOT核心适配。DBTune不在本次贴稿的七个基线清单中。程序choices中没有CogniTune/JointTune。

```bash
bash scripts/run_remaining_baselines.sh status
```

启动或断开终端后恢复队列：

```bash
bash scripts/run_remaining_baselines.sh start
```

同一端口的所有实验使用 `.runtime/real-comparison.lock`；已有队列持有锁时，另一个队列会拒绝进入，
不要并发运行Twitter/消融/其他数据库调优。输出为 `artifacts/remaining_baselines_100_v1/`。
本次客户端的监督进程还由工具执行会话保活，实际PID记在supervisor.json；用户终端可用上面的nohup入口。

## 测量协议

- PostgreSQL 14.9，独立端口55432；实际已有安装取代旧说明中的/root/GPTuner路径。
- 60个合法诊断旋钮域；直接复用Twitter stable已持久化的10个共享IID配置，按负载重新测量。
- 每次15秒预热、60秒测量；每轮从该负载独立模板恢复。配置、测量、失败、恢复和源码来源分别保存。
- YCSB scale=100（100,000行）、16 terminals，沿BenchBase样例的50/5/15/10/10/10混合。
- TPC-H scale=0.1、1 terminal、22类查询串行循环；优化同一测量窗口成功分析查询的p95延迟。
- CH-benCHmark scale=1，TPCC侧12 terminals、分析侧4；同一数据库并发执行两类负载。
  分开记录窗口内事务TPS及所有成功分析请求合并后的p95，不使用“各查询p95的最大值”。
  优化目标统一为0.5*log(TPS/defaultTPS)+0.5*log(defaultP95/P95)，全部方法使用相同权重。
- 独立BenchBase jar增加成功请求按事务类型的计数与延迟；旧jar/source/历史Twitter计量不变。
- 开始前3次默认测量（CV<=15%）、12轮默认健康探针；共享初始点只测一次，供各方法复用。
- 各方法最佳配置3次复测，之后3次默认漂移检查；漂移超过20%标记comparison_valid=false，
  不把该结果当稳定胜出。复测与诊断均保留，不能挑最高峰替代复测。

100是优化配置槽位，含确认为候选失败的轮次。基础设施重试、默认对照、预检和复测是额外真实
执行，全部单独记账；不能在论文中把其总物理执行数也写成“严格100次”。共享初始测量复用
亦明确标记。总耗时、恢复次数和额外执行开销应与曲线一起报告。

## 恢复与续跑

`DurableEvaluator` 在真实执行前原子保存候选，成功或确认为候选失败后再原子保存结果。
进程中断后基线从固定种子重建优化器状态，已完成前缀直接读缓存，不重做数据库测量。
重放若提出不同配置，会报ReplayDivergence并保留原始记录，避免偷偷续成另一条轨迹。
每次重建输出到独立sessions目录；权威结果在ledger目录，未完成请求与所有attempt目录保留。

非正/非有限指标、断连、超时、配置无法启动等都会保存原始错误。随后恢复默认配置并真正
执行同一负载作对照；仅SELECT 1成功不能认定负载恢复。对照正常后同一候选最多试两次；
若它仍失败，记为候选失败并消耗一个配置槽位。默认负载也失败则暂停该worker并返回75，
不把基础设施异常喂成一串零值训练模型。

监督进程会继续其他基线、冷却300秒后再处理延期任务，最多3轮队列扫描。单任务1小时无完成
测量/重放进展会触发watchdog，终止该worker、保存现场并延期。只终止自己的benchmark/worker，
不强杀数据库，不在未确认旧postmaster退出时启动另一个实例。

永久I/O故障、磁盘耗尽、持续认证失败和确定性算法错误不能靠无限重试修好；三轮后仍失败的
任务保持pending/DEFERRED，需要依据日志修正后恢复，不能把零值或模拟数据当完成结果。

## 适配边界与来源

- GP-BO/SMAC/CDBTune/OtterTune/ResTune沿用当前仓库实现；OtterTune用ARD重要性而非原FA/工作负载映射，
  CDBTune为在线DDPG适配，ResTune为加权源GP适配。不可据此声称完整复现了原论文所有模块。
- ResTune不读取当前目标工作负载GP-BO的未来结果；只快照其他历史任务，首次快照后固定，避免续跑改变来源。
- MCTuner：官方提示、排名、空间树、分类器和GP/EI核心。负载描述和目标已参数化，不能把Twitter的推荐挪给TPC-H。
- DOT：导入官方GP epoch、RFECV、TwoActionLRT函数；替换MySQL/SSH执行与混合旋钮编码，采用共享初始历史。
  为统一测量窗口未启用官方partial/adaptive benchmarking；这是官方核心适配，不是原生MySQL端到端复现。
- 各方法版本、源码会话、数据模板和jar来源都保存在输出目录；原始模型接口不因某个方法效果差而偷偷换成随机搜索。

官方DOT仓库：https://github.com/Orange-OpenSource/DOT
官方MCTuner来源见 `.runtime/MCTuner-official` 的远端与commit记录。

## 论文导入

用户附件已逐字覆盖 `docs/paper_revision/cognitune_method.tex`，旧文件备份在
`.runtime/paper-latex/cognitune_method_before_user_manuscript_20261106.tex`。
新文件是JointTune整篇文档的开头至实验设置片段，不再是可供旧preview直接input的方法片段。
本次不自动改写用户原文；原文的LHS/100次、RL不训练、DBTune使用LLM等表述与当前实际实现
存在需要后续对齐之处，不能仅靠覆盖论文就把实验称为该协议的完整复现。
附件末尾没有end{document}/完整参考文献，本轮未声称它已独立编译通过；旧方法PDF是历史预览。

## 验证

新增测试覆盖：持久化重放和配置一致性、断连后真实默认对照、重复零TPS分类、坏对照暂停、
共享初始校验、非有限效用隔离、HTAP两种指标分离、禁止运行我们的方法、上一worker断连后的启动恢复。
DOT官方核心另做12轮合成冒烟检查，未用真实数据库/LLM，不能作为性能结论。
