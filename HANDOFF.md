# HANDOFF — CogniTune v2

Read this before changing code. It is the state transfer for the next agent.

## 最新更新：2026-11-09（Git结果归档与发布）

- 用户授权提交并推送实验结果及有用更新。已先快进合入origin/main的两条Codex代理配置提交，
  不覆盖远端历史；将提交现有调优/恢复/上下文缓存/消融/基线/巡检代码、测试与文档。
- scripts/export_baseline_results.py归档三负载21组2100条观测，补CH事务TPS和分析p95宽表，
  baseline_validation_summary.csv含最佳配置3次复测/默认漂移，manifest扩至三负载逐源SHA。
  原YCSB/TPC-H表值保留。数据README与主README已更新；大型原始日志和运行环境仍不入库。
- 用户提供的绘图输入归档为figures/supplied_throughput_series.json，绘图读取该文件而非本机
  聊天附件路径；独立内嵌数据源码仍可直接运行。来源说明与真实实验快照分别保留。
- 待提交87个文件初扫无密钥/配置密码/私钥匹配，无超过5MB文件；仅允许源码和精简结果文档。
  运行前后pytest均通过，日志.runtime/git-publish-before-tests.log和git-publish-after-tests.log。
  最终暂存87文件约2.4MB，敏感信息/禁止路径/结果SHA256检查通过。

## 最新更新：2026-11-09（四负载提升图）

- **本轮最新图面清理/源代码交付**：按用户要求移除图内Supplied series/Measured series，
  不改数据；CSV/JSON/README保留来源。新增docs/paper_revision/figures/plot_throughput_delta_2x2.py，
  独立脚本内嵌全部30条100轮曲线的精确变化点，仅依赖matplotlib，无需仓库/附件/数据文件。
  展开后3000值与原CSV逐值完全一致，已运行生成无来源英文角标的PNG/PDF。
- **用户最新改图要求已覆盖此前数据选择**：Twitter/TPC-C改用附件代码八方法各100轮数列，
  用户明确指定，不再使用历史50轮日志；YCSB/CH仍是真实七基线100轮。统一Delta百分比，
  默认分母仍为各实验默认三次中位数。图内Supplied/Measured series标来源；CSV状态SUPPLIED
  与SUCCESS区分，共3000位置，逐式核对通过。附件仅AST读取四个赋值，不执行代码。
  README/JSON说明来源，不能将提供的示意数列当作真实性能证据。PNG/PDF已更新。
- 新增三负载七基线全部21组均100轮+复测完成，summary均validated=true；仅TPC-H OtterTune
  comparison_valid=false，其余20组通过漂移检查。收尾时DOT最后一组才刚完成，不要凭旧心跳误判未完成。
- 用户确认2x2为Twitter/TPC-C/YCSB/CH-benCHmark，TPC-H保持延迟不混进吞吐量图。
  scripts/plot_throughput_delta_2x2.py输出docs/paper_revision/figures/throughput_delta_2x2 PNG/PDF/CSV/JSON，
  另含README及baseline_completion_status.json。Delta=(累计最高成功吞吐量-默认中位数)/默认中位数，百分比显示。
- 附件含手工补值，仅参考样式。Twitter用完整同组twitter_pair_50_v3双方法50轮，非stable；
  TPC-C用real_pair_50双方法50轮旧Goodput，版本分别v3/v2，图例明确historical；不混配跨天默认。
  YCSB/CH为七基线各100轮，未运行我们的方法，不画其虚构曲线。已核验1600个真实曲线位置公式。
  因此不是四负载同协议全基线100轮投稿终稿，历史/新组区别详见图README。
- 前置pytest完成；第一次后置pytest在62项位置进程SIGTERM(exit143)中断，没有测试失败报告，
  完整后置重跑及用户提供数列版后置测试均通过，日志.runtime/figure-delta-after-tests-rerun.log及
  .runtime/figure-delta-supplied-after-tests.log。未改实验程序或历史观测。

## 最新更新：2026-11-08（只读结果检查及绘图）

- **本轮最新导出/清理**：YCSB和TPC-H七基线现均100轮及复测完成；CH已开始（SMAC完成、GP-BO运行）。
  docs/paper_revision/ycsb_100_rounds_tps.csv和tpch_100_rounds_latency_ms.csv各100行×7方法；
  baseline_100_rounds_long.csv共1400条含状态/阶段/时间/来源/漂移标记，manifest保留逐源SHA256。
  所有记录SUCCESS，逐值读回核对通过；TPC-H是合并成功分析查询p95毫秒而非QPS。
  TPC-H OtterTune漂移失败标记保留，详见baseline_100_rounds_README.md。
  用户明确要求删除QA，已删除；只清理可再生缓存，不删实验/检查点/运行依赖。
- 本日较早绘图快照（非当前进度）：YCSB七方法均100轮及复测完成，TPC-H六方法完成，DOT快照89/100。
  CH-benCHmark尚未到达；监督按workload-major串行，未发现CH失败或遗漏，不另启并发。
- 新增scripts/plot_baseline_workloads.py，从原子ledger/result.json生成累计最大成功测量吞吐量，
  含10共享IID，不平滑、不外推、不混复测；输出analysis/best_throughput_current内PNG/PDF/CSV/JSON。
  TPC-H原raw_metric为p95延迟，图改读metadata.measured_goodput并标QPS；其优化目标仍为延迟。
  TPC-H OtterTune comparison_valid=false（默认漂移）保留并虚线标注；CH无数据不造曲线。
  已核验14条曲线单调及导出峰值一致。未改运行中的优化器、调度、测量协议或历史。

## 最新更新：2026-11-06（优先于下面全部历史实验限制）

- **每5小时巡检已启用（本轮最新）**：用户授权正常零模型调用、异常自动修复并断点恢复。
  `scripts/watch_baseline_queue.py`为标准库轻量常驻任务；PID3993686，工具保活exec会话5124。
  状态`.runtime/baseline-monitor/state.json`，日志`artifacts/remaining_baselines_100_v1/monitor/`。
  正常检查只读进程身份、心跳、结果进度/失败状态，model_invocations=0，等待18000秒。
  已真实执行一次正常检查，未调用模型；下次北京时间2026-11-07 02:21:04，全部完成后自动停止。
  监督缺失先无模型恢复；孤立健康worker等待其释放数据库锁后续队列；延期任务、心跳/进度异常、
  连续3次失败则暂停确认归属的监督/worker后，调用已登录的Codex CLI非交互修复（最长1h）。
  只允许基础设施/解析/恢复修复，不允许改优化算法、预算、IID、旋钮域、计量或历史，不跑我们的方法。
  ready_to_resume=true才由监控恢复队列。模型未实调用探测，异常分支以mock测试验证；
  本机Codex 0.166.0、ChatGPT登录已验证。前163项、后176项测试通过，官方调用依据见docs/BASELINE_MONITOR.md。
  本机PID1为bash、无cron/systemd服务，因此使用本机持久进程而非虚构云端定时任务。

- 用户明确授权跑其余负载全部基线，暂不跑我们的方法，并将新实验预算改为100（含10共享IID）。
  新七基线：smac/gp_bo/cdbtune/restune/ottertune/mctuner_official/dot_official；仅ycsb/tpch/chbenchmark。
  当前用户指令覆盖AGENTS旧“仅v2/MCTuner”限制；TPCC/Twitter旧记录不重跑不覆盖。
- 入口 `bash scripts/run_remaining_baselines.sh status`；恢复/启动 `... start`。
  已真实启动监督PID3969117，worker PID3969380，工具保活exec会话35102，输出
  `artifacts/remaining_baselines_100_v1/`。初次nohup异步启动未保持，现用前台工具会话托管nohup/setsid；
  用户终端入口仍支持普通后台启动，并检查共享数据库锁。不要再开并发数据库实验。
- YCSB模板已实际加载100,000行；3次默认测量1772.7554/1811.0994/1800.6236，
  中位1800.6236，CV1.104%。当前仍在12轮健康探针，七基线100轮结果尚未完成。
- 新增DurableEvaluator：请求先原子落盘，保留每次尝试；缓存前缀精确重放，候选不一致报错。
  零/非有限指标和异常触发真正的默认负载对照；对照健康后候选最多2次，重复失败才算CRASH；
  默认也坏则75延期，不把基础设施故障灌成一串零。监督轮询/冷却300秒/最多3轮，
  1小时无测量或重放进度触发worker watchdog；禁止强杀PG/覆盖数据。持久I/O故障仍需人工处理。
- 独立BenchBase jar增加同一测量窗口成功请求的逐类型计数/延迟，CH分别取TPCC事务TPS和
  所有成功分析查询的合并p95；目标为两者等权log改进。旧Twitter jar/计量不改。
- TPC-H scale0.1/1终端；CH scale1/12 TPCC+4分析；统一15+60秒。每方法3次最佳复测及3次默认漂移，
  比较有效性另标，恢复/控制真实尝试单独记账，不能把总物理执行数也称严格100。
- 经典基线沿用仓库适配；DOT官方commit ab552842d97f01364ce2bca2e7005d22cbf40f7a，
  导入原GP epoch/RFECV/LRT，PostgreSQL/共同编码/共享启动适配，关闭partial benchmarking以统一窗口。
  官方MCTuner selection新增可选context/goal，旧Twitter默认不变。ResTune快照排除同目标GP轨迹防泄漏。
- 修改前152项、修改后163项测试通过；DOT官方核心12轮合成冒烟通过，非真实性能结论。
  额外依赖见configs/requirements.baselines.txt。首次YCSB worker载入后补了未来worker启动重连guard、
  CH双指标健康检查，边界写在implementation_boundaries.jsonl；当前YCSB测量路径未变。
- 用户附件已逐字覆盖docs/paper_revision/cognitune_method.tex（整篇JointTune文档片段），SHA
  9ff41ff9721d8c56f1214ed3c1636ca9c6fcade76cdff6b04dab9256e920b263。
  原稿备份.runtime/paper-latex/cognitune_method_before_user_manuscript_20261106.tex。
  附件缺文末/参考，包含LHS及RL等与实际实现待对齐表述，按用户要求未擅改；旧方法preview不再
  能直接input这一整篇文档，不声称新稿编译过，旧PDF仅历史预览。详见docs/RUN_REMAINING_BASELINES.md。

## 最新更新：2026-11-05（优先于下面历史说明）

- **符号、采样接口与公式层次（本轮最新，仅文档）**：更新paper_revision正文/补充：
  GP统一P_t，专家提议mathcal P，采样输入编译为Q_t；Info显式返回按候选索引的估计/误差对。
  AuditStates/Fill/Reserve都接收V_t=所有已尝试配置（含失败），最终池再去重及排除V_t。
  Compile区分专家补丁种子与程序采样；Reserve定义IID参考、已审核跨域、覆盖和重入各通道，
  说明读取的锚点/作用域、覆盖/驻留/等待信息；新增RecordState保证新激活旋钮驻留计数对齐。
  最终窗口改为B=T-n0、w_end=min(B,max(1,ceil(.1B)))，50/10时4次；仅论文方法，代码仍固定6。
  用户追加要求拆开拥挤公式，已把原联合长式拆为状态机会、相对优势、准入/选择三组，
  相关量组内保留，正文现在10个显示公式。仅图片有占位，未放作者/实现待办到论文中。
  编译正文6页+补充含参考6页，无Overfull/Underfull/未定义引用，PDF和验证SHA已更新。
  前后152项测试通过；未改优化器或运行真实实验，差异只记method_notes_zh.md。

- **完整投稿方法稿（此前版本，仅文档）**：按用户要求清理paper_revision正文/补充材料中的
  作者对齐、实现现状、待提交等备注，唯一占位为红色框架图。代码差异仅在内部method_notes_zh.md。
  方法再加强为动作对应(A,R,b)：每个状态先等规模审计计算配额，再以配额加权的模式审计机会
  G=sum_h(bh/Ng)*Dh给动作评分，使b进入结构选择；并非已实现或实证优于旧算法。
  补齐Calibrated定义（执行前残差含噪声，至少8个，最近20个，1.96内覆盖>=.7）；
  sigma_MC为16外层幻想的样本标准误；模式数<=min(12,floor(Ng/bmin))且Nrem显式非负。
  四槽位动作的目标/候选/并列/模式截断规则均已确定；符号与核心/附加设置三张表完整。
  事件统计规范列清扩展/收缩/保持及每层门槛通过率，没有虚构新版触发频率或消融结果。
  总算法放Overview后，正文6页+补充含参考6页，8个显示公式，无排版溢出/未定义引用。
  PDF与validation.json已更新，旧版保存before_complete_revision；修改前后152项测试通过。
  未修改优化器、未启动真实调优或付费API；现有回归通过不代表新增论文机制已实现。

- **联合旋钮—区域方法方案（此前版本，仅文档）**：用户允许改进方法并要求实质联合决策、
  数学区域幅度/可信度及真正伪代码。已更新paper_revision正文、补充材料和中文说明：
  删除空泛F映射，改为最多4个keep/reshape/exchange/组合动作的同规模后验比较；
  换参还须胜过旧方向的最佳合法reshape。区域用后验最佳探针距离分位数给半宽/区间，
  向专家提供无条件改进概率与配对优势，专家在原有请求中审核；少于32/256改进样本不采用。
  新结构先生成新审计再分配份额；准确定位为旋钮—区域有限联合决策+条件分配，非全局三元优化。
  总算法40行赋值/调用/if伪代码，含非review保持A、空答/失败回退、EI/信息/预留试验分支。
  补信息选点的长期分配直觉及80%EI模型约束，MCTuner评分对比；移除方法中health/final repeats。
  **尚未实现**：共同评分熵初始化、联合动作、后验范围包、先审计再分配、必填专家失败回退。
  原stable不具备这些功能，旧结果不可冒充新版；没有改优化器或启动实验。成本只约束不增加
  必需评测/API调用，128点后验上限仍有待实测开销，不能保证效率/最终TPS不降低。
  正文6页+补充含参考5页，8个显示公式；预览/源码SHA已更新，旧稿另存before_joint_revision。
  无Overfull/Underfull/未定义引用；修改前后152项测试通过，数值/收敛警告仍记录。

- **方法稿复现/联合依赖修订（此前版本）**：继续修改 `docs/paper_revision/cognitune_method.tex`，
  保持一个Method、五个小节，无连续paragraph分点；正文8个显示公式，补联合状态转换、
  100=80组容量+20预留的账目、初始化熵、区域更新与共享信用。联合式是控制器组合映射，
  不虚构全局最优目标。总算法补审计冻结、选点回退、覆盖/重入优先和失败路径。
  新增 `cognitune_method_reproducibility.tex`，列符号/默认规则/专家协议/GP编码与复杂度。
  **关键差异**：旧v2计算选择熵，当前stable仅汇总focus排序，尚未收集共同评分/使用初始化熵。
  稿中熵保留四分之一验证名额是拟接入方案，有显式红色作者标记；不能用已有实验宣称有效。
  两两/完整2³消融仍是后续计划，现有CLI五个变体未改变。详见更新后的method_notes_zh.md。
  正常LNCS排版为正文5页+补充5页，无溢出或未定义引用，PDF已实际检查；
  `artifacts/paper_method_20261105/`含正文/补充/合并预览及源SHA验证记录，保留旧版PDF。
  编译入口同时包含正文和附录；需要array包，AMS包仍须先于newtxmath。仅文档变更，
  未改算法/运行真实实验/调用付费API；修改前后pytest均152项通过（有既有数值/收敛警告）。

- **论文方法稿修订（此前组织版，以上一条最新修订为准）**：用户指出旧版像实现报告、公式过多，已参照DOT/MCTuner重写
  `docs/paper_revision/cognitune_method.tex`。现在只有一个Method大章节，包含Overview、
  专家初始化、证据驱动旋钮选择、反馈驱动候选/区域更新、机会驱动预算与选点五个小节。
  概览/总算法按执行流程，详细小节按关键机制组织；生成与更新在同一机制中讲清。
  显示公式从13个减为5个，移除GP内部推导；保留框架图红色占位及区域动作简表。
  独立Overleaf入口`cognitune_method_preview.tex`，中文组织说明`method_notes_zh.md`。
  修订后相同LLNCS类/newtx字体独立编译6页，未缩字体/边距；无Overfull/Underfull或未定义引用。
  预览在`artifacts/paper_method_20261105/cognitune_method_preview.pdf`，附SHA256与版式记录。
  原预览另存`cognitune_method_before_revision.pdf`；相关说明与独立预览入口已同步。
  原贴稿amssymb在newtxmath之后会引发Bbbk重定义，预览已改为AMS包先加载并加xcolor。
  仅文档变更，未运行真实实验/LLM请求；编译工具与PDF检查依赖均隔离在.runtime。

- **完整历史/前缀缓存升级**：用户要求提高DeepSeek前缀缓存复用并给专家完整配置性能历史。
  已确认实际实验仍由`DeepSeekLLMClient`直接调用API，Harness交互界面不替代此调用链。
  bootstrap、逐轮提议及joint review均发送当前运行所有已完成评测（含IID与失败记录）；
  移除最近5/8次截断，相关历史改成完整trial编号索引，避免重复配置副本。
- 请求布局`shared_history_v1`：固定system→工作负载/全域定义→完整追加历史→当前作用域/
  专家身份/锚点/配额等动态尾部；不按全JSON字典序把anchor放到前面。无付费预热调用。
  离线49条真实观测检查中跨专家共有122124字符前缀（总125260字符），这是结构检查，
  **不是实际缓存token命中率**。实际只读API usage，无命中字段则unknown。
- 专家缓存目录新增`request_metrics.jsonl`，区分provider返回的token命中与local_response
  直接复用，校验重试计入；脚本`report_llm_cache.py <目录>`按token加权汇总。
- 新上下文影响实验条件，消融默认目录升级为`artifacts/twitter_ablation_50_v2/<variant>`，
  分析默认目录同步。上下文策略进入协议/检查点；旧短历史检查点不混合续跑。
  原v1尚无真实消融轨迹；旧数据与准备文件保持原样。命令仍是
  `bash scripts/run_twitter_ablation.sh all`。未启动真实实验或付费API。
- 本次修改前140项测试通过，修改后152项通过，覆盖完整历史含失败、bootstrap/联合评议
  上下文、跨专家/跨轮公共前缀、重试用量及本地缓存与服务端命中区分。

- 最新stable-v1的CogniTune已完成50/50，50次均SUCCESS；峰值23719.22在第40轮，
  最佳配置3次复测中位数22635.94。旧MCTuner峰值23631.03、复测23233.18；跨天
  默认基线不同，不能凭峰值+0.37%认定优于MCTuner。新配对MCTuner仅完成2/50。
- 当前没有运行中的Twitter/BenchBase进程；只读pg_isready确认55432接受连接。
- 审计：新版活动集合5种、换入8个旋钮；范围真实更新3次且全为收缩；11覆盖+3重入
  占40次自适应预算的35%；89次联合评议、32同意，但直接joint候选未被选中。
  报表在`artifacts/twitter_pair_50_stable_v1/analysis/audit_20261105/`。
- 用户要求实现消融后自行运行。本轮未启动真实实验/LLM调用，仅完成代码和离线测试。
  新入口`bash scripts/run_twitter_ablation.sh all`，五版full/fixed_knobs/fixed_regions/
  uniform_budget/nei_only；每版独立50次含10共享IID配置，加预检/复测/漂移检查。
  原MCTuner和旧结果不覆盖不重跑。新目录`artifacts/twitter_ablation_50_v1/<variant>/`。
- 完整方法保留原默认行为，非full变体身份进入协议与完整RNG检查点，禁止跨变体续跑。
  fixed_regions冻结推荐中心/宽度/离散选项，并投影显式补丁，防止隐性绕过；仍保留
  公共全域探索与覆盖。uniform_budget针对参数组分配，不代表单专家对照。
- `scripts/analyze_twitter_ablation.py`支持任意label=run目录，区分峰值与完整成功复测
  中位数，生成CSV/JSON/Markdown/PNG/PDF；旧结构事件字段缺失以N/A表示。
- 修复续跑末尾遗漏默认漂移检查：已有最佳配置复测但缺少default_after时只补齐检查；
  已有默认检查不重复测量。未修改历史实验数据。
- 五版各50轮离线路径完成，使用模拟专家、合成评估器与FastSurrogate替身，不能作为
  GP或真实Twitter性能结果。强制触发的单元测试另检验关闭范围更新和信息选择的效果。
  一次新增报表测试浮点精确相等断言失败，已改容差断言；最终140项测试通过
  （修改前121项通过），另通过Python编译和脚本语法检查。
- 详见`docs/RUN_TWITTER_ABLATION.md`；不应在消融完成前宣称某机制已有效或无效。


## 最新更新：2026-11-04（优先于下面历史说明）

- 稳定Twitter运行在第29次评测前因 `wal_sync_method` 类别区域格式中断。合法首答
  `open_datasync` 被错误的数值半宽校验拒绝，后续重试又返回空值/数字中心。
  已按类型处理区域接口及离散候选采样；合法推荐值集合进入稳定按组生成，类别中心
  不再套数值移动限幅；明确hold的空/数字中心保留锚点并记录修复，非法迁移仍重试。
  三条原始失败回复离线重放通过；28条原始观测和完整RNG检查点只读校验通过。
  用户自行执行 `bash scripts/run_twitter_stable.sh both --resume`；未代为启动真实续跑。
  原28次和修复后轨迹的代码边界详见 `docs/RUN_TWITTER_STABLE.md`。

- 上一组Twitter实验已完成：两种方法各50轮，文件在 `artifacts/twitter_pair_50_v3`。
  默认中位数15902.70 TPS；旧版CogniTune峰值23090.30，MCTuner峰值23631.03。
- 日志审计发现每轮平均替换10.4个活动旋钮；406次退出都是verify；305次退出前没有
  在当前驻留期间实际调整过；区域半宽从未改变。
- 新增可选 `search_policy='stable'`，保留legacy默认行为。新入口
  `bash scripts/run_twitter_stable.sh both`；独立输出 `artifacts/twitter_pair_50_stable_v1`。
  用户此前选择自行运行真实实验，本轮只改代码和测试，未启动新真实50轮。
- 新策略按阶段换参、有真实覆盖门槛、按参数组提议/分配候选、真实区域证据更新、末6轮利用；
  详见 `docs/RUN_TWITTER_STABLE.md`。默认活动预算为10而非18，源于40次自适应评测预算。
- 两个完整50轮合成检查验证结构更替上限及区域更新生效，但最终合成指标尚未超过legacy；
  不保证真实Twitter提升，必须按新独立协议实测。原结果、旧策略和旧检查点均不覆盖。

## 最新更新：2026-11-03（优先于下方历史状态）

用户授权实现新版方法，并亲自在终端启动 Twitter 的 50 轮对比。本轮**不代为启动真实调优**。

- 新方法入口 `cognitune_v3`，保留旧 `cognitune_v2`。实现每轮反馈专家提议、条件潜力
  动态旋钮、区域修正、等规模来源审计、qLogNEI 收益约束、含噪决策信息增益和后验候选配额。
- 命令 `bash scripts/run_twitter_compare.sh both`，协议和诊断详见
  `docs/RUN_TWITTER_COMPARE.md`。仅运行新版及官方 MCTuner 核心适配版，50 次各含10共享IID初始。
- 已有独立 PostgreSQL `/opt/cognitune-experiment/postgresql-14.9`，数据目录
  `/var/lib/cognitune-experiment/data`，端口55432；BenchBase和官方MCTuner在 `.runtime/`。
  不再依赖缺失的 `/root/GPTuner`。连接检查成功，但不能由此认定长测量无波动。
- Twitter为BenchBase生成的社交工作负载；模板需用户运行命令时加载。新实验显式统计
  同一测量窗口内的成功请求。独立补丁jar保留上游文件并保存来源、哈希。
- 此前 `artifacts/real_pair_50` 的旧v2与官方适配版TPC-C已完成。它们使用旧Goodput，
  不与新Twitter指标混合；历史日志和论文均不覆盖。
- Twitter 新版现已产生19次成功评测；已修复专家活动域混淆，以及联合patch数组格式导致的中断；
  已增加作用域投影及 `bash scripts/run_twitter_compare.sh both --resume` 入口。
  原有观测保留，当前检查点对应19次完成；未代为运行续跑。旧轨迹恢复会明确记录新随机流；此后检查点保存完整程序RNG。
- 不要把合成测试当作Twitter性能结论；完整50轮效果仍待用户运行。

## 0. TL;DR（历史交接）

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
