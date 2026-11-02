# CogniTune：面向数据库旋钮调优的多智能体认知先验与熵驱动搜索方法

> 论文草稿（基于当前仓库实现，2026-10-25）。当前 TPC-C/YCSB 对比实验仍在运行，本文只描述方法与工程修正，不把中间结果当作最终实验结论。

## 摘要

数据库旋钮调优面临三个困难：搜索空间高维且强约束、每次真实评测昂贵、参数之间的条件关系难以从文档中直接获得。大语言模型（LLM）可以给出旋钮重要性和推荐范围，但它不是性能代理模型，直接执行 LLM 推荐既不可靠也无法保证收敛。本文提出 CogniTune：用五个领域专家、三个提示模板构造认知样本，从优先级分歧、推荐取值分歧和条件推荐敏感性中提取三类熵信号；将硬约束、激活依赖和 LLM 软关系分离为类型化认知图；用高置信软边构造动态协同块；用 ACTIVE/WATCH/DORMANT 三态管理旋钮；在认知约束候选池上用 BoTorch GP 与 LogEI 选择下一次真实评测。CogniTune 与 GP-BO、SMAC、OtterTune、CDBTune、DBTune、ResTune 共用同一套 60 维 PostgreSQL 旋钮空间、同一 IID 初始设计、同一 BenchBase 评测器和同一 50 轮协议。针对 TPC-C SERIALIZABLE 下请求吞吐被重试放大的问题，本文明确采用 Goodput（完成事务率）作为优化目标，并把请求吞吐保留为诊断量。工程上修复了 LLM JSON 尾随数据、状态机全 ACTIVE、候选池非 IID、挑战不触发、PostgreSQL 恢复竞态和指标缓存混用等问题。当前 43 个单元测试与七方法 synthetic smoke 均通过。

## 1 引言

数据库参数调优的本质是一个带约束的黑盒优化问题：配置 $x$ 在高维空间中取值，真实性能 $f(x\mid w,e)$ 受工作负载 $w$、硬件与数据库环境 $e$ 以及测量噪声 $\epsilon$ 共同影响。传统做法依赖 DBA 经验，或者用高斯过程、随机森林、强化学习在真实数据库上反复试错。前者不可扩展，后者在 50–100 次评测预算内很难覆盖 60 维以上的空间。

LLM 提供了一种新的先验来源：它可以读手册、理解工作负载语义，并给出“哪些旋钮重要、大概取什么范围、哪些旋钮之间存在条件关系”。但把 LLM 直接当作调优器会带来三个问题。第一，LLM 没有见过当前机器的真实测量，它的推荐可能系统性偏离；第二，LLM 输出是自然语言或松散 JSON，存在幻觉、越界和格式错误；第三，LLM 对同一旋钮的判断可能自相矛盾，直接取平均会掩盖分歧。

CogniTune 的核心观点是：**LLM 只提供认知先验，性能函数必须由真实执行结果学习；熵不是性能，而是认知分歧的度量。** 具体地，本文把 LLM 输出拆成三类可计算的信号：

1. 选择熵：专家对“这个旋钮属于高/中/低优先级”的分歧；
2. 取值熵：专家对“这个旋钮应该取什么值/范围”的分歧；
3. 条件推荐敏感性：目标旋钮的推荐是否随上游旋钮的分支发生显著变化。

三类信号分别服务于旋钮筛选、搜索范围调整和跨类别联合调优。它们不直接相加成一个“重要性分数”，而是进入不同的决策环节：优先级得分决定 exploitation，熵和关系不确定性决定 challenge，GP 后验决定局部改进方向。

本文的贡献可以概括为：

- 一个不微调 LLM 的多专家认知先验采集与校验流程，包含严格 schema、缓存、重试和原始响应审计；
- 三类熵信号的形式化定义，以及“先验排序 + 分歧挑战”的旋钮选择机制；
- 硬约束、激活依赖、软关系分离的类型化认知图，以及确定性的动态协同块算法；
- ACTIVE/WATCH/DORMANT 三态管理与熵驱动的挑战预算分配；
- 与 GP-BO、SMAC、OtterTune、CDBTune、DBTune、ResTune 的统一评测协议，以及 Goodput 指标修正；
- 一套可复现的工程实现与回归测试。

## 2 方法

### 2.1 问题定义与统一评测口径

设数据库配置为 $x=(x_1,\ldots,x_d)$，真实性能为

$$
y=f(x\mid w,e)+\epsilon .
$$

所有方法最大化 utility。对吞吐目标：

$$
u(x)=\log\frac{\mathrm{Goodput}(x)}{\mathrm{Goodput}(x_{\mathrm{default}})},
$$

其中 Goodput 是 BenchBase 统计的**成功完成事务率**。对延迟目标使用 $u(x)=-\log(\mathrm{Latency}(x)/\mathrm{Latency}(x_{\mathrm{default}}))$。失败配置使用统一惩罚：

$$
u_{\mathrm{fail}}=u_{\min}-\mathrm{failure\_penalty}.
$$

这里必须强调：BenchBase 的 `Throughput (requests/second)` 统计所有 measured requests，包含 TPC-C 在 SERIALIZABLE 下被服务器拒绝后重试的事务。高并发/高锁表配置可以把请求吞吐抬高数倍，但成功事务并没有增加。本文因此以 `Goodput (requests/second)` 作为主指标，把请求吞吐、`Measured Requests` 和重试计数保留在 metadata 中用于诊断。SMAC 最小化框架接收 $\mathrm{cost}=-u$。

三种核心方法（CogniTune、SMAC、GP-BO）以及四个扩展基线（OtterTune、CDBTune、DBTune、ResTune）共享：相同搜索空间、相同硬约束、相同初始配置、相同总预算、相同数据库评测器、相同 benchmark 与 warm-up 时间、相同失败惩罚。

### 2.2 搜索空间与共享初始设计

搜索空间由 YAML 描述，支持 integer、float、categorical、ordinal、bool，支持 log scale、default、unit、category、restart_required、activation condition、forbidden combinations、description。SearchSpace 统一实现：

- `sample_iid(seed, n)`：按类型做 IID 采样；
- `encode/decode`：数值线性或对数映射到 $[0,1]$，categorical 做 one-hot，ordinal 按序数映射，bool 做二值 one-hot；条件未激活的旋钮写零向量并置 active mask 为 0；
- `repair`：裁剪越界值、处理条件激活、回退 forbidden 组合、按 hard constraint 缩放内存类参数；
- `validate`、`get_default`、`get_active_knobs`、`to_configspace`、`fingerprint`。

所有方法读取同一个持久化初始设计文件 `<dbms>_<workload>_seed_<seed>.json`：

$$
X_{\mathrm{init}}=\{x_{\mathrm{default}}\}\cup\{n_{\mathrm{init}}-1\ \text{个同 seed 的 IID 随机配置}\}.
$$

默认 $n_{\mathrm{init}}=5$，trial 0 是数据库默认配置，trial 1–4 是 IID 随机配置。项目明确禁止 LHS、Sobol、Halton、正交表和 Hammersley 等准随机设计；初始设计 metadata 写入 `initial_design: iid_random`、`lhs_used: false`、`sobol_used: false`、`halton_used: false`。三种核心方法必须读取同一个文件，不允许各自重新生成。

### 2.3 多智能体认知先验

系统使用五个专家：`memory_expert`、`concurrency_expert`、`optimizer_expert`、`execution_expert`、`storage_wal_expert`。每个专家使用三个固定模板：`mechanism_template`（数据库内部机制）、`workload_template`（当前工作负载）、`resource_template`（硬件资源与竞争）。五个专家 × 三个模板得到 15 个认知样本。同一专家的多个模板不是三个独立真实专家，而是 prompt ensemble 的认知样本；这一点在统计时通过样本权重和分歧度量体现。

每个响应必须是严格 JSON，包含：

- `expert`、`template`；
- `priority` 的 high/medium/low 划分；
- 每个旋钮的 `recommended_values`、`recommended_interval`、`confidence`、`reason`；
- `conditional_relations`，每条含 source、target、reason 和至少两个 branch；每个 branch 给出 target_values 或 target_interval。

解析器验证：每个旋钮恰好出现在一个优先级集合；不能出现空间外旋钮；推荐值必须合法；条件关系端点必须存在；branch 少于两个不计算关系敏感性；只有自然语言、没有分支的关系保存为 unverified_relation。每次调用保存原始提示、原始响应、解析结果、模型名、token 数和时间戳。没有 API Key 时 MockLLMClient 可以完成整个 synthetic 流程。

LLM 不是性能代理模型。它只提供：旋钮重要性先验、推荐范围、条件关系、候选配置和解释理由。真实性能函数由 GP 从数据库执行结果学习。

### 2.4 旋钮选择

对旋钮 $k$，统计 15 个认知样本中投 high/medium/low 的次数 $n_{k,H},n_{k,M},n_{k,L}$。使用 Dirichlet 平滑：

$$
p_{k,c}=\frac{n_{k,c}+\alpha_{\mathrm{priority}}}{J+3\alpha_{\mathrm{priority}}},\quad c\in\{H,M,L\},
$$

默认 $\alpha_{\mathrm{priority}}=0.5$。归一化选择熵：

$$
H_{\mathrm{sel}}(k)=-\frac{\sum_c p_{k,c}\log p_{k,c}}{\log 3}\in[0,1],
$$

序数优先级得分：

$$
S_{\mathrm{pri}}(k)=1.0\,p_{k,H}+0.5\,p_{k,M}+0.0\,p_{k,L}.
$$

$S_{\mathrm{pri}}$ 表示专家总体认为该旋钮有多重要；$H_{\mathrm{sel}}$ 表示专家对该旋钮优先级的分歧。二者不能简单相加成 Top-K 排序：全体 high 和全体 low 的熵都接近 0，但优先级得分分别是 1 和 0；只有 high/medium/low 均匀分布时熵才接近 1。

初始旋钮选择分为 exploitation 和 challenge 两部分，默认 active budget = 12，其中 exploit = 9、challenge = 3：

- exploitation：按 $S_{\mathrm{pri}}$ 从高到低选择；
- challenge：从高 $H_{\mathrm{sel}}$、处于重要关系中、长期未验证、可能影响协同块、GP 不确定性高的旋钮中选择。

实现上，StateManager 初始化时选 9 个 exploit 和 3 个 challenge 旋钮为 ACTIVE；challenge 旋钮在最初 3 个 challenge interval 内保留，避免被 active cap 立即降级。硬约束涉及的旋钮和 activation parent 作为 protected，始终保留在搜索空间中，但不占用 challenge 配额。

### 2.5 取值范围调整

对数值旋钮 $k$ 建立固定候选网格 $V_k=\{v_{k,1},\ldots,v_{k,M}\}$，默认 $M=16$；普通数值在线性空间建网格，log 旋钮在对数空间建网格。把第 $j$ 个样本的推荐映射为离散分布 $q_{k,j}$：

1. `recommended_values` 映射到最近网格单元；
2. `recommended_interval` 对覆盖的网格单元均匀分配概率；
3. values 与 interval 同时存在时，values 权重 0.7、interval 权重 0.3；
4. 加 Dirichlet 平滑 $\alpha_{\mathrm{value}}=0.1$；
5. 按专家 confidence 归一化得到样本权重 $\omega_j$。

加权平均分布：

$$
\bar q_k=\sum_j \omega_j q_{k,j}.
$$

取值推荐分歧：

$$
H_{\mathrm{val}}(k)=H(\bar q_k)-\sum_j \omega_j H(q_{k,j}),
$$

等价于加权 Jensen–Shannon divergence。它的用途不是判断旋钮是否重要，而是判断推荐范围是否分散：$H_{\mathrm{val}}$ 低且推荐集中时，候选采样更集中；$H_{\mathrm{val}}$ 高时，采样分布更宽。重要性仍由 $S_{\mathrm{pri}}$ 决定。

### 2.6 条件关系敏感性

对候选关系 $X\to Y$，每个 source branch $b$ 给出 $Y$ 的推荐分布 $q_Y^{(b)}$。定义：

$$
R_{X\to Y}=H\left(\sum_b \rho_b q_Y^{(b)}\right)-\sum_b \rho_b H\left(q_Y^{(b)}\right),
$$

默认 $\rho_b=1/|\mathcal B|$。它衡量的是“LLM 对 $Y$ 的推荐是否随 $X$ 的分支发生明显变化”，不是真实数据库互信息，也不能在论文中表述为因果交互。

测试性质：所有分支推荐相同 $Y$ 时 $R\approx 0$；每个分支内部确定但分支间不同时 $R>0$；缺少分支时不计算；只有自然语言关系时保存为 unverified_relation。当前实现用跨模板样本的均值和标准差估计关系强度：

$$
R_{\mathrm{lower}}=\mathrm{mean}(R)-\kappa_{\mathrm{relation}}\cdot\mathrm{std}(R),
$$

默认 $\kappa_{\mathrm{relation}}=1.0$。只有 $R_{\mathrm{lower}}\ge\mathrm{relation\_threshold}$ 的软边进入高置信认知图。严格的重采样 bootstrap 未实现，当前是解析的均值–标准差下界；这是已知实现限制。

### 2.7 类型化认知图

类型化图定义为

$$
G=(V,F_{\mathrm{hard}},E_{\mathrm{activation}},E_{\mathrm{soft}}).
$$

- $F_{\mathrm{hard}}$：硬约束因子，包括合法范围、禁止组合、总内存约束等；
- $E_{\mathrm{activation}}$：一个开关决定另一个旋钮是否生效；
- $E_{\mathrm{soft}}$：LLM 条件推荐产生的软依赖。

三类关系不能合并成普通图后统一跑 SCC。处理规则是：hard constraint 始终用于过滤候选；activation dependency 控制旋钮是否激活；soft dependency 只影响联合候选、协同块和挑战预算；soft edge 可以被真实执行结果降权或删除。

软边记录 source、target、weight、support_count、mean、std、lower_bound、status。输出 `cognitive_graph.json`、`cognitive_graph.graphml`、`coordination_blocks.json`。图中 soft edge 的 `status` 初始为 `unverified`，后续由协同块校准更新为 `retained` 或 `demoted`。

### 2.8 动态协同块

根据高置信软边构造协同块划分 $\Pi=\{B_1,\ldots,B_m\}$，优化目标：

$$
\min \sum_{(i,j)\ \mathrm{cut}} R_{\mathrm{lower}}(i,j)+\lambda_{\mathrm{block}}\sum_B C(B),
\quad |B|\le \mathrm{max\_block\_size}.
$$

默认 `max_block_size=4`。第一版使用确定性贪心：

1. 按 $R_{\mathrm{lower}}$ 从高到低排序软边；
2. 每个旋钮开始时是独立块；
3. 若合并后大小不超过 4，且边下界大于复杂度增量，则合并；
4. 记录每次合并/拒绝的决策与原因；
5. 相同输入产生相同划分。

协同块的用途是：块内旋钮联合生成候选；块间旋钮可以独立采样；对高影响但未验证的块安排联合挑战；协同块不是永久结构，会随校准结果变化。

### 2.9 旋钮三态管理

每个旋钮处于 ACTIVE、WATCH、DORMANT 之一：

- ACTIVE：进入当前主要搜索空间，可以被 GP 候选生成器改变；
- WATCH：默认固定为 incumbent，可以在 challenge 轮次被改变，保留重新激活机会；
- DORMANT：当前阶段固定为 incumbent，仅在周期性恢复检查或新关系出现时重新评估。

状态判断使用 $S_{\mathrm{pri}}$、$H_{\mathrm{sel}}$、$H_{\mathrm{val}}$、是否属于协同块、是否受 hard constraint 保护、是否受 activation dependency 保护、GP 条件改进上界、GP 后验不确定性、真实评估数量、最近 challenge 结果。

对旋钮 $k$ 的条件性能差：

$$
d_k(v)=f(x_{-k},v)-f(x_{\mathrm{incumbent}}),
$$

GP 给出 $m_{d,k}(v),s_{d,k}(v)$，乐观改进上界：

$$
\mu_+(k)=\max_v\left[m_{d,k}(v)+\beta_{\mathrm{state}}s_{d,k}(v)\right].
$$

规则：

- ACTIVE：$S_{\mathrm{pri}}$ 高，或 $\mu_+$ 高，或属于高置信协同块，或 challenge 获得明显提升；
- WATCH：$S_{\mathrm{pri}}$ 较低但 $H_{\mathrm{sel}}$ 高，或关系尚未验证，或 GP 不确定性高，或真实样本不足；
- DORMANT：只有 $S_{\mathrm{pri}}$ 低、$H_{\mathrm{sel}}$ 低、$\mu_+$ 低、GP 不确定性低、不属于受保护关系且样本足够时才进入。

所有状态可逆。实现中 active cap 为 `exploit_budget + challenge_budget`，并计入 protected 旋钮；高不确定性只把 DORMANT 提升为 WATCH，不直接把所有旋钮提升为 ACTIVE；高不确定性旋钮不会被直接降为 DORMANT。默认 `challenge_interval=5`。

### 2.10 挑战调度与协同块校准

动作 $a$ 的认知探索优先级：

$$
A(a)=\frac{U(a)D(a)}{C(a)},
$$

其中 $U(a)$ 是动作涉及的后验不确定性（$H_{\mathrm{sel}}$、$H_{\mathrm{val}}$、关系不确定性、GP 方差），$D(a)$ 是动作改变当前决策的可能性（是否改变状态、协同块、推荐范围或 incumbent），$C(a)$ 是执行成本，第一版统一设为 1。挑战排序使用 $U\cdot D/C$；熵只表示认知分歧和探索路由，不能说成“真实信息增益最大”。

对需要验证的协同块 $B$，构造 independent configuration（每个旋钮采用独立推荐后组合）与 joint configuration（协同块联合推荐），定义：

$$
\Gamma_B=f(x_{\mathrm{joint}})-f(x_{\mathrm{independent}}).
$$

预算有限时不要求每个块都成对测试，优先验证关系强、影响高、不确定性高、可能改变当前最优的块。$\Gamma_B$ 明显为正时提高软边可靠度并置 `retained`；接近零或为负时降低权重并置 `demoted`；数据不足时保持 `unverified`。$\Gamma_B$ 只解释为联合推荐相对独立组合的额外收益，不是严格因果交互。

### 2.11 GP 性能代理与 LogEI

CogniTune 与传统 GP-BO 复用同一个 GPPerformanceSurrogate：

- BoTorch `SingleTaskGP`；
- double precision；
- Matérn 5/2 kernel，ARD；
- `Standardize` outcome；
- `ExactMarginalLogLikelihood`；
- `LogExpectedImprovement`，$q=1$。

输入编码按 2.2 节：数值线性/对数映射，categorical one-hot，ordinal 序号，inactive 条件旋钮零向量 + active mask。每获得一次真实执行结果后重新拟合：

$$
f(x)\mid\mathcal D_t\sim\mathcal N(\mu_t(x),\sigma_t^2(x)),
$$

$$
x_{\mathrm{next}}=\arg\max_{x\in\mathcal C_t}\mathrm{LogEI}_t(x).
$$

CogniTune 与 GP-BO 的唯一区别是候选池 $\mathcal C_t$ 的构造：CogniTune 使用认知约束候选池，GP-BO 在完整合法空间生成 IID 候选。GP 类、编码、核函数、Standardize、LogEI 和真实预算完全相同。

### 2.12 认知约束候选池

CogniTune 不直接执行 LLM 输出的配置；LLM 推荐只是候选之一。每轮构造 $\mathcal C_t^{\mathrm{Cogni}}$，默认 `candidate_pool_size=10000`，组成比例：

1. 35% `llm_recommended`：在 LLM 推荐范围内采样，优先高 $S_{\mathrm{pri}}$ 旋钮，采样宽度受 $H_{\mathrm{val}}$ 影响；
2. 25% `local_perturbation`：围绕 incumbent 局部扰动，只改变 ACTIVE 旋钮；
3. 20% `coordination_block`：对协同块中的旋钮联合采样，使用条件推荐分支生成联合元组；
4. 10% `active_global_random`：只在 ACTIVE 空间做 IID 随机采样，WATCH/DORMANT 固定为 incumbent；
5. 10% `challenge`：改变一个 WATCH/DORMANT 旋钮，或验证一个未确认协同块。

所有候选经过 `repair()`、`validate()`、hard constraint 过滤、去重、排除已执行配置。每个候选记录来源。普通 exploitation 轮次选 LogEI 最大候选；challenge 轮次先由熵和决策影响选择挑战目标，再在对应候选中选 LogEI 最大配置。

### 2.13 整体流程

1. 读取搜索空间，生成/加载共享 IID 初始设计；
2. 调用 15 个 LLM 认知样本，解析、校验、缓存；
3. 计算 $S_{\mathrm{pri}}$、$H_{\mathrm{sel}}$、$H_{\mathrm{val}}$、关系敏感性；
4. 构建类型化认知图与协同块；
5. 初始化三态，前 $n_{\mathrm{init}}$ 轮执行共享初始配置；
6. 对 $t=n_{\mathrm{init}},\ldots,\mathrm{budget}-1$：
   - 拟合 GP；
   - 用 GP 条件改进和不确定性更新三态；
   - 构造认知约束候选池；
   - 若挑战到期，先选挑战目标；
   - 用 LogEI 选配置，执行真实数据库，记录 Goodput、状态、来源和旋钮；
   - 必要时做协同块校准并更新软边；
7. 训练结束后对 best config 额外执行 3 次 final validation（不计入调优预算）；
8. 输出 trials.jsonl、knobs.jsonl、knobs.csv、summary.json、final_validation.json、认知图和曲线。

### 2.14 对比方法

所有方法共用 2.1–2.2 的协议，区别只在配置选择策略：

- **CogniTune**：LLM 认知先验 + 三类熵 + 类型化图 + 协同块 + 三态 + GP/LogEI；
- **GP-BO**：不使用 LLM、不使用优先级/熵/协同块/三态；在完整合法空间生成 80% IID + 20% incumbent 局部扰动候选，用同一个 GP 和 LogEI；
- **SMAC**：官方 SMAC3 `HyperparameterOptimizationFacade`、`Scenario`、`ask/tell`、标准随机森林代理和 EI；通过 `DefaultInitialDesign + additional_configs` 注入共享初始配置；cost = $-u$；
- **OtterTune**：GP + ARD 长度尺度旋钮重要性选择 + LogEI；高层旋钮选择对应 OtterTune 的 factor analysis/k-means 步骤；
- **CDBTune**：DDPG actor-critic；状态为编码配置 + 归一化吞吐，动作为每个旋钮的连续位置，解码后 repair；使用 replay buffer、目标网络和高斯探索噪声；
- **DBTune**：按其高维默认的 probabilistic random forest + EI；候选池采用 local-first（70% incumbent 局部扰动 + 30% 全局随机）；
- **ResTune**：WGPE 式加权 GP 集成；目标 GP 使用当前观测，source GP 来自前一个负载的 GP-BO 结果；权重由 source 预测与当前观测的 rank correlation 决定，均值和方差按高斯混合合成后做 EI。

扩展基线是在统一评测器内按论文算法移植的，不是直接运行各自官方仓库；官方实现绑定各自的 MySQL/工作负载接口，无法直接共享当前 PostgreSQL 60 维空间和 BenchBase 评测器。论文中应明确这一点。

### 2.15 复杂度与可复现性

GP 拟合复杂度约 $O(n^3)$，$n\le 50$；每轮 10000 个候选的 LogEI 评分是主要计算开销，但相对真实数据库评测（分钟级）可以忽略。LLM 调用固定 15 次/负载/seed，响应缓存到 `artifacts/llm_cache`。所有随机性来自显式 seed：SearchSpace 采样、GP 初始化、SMAC、DDPG、RF、ResTune source 选择。方法运行顺序按 seed 打乱，避免数据库状态或系统负载造成方法间偏差。

## 3 实现过程中的问题与修改

### 3.1 LLM 返回尾随 JSON 导致 `Extra data`

**现象**：DeepSeek 返回一个完整 JSON 对象后追加了第二个对象/右括号/自然语言，`json.loads` 抛出 `JSONDecodeError: Extra data: line 1 column 16243`，整个实验在第一个专家响应处中断。

**原因**：模型在 JSON mode 下仍可能拼接多个对象；旧代码只调用一次 `json.loads`，且解析失败发生在 coordinator 的 retry 之外。

**修改**：新增 `_parse_json_object()`，去除 BOM、markdown fence、前导 prose；用 `JSONDecoder(strict=False)` 逐个解码顶层值；若有多个对象，按 recommendations/priority 完整度保留最完整的一个；尾随不可解析内容丢弃并记录 repair；只有完全无对象时才抛 `ValueError`。coordinator 把 `client.complete()` 移入 retry 循环，解析失败会带着 `validation_feedback` 重试最多 3 次，并保存 invalid 原始响应。

**验证**：新增重复对象、尾随括号、markdown fence、最完整对象、完全非法内容、coordinator 重试等回归测试。

### 3.2 状态机退化为“全部 ACTIVE”

**现象**：TPC-C 50 轮结束时 60 个旋钮全部 ACTIVE，`active_knob_budget=12` 完全失效。

**原因**：旧代码把 `selection_entropy + value_entropy`（0–2）与阈值 0.85 比较，几乎所有旋钮每轮都被提升为 ACTIVE；同时所有 memory 类别旋钮被当作 protected，绕过 active cap。

**修改**：uncertainty 归一化为 $0.5H_{\mathrm{sel}}+0.5H_{\mathrm{val}}$；高不确定性只把 DORMANT 提升为 WATCH；protected 仅包含 hard-constraint 参与旋钮和 activation parent；active cap 计入 protected；challenge 旋钮在最初 3 个 interval 内保留；高不确定性旋钮不会被直接降为 DORMANT。

**验证**：新增 active budget 测试；60 维 synthetic 运行中 active 数量稳定在 12（含 protected），challenge 旋钮早期保留。

### 3.3 `active_global_random` 不是 IID

**现象**：日志中 45 个非初始轮有 43 个来源是 `active_global_random`，但实际是 incumbent 的随机子集变异，不是规范要求的 IID 采样。

**原因**：旧实现调用 `local_mutations(..., max_changes=len(active))`，只改变部分旋钮，其他旋钮保持 incumbent。

**修改**：新增 `_active_iid_samples()`：在完整空间 `sample_iid` 后只覆盖 ACTIVE 旋钮，WATCH/DORMANT 固定为 incumbent；仍经过 repair/validate/去重。

### 3.4 Challenge 从未真正发生

**现象**：`challenge_names()` 长期为空，候选池的 challenge 份额实际退化为普通局部扰动。

**原因**：旧逻辑只对 DORMANT 设置 `challenge_due`，而状态机几乎不产生 DORMANT。

**修改**：WATCH/DORMANT 在 `trial % challenge_interval == 0` 时都标记 due；challenge reserve 保证高 $H_{\mathrm{sel}}$ 旋钮早期进入 ACTIVE；challenge 候选与普通候选共享 LogEI，但先选目标。

### 3.5 请求吞吐被重试放大（SMAC 1000+ 的根因）

**现象**：TPC-C 上默认配置约 356 request TPS，SMAC 某轮 1166，其他方法约 350；看起来 SMAC 找到了 3 倍优配置。

**原因**：BenchBase `Throughput` 统计所有 measured requests，包括 SERIALIZABLE 下被拒绝后重试的 TPC-C 事务。SMAC 高分配置的 server retry 从默认 229,900 增到 523,660，请求吞吐被重试放大；其 Goodput 只有 211.6。GP 的 1862 request TPS 配置对应 Goodput 310.5、retry 710,775；CDBTune 的真实最佳 Goodput 407.2 只有 573 retry。修复后现场验证：默认 Goodput 160.2；SMAC 高吞吐配置 request throughput 789.1、Goodput 193.9，即请求吞吐约 9 倍但完成事务仅 +21%。

**修改**：`_extract_metric()` 优先取 `Goodput (requests/second)`，把 `Throughput`、`Measured Requests` 写入 metadata；utility 用 Goodput 计算；旧 shared-initial 缓存没有 `metric` 字段会自动失效重测。

**遗留**：BenchBase Goodput 的分子包含 warm-up 完成事务、分母是 measured 时间，绝对值偏高约 warmup/measure 比例；所有方法同协议可比，但论文不应把绝对值写成稳态 TPS。

### 3.6 DBTune 全零与 PostgreSQL 恢复竞态

**现象**：DBTune 前 10 轮 TPS 为 356.8、260.1、115.0、310.6、240.8、100.8、111.9、100.6、75.0、20.0；第 10 轮起全部 0，最后 suite 在 `restore_default` 处中断，YCSB 没跑。

**原因**：候选触发的 fast shutdown 卡住约 4 分钟，legacy `pg_ctl restart -m fast -w` 60 秒后报 `server does not shut down`；recovery 移走 `postgresql.auto.conf` 后重启，但旧 postmaster 仍在关闭，启动失败；之后每个 trial 在 `write_candidate` 连接时 `Connection refused`，全部 CRASH；`restore_default` 也因数据库不可达抛异常。

**修改**：`_ensure_ready()` 在正常 recover 失败后执行 `pg_ctl stop -D <data> -m immediate -w -t 30` 强制停库，再用空 auto.conf recover；`prepare/evaluate` 前确保数据库可达；`restore_default` 捕获异常不再中断 suite；DBTune 候选池改为 70% 局部 + 30% 全局，减少 RF 外推导致的极端配置。

### 3.7 共享初始缓存混用指标

**现象**：旧 `shared_initial/trial_000.json` 保存的是 request throughput；切换到 Goodput 后，如果直接复用，default metric 会变成 356.8，而新 trial 的 Goodput 约 160，utility 全为负。

**修改**：复用缓存前检查 `metadata.metric in {"goodput","throughput"}`；没有该字段的旧缓存自动重新测量并覆盖。这样无需手工删除旧缓存。

### 3.8 方法顺序、旧 run 跳过与 knobs 落盘

- 方法顺序：原 suite 固定方法顺序；现在每个 seed 用 `random.Random(seed)` 打乱，满足公平性要求。
- 旧 run 跳过：suite 原本只要 `summary.json` 和 `final_validation.json` 存在就跳过；指标修复后旧 TPC-C 结果仍会被跳过，因此复跑必须 `--force`，旧数据先归档到 `results/archive/`。
- 每轮旋钮：RunRecorder 新增 `knobs.jsonl`（逐轮追加）和 `knobs.csv`（finalize 时写），包含 trial、status、raw_metric、utility、best_so_far、candidate_source 和完整 config；进程中途崩溃也能保留已跑轮次。

### 3.9 负载 profile 名称硬编码

**现象**：`_workload_profile()` 把 `name` 硬编码为 `"tpcc"`，Twitter/YCSB 的 profile 会标错负载。

**修改**：从 BenchBase XML 的 JDBC URL 中解析 `ApplicationName`，回退到文件名；重新生成 Twitter/YCSB profile 并验证 terminals、database、measurement_seconds。

### 3.10 外部基线移植与实现边界

OtterTune、CDBTune、DBTune、ResTune 的官方仓库分别绑定 MySQL、各自 workload runner 和采集接口，不能直接接入当前 PostgreSQL + BenchBase + 60 维空间。本文在统一评测器内按论文算法移植：

- OtterTune：GP + ARD 重要性 Top-k + 受限候选池 + LogEI；
- CDBTune：DDPG actor-critic、replay buffer、目标网络、Gaussian 探索；
- DBTune：高维默认 PRF + EI，local-first 候选；
- ResTune：WGPE 加权 GP 集成，source GP 来自前一个负载的 GP-BO 结果。

这不是官方代码的逐行复刻，论文中必须说明“统一评测器内的算法级重实现”，否则会被质疑复现性。

### 3.11 其他工程问题

| 问题 | 修改 |
|---|---|
| `EvaluationResult` 没有 `trial` 字段，复测 metadata 试图读取 `second.trial` | 删除该字段，只记录 first/second raw metric |
| DDPG batch=32、50 轮内几乎不训练 | batch=8、每轮 4 次更新、用初始 transition 预填 replay |
| ResTune source GP 的 sklearn 收敛警告 | 限制 kernel 上界、忽略警告；权重用 rank correlation 截断到非负 |
| runtime config 目录不存在 | `_write_runtime_config` 内 `mkdir(parents=True, exist_ok=True)` |
| suite 崩溃后 `restore_default` 中断整个实验 | `restore_default` 改为 best-effort，下一轮 `_ensure_ready` 自动恢复 |
| final validation 重复次数与调优预算混淆 | final validation 单独写 `final_validation.json`，不计入 50 轮 |
| 基线“新最优复测”与 CogniTune 不对称 | 保留该协议但明确记录；论文中作为限制讨论，或后续改为全方法统一规则 |

## 4 讨论与局限

**指标口径**。Goodput 修正了请求吞吐的重试放大，但 BenchBase Goodput 仍包含 warm-up 完成事务；绝对数值偏高。本文的比较在同一 warm-up/measure 比例下成立，论文应报告相对默认配置的提升，而不是把 Goodput 绝对值当作稳态 TPS。

**基线公平性**。基线在“刷新最优”时复测两次并取均值，CogniTune 不复测；这给基线额外一次数据库观测信息。当前按实验需求保留，但论文中必须明确，或改为所有方法统一规则。

**方法范围**。方案文档最初只要求 CogniTune/SMAC/GP-BO 三种方法，并禁止 OtterTune 和强化学习；当前实现按后续需求扩展到七种方法，CDBTune 属于 DDPG（RL）。建议论文主表保留三种核心方法，七种方法作为扩展实验；或者正式修订方案范围。

**seed 数**。当前只跑 seed 0，适合流程验证；最终论文结论至少需要 3 个 seed 的均值与置信区间。

**外部基线是算法级重实现**。OtterTune/CDBTune/DBTune/ResTune 未直接运行官方仓库；这会影响“官方实现”层面的可复现性，需要在论文中明确并尽量开源适配代码。

**GP 高维性**。60 维、50 个观测对 ARD GP 仍然困难；CogniTune 通过先验和三态降维，但 GP 仍可能把探索引向不确定的坏区域。后续可以加入 trust region 或分批坐标搜索。

## 5 结论

CogniTune 把 LLM 从“调优器”降级为“认知先验源”，把真实性能学习交给 GP/LogEI，把认知分歧转化为旋钮选择、取值范围、条件关系、协同块和挑战调度。工程上，本文修复了 LLM JSON 解析、状态机、候选池、挑战调度、Goodput 指标、PostgreSQL 恢复竞态、缓存混用和可复现性等问题。当前七方法 synthetic smoke 与 43 个单元测试通过；TPC-C/YCSB 的 Goodput 对比实验仍在进行，最终结论需要多 seed 结果支撑。
