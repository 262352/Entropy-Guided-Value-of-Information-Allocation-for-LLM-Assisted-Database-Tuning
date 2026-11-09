# 四负载相对默认吞吐量提升

纵轴为 Δ(r)=(best_TPS(r)-TPS_default)/TPS_default，以百分比显示。50%表示比默认提升50%。默认分母取对应实验开始前三次默认配置测量的中位数：Twitter 15902.699694138077；TPC-C 174.59999861775；YCSB 1800.6236091689893；CH-benCHmark 68.2499637183193。

按用户最新明确要求，Twitter和TPC-C使用附件绘图代码里的八方法数列，保留该代码补齐/截断至100轮的规则，然后计算累计最高值。这两个子图是用户提供的示意数列，不是原始实验轨迹；CSV状态SUPPLIED、来源user_supplied_arrays。附件中的LHS命名仅为原变量名称，不表示实际执行了LHS实验。各曲线没有以首个启动点作为默认分母。

YCSB、CH-benCHmark使用artifacts/remaining_baselines_100_v1的真实ledger，各七基线100轮；没有我们方法的数据，故无Ours曲线。CSV来源measured_trial_ledger。CH展示事务吞吐量，实际调优目标同时包含分析延迟；本图不代替综合HTAP性能评估。TPC-H以延迟为指标，未放入吞吐量四图。

附件中的数列已归档为supplied_throughput_series.json。仓库绘图脚本读取此JSON，无需聊天附件，也不执行附件代码。CSV同时保存原始数列值、默认值、累计最佳、提升比例与来源；JSON附方法与分母信息。共3000个数据位置。不能将Supplied series作为实测结论用于论文性能声明。

重绘：`.venv/bin/python scripts/plot_throughput_delta_2x2.py`。PNG/PDF为图，CSV/JSON为数据与元信息。

用户要求移除图内的来源英文标注，图片已移除；数据来源仍由本文、CSV和JSON保留。

独立绘图源代码：`plot_throughput_delta_2x2.py`。所有曲线数据已内嵌，无需原仓库、附件或CSV；仅需matplotlib。运行 `python plot_throughput_delta_2x2.py` 会在脚本同目录生成PNG/PDF。
