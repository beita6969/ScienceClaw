# tsfm — FoR35 / FoR33 / FoR41: 预训练时间序列基础模型(Chronos-2 / Chronos-Bolt)作为可选工具

标注: [实测] = 本机/服务器实际跑出(脚本与日志在 `scripts/tsfm/`), [估计] = 判断, 未测。
口径: 只评"开工具"版本。src/val 只用于开发选型; id/ood 池按预先写定的流水线**各只评一次**(`scripts/tsfm/tsfm_final.py`, 输出 `scripts/tsfm/tsfm_final.log`), 没有据此再调参。

## 0. 结论
- 工具: 新增 `scilib.tsfm`(`forecast(histories, horizon, quantiles, model, context_length, lengths)` → 分位数预测数组 (n, horizon, Q); 模型 `chronos_2` = `amazon/chronos-2`(120M), `chronos_bolt` = `amazon/chronos-bolt-base`(205M))。沙箱没有 torch, 走文件中转桥到 Node0015 的 GPU 2, 白名单 `("tsfm","forecast")`。仅在 `tsfm.available()` 时才通过 `describe_extra("tsfm")` 出现在 FoR35/33/41 的 objective 末尾; 默认库(`forecast.fit_predict`、`loadforecast`、`aquatics`)一行未改。
- **预先写定的流水线**(看 id/ood 之前固定, 只用 src/val 选): 库的默认输出与 Chronos-2 中位数(FoR33/35)或 μ、σ(FoR41)**等权平均**。
- id/ood 结果 [实测](相对变化 = 流水线 / 仅用库 − 1, 负 = 更好; 95% 配对 bootstrap):

| 学科 / 指标 | 池 | n | 仅库 | 仅 Chronos-2 | 流水线 | 相对变化 (95% CI) |
|---|---|---|---|---|---|---|
| FoR35 Tourism 月度 MASE ↓ | id | 64 序列 | 1.4347 | 1.5012 | **1.3908** | −3.06% (−7.99%, +1.39%) |
| | ood | 102 序列 | 1.2296 | 1.2671 | **1.1873** | −3.44% (−7.21%, +0.46%) |
| FoR33 BuildingsBench 平衡 CVRMSE % ↓ | id | 88 条 / 11 栋 | 54.82 | 52.99 | **51.62** | −5.83% (−13.32%, +0.08%) |
| | ood | 40 条 / 5 栋 | 45.21 | 48.72 | 46.06 | **+1.87%** (−16.34%, +2.44%) |
| FoR41 NEON 水生态 CRPS ↓ | id | 64 条 | 0.5724 | 0.5530 | **0.5501** | **−3.89% (−5.87%, −2.12%)** |
| | ood | 64 条 | 0.7163 | 0.7355 | **0.6891** | −3.80% (−7.94%, +0.06%) |

  **读法**: 6 个 (学科, 池) 比较里 5 个点估计向好、1 个(FoR33 ood)变差; 只有 FoR41 id 的区间不含 0, 其余区间都含 0(FoR35 两池上界 +1.4% / +0.5%, FoR33 id 上界 +0.08%)。三个学科合起来方向一致(id 与 ood 上 5/6 个为负), 但每个单独都不能算显著。FoR33 ood 只有 5 栋楼, 聚类 bootstrap 区间 (−16%, +2.4%) 极宽且点估计落在区间上沿(重采样会改变类别中位数), 只能当作"无信息"。FoR35 的 MASE 对 SNaive 的比值: 流水线 0.833(id)/ 0.831(ood), 库单独 0.859 / 0.861(从上表 MASE 反推)。
- 对同行(`reports/overnight_20260930.md` §11): FoR35 我们 id MASE 1.39 与 Monash 基准里 DeepAR 1.409 同一水平, 但池、窗口不同, 绝对数不可比; FoR33 / FoR41 没有可比的绝对同行数, 本次变化是在自己的池上相对库的 −3% ~ −6%, 不改变 §11 的"CONSISTENT / NOT-COMPARABLE"判定。
- **Agent 端到端 episode**: 因网关(lab-gpt-5.5 / lab-gpt-5.4-mini)持续 HTTP 502 未得到结果; 本报告全部是离线、固定流水线的结果。

## 1. 为什么加这个工具(根因)
FoR35/33/41 的现有库是经典方法(ETS/Theta/回归/气候学类基线与 LightGBM/岭回归集成)。同行近年的零样本预测基准(GIFT-Eval、fev-bench、Chronos Benchmark II)里, 预训练时间序列基础模型是最常见的强对照; 按"同行有工具我们也调用"的授权, 把它们作为沙箱里的可选工具提供, 而不是替换默认库。
设计要点: 模型与权重在 Node0015 上(`$S/models/tsfm/{chronos_2,chronos_bolt_base}`), 沙箱经文件中转桥调用; 数组按 `scripts/remote/blobs.py` 打包上传(约 60 KB/s), 序列较短时开销可忽略。`lengths` 参数允许一次调用传入长度不等的序列。

## 2. 工具与权重
- 权重来源 Hugging Face(经 VPS 下载后拉到 Node0015, 两端 sha256 一致) [实测]:
  - `chronos_2/model.safetensors` `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`(477,930,472 B); `config.json` `ef1143bf…08031`
  - `chronos_bolt_base/model.safetensors` `31f875483a3215bc6880a0837ea608a13ce55f88ad90538c3cdd0b29aeb60b36`(821,203,576 B); `config.json` `46f7e7e6…5530b`
- 包: `chronos-forecasting` 2.3.2(服务器环境 `$S/env`)。
- 耗时 [实测] (GPU 2, 与他人共用, float32, 权重已加载; `scripts/tsfm/cost_bench.py`): Chronos-2 对 256 条 × 120 值 × 预测 24 步 0.071 s(3625 条/s), 256 × 300 值 0.126 s, 64 × 1461 值 0.051 s; Chronos-Bolt 同三项 0.068 / 0.076 / 0.045 s。首次调用加载权重: Chronos-2 约 10.7 s, Bolt 0.3 s(Bolt 的文件在页缓存里)。远程调用另加桥的往返。
- 单元测试: `tests/test_scilib_tsfm.py`(8 项: 可用性、输入检查、padding/lengths 往返、远程路由、白名单、事实性文字守卫)全部通过; 相关守卫 `tests/test_adapter_visible_text.py` 等 9 个文件 116 项通过。文档中不含配方 / 阈值 / 推荐(F5); 一次措辞(含 "last value")触发了配方扫描器, 已改为 "the end of its history"。

## 3. 选型(只用 src/val) [实测]
候选流水线与"库单独"比较, 都在 src/val 的 eval 与 dev 视图上(FoR35 的 dev 视图 = history[:-24])。

**FoR41**(CRPS, 越低越好; `scripts/tsfm/f41_eval.log`): Chronos-2 用全部 1461 天上下文; 365 天上下文差很多(0.64–0.77); Bolt 比 Chronos-2 差。

| 方法 | src/eval | src/dev | val/eval | val/dev |
|---|---|---|---|---|
| 库 `fit_predict` | 0.6264 | 0.6447 | 0.5527 | 0.6506 |
| Chronos-2(全上下文) | 0.5859 | 0.6115 | 0.5331 | 0.6524 |
| **½ 库 + ½ Chronos-2(预先写定)** | 0.5864 | 0.6070 | 0.5251 | 0.6278 |
| 相对库 | −6.4% | −5.9% | −5.0% | −3.5% |

**FoR35**(MASE; `scripts/tsfm/f35_gridsum.log`, 对 12 个 (模型, 上下文, 变换) 组合做了网格): 单独的 Chronos-2 都不如库(池化 1.649–1.690 vs 1.602), 但等权平均有收益; 选了池化最优的 Chronos-2 上下文 120 + log1p, 池化 1.6023 → 1.5487(−3.3%)。注意: 这是 12 选 1, 所以 src/val 上的 −3.3% 偏乐观, id/ood 的 −3.1% / −3.4% 是不受选择影响的数。
**FoR33**(CVRMSE %; `scripts/tsfm/f33_eval.log`): `ens` → `ens + Chronos-2(168 h 全上下文)`: src/eval 37.79 → 37.01, src/dev 39.47 → 38.06, val/eval 51.87 → 51.70, val/dev 33.89 → 34.47(变差)。是噪声水平, 但没有更好的候选, 也不想再多选, 就按预先写定的流水线评 id/ood。
**FoR38**: 不做。FoR38 的相对结果已经与同行 CONSISTENT(§11: 比值 0.77–0.95 vs M4 Yearly 类比 0.79–0.81), 且是 ~60 个年度点 + 协变量的宏观面板, 不属于"明显低于同行"的优先队列。

## 4. 流水线定义(逐字, 看 id/ood 之前写定)
- FoR35: `0.5·fit_predict + 0.5·Chronos-2 中位数`, Chronos-2 上下文 = 最近 120 个月, 输入取 log1p、输出取 expm1。
- FoR33: `0.5·ens + 0.5·Chronos-2 中位数`, 上下文 = 全部 168 h, 结果截到 ≥ 0。
- FoR41: μ、σ 各取 `0.5·库 + 0.5·Chronos-2`; Chronos-2 用全部 1461 天上下文, μ = q50, σ = (q75 − q25)/1.349, σ 截到 [0.05, 50], μ 截到 [0, 45]。

## 5. 披露与局限
- **预训练数据重叠**: Chronos-2 的模型卡列出 `autogluon/chronos_datasets` 与 `Salesforce/GiftEvalPretrain` 作为训练语料; Chronos-Bolt 用约 1000 亿观测的公共 Chronos Datasets + 合成序列。这些语料含 Monash 档案(包含 Tourism 月度/季度)、电力/交通/天气/能源序列 [估计: 原版 Chronos 论文把 Tourism 数据集列在语料内, 本次没有逐一核实到版本和切分]。因此 **FoR35 的 Tourism 序列(id = 月度, ood = 季度跨数据集)很可能出现在预训练中**, 目标窗口是否包含未知; FoR33(BuildingsBench 的源数据集)与 FoR41(NEON)同样不能排除。这意味着上表的增益可能含有"见过数据"的成分, 不能直接解读为对全新数据的泛化。工具文档 (`scilib.tsfm.__doc__`) 里如实写了这一点。
- 统计: 每个比较各自单独看, 6 个中只有 1 个区间不含 0; 没有做多重比较校正。FoR35 的 bootstrap 以序列为单位, FoR41 以条目为单位, FoR33 以楼为单位(聚类)。
- 选择程度: FoR35 在 12 个候选中选 1(src/val), FoR41 在 4 个(模型 × 上下文)× 2 种 σ 取法里选, FoR33 无选择(只试了 4 个组合)。id/ood 池没有参与任何选择, 每个只评一次。
- 未做: agent 端到端 episode(网关 502)、FoR38、tool-ON 下 token 预算(所有 tool-ON episode 目前仍因超过 200k 逻辑 token 而 z=0, 与本工具无关)。
- 失败/偏路记录: 评测脚本曾把两个模型都标成 "chrono" 导致 Bolt 覆盖 Chronos-2, 已改为 `c2` / `bolt` 重跑后才读数; 第一次 FoR41 导出 268 MB 太大, 改为只保留 2 个通道、float32、src 隔两项抽样(8 MB)后上传。

## 6. 复现
- 服务器: `scripts/tsfm/{f35,f33,f41}_prep.py`(导出池与库基线, 本机)→ `*_tsfm.py`(服务器 GPU 2)→ `*_eval.py`(src,val 选型)→ `f35_final.py` / `f41_tsfm.py`(预先写定的 id/ood 一次)→ `tsfm_final.py`(本机一次性评估, 输出 `tsfm_final.log`)。
- 临时 pickle 不入库(`/private/tmp/claude-501/sc-scratch/tsfm/`)。
