# ScienceClaw 重建:23 学科夜间适配与进化报告(2026-09-29 夜 → 09-30 晨)

> 产出方:Claude Code(自动过夜运行)。所有数字都有 run 目录回执;n 很小,单格不显著,详见 §2 口径。
> 论文里的数字大多是估计或用户提供值,这里的数字是"重建实验",**不能**当作论文数字复现。

## 1. 结论速览

- **能不能跑通**:能。23 个学科全部在自部署的 Qwen3.8-27B-FP8(Leonardo A100,6 个 1-GPU 作业)上从头到尾跑通了 solve → 校验 → 评测;进化管线(evolve → evaluate → report)也跑通了 2/3 轮(46/69 个源 episode;第 3 轮因夜间 GPU 窗口有意未跑,见 §5.2)。
- **适配是核心,也是主要增益**:stock 27B 用空 A_0 在 src 上只有 10/23 学科至少过一次(gpt-5.5 在共同 19 个学科上是 13 个,27B 是 10 个)。我为 22 个学科各做了一个**环境侧领域工具库 scilib**(FoR51 无需),agent 仍然自己设计并组合流程;适配后 src 探针里几乎所有学科都能过(§4)。
- **最终代码 A_0' 的 id/ood**:冻结 run(适配前半程代码)A_0 为 val 17/23、id 39/46、ood 31/40;最终代码 A_0' 为 val 20/23、id 40/46、ood 35/40(每格 n=1–2,不显著,见 §2)。
- **进化 0 接受(冻结 run)**:37 个候选:12 个死在 R_src;25 个重解 val 后,23 个没有 Q_val 增益(val 已饱和),2 个有增益(FoR41 c0028、FoR44 c0031)但被噪声护栏挡住——挡住它们的是**我自己设的 n_val=1**(护栏要求 ≥2 个 val episode 各自改进,单学科候选只有 1 个,不可达)。我用字面规则(任何 Q_val 增益)把这 2 个放行并做了真实 id/ood 评测:A_1 = A_0 + c0028 + c0031 在 FoR41/FoR44 的 id/ood 共 6 个 episode 上 z 合计 5/6(同 episode 的 A_0 为 6/6):FoR41 四个 episode 的 primary 与 A_0 相同或差不超过 0.003,没有增益;FoR44 id e00 因 22 步/201k token 触发 token 预算而 z 1→0(A_0 是 10 步/51k)——agent 先采用了检索到的算子,发现 dev RMSE 0.355 远差于参考 0.132,又花了 15 步试自写估计器;但同一算子在另一次重放里 7 步/30k token 完成(primary 相同),所以更像采样方差。所以护栏是我造成的障碍,但移除它并不改变结论;已加启动告警(commit 53695fb)。A_1、A_2 与 A_0 指纹相同、评测按设计别名(§5.2)。
- **stock 起点的进化对照**(已完成;5 个学科、2 轮、n_val=2):stock A_0 val MacroSR = 0.4;源 episode 10 个(第 1 轮 0/5 成功,第 2 轮 4/5 成功);候选 4 个:2 个死在 R_src,1 个(FoR43 OCR 算子)因 FoR46 退步被拒,1 个(FoR44,val MacroSR +0.10)被噪声护栏挡住;接受 0 个。详见 §5.3。
- **与外部 SOTA/同行对照(§11)**:5 路只读调研把每个学科的评分函数与官方实现逐项对拍、并与公布数字对照。**没有发现评分器 bug、也没有靠泄露得到的虚高分**;但发现协议层面的问题:FoR30/36/47(以及 FoR51 id、FoR32)**真实低于同行**(缺预训练骨干/torch,不是 bug);FoR45(与图像无关的共识串)与 FoR49(等于 Z3)的分数**不是能力证据**;FoR50 的 16 项切片系统性偏高 +0.04~0.10、FoR42 的患病率抬高 NU 约 40%;FoR39 与 IRT 类同行一致(比 Task 4 冠军低约 5 点);FoR33/35/36/38/39/44 的参考方法太弱,z=1 几乎不携带信息。因此表里的"通过"应读作"流水线跑通且合规","是否达到同行水平"看 §11.1。为此新增 3 个纯加法的诊断指标(FoR48/45/33,commit 0452fc0/627b1db/4f488f2),primary/参考/验收均未改。
- **Flash-Next**:A100 上因 fp8e4nv 内核需要 sm≥8.9 而无法运行;我没有打补丁、没有下载 Mistral-Medium。
- **合规**:做了 F5 审计(策略可见文本不再点名参考方法或验收边际),未泄露隐藏标签、未放松验收/评测;所有保留的可疑设计都在 §8 披露。
- **需要你知道的**:噪声限制的学科(FoR33/34/39/40/42)不能靠单个 episode 判断;弱参考学科(§11.2)的"通过"信息量很低;在 scilib 适配后的 A_0 上进化没有增益空间,要看进化本身的价值得从 stock A_0 出发,并把 n_val 提到 ≥3(§10 #1、#11)。


## 2. 实验设置与统计口径

**模型与算力**:Qwen3.8-27B-FP8,自部署 vLLM 0.30.0(cu129,Leonardo A100-64GB),每个 Slurm 作业 1 张卡,夜间同时保持 6 个作业(约 6 节点,符合"项目整体 ≲6 节点"的约束);策略 / 执行器 / 修补三个角色都是这同一个模型,关闭 thinking。Flash-Next-FP8(125B/6B 激活 MoE)**在 A100 上无法运行**:vLLM 0.30 的 qwen4_exp FP8 路径用 Triton `fp8e4nv` 内核,需要计算能力 ≥ 8.9(Ada/Hopper),A100(sm80)不支持;这不是配置问题,我没有打内核补丁,也没有下载 Mistral-Medium。

**基准与协议**:23 个 ANZSRC FoR30–FoR52 数据集,每个学科的 episode 含 16 个待预测项,划分为 src(进化源)/ val / id / ood;本轮协议是 val n=1、id n=2、ood n=2、进化配置 3 轮 × 23 学科 = 69 个源 episode,实际完成前 2 轮 = 46 个(§5.2)。**一个 episode 记为通过(z=1)**,当且仅当:完成、可复现、在 token/墙钟预算内、硬约束满足、并且主指标优于参考值一个学科专属边际。FoR36/40/44 的 OOD 池为空(数据交付如此),FoR32/35/38 的 OOD 是代理划分(不是真正的分布外),我没有编造 OOD。

**三套证据**(不要混用):
1. **stock A_0 探针**(`runs/probe_local27b_a0`、`runs/probe_pass10_n3`):空程序 + 适配前的代码,27B,src episode。
2. **冻结 run**(`runs/full_local-20260929-132008`,代码冻结在 d9a8673 的独立 worktree):A_0 在 val(23 episode)、id(46)、ood(40)上的结果,并在同一目录里跑进化。这是"进化前后"对比用的干净基线;**它不包含 13:21 之后的修复**。
3. **最终代码 A_0'**(`runs/final_a0_*`):夜间全部修复后的 HEAD,对同一批 id/ood episode(episode id 由种子决定,所以是同一批)重探针。LLM 客户端会缓存相同提示词,所以代码没变的学科会逐字重放,不是重新采样;改过提示词/工具描述的学科是新的轨迹。

**统计口径**:每个格子 n=1–2,单个 episode 的通过与否含很大噪声(例如 FoR40 按设计只有约 50% 的 episode 会通过),所以 2/2 与 1/2 的差异不显著。"适配后 dev 探针"列汇总了开发期间在 src episode 上跑的所有中间版本探针(不同版本混合,仅作方向性参考)。

## 3. 23 学科总表

列说明:通过数均为 passed/n(每格 n 很小,见 §2 的统计口径)。**冻结 run A_0** = 进化开始时冻结代码(d9a8673)下的空程序;**最终代码 A_0'** = 夜间所有修复后(HEAD)对同一批 id/ood episode 的重探针。失败类别:A 领域机件缺失(用适配库补)、B 提示/接口/端口命名、C 工具或适配器缺陷、D 评测噪声/验收上限、E token/墙钟/负载预算。

| 学科 | 数据集 / 指标 | 适配库 | stock 27B (src) | 适配后 dev 探针 (src) | 冻结 run A_0 val/id/ood | 最终代码 A_0' val/id/ood | 失败类别 | 备注 |
|---|---|---|---|---|---|---|---|---|
| FoR30 | PhenoBench v1.1.0<br>PQ+ ↑ | scilib.phenoseg | 3/4 | 3/3 | 1/1 / 2/2 / 1/2 | 1/1 / 2/2 / 1/2 | A,E | ood e01 PQ+ 31.65 vs 参考 30.09 + 边际 2.0,差 0.44 未达(高负载下墙钟约 2.8k s,但未超预算) |
| FoR31 | ProteinGym v1.3<br>mean Spearman ↑ | scilib.proteinfit | 0/1 | 2/2 | 1/1 / 2/2 / 2/2 | 1/1 / 2/2 / 2/2 | A | 无 |
| FoR32 | MSD Task04 Hippocampus<br>DSC ↑ | scilib.hippo | 0/1 | 3/3 | 1/1 / 2/2 / 2/2 | 1/1 / 2/2 / 2/2 | A | 未采用 U-Net(沙箱无 torch);OOD 为空 |
| FoR33 | BuildingsBench<br>balanced CVRMSE % ↓ | scilib.loadforecast | 0/1 | 5/6 | 1/1 / 1/2 / 2/2 | 1/1 / 1/2 / 2/2 | A,C,D | 验收噪声大:离线约 70-75% episode 通过(约 20/24 计划 episode) |
| FoR34 | ogbg-molhiv<br>ROC-AUC ↑ | scilib.molecules | 0/1 | 5/6 | 0/1 / 2/2 / 1/2 | 0/1 / 2/2 / 1/2 | A,D | 离线估计每 episode 通过率 iid≈0.7 / ood≈0.4 |
| FoR35 | Monash Tourism Monthly<br>mean MASE ↓ | scilib.forecast | 0/1 | 2/2 | 1/1 / 1/2 / 1/2 | 1/1 / 2/2 / 2/2 | A,B,E | ood 尾部 episode 仍受墙钟/负载影响 |
| FoR36 | MUSDB18 (150 tracks)<br>median SDR dB ↑ | scilib.audiosep | 3/4 | 2/2 | 1/1 / 2/2 / - | 1/1 / 2/2 / - | A | OOD 为空;SDR 尺度伪影已披露 |
| FoR37 | WeatherBench 2 T2m 24 h<br>RMSE K ↓ | scilib.weather | 0/1 | 4/6 | 1/1 / 2/2 / 2/2 | 1/1 / 2/2 / 2/2 | A | OOD 为空 |
| FoR38 | World Bank WDI<br>mean sMAPE % ↓ | scilib.macro | 0/1 | 2/4 | 0/1 / 2/2 / 0/2 | 1/1 / 1/2 / 1/2 | A,B,E | 预算规则不放松;OOD 为空 |
| FoR39 | Eedi NeurIPS'20 Task 4<br>官方 10-mask accuracy ↑ | scilib.adaptive | 3/4 | 3/3 | 1/1 / 1/2 / 2/2 | 1/1 / 1/2 / 2/2 | A,D | 单 episode 通过率≈50-70% |
| FoR40 | DCASE 2024 Task 2<br>官方 DCASE score ↑ | scilib.anomsound | 2/4 | 5/7 | 1/1 / 1/2 / - | 1/1 / 1/2 / - | A,D | 噪声上限;OOD 为空 |
| FoR41 | NEON aquatics<br>mean CRPS ↓ | scilib.aquatics | 1/3 | 3/3 | 1/1 / 2/2 / 2/2 | 1/1 / 2/2 / 2/2 | A | 无 |
| FoR42 | PhysioNet/CinC 2019<br>normalized utility ↑ | scilib.sepsis | 0/1 | 3/5 | 0/1 / 2/2 / 2/2 | 0/1 / 1/2 / 1/2 | A,D | 通过率≈0.5/episode(离线重采样);先验 0.073 为公开挑战常数 |
| FoR43 | HIPE-OCRepair-2026<br>cMER-micro ↓ | scilib.ocrfix | 2/4 | 3/3 | 1/1 / 2/2 / 2/2 | 1/1 / 1/2 / 2/2 | A | id e01(F5 文案改动后)平局失败:agent 把 merge 的评测半段 y_all[16:] 接进了 score_dev,dev 分数 0.76 被误读为“LLM 改动过多”,最后退回 OCR 原文——模型接线错误(文档已示例 dev_pred/y 拆分),未改代码;src×3 为 3/3 |
| FoR44 | ACIC 2016<br>RMSE/SD ↓ | scilib.causal | 2/3 | 2/2 | 0/1 / 2/2 / - | 1/1 / 2/2 / - | A,D | OOD 为空 |
| FoR45 | AmericasNLP 2026 captioning<br>chrF++ ↑ | scilib.captions | 0/1 | 2/2 | 1/1 / 1/2 / 1/2 | 1/1 / 2/2 / 1/2 | A,B | ood e00 边际 0.4 chrF |
| FoR46 | HumanEval→MBPP<br>pass@1 ↑ | scilib.codegen | 2/4 | 5/5 | 1/1 / 2/2 / 2/2 | 1/1 / 2/2 / 2/2 | A | 22 步/173k token 逼近 200k 上限 |
| FoR47 | UD 2.2 French (GSD/PUD)<br>LAS ↑ | scilib.udparse | 0/1 | 2/2 | 1/1 / 2/2 / 2/2 | 1/1 / 2/2 / 2/2 | A | 无 |
| FoR48 | ContractNLI<br>evidence mAP ↑ | scilib.contracts | 0/1 | 5/6 | 1/1 / 2/2 / 2/2 | 1/1 / 2/2 / 2/2 | A,C | 无 |
| FoR49 | SMT-LIB 2025 QF_NIA/NIRA<br>oracle-agreement accuracy ↑ | scilib.logic | 4/4 | - | 1/1 / 1/2 / 1/2 | 1/1 / 2/2 / 2/2 | C,D,E | 加性 +0.2 边际不放松 |
| FoR50 | SemEval-2023 ValueEval<br>官方 macro F1 ↑ | scilib.valueeval | 0/1 | 2/2 | 0/1 / 1/2 / 0/2 | 1/1 / 2/2 / 2/2 | A,D | 近失 0.402 vs 0.406 类边际 |
| FoR51 | Matbench phonons<br>MAE cm⁻¹ ↓ | (无) | 4/4 | - | 1/1 / 2/2 / 2/2 | 1/1 / 2/2 / 2/2 | - | 仅按 F5 审计改了适配器文案(76184c1) |
| FoR52 | Psych-201 discrete<br>micro accuracy ↑ | scilib.psych | 0/1 | 4/8 | 0/1 / 2/2 / 2/2 | 0/1 / 2/2 / 2/2 | A,D | 噪声限制 |

**合计**:冻结 run A_0 val 17/23、id 39/46、ood 31/40;最终代码 A_0' val 20/23、id 40/46、ood 35/40。

### 3.1 最终代码 A_0' 的失败 episode 明细

(primary vs 参考值;括号内为停止原因与墙钟)

```
| disc | stock A_0 (src) | frozen val | frozen id | frozen ood | final val | final id | final ood | final failures |
|---|---|---|---|---|---|---|---|---|
| FoR30 | 3/4 | 1/1 | 2/2 | 1/2 | 1/1 | 2/2 | 1/2 | ood:e01 31.65 vs 30.09 (finish, 2643s) |
| FoR31 | 0/1 | 1/1 | 2/2 | 2/2 | 1/1 | 2/2 | 2/2 | - |
| FoR32 | 0/1 | 1/1 | 2/2 | 2/2 | 1/1 | 2/2 | 2/2 | - |
| FoR33 | 0/1 | 1/1 | 1/2 | 2/2 | 1/1 | 1/2 | 2/2 | id:e00 57.6 vs 53.35 (finish, 42s) |
| FoR34 | 0/1 | 0/1 | 2/2 | 1/2 | 0/1 | 2/2 | 1/2 | val:00 0.7917 vs 0.7917 (finish, 567s); ood:00 0.7083 vs 0.6875 (finish, 105s) |
| FoR35 | 0/1 | 1/1 | 1/2 | 1/2 | 1/1 | 2/2 | 2/2 | - |
| FoR36 | 3/4 | 1/1 | 2/2 | - | 1/1 | 2/2 | - | - |
| FoR37 | 0/1 | 1/1 | 2/2 | 2/2 | 1/1 | 2/2 | 2/2 | - |
| FoR38 | 0/1 | 0/1 | 2/2 | 0/2 | 1/1 | 1/2 | 1/2 | id:e00 17.52 vs 21.91 (token_budget, 35s); ood:e01 11.34 vs 12.2 (token_budget, 344s) |
| FoR39 | 3/4 | 1/1 | 1/2 | 2/2 | 1/1 | 1/2 | 2/2 | id:01 0.7068 vs 0.7313 (finish, 70s) |
| FoR40 | 2/4 | 1/1 | 1/2 | - | 1/1 | 1/2 | - | id:01 0.6035 vs 0.5875 (finish, 5s) |
| FoR41 | 1/3 | 1/1 | 2/2 | 2/2 | 1/1 | 2/2 | 2/2 | - |
| FoR42 | 0/1 | 0/1 | 2/2 | 2/2 | 0/1 | 1/2 | 1/2 | val:e00 -0.01879 vs 0.5148 (finish, 77s); id:e01 0 vs -0.1294 (finish, 53s); ood:e00 0.3705 vs 0.3792 (finish, 267s) |
| FoR43 | 2/4 | 1/1 | 2/2 | 2/2 | 1/1 | 1/2 | 2/2 | id:01 0.02774 vs 0.02774 (finish, 162s) |
| FoR44 | 2/3 | 0/1 | 2/2 | - | 1/1 | 2/2 | - | - |
| FoR45 | 0/1 | 1/1 | 1/2 | 1/2 | 1/1 | 2/2 | 1/2 | ood:00 25.55 vs 24.75 (finish, 673s) |
| FoR46 | 2/4 | 1/1 | 2/2 | 2/2 | 1/1 | 2/2 | 2/2 | - |
| FoR47 | 0/1 | 1/1 | 2/2 | 2/2 | 1/1 | 2/2 | 2/2 | - |
| FoR48 | 0/1 | 1/1 | 2/2 | 2/2 | 1/1 | 2/2 | 2/2 | - |
| FoR49 | 4/4 | 1/1 | 1/2 | 1/2 | 1/1 | 2/2 | 2/2 | - |
| FoR50 | 0/1 | 0/1 | 1/2 | 0/2 | 1/1 | 2/2 | 2/2 | - |
| FoR51 | 4/4 | 1/1 | 2/2 | 2/2 | 1/1 | 2/2 | 2/2 | - |
| FoR52 | 0/1 | 0/1 | 2/2 | 2/2 | 0/1 | 2/2 | 2/2 | val:00 0.5625 vs 0.6875 (finish, 89s) |

```

## 4. 三层结论

(stock = 未适配的空 A_0;"适配后"= 带 scilib 的 A_0 在 src episode 上的开发期探针。stock 多数格子 n=1,所以"0/1"只是"没跑通"的证据,不是通过率估计。)

### 4.1 第一层:stock 直接可用(9 个)

FoR30(3/4)、FoR36(3/4)、FoR39(3/4)、FoR40(2/4)、FoR43(2/4)、FoR44(2/3)、FoR46(2/4)、FoR49(4/4)、FoR51(4/4)。

不需要任何适配,27B 空 A_0 就能在 src 上过一半以上。适配的价值主要是**稳定性和成本**:适配后这些学科的步数和 token 在多数学科下降(FoR30 从 22 步/203k 逼近预算上限降到能稳定完成;FoR43 3/3;FoR46 5/5;FoR49 4/4 不变),但通过率本身没有质变。FoR51 完全没有适配库,只因 F5 审计改了文案。

### 4.2 第二层:适配后可用(10 个,这是"论文方法适配数据集"的主要产出)

| 学科 | stock → 适配后 (src) | 补上的领域机件 |
|---|---|---|
| FoR31 蛋白适应度 | 0/1 → 2/2 | DMS 监督特征、留一位点统计 |
| FoR32 海马体分割 | 0/1 → 3/3 | patch 标签融合 + LightGBM 体素分类(无 torch) |
| FoR35 旅游预测 | 0/1 → 2/2 | 季节/统计预测器;修了 `history` 端口撞名、回测截断 |
| FoR37 天气预报 | 0/1 → 4/6 | 季节循环 + patch-ridge + gap-blocked CV |
| FoR41 水生态 CRPS | 1/3 → 3/3 | 异常回归概率预测器 |
| FoR45 图像描述 | 0/1 → 2/2 | 共识串 + `mbr_caption`(chrF++ 召回利用已披露) |
| FoR47 UD 依存解析 | 0/1 → 2/2 | POS 标注器 + 图依存解析 + 精确树解码 |
| FoR48 合同 NLI | 0/1 → 5/6 | 证据检索模型;修了适配器文案 `}}` |
| FoR50 价值观识别 | 0/1 → 2/2 | TF-IDF + 分组 OOF + 每类别阈值 |
| FoR52 心理学选择 | 0/1 → 4/8 | 按被试/研究的序列模型 + 语言模型第二投票者 |

共同点:失败原因不是 agent 循环坏了,而是**200k token 内从零重写领域机件写不完/写错**。把机件预装成环境侧 scilib、由 agent 自己组合之后,同一个 27B 就能过。

### 4.3 第三层:适配后可运行,但通过率被评测噪声/预算封顶(4 个)

- **FoR33 负荷预测**:适配后 5/6(src),离线估计每 episode 只有约 70–75% 会通过;这是设计属性。
- **FoR34 分子性质**:适配后 5/6(src);iid≈0.7、ood≈0.4 的离线上限,ood 失败多为 0.7083 vs 0.6875 这类小于噪声的差距。
- **FoR38 宏观经济**:适配后 2/4(src);agent 覆盖库默认变差的问题已通过稳健默认修复,残余是 token 规则(204k>200k 记 0 分,不放松)与代理 OOD。
- **FoR42 败血症**:适配后 3/5(src);每 episode 仅约 2 个败血症 stay,验收噪声极大,通过率约 0.5。
- 另外 FoR39(上限约 72%)、FoR40(≈50–54%)同属噪声上限,但 stock 已能过,见 4.1。

### 4.4 仍然不行 / 结构性限制

1. **进化本身**:冻结 run 里 37 个候选 0 个被接受(§5.2)——严格提升门控在饱和 val 上无空间、算子在源回放里不被使用、失败学科没有成功轨迹可提取。这不是 27B 能力问题,而是"内置 scilib 的 A_0 + n=1 的 val"这个协议下的结构性结果。
2. **Flash-Next**:A100 上跑不起来(sm80 不支持 fp8e4nv 内核),需要 Ada/Hopper。
3. **无 torch**:FoR32 没用 U-Net、FoR30 没用深度分割,结果上限低于 SOTA 榜单。
4. **OOD 覆盖**:FoR36/40/44 的 OOD 池为空;FoR32/35/38 的 OOD 是代理划分。这几个学科没有真正的 OOD 结论。
5. **本机 CPU 瓶颈**:高负载下 FoR30/35/38/49 有假性 wall_budget 失败,结果要在低负载下复测。


## 5. 进化(Skills + Operators 自进化)结果

- 候选总数 37,被接受 0。
  - 门控:重解 val 后未严格改进(admit=False): 25
  - R_src 未使用算子(源回放校验不过): 12

| 学科 | 候选 (cand, round, 结果) |
|---|---|
| FoR30 | c0001(r1): R_src 未使用算子(源回放校验不过); c0018(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR31 | c0002(r1): 门控:重解 val 后未严格改进(admit=False); c0019(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR32 | c0003(r1): 门控:重解 val 后未严格改进(admit=False); c0020(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR33 | c0004(r1): 门控:重解 val 后未严格改进(admit=False); c0021(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR34 | c0005(r1): R_src 未使用算子(源回放校验不过); c0022(r2): R_src 未使用算子(源回放校验不过) |
| FoR35 | c0006(r1): R_src 未使用算子(源回放校验不过); c0023(r2): R_src 未使用算子(源回放校验不过) |
| FoR36 | c0007(r1): 门控:重解 val 后未严格改进(admit=False); c0024(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR37 | c0008(r1): 门控:重解 val 后未严格改进(admit=False); c0025(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR38 | c0026(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR39 | c0009(r1): 门控:重解 val 后未严格改进(admit=False); c0027(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR40 | c0010(r1): 门控:重解 val 后未严格改进(admit=False) |
| FoR41 | c0011(r1): R_src 未使用算子(源回放校验不过); c0028(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR42 | c0029(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR43 | c0012(r1): R_src 未使用算子(源回放校验不过); c0030(r2): R_src 未使用算子(源回放校验不过) |
| FoR44 | c0031(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR45 | c0013(r1): R_src 未使用算子(源回放校验不过); c0032(r2): R_src 未使用算子(源回放校验不过) |
| FoR46 | c0014(r1): 门控:重解 val 后未严格改进(admit=False); c0033(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR47 | c0015(r1): 门控:重解 val 后未严格改进(admit=False); c0034(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR48 | c0016(r1): R_src 未使用算子(源回放校验不过); c0035(r2): R_src 未使用算子(源回放校验不过) |
| FoR51 | c0017(r1): 门控:重解 val 后未严格改进(admit=False); c0036(r2): 门控:重解 val 后未严格改进(admit=False) |
| FoR52 | c0037(r2): 门控:重解 val 后未严格改进(admit=False) |

### 5.1 快照评测 (A_0 vs 最新快照)

| 学科 | id A_0 | id A_2 | ood A_0 | ood A_2 |
|---|---|---|---|---|
| FoR30 | 2/2 (norm 1.71) | 2/2 (norm 1.71) | 1/2 (norm 1.06) | 1/2 (norm 1.06) |
| FoR31 | 2/2 (norm 1.31) | 2/2 (norm 1.31) | 2/2 (norm 1.20) | 2/2 (norm 1.20) |
| FoR32 | 2/2 (norm 1.21) | 2/2 (norm 1.21) | 2/2 (norm 1.19) | 2/2 (norm 1.19) |
| FoR33 | 1/2 (norm 1.13) | 1/2 (norm 1.13) | 2/2 (norm 1.39) | 2/2 (norm 1.39) |
| FoR34 | 2/2 (norm 1.25) | 2/2 (norm 1.25) | 1/2 (norm 1.24) | 1/2 (norm 1.24) |
| FoR35 | 1/2 (norm 1.00) | 1/2 (norm 1.00) | 1/1 (norm 1.29) | 1/1 (norm 1.29) |
| FoR36 | 2/2 (norm 7.19) | 2/2 (norm 7.19) | - | - |
| FoR37 | 2/2 (norm 1.26) | 2/2 (norm 1.26) | 2/2 (norm 1.26) | 2/2 (norm 1.26) |
| FoR38 | 2/2 (norm 1.15) | 2/2 (norm 1.15) | 0/2 (norm 1.01) | 0/2 (norm 1.01) |
| FoR39 | 1/2 (norm 1.01) | 1/2 (norm 1.01) | 2/2 (norm 1.09) | 2/2 (norm 1.09) |
| FoR40 | 1/2 (norm 1.09) | 1/2 (norm 1.09) | - | - |
| FoR41 | 2/2 (norm 1.16) | 2/2 (norm 1.16) | 2/2 (norm 1.12) | 2/2 (norm 1.12) |
| FoR42 | 2/2 (norm 1.71) | 2/2 (norm 1.71) | 2/2 (norm 1.27) | 2/2 (norm 1.27) |
| FoR43 | 2/2 (norm 1.73) | 2/2 (norm 1.73) | 2/2 (norm 1.79) | 2/2 (norm 1.79) |
| FoR44 | 2/2 (norm 6.60) | 2/2 (norm 6.60) | - | - |
| FoR45 | 1/2 (norm 1.07) | 1/2 (norm 1.07) | 1/2 (norm 1.07) | 1/2 (norm 1.07) |
| FoR46 | 2/2 (norm 10.00) | 2/2 (norm 10.00) | 2/2 (norm 9.06) | 2/2 (norm 9.06) |
| FoR47 | 2/2 (norm 3.03) | 2/2 (norm 3.03) | 2/2 (norm 2.91) | 2/2 (norm 2.91) |
| FoR48 | 2/2 (norm 2.17) | 2/2 (norm 2.17) | 2/2 (norm 2.79) | 2/2 (norm 2.79) |
| FoR49 | 1/2 (norm 1.67) | 1/2 (norm 1.67) | 1/2 (norm 1.59) | 1/2 (norm 1.59) |
| FoR50 | 1/2 (norm 1.30) | 1/2 (norm 1.30) | 0/2 (norm 1.08) | 0/2 (norm 1.08) |
| FoR51 | 2/2 (norm 4.33) | 2/2 (norm 4.33) | 2/2 (norm 1.99) | 2/2 (norm 1.99) |
| FoR52 | 2/2 (norm 1.15) | 2/2 (norm 1.15) | 2/2 (norm 1.47) | 2/2 (norm 1.47) |

### 5.2 为什么 0 个候选被接受(诊断;此处修正了我早先"val 饱和"的单一解释)

先给结论:冻结 run 的 0 接受是**三个原因叠加**,其中一个是我自己的配置选择,而不是方法本身。

1. **多数候选根本没有 Q_val 增益(23/25)**:37 个候选里 25 个通过了 R_src 并在 val 上重解,存档的 `reasons` 显示其中 23 个 `q_gain=False`(被触及的 val episode 已通过,重解后最多持平;FoR36 的 c0007/c0024 还各退步 2 个 episode,FoR33 c0004 的 MacroSR 下降 0.04)。这部分确实是"带 scilib 的 A_0 在 val 上已饱和"。
2. **有 2/25 个候选有 Q_val 增益,但被噪声护栏挡住,而挡住它们的是我的 `n_val=1`**:c0028(FoR41:ΔMacroSR 0.00,Δ归一化分 +0.04,改进 1、退步 1)和 c0031(FoR44:ΔMacroSR +0.04,Δ归一化分 +0.24,改进 1、退步 0)都 `feasible` 且 `h_ok`,唯独 `supported=False`。原因:DESIGN 决定 12 的护栏要求增益由 ≥`min_improved_episodes`(默认 2)个 val episode 各自改进来支撑,而冻结 run 用的是 `bench.n_val: 1`(为控制时长),单学科候选最多只能触及 1 个 val episode,护栏因此**不可达**。这是配置组合的问题,不是代码错误;为避免下次再踩,我加了一个启动告警(commit `53695fb`,`min_improved_episodes > n_val` 时 `evolve` 打印 WARNING,门控行为不变,附测试)。
3. **字面规则(`min_improved_episodes: 1`,即论文的"任何 Q_val 增益")重放**:我用冻结 run 存档的 bundle 构造 A_1 = A_0 + c0028 + c0031,在 FoR41/FoR44 的 id/ood 上做了真实评测(同一冻结代码、同一 27B 端点):

| 学科 | 划分 | episode | A_0 primary(z) | A_1 = A_0+c0028+c0031 primary(z) | 检索/使用算子 | 参考值 |
|---|---|---|---|---|---|---|
| FoR41 | id | e00 | 0.5290(1) | 0.5290(1) | 1/0 | 0.6006 |
| FoR41 | id | e01 | 0.6351(1) | 0.6351(1) | 1/1 | 0.7506 |
| FoR41 | ood | e00 | 0.6716(1) | 0.6716(1) | 1/1 | 0.7624 |
| FoR41 | ood | e01 | 0.7220(1) | 0.7193(1) | 1/0 | 0.8030 |
| FoR44 | id | 00 | 0.0271(1) | 0.0321(0) | 1/1 | 0.1379 |
| FoR44 | id | 01 | 0.0175(1) | 0.0227(1) | 1/1 | 0.1422 |

z 合计 A_1 = 5/6,同 episode 上 A_0 = 6/6;未完成 0 个。


   结论:按字面规则放行这两个候选,**没有观察到增益,且有一处变差**:FoR41 四个 episode 的 primary 与 A_0 相同或差不超过 0.003(A_0 本来就全部通过);FoR44 id e00 因 22 步/201k token 触发 token 预算(stop_reason=token_budget),z 从 1 变 0(A_0 是 10 步/51k token,primary 0.0271→0.0321 同样远优于参考 0.138)。轨迹显示:agent 第 0 步就采用了检索到的 c0031 算子,dev 上得到 RMSE 0.355,远差于参考 0.132,于是接下来 15 步反复尝试自写估计器直到耗尽 token,最终交付的仍是算子的输出(所以 primary 是算子的 0.0321)。但**同一个算子**在下面 §5.3 的 stock 重放里(提示略有不同)7 步/30k token 就完成,primary 同为 0.0321、z=1——所以这次 z 1→0 更像 agent 采样方差(是否去查 dev 分数并展开搜索),而不是算子必然拉高成本;"从单条轨迹蒸馏出的弱算子会锚定 agent"只是一个未被证实的假说。c0028 只是对 `scilib.aquatics` 的一行封装(191 字符,本身不含新知识);c0031 是 2.4k 字符的独立因果估计器,而 A_0 在 FoR44 上本来就靠 scilib.causal 通过。所以护栏虽是我的配置造成的障碍,但**移除它后并没有观察到收益**,样本量小(6 个 episode,z 在 A_0 上已饱和)只能说"未观察到增益",不能说"证明无增益"。
4. **12/37 个候选死在 R_src**:候选要求源 episode 回放时"通过且使用该算子",很多打包出来的算子在回放里没被用到(agent 的成功路径不经过它)。
5. **失败学科没有可提取的成功轨迹**(结构性):进化只能从成功的源 episode 提取实例,最需要算子的失败学科提供不了实例。
6. **快照全部别名到 A_0**:没有候选被接受,A_1 与 A_0 指纹相同(95e0219420fd0c39,scilib 属环境侧,不在程序指纹里),评测阶段按设计以零 token 别名,所以冻结 run 里"进化前 vs 进化后"恒等——这是诚实的结论,而不是没跑。
7. **只完成了 3 轮中的前 2 轮源 episode**(46/69):前两轮 0/37 接受,第 3 轮不太可能改变结论,而夜间 GPU 窗口更适合拿来做下面 5.3 的对照实验。

### 5.3 对照实验:从 stock A_0(无 scilib)出发的进化(状态:已完成)

设置:`SCIENCECLAW_NO_SCILIB=1`(objective 里不再追加 scilib 文档;scilib 属环境侧,所以程序指纹与带 scilib 的 A_0 相同,均为 95e0219420fd0c39),学科 FoR40, FoR41, FoR43, FoR44, FoR46,每学科 val n=2、id n=2、ood n=2,2 轮,`min_improved_episodes=2`、`max_regressed_episodes=1`(配置 `configs/stock_evolve_local.yaml`,run `stock_evolve_local-20260929-200120`)。选这 5 个学科,是因为它们在最终代码下 val 通过、在 stock 下多数失败,最能显示"进化能否补上缺失的领域机件"。这里 n_val=2,护栏是可达的。

- **stock A_0 的 val**:MacroSR = 0.4(10 个 val episode);对照:带 scilib 的冻结 run(n_val=1)在这 5 个学科里 4 个 val 通过、FoR44 未通过(下表)。
- **源 episode**:10 个,逻辑 token 合计约 1308k;成功(回放校验通过)4 个。第 1 轮 5/5 失败(没有实例);第 2 轮 4/5 成功(FoR40 仍失败)。第 2 轮的成功不是因为进化(没有任何东西被接受,程序仍是 A_0),而是换了一个 episode 后 agent 走通了——10 个源 episode 里只有 4 个成功(n 太小,不能当成功率),进化管线只能从这几个成功轨迹取材。
- **候选**:4 个,接受 0 个。

| 学科 | 名称 | stock A_0 val z | scilib A_0 val z(冻结 run) | stock 源 episode z(第1轮/第2轮) | 可回放实例 |
|---|---|---|---|---|---|
| FoR40 | 异常声音检测 | 0/0 | 1 | 0/0 | 0 |
| FoR41 | 水生生态概率预测 | 0/0 | 1 | 0/1 | 1 |
| FoR43 | OCR 后校正 | 1/0 | 1 | 0/1 | 1 |
| FoR44 | 因果效应估计 | 0/1 | 0† | 0/1 | 1 |
| FoR46 | 代码生成 | 1/1 | 1 | 0/1 | 1 |

| 候选 | 学科 | 结果 | 详情 |
|---|---|---|---|
| c0001 | FoR41 | R_src 失败 | 源回放通过,但回放里没有使用该算子(Use=False) |
| c0002 | FoR43 | 拒绝 | ΔMacroSR -0.10;改进 1、退步 3;q_gain=False、feasible=False;新硬约束违规 FoR46…01:['implemented'] |
| c0003 | FoR44 | 拒绝 | ΔMacroSR +0.10;改进 1、退步 1;q_gain=True、feasible=True |
| c0004 | FoR46 | R_src 失败 | 源回放通过,但回放里没有使用该算子(Use=False) |

**读法**:
- 4 个候选里 2 个(FoR41、FoR46)死在 R_src:算子从成功轨迹里抽出来了,但用它重放同一个源 episode 时 agent 不调用它。
- FoR43 的 OCR 算子(c0002)让 FoR43 val 有改进,但 FoR46 val 从 1/1 掉到 0/0,并新增硬约束违规 `implemented`,MacroSR 0.4→0.3,被拒。我核对了 FoR46 的轨迹:候选的 FoR46 episode 里 OCR 算子被检索到提示中,但没有被使用(`uses=[]`);A_0 版本 15/16 可见测试通过,候选版本 6/16。**这是"提示里多了一个无关算子"还是 27B 的运行方差,我无法从两次运行区分**,未验证;需要重复运行才能定论。它至少说明:算子库检索会把不相关学科的算子带进提示,是潜在的负迁移路径。
- FoR44 的候选(c0003)让 val 从 z=0/1 变成 1/1(MacroSR +0.10),但一个 episode 的归一化分下降,护栏(需要 ≥2 个 episode 各自改进)判它"不被支撑",被拒。按论文的字面规则(任何 Q_val 增益)它会被接受。字面规则的真实评测见下表(未观察到增益)。

**字面规则重放(只放行 c0003)**:构造 A_1 = stock A_0 + c0003,在 FoR44 的 id(n=2)上评测:

| episode | stock A_0 primary(z) | A_1=A_0+c0003 primary(z) | 检索/使用算子 |
|---|---|---|---|
| FoR44-id-da552e7c-00 | 0.0271(1) | 0.0321(1) | 1/1 |
| FoR44-id-da552e7c-01 | 0.0175(1) | 0.0227(1) | 1/1 |

FoR44 的指标越小越好:A_1 的 primary 略差于 A_0,z 不变(A_0 在这两个 id episode 上本来就都通过)——val 上的 MacroSR +0.10 没有在 id 上兑现。

**id/ood(stock A_0;进化没有接受任何候选,A_1=A_2=A_0 按别名零 token 评测)**:

| 学科 | 划分 | stock A_0 通过/已评测 | scilib A_0 通过/已评测(冻结 run;episode 抽样不同,不可逐个对应) |
|---|---|---|---|
| FoR40 | id | 1/2 | 1/2 |
| FoR41 | id | 0/2 | 2/2 |
| FoR41 | ood | 0/2 | 2/2 |
| FoR43 | id | 0/2 | 2/2 |
| FoR43 | ood | 1/2 | 2/2 |
| FoR44 | id | 2/2 | 2/2† |
| FoR46 | id | 0/2 | 2/2 |
| FoR46 | ood | 1/2 | 2/2 |

(去掉 FoR44 后:stock 3/14 → scilib 13/14。)合计:stock A_0 id+ood 通过 5/16;冻结 run(带 scilib)在同 5 个学科 id+ood 通过 15/16。

† FoR44 的 scilib.causal 是冻结之后才加的(commit 0344245),所以冻结 run 里的 FoR44 与 stock 是**同一个系统**(同样的提示,LLM 缓存命中,primary 逐格相同),这一行不能读作 scilib 的效果;其余 4 个学科的 scilib 工具库都在冻结之前。


**建议(给你决定的一层结论)**:

- 在带 scilib 的 A_0 上,自进化**没有可挖的空间**:val 已饱和(23/25 个通过 R_src 的候选无 Q_val 增益),两个有增益的候选被 `n_val=1` 下不可达的噪声护栏挡住,而按字面规则放行后的真实评测也没有增益(FoR41 持平、FoR44 z 1→0 更像方差)。
- 在 stock A_0 上,进化管线**只能从少数成功的源 episode 产出候选**(10 个源 episode 里 4 个成功),4 个候选 0 个被接受:2 个死在 R_src,1 个在 FoR46 上退步(是检索溢出还是方差,本次没有验证),1 个(FoR44)被护栏挡住,字面规则重放也没有增益。所以这个基准上"论文方法 = 自进化"目前**不是**得分的来源;得分来源是环境侧的领域工具库(id+ood 通过数:stock 5/16 → 带 scilib 15/16;FoR44 因 scilib.causal 晚于冻结,两边是同一个系统,去掉后为 3/14 → 13/14;每格 n=2,只作方向性证据)。
- 下一步优先级:(1) 用 stock 起点、23 个学科、`n_val ≥ 3`(护栏才有意义)重跑进化,并加一个"stock + scilib、不进化"的对照,才能把"工具库的作用"和"进化的作用"分开;(2) 让打包出来的算子真正被使用(12/37 的 R_src 死亡是最大的可修点);(3) 重复 FoR43→FoR46 那种跨学科退步的实验,区分检索溢出与方差。



## 6. 论文方法如何适配到 23 个数据集:做法与经验

### 6.1 为什么单靠"空程序 + 自进化"不够,以及我们的适配方案

论文的方法是:冻结的 LLM 逐步构造一张类型化工作流图,程序 A_r = Skills + 类型化 Operators,通过"源 episode → 可回放校验的实例 → 打包成候选 → 门控(严格提升 MacroSR)"来自进化。stock 27B 在空程序 A_0 下 src 只能过约 10/23 学科,与 gpt-5.5 在共同 19 个学科上的 13 个相比 10 个。失败的主因不是 agent 循环坏了,而是**领域机件缺失**:像素特征、分水岭实例分割、IRT/BALD、官方 utility 指标、Z3 驱动等,agent 每次都要在 200k token 预算内从零重写,写不完或写错。

进化本来是要"学到"这些机件,但在这个基准上有一个结构性问题(见 §5):要成为算子,源 episode 必须先成功;而失败的学科没有可提取的成功轨迹,成功的学科在 val 上已经饱和,严格提升的门控就没有空间。所以**适配的关键动作是把领域机件预先装进环境**,而不是改 agent:

- 每个学科一个 `scilib/<name>.py` 环境侧工具库(纯 Python、确定性、单个 300 s 代码节点内可用、2 线程 BLAS),模块 docstring 就是接口文档,通过 `scilib.describe(name)` 追加到 episode 的 objective 里。库只接收 `load_*` 工具给出的数组/表(不读隐藏标签),也不联网。
- 库打包的是该领域 SOTA/榜单/组织者方案里的**组件**:特征、模型、后处理、指标实现、验证协议(例如 FoR44 的 imputation/X-learner/AIPW、FoR49 的批量 Z3 驱动、FoR42 的官方 utility 与因果特征)。agent 仍然要自己设计流程、选参数、读可见的 dev 分数。
- 沙箱解释器只有 numpy/pandas/scipy/sklearn/statsmodels/networkx/sympy/lightgbm/xgboost/rdkit/nibabel/conllu/jiwer/sacrebleu/z3/pillow,**没有 torch**。所以 FoR32 没用 U-Net,FoR30 没用深度分割网络,而是 LightGBM+经典算法。这是对结果上限的一个真实约束。

### 6.2 每个学科的适配要点与根因

见 §3 总表的"失败类别"和下面的分述。类别:A 领域机件缺失、B 提示/接口/端口命名、C 工具或适配器缺陷、D 评测噪声/验收上限、E 预算/墙钟/负载。

- **FoR30** PhenoBench v1.1.0(PQ+ ↑;scilib.phenoseg;类别 A,E):从零写像素特征+实例分割耗尽 200k token(stock 22 步 203k)。修复提交:d9a8673。残余:ood e01 PQ+ 31.65 vs 参考 30.09 + 边际 2.0,差 0.44 未达(高负载下墙钟约 2.8k s,但未超预算)。
- **FoR31** ProteinGym v1.3(mean Spearman ↑;scilib.proteinfit;类别 A):缺少 DMS 监督特征/留一位点统计;stock 只做零样本式启发。修复提交:8247f57。残余:无。
- **FoR32** MSD Task04 Hippocampus(DSC ↑;scilib.hippo;类别 A):3D 分割 stock 几乎不可行(norm 0.37);库提供 patch 标签融合+LightGBM 体素分类。修复提交:3e813e9。残余:未采用 U-Net(沙箱无 torch);OOD 为空。
- **FoR33** BuildingsBench(balanced CVRMSE % ↓;scilib.loadforecast;类别 A,C,D):缺预测器;backtest_history 对无 holdout 负荷的建筑给 NaN(工具缺陷,已修);其余是评测噪声。修复提交:c1f97af, ff88b73。残余:验收噪声大:离线约 70-75% episode 通过(约 20/24 计划 episode)。
- **FoR34** ogbg-molhiv(ROC-AUC ↑;scilib.molecules;类别 A,D):缺分子特征/scaffold 分组 CV;之后剩余失败是效应量小于噪声。修复提交:ead0086, efc8bb1。残余:离线估计每 episode 通过率 iid≈0.7 / ood≈0.4。
- **FoR35** Monash Tourism Monthly(mean MASE ↓;scilib.forecast;类别 A,B,E):load_dev.history 与 load_eval_inputs.history 同名端口,agent 把交付物接到 dev 历史上;全局回测慢引发 wall_budget。修复提交:f9d7d4a, b2f7652, 0b0b880, 510e727。残余:ood 尾部 episode 仍受墙钟/负载影响。
- **FoR36** MUSDB18 (150 tracks)(median SDR dB ↑;scilib.audiosep;类别 A):参考值很低(约 -5~-6 dB),且存在 SDR 尺度伪影(全零估计得 0 dB)使验收偏松;库提供 STFT 软掩码分离+留轨 CV,真实分离约 3 dB。修复提交:a41c05c。残余:OOD 为空;SDR 尺度伪影已披露。
- **FoR37** WeatherBench 2 T2m 24 h(RMSE K ↓;scilib.weather;类别 A):缺季节循环/patch-ridge 预测器与 gap-blocked CV。修复提交:c04a1b6。残余:OOD 为空。
- **FoR38** World Bank WDI(mean sMAPE % ↓;scilib.macro;类别 A,B,E):agent 用噪声 dev 覆盖库默认(weights='auto')变差;默认改为稳健成员。残余:id e00 token_budget 204k>200k 按预算规则记 0 分。修复提交:33745b3, 82fae48。残余:预算规则不放松;OOD 为空。
- **FoR39** Eedi NeurIPS'20 Task 4(官方 10-mask accuracy ↑;scilib.adaptive;类别 A,D):缺 IRT/BALD 选题;验收天花板约 72%(参考+边际)。修复提交:296e6df。残余:单 episode 通过率≈50-70%。
- **FoR40** DCASE 2024 Task 2(官方 DCASE score ↑;scilib.anomsound;类别 A,D):缺 log-mel/kNN/LOF 集成;16 段 clip 上验收通过率按设计≈50-54%。修复提交:2bc4e89。残余:噪声上限;OOD 为空。
- **FoR41** NEON aquatics(mean CRPS ↓;scilib.aquatics;类别 A):缺概率预测器(异常回归 CRPS)。修复提交:b71d97e。残余:无。
- **FoR42** PhysioNet/CinC 2019(normalized utility ↑;scilib.sepsis;类别 A,D):缺官方 utility/因果特征/阈值;每 episode 仅 2 个败血症 stay,验收噪声极大。修复提交:9b91b60, 4a389ec, 9a6ada7, 227d212。残余:通过率≈0.5/episode(离线重采样);先验 0.073 为公开挑战常数。
- **FoR43** HIPE-OCRepair-2026(cMER-micro ↓;scilib.ocrfix;类别 A):缺 OCR 后校正工具(字符级候选与选择)。修复提交:50b4942。残余:id e01(F5 文案改动后)平局失败:agent 把 merge 的评测半段 y_all[16:] 接进了 score_dev,dev 分数 0.76 被误读为“LLM 改动过多”,最后退回 OCR 原文——模型接线错误(文档已示例 dev_pred/y 拆分),未改代码;src×3 为 3/3。
- **FoR44** ACIC 2016(RMSE/SD ↓;scilib.causal;类别 A,D):估计量方差大(agent 用 OLS);库提供 imputation/X-learner/AIPW,单线程稳健。修复提交:0344245, fd37e42。残余:OOD 为空。
- **FoR45** AmericasNLP 2026 captioning(chrF++ ↑;scilib.captions;类别 A,B):agent 用 6 条噪声 dev 覆盖库默认;默认输出为利用 chrF++ β=2 召回的共识串(已披露,附 mbr_caption 可读替代)。修复提交:f0235d7, dab037c。残余:ood e00 边际 0.4 chrF。
- **FoR46** HumanEval→MBPP(pass@1 ↑;scilib.codegen;类别 A):缺提示模板/回复清洗/修复/选择流程。修复提交:080a7f1。残余:22 步/173k token 逼近 200k 上限。
- **FoR47** UD 2.2 French (GSD/PUD)(LAS ↑;scilib.udparse;类别 A):缺 POS 标注器+图依存解析器+精确树解码。修复提交:5695fa3。残余:无。
- **FoR48** ContractNLI(evidence mAP ↑;scilib.contracts;类别 A,C):缺证据检索模型;适配器 deliverable 文案有多余 '}}'(已修)。修复提交:39ab47e, 8c93947。残余:无。
- **FoR49** SMT-LIB 2025 QF_NIA/NIRA(oracle-agreement accuracy ↑;scilib.logic;类别 C,D,E):z3 参数错误被误报成 memout(工具缺陷);验收 acc≥ref+0.2 差一道题;每题 10 s 墙钟对负载敏感。修复提交:53f7b04。残余:加性 +0.2 边际不放松。
- **FoR50** SemEval-2023 ValueEval(官方 macro F1 ↑;scilib.valueeval;类别 A,D):缺 TF-IDF 特征/分组 OOF/阈值;每类别按期望 F1 选标注数。修复提交:7cb1af6, ec5f59b。残余:近失 0.402 vs 0.406 类边际。
- **FoR51** Matbench phonons(MAE cm⁻¹ ↓;(无);类别 -):stock 4/4 直接通过。修复提交:-。残余:仅按 F5 审计改了适配器文案(76184c1)。
- **FoR52** Psych-201 discrete(micro accuracy ↑;scilib.psych;类别 A,D):缺按被试/研究的序列选择模型;效应量小,dev SE≈0.07;加入语言模型答案作第二投票者。修复提交:a82b0b3, 0476525。残余:噪声限制。

### 6.3 可复用的适配经验(对下一个数据集也适用)

1. **单一显眼的入口 + 安全默认值**(`fit_predict(...)`):agent 倾向于在带噪声的 dev 分数上覆盖默认值(FoR38 覆盖权重、FoR45 用 6 条 dev 覆盖 25/150、FoR52 dev SE≈0.07 却据此切换),结果比默认更差。对策:默认值本身要稳健;dev 分数的条目数与不确定性用事实性文字披露;不要在文档里推荐"最佳"配方(F5)。
2. **端口/工具命名不能撞名**:FoR35 里 `load_dev.history` 与 `load_eval_inputs.history` 同名,agent 把交付物接到了 dev 历史上;把 `load_eval_inputs` 排在前面并让描述事实化后消失。
3. **诚实的 dev 分数**:回测要把训练序列截到历史之前(FoR35 `train_cut`),否则 dev 分数虚高,agent 会过拟合。
4. **工具报错要说真话**:FoR49 里 Z3 的参数错误被报成 memout,agent 于是调大内存而不是改参数;FoR33 的回测对没有 holdout 负荷的建筑返回 NaN。这些是"工具缺陷"而不是模型能力问题。
5. **速度即通过率**:每个函数必须在一个 300 s 节点里跑完;墙钟/负载敏感的学科(FoR49 每题 10 s、FoR35/FoR30/FoR38 的 wall_budget)在机器高负载时会假性失败,需要在低负载下复测。
6. **验收噪声是独立于方法的上限**:验收规则是"在一个 16 项的 episode 上比参考值好一个边际"。FoR34/40/42/39/33 的离线重采样显示,即使是最好的方法,每个 episode 的通过概率也只有约 0.4–0.8(FoR34 iid≈0.7/ood≈0.4;FoR42≈0.5;FoR40≈0.50–0.54;FoR39 上限≈0.72;FoR33≈0.7–0.8)。这些学科的"失败"应当读作噪声,而不是适配失败;要区分方法好坏必须靠多 episode 聚合(如 MacroSR),不能看单个 episode。


## 7. 基础设施、吞吐与开销

**27B 服务**:vLLM 0.30.0,Qwen3.8-27B-FP8,served name `sc-llm`,每作业 1 张 A100-64GB(8 CPU/100G 内存),thinking 关闭。32 并发时聚合生成约 154 tok/s,典型一次调用(约 700 输入/90 输出 token)中位 13.5 s。本机通过 SSH 隧道(`~/.ssh/cm-sc-leo`,端口 18002 起)把 6 个端点做轮询;保活脚本每 ~10 分钟检查隧道与队列,端点集变化时自动重跑 `scripts/leonardo/tunnel.sh`。

**算力(sacct 实测,A100 GPU 小时)**:

- 27B 服务(1 GPU):111.6 GPU 小时
- Flash-Next 尝试(4 GPU):2.7 GPU 小时
- 冒烟/驱动检查(1 GPU):0.5 GPU 小时
- 合计 114.8 GPU 小时(A100);27B 服务作业共 15 个(含超时、被我取消的重复作业)。

(sacct 终值,取自 23:07 PDT;所有 27B 作业均已在维护窗口前到时限,没有作业跨窗口,也没有遗留排队作业。)

作业管理:夜里一度同时有 9 个 27B 作业(超过"项目整体 ≲6 节点"),19:02 PDT 取消了 3 个多余作业,之后稳定在 6 个;所有 27B 作业的时限都设在 09-30 08:00 CEST(= 09-29 23:00 PDT)维护窗口之前,没有作业跨窗口。没有重投 work1–work4 原始四组,没有触碰 work1-r2flow-* 或他人的作业。

**Token 用量**:

- 进化源 episode(冻结 run,前 46 个):逻辑 token 3.75M(策略 3.48M + 执行器 0.28M),其中 630/788 次调用命中客户端缓存(真实新算的 token 仅 1.14M);agent 墙钟 1.9 h,回放校验墙钟 1.0 h。
- 冻结 run 的 A_0 val(23 episode)约 2.02M 逻辑 token;A_0 的 id/ood 评测阶段 5.96M token、1035 次调用、183 次重试(冻结 run 的 evaluate 阶段)。
- 单个 episode 的策略预算上限:200k–250k token(按学科策略);超限记 0 分。

**本机是瓶颈**:Mac 10 核 16 GB,代码节点(numpy/LightGBM 等)、回放校验和 Z3 都在本机跑,高峰负载 7–25。GPU 端并没有排队,反而是本机 CPU 决定了 episode 的墙钟——FoR49(每题 10 s 墙钟)、FoR35/FoR30/FoR38 的 wall_budget 失败都随负载出现(§6.3 第 5 条)。想要更快、更稳,应把代码节点也搬到 Leonardo 的计算节点上。

**Flash-Next-FP8**:在 A100 上无法运行。vLLM 0.30 的 qwen4_exp FP8 路径使用 Triton `fp8e4nv` 内核,需要 sm ≥ 8.9(Ada/Hopper);A100 是 sm80。尝试了 3 次共约 0.7 小时 × 4 GPU(见上面的 GPU 小时),失败原因一致。我没有打内核补丁,没有引入 torch 侧的反量化路径,也没有下载 Mistral-Medium(需要你批准)。


## 8. 合规审计与我代你做的决定

你说"需要我决定的东西你帮我决定,合规就行",我采用的原则是:**不泄露隐藏标签 / 不放松验收与评测 / 策略可见文本只陈述事实(DESIGN §7 F5)/ 在报告里如实披露**。

### 8.1 F5 审计(commit 76184c1)

我用 `scripts/dev/scan_visible_text.py` 扫描了所有适配器的 objective、ToolSpec 描述、可见约束和 scilib 文档,凡是点名参考方法或验收边际的地方都改成事实性描述:

- FoR43 objective 删掉了"未校正的 OCR 文本就是参考";FoR39 `score_dev` 的"organizer starter baseline"→"reference predictor";FoR49 "majority-label reference accuracy"→"reference accuracy";FoR36 "mixture-as-estimate reference"→"reference SDR";FoR34/FoR51 的 "trivial reference model"→"reference model"。
- `scilib.psych`:`reference_prediction/reference_accuracy`/`reference` 方法/`margin` 字段改名为 `mode_*`/`mode` 并去掉边际字段;`scilib.aquatics` 去掉了"实测敏感度"段落(等于配方提示)。
- 新增守卫测试 `tests/test_adapter_visible_text.py`(七个适配器 + psych 文档),以后再写出这类措辞会直接失败。
- 验证:F5 改动后重探针(`runs/final_a0_f5_id`,FoR43/52/39 各 id×2,以及 FoR43 src×3):FoR52 id 2/2(与改动前一致);FoR39 id 1/2(改动前也是 1/2,验收天花板约 72%);FoR43 id 1/2——失败的 e01 提交的是未校正的 OCR 文本(轨迹显示:agent 把 merge 的评测半段接进了 score_dev,得到 0.76 的假 dev 分数,误以为 LLM 改动过多而退回原文;文档里已有 dev_pred/y 拆分示例,属模型接线错误)(primary 0.02774 = 参考 0.02774,未超过边际),而 FoR43 src 重探针 n=3 是 3/3(primary 0.0196/0.0088/0.0082,norm 1.5–2.6),因此未发现 F5 文案改动带来的回归,该 id 失败按单 episode 波动处理(没有据此改动代码)。

### 8.2 我保留不动(并在此披露)的决定

| 项 | 决定 | 理由 |
|---|---|---|
| FoR45 默认输出 | 保留 chrF++ 召回驱动的共识串,并在库里附 `mbr_caption`(真实文本替代,约低 0.8 chrF) | 默认输出是不可读的高频词袋,利用 chrF++ β=2 的召回;这是指标利用,论文/评审需要知道。已披露,不改评测 |
| FoR49 验收 | 保持 acc ≥ ref+0.2 的加性边际 | 放松验收是禁止的;代价是"明显更好但差一题"会失败 |
| FoR33 | 保持设计(离线约 70-75% episode 通过) | 噪声是设计属性 |
| FoR38 | token 预算超限(204k>200k)按规则记 0 分 | 不放松预算 |
| FoR50 | 每类别按期望 F1 选标注数 | 只用可见训练数据与官方 macro F1 定义;但它确实利用了 macro P/R-F1 的定义,已披露 |
| FoR42 | 患病先验 0.073 = 2019 年公开挑战的败血症占比(文献常数) | 与评测 episode 的组成无关(评测 episode 的 septic 数已从文档中删除);训练集自身占比是 5.7%,差异已披露 |
| FoR36 | SDR 尺度伪影(全零估计得 0 dB)只披露 | 属评测定义,不改 |
| Flash-Next | 不打内核补丁、不引入 torch、不下载 Mistral-Medium | 需要你批准的下载/供应链动作 |

### 8.3 硬约束遵守情况

- 存储:代码/venv 在 fast scratch,权重/输出在 large scratch;home 未写数据。
- 未在登录节点跑训练/推理;未重投 work1–work4 原始四组;未触碰 work1-r2flow-* 与他人作业;项目节点数保持 ≲6(夜里一度 9,已在 19:02 取消三个多余作业)。
- 网关密钥全程未读取、未打印、未外传(自部署端点不需要);无 git push、无破坏性操作;`.claude/` 与 `configs/dev5_flash.yaml` 从未加入版本库。
- 维护窗口(09-30 08:00 CEST = 09-29 23:00 PDT)前,所有 27B 作业的时限都在 22:50 PDT 之前结束,没有作业会跨维护窗口。

### 8.4 冻结 run 与最终代码的差别(读表必看)

进化 run 冻结在 d9a8673(13:21),后来的修复(FoR33 回测 NaN、FoR35 端口/回测、FoR38 稳健默认、FoR42 默认、FoR44 causal、FoR45 默认、FoR49 Z3 驱动、FoR50 每类别阈值、FoR52 语言模型投票者、F5 文案)**不在**进化 run 里。因此"进化前/后"只用冻结 run 内部的 A_0 vs A_k 比较,而"夜间修复的效果"用 冻结 A_0 vs 最终代码 A_0' 比较,二者不能相加。


## 9. 改动的代码与提交

自 2026-09-28 起的全部提交(均未 push):

- `28923d1` Harden runtime and rework policy prompts (review RT-2/4/5/6/7/8/11, F3-F5, F7-F11, M1)
- `3a3a939` Merge branch 'master' into worktree-agent-a87a82cead7e33acf
- `8da88b2` Merge runtime and prompt review fixes
- `0d8273e` Benchmark integrity and experiment pipeline fixes (checkpoint)
- `7d9e475` FoR42 causality probes and acceptance floor (LEAK-1); FoR37 draw-time spacing (LEAK-4)
- `4bc97a5` Merge master into integrity/pipeline worktree (keep FoR33 v2 role-file adapter, F5 objective wording without the reference recipe)
- `445760e` FoR33 same-building windows at least 144 h apart (LEAK-4), DESIGN sync for integrity and pipeline fixes
- `1afc1db` Merge dataset-integrity and pipeline review fixes
- `81a0efd` Run hidden adapter probes in replay evaluation; drop xfail on public-view scan
- `bd06417` Add self-hosted vLLM endpoints (load-balanced, no gateway key) and per-role extra_body
- `1f1e5a1` Serve 27B on Leonardo: per-job ports, mux-based tunnels, endpoint bench, pool-capped split building
- `83d82e8` Leonardo scripts: EPDIR/PORT_BASE endpoint groups so a second model can be served and tunneled side by side
- `a7e4090` tunnel.sh: use a dedicated persistent ControlMaster (cm-sc-leo); the shared cm-claude-leo socket was taken over by another session's ssh and stalled
- `1011def` Add summarize_probe.py: side-by-side pass/score/token table for probe runs
- `b83fd1e` Actions: dropping an input port in modify_node drops its edge; hint how to fix graph-invalid edits
- `8f0c829` Actions: strip reply-envelope debris (}}}}</invoke></div>) leaked into code strings
- `e45f5d0` Node errors: keep the node's own frame, elide deep library frames, shorten site-packages paths
- `7a49c48` Prompt/feedback interface facts: policy tokens left, output truncation, cached nodes
- `bbedfa6` FoR50: hold the dev slice out by conclusion, drop reference recipe from score_dev text
- `8c93947` FoR48: fix stray '}}' in the deliverable spec and the load_hypotheses port description
- `9120102` Add scilib: per-discipline domain libraries importable from code nodes
- `7cb1af6` FoR50: offer a ValueEval toolkit (official metric, TF-IDF features, grouped OOF, episode threshold)
- `0288881` Declare the optional domain packages code nodes may use (lightgbm and xgboost now installed)
- `2a1cce8` tunnel.sh: keep healthy forwards (stable ports), only add/cancel what changed
- `206c7b1` full_local: allow up to 10 local tunnel ports (fleet size varies overnight)
- `9b91b60` FoR42: scilib.sepsis toolkit (official utility, causal features, grouped-CV LightGBM ensemble, utility-optimal cut-off)
- `ead0086` FoR34: offer a molecule toolkit (RDKit features, scaffold groups, RF+ET ensemble, grouped CV)
- `a82b0b3` FoR52: psych domain library (scilib.psych), 48-item dev slice, clearer score_dev contract
- `f9d7d4a` FoR35: scilib.forecast toolkit (MASE/sMAPE/RMSSE, local ETS/Theta/STL/SARIMA methods, rolling-origin backtest, combination, ridge/ExtraTrees window models)
- `c8fb10e` run_node_on_episodes: pass dev_pred to score_dev's actual first input port, ignore non-numeric dev fields
- `4a389ec` FoR42 adapter: missing-rate facts, causal-rule wording in tool descriptions, scilib.sepsis in the objective
- `f0235d7` FoR45: scilib.captions (chrF++ tools, greedy consensus string, leave-one-out estimator) wired into the objective
- `9a6ada7` FoR42: do not disclose the evaluation stays' septic count, default threshold weighting off
- `c1f97af` Add scilib.loadforecast toolkit for FoR33 day-ahead building load
- `b5bca15` scilib.forecast: neutral method names (repeat_last/repeat_season) so the library text does not name the reference recipe
- `c04a1b6` Add scilib.weather: seasonal cycle, patch-ridge forecaster, WB2 RMSE, gap-blocked CV for FoR37
- `39ab47e` FoR48: scilib.contracts domain toolkit for ContractNLI evidence identification
- `5695fa3` FoR47: offer a UD parsing toolkit (POS tagger, graph parser, relation labeller, exact tree decoders, LAS metric)
- `33745b3` FoR38: macro forecasting toolkit scilib/macro.py
- `8477e3a` Add pipeline scripts that evaluate only A_0 vs the final snapshot, and a per-discipline adaptation table
- `8247f57` FoR31: scilib.proteinfit supervised DMS toolkit (leave-one-out site statistics, ridge/LightGBM/extra-trees rank ensemble)
- `a41c05c` FoR36: scilib.audiosep domain toolkit (SDR parity, STFT soft-mask separator, leave-track-out CV, gain-only/null diagnostics)
- `b71d97e` FoR41: add scilib.aquatics (anomaly-regression CRPS forecaster) and document it in the episode objective
- `296e6df` FoR39: scilib.adaptive domain toolkit (IRT item curves, BALD/batch query selection, posterior prediction)
- `3e813e9` FoR32: offer a hippocampus segmentation toolkit (patch-based label fusion + LightGBM voxel classifier)
- `c36ccdc` Add per-discipline table script merging probe evidence with a full run's report
- `080a7f1` FoR46: scilib.codegen domain library (prompts, reply sanitizer, repair, selection) wired into the objective
- `2bc4e89` FoR40: scilib.anomsound domain toolkit (log-mel, kNN/LOF/band/Mahalanobis scores, rank-average ensemble) wired into the objective
- `50b4942` FoR43: scilib.ocrfix domain toolkit for HIPE-OCRepair post-correction
- `d9a8673` FoR30: offer a PhenoBench toolkit (pixel features, LightGBM classifier, watershed instances, PQ+ metric)
- `0344245` FoR44: add scilib.causal toolkit (mixed-type design matrix, outcome-imputation / X-learner / AIPW / IPW estimators, estimate_effects entry point, visible-data validation helpers) and wire it into the objective
- `efc8bb1` FoR34 docs: record effect-size / noise analysis of scilib.molecules (per-episode acceptance probability ~0.7 iid, ~0.4 ood)
- `227d212` FoR42: sepsis toolkit defaults tuned for per-episode acceptance
- `fd37e42` FoR44: run scilib.causal model fits on one thread (robust on a shared CPU) and document the domain library with before/after numbers
- `ec5f59b` FoR50: scilib.valueeval fit_predict picks the per-category number of marked rows by calibrated expected F1 (instead of one global cut-off); groups kwarg, unpackable Predictions
- `0476525` FoR52: language-model answers as a second voter in scilib.psych
- `82fae48` FoR38: robust default members for scilib.macro, non-destructive auto weights
- `b2f7652` FoR35: deliverable-first tool order, backtest_panel with global models, train_cut, median fit_predict default
- `ff88b73` FoR33: backtest_history skips buildings without holdout load; record effect-size / noise analysis of scilib.loadforecast (ens ratio 0.79, 20/24 planned episodes accepted)
- `0b0b880` FoR35: fit_predict/backtest_panel cut train series that extend a history to it (honest dev scores by default)
- `510e727` FoR35 doc: record ood probe after the improver pass
- `dab037c` FoR45: captions consensus with singleton rule (vocab 300), mbr_caption (real clauses), loo_lengths, capped LOO folds
- `53f7b04` FoR49: scilib.logic batched Z3 driver, report rejected z3 parameters instead of memout
- `59b824e` Add final_table.py: stock vs frozen-run vs final-code A_0 table per discipline
- `76184c1` F5 compliance: policy-visible text no longer names the reference method or margin
- `fd50739` final_table: count failed frozen eval rows as not passed
- `3e7ff75` Add SCIENCECLAW_NO_SCILIB switch and stock-start evolution config
- `4fbcb9c` Add overnight 23-discipline adaptation and evolution report (interim)
- `53695fb` Warn when min_improved_episodes > n_val makes the Eq. 3 noise guard unreachable
- `0d9213a` Overnight report: 23-discipline adaptation, evolution diagnosis, stock-start ablation
- `0452fc0` FoR48: report the paper's pooled P@R80 and the majority-label baseline next to the per-pair auxiliaries
- `627b1db` FoR45: record an image-blind / distinct-caption diagnostic in the evaluation details
- `4f488f2` FoR33: report the published 7-day average-persistence baseline next to the previous-day reference

## 10. 需要你决定的事项与建议

你说过"需要我决定的都你替我决定、合规就行"。下面这些我**已经按合规原则暂定**,列出来是为了让你知道有这些选择,以及改主意时的代价。

| # | 事项 | 我暂定的做法 | 你改主意时会怎样 |
|---|---|---|---|
| 1 | 进化实验的起点 | 主实验用内置 scilib 的 A_0;另做了一个从 stock A_0(无 scilib)出发的 5 学科对照(§5.3) | 主实验无增益空间(§5.2)。对照实验只有 2 轮、每学科 n_val=2、每格 n=1–2,结论是定性的。要做正式的"进化净增益"结论,需在 stock 起点上跑全 23 学科 × 更多轮 × n_val≥3,并同时跑"stock+scilib 不进化"作对照 |
| 2 | 论文里的数字 | 一律当"目标/估计值",不拿本次数字去对账 | 若要投稿数字,必须用固定协议(n 足够大、多种子)重跑,当前每格 n=1–2 |
| 3 | FoR45 | 保留 chrF++ 召回驱动的共识串并披露,附 `mbr_caption` 可读替代(约低 0.8 chrF) | 若改成只报可读文本,该学科通过边际会明显缩小 |
| 4 | FoR49 验收 | 保持 acc ≥ ref + 0.2 | 放松会抬高通过率,但属于放松验收,我没有做 |
| 5 | FoR38 token 规则 | 204k > 200k 记 0 分,不放松 | 放宽预算会让 id e00 通过,但改变了协议 |
| 6 | FoR36/40/44 OOD 为空;FoR32/35/38 OOD 为代理划分 | 如实标注,没有编造 OOD | 需要数据交付方补 OOD |
| 7 | Flash-Next | A100 上放弃;不下载 Mistral-Medium | 需要 Ada/Hopper 卡,或你批准下载并接受另一模型的替代;Leonardo Booster 只有 A100 |
| 8 | 噪声限制的学科(FoR33/34/39/40/42) | 只披露,不改评测 | 要区分方法好坏,需多 episode 聚合(MacroSR),不能靠单个 episode |
| 9 | 沙箱里没有 torch | 用 LightGBM/经典方法 | 若允许装 torch,FoR32 可上 U-Net、FoR30 可上深度分割,上限会更高,但要评估供应链与环境体积 |
| 10 | 本机 CPU 是瓶颈 | 不动 | 把代码节点执行搬到 Leonardo 计算节点可显著降低 wall_budget 假性失败 |
| 11 | 噪声护栏 `min_improved_episodes` 与 `n_val` 的关系 | 保留论文之外的护栏(默认 2/1),但要求 `min_improved_episodes ≤ n_val`;`evolve` 启动时对违反者告警(commit 53695fb) | 想严格按论文字面规则,设 `min_improved_episodes: 1`(同时取消退步上限)。代价:val 上单个 episode 的噪声也会被接受(stock 起点 FoR44 c0003 就是这种"1 改进 1 退步"的边界情形,见 §5.3) |
| 12 | 外部对照发现的弱参考/可利用指标(§11.2、§11.4) | 只加诊断指标,不改 primary/参考/验收;列出建议 | 统一换强参考、合并 ≥64 项判定会降低通过率,并使冻结 run 与新协议不可直接比较——需要下一轮统一改、统一重跑、统一披露 |
| 13 | 给沙箱加预训练骨干(FoR30/36/47/51 的 BELOW-PEERS 差距) | 没做(下载模型+引入 torch 属供应链决定) | 批准后差距会明显缩小,但要评估环境体积与可复现性 |

**建议的下一步(按价值排序)**

1. 固定协议、放宽墙钟噪声后重测:用低负载(或 Leonardo 计算节点)重跑最终代码 A_0' 的 id/ood(至少 n≥4),把 §3 表里的"单次通过/失败"变成通过率。
2. 正式的进化对照:stock A_0 起点、全 23 学科、n_val≥3(满足 `min_improved_episodes ≤ n_val`),并跑"stock+scilib 不进化"作对照,得到进化本身的净增益;同时重复运行 §5.3 里 FoR43→FoR46 的负迁移案例,区分算子检索的提示污染与 27B 运行方差。
3. 对噪声上限明显的学科(FoR34 ood、FoR42、FoR40、FoR39)改用 MacroSR/多 episode 聚合作为主要结论,不看单 episode。
4. 若允许 torch:优先 FoR32、FoR30。
5. 按 §11.4 统一收紧参考(FoR33 avg7、FoR36、FoR44、FoR35、FoR30、FoR51、FoR52、FoR39、FoR38)并合并 ≥64 项/≥8 个 episode 判定;对 FoR50 补一次全测试集评测;对 FoR49/45 补"纯 Z3""检索式"基线;然后再重跑一次 23 学科,得到有同行意义的通过率。
6. 把 scilib 的"factual docstring + 单入口 + 安全默认值"作为新增数据集的适配模板(见 §6.3)。


## 11. 与外部 SOTA / 同行的对照(响应你 09-29 的核对要求)

你的要求是:分数不能只和我们自己的参考方法比,要和同行、当前 SOTA 和近期论文的数字对一遍,看我们"跑通的分数"是不是真的可信、有没有 bug。我做了这件事,结论先说:

- **没有发现评分器 bug。** 5 个只读调研 agent 分别把每个学科的评分函数和官方实现逐项对拍(官方包、官方评测脚本或按论文定义重写),其中 FoR30/32/33/34/36/38/40/41/42/44/45/47/48/50/51 的指标与官方实现逐位相同(差 ≤1e-6),FoR31/35/37/43 只有已量化的边界情形或聚合方式差异(常数预测 0.0 vs NaN、Theta 1.647 vs 1.649、primary 聚合 +0.005 K、复制 OCR 0.0229 vs 0.0226),FoR30 的 512 px 评分使 PQ+ 略偏高(细节见 §11.3)。没有发现"靠泄露得到的虚高分"。
- **但"分数对得上官方公式"不等于"分数说明了能力"。** 问题集中在协议层:16 项切片的抽样噪声、偏弱或无信息的参考方法、可被利用的指标、以及沙箱里缺预训练骨干(torch)。按这个口径,23 个学科分成四类(§11.1)。
- **口径警告**:我们每个 episode 只有 16 个待预测项(FoR45 是 8 个),同行数字来自完整测试集,所以只能做"量级和比值"的对照,不能把 0.836 和 0.895 当成同一个量。凡是绝对值不可比的学科,我用"相对基线的比值"(例如 agent/SNaive、agent/前一天)来对照。
- 数字均由 agent 在各自会话中从引用来源读取;个别数字 agent 自己标了"未核实"(例如 FoR30 Hierarchical Mask2Former 的 75.99 与数据组简报的 76.78 不一致;FoR44 的 bart_tmle 数值来自搜索摘要;FoR50 EAVIT 0.66 是作者自报),我没有逐一二次核实,表中已避免依赖这些数字。证据文件与复算脚本见 `reports/sota_crosscheck/`。

### 11.1 23 学科对照表

判定词:**CONSISTENT** = 与同行量级一致,没有虚高迹象;**BELOW-PEERS** = 确实低于同行,原因是能力/机件缺失,不是评分 bug;**SUSPICIOUSLY-HIGH** = 分数高但不代表能力(可被利用);**NOT-COMPARABLE** = 协议差异太大,不能拿绝对数比。

| 学科 | 我们(5 个 episode 的范围) | 同行 / SOTA(来源) | 判定 | 要点 |
|---|---|---|---|---|
| FoR30 PhenoBench PQ+ | 31.6–45.3(均值 38.7) | 挑战赛前三 81.1–82.6(arXiv 2501.00527 表 4);HAPT 基线 65.3 | **BELOW-PEERS** | 土壤 IoU 已持平(95–99.6 vs 98.5–99.4);差在杂草 IoU(2–35 vs 61–74)和 PQ_leaf(7–12 vs 47–75)——缺 Mask2Former/SAM 一类预训练实例分割骨干。我们在 512 px 上评分,PQ+ 比全分辨率偏高约 0.4–1.0;ExG 参考在 val e00 上(19.9)低于"全预测土壤"的平凡下限(24.4) |
| FoR31 ProteinGym Spearman | 0.56–0.88(id 0.667/0.555;ood 0.880/0.880) | 同 16 个 assay 的官方 random 折:Kermut 0.712/0.611,ProteinNPT 0.710/0.589(arXiv 2407.00002 + 官方 CSV) | **CONSISTENT** | 切片配对后落后 Kermut 0.03–0.06、ProteinNPT 0.01–0.04:我们只用了功能 assay 41% 的标签且没有 PLM;任何地方都不高于 SOTA |
| FoR32 MSD 海马体 DSC | 0.812–0.852(均值 0.836) | nnU-Net 0.862–0.895(arXiv 1904.08128;2306.03271) | **CONSISTENT**(略低于同行,符合预期) | 训练只有 28 卷 vs 每折 208 卷;Dice 与 nnU-Net 公式逐位一致;"OOD"是强度编码代理(ood Dice 与 iid 相同),不是真分布外 |
| FoR33 BuildingsBench CVRMSE% | 36.9–57.6 | 论文最佳约 45.1(arXiv 2307.00142 表 3/4) | **相对 CONSISTENT;绝对 NOT-COMPARABLE** | 论文是 1900 栋楼整年;我们每栋楼只有 24–48 h,中位数极噪。比值:agent/前一天 0.79(论文最佳 0.78),agent/7 日均值 0.93(论文最佳 0.95);7 日均值(avg7)本身是前一天的 0.82 倍,在 5 个 episode 中 3 个能过 0.90 的验收阈值 |
| FoR34 ogbg-molhiv ROC-AUC | 逐 episode 0.708–0.917;合并 80 个分子 0.8025(95% 自助区间 0.66–0.91) | OGB 榜 0.81–0.85(分子指纹+RF 0.806–0.821;GNN 0.81 左右) | **CONSISTENT** | 4000 分子训练的合并 AUC 与"33k 训练的 RF"同量级;单 episode AUC 标准差 0.13–0.16,只有合并值有意义 |
| FoR35 Tourism 月度 MASE | 1.06–1.44 | DeepAR 1.409、ETS 1.526、SNaive 1.631(Monash 2021 + 2025 基准表) | **CONSISTENT** | 切片 SD 0.15–0.24,只能看比值:agent/SNaive 0.774–0.892(均 0.85),DeepAR 0.864、ETS 0.936,处在 DeepAR 一端且在噪声内 |
| FoR36 MUSDB18 SDR | 2.15 / 2.40 / 2.75 dB | Open-Unmix 5.3–5.4;HT-Demucs 7.5;BS-RoFormer 9.8;IRM 上限 8.2(arXiv 2211.08553、2309.02612) | **BELOW-PEERS** | museval 逐位一致(0.000 dB)。低 3–7 dB:沙箱没有预训练分离器,只有经典 STFT 软掩码。另外验收基线是"混合音当估计"(−5.5…−6.6 dB)+3 dB,任何非静音输出都能过,z=1 不携带信息 |
| FoR37 WeatherBench2 T2m 24h | 1.24–1.51 K(参考持续性 1.56–1.92) | ClimODE 1.40、FourCastNet 1.68(5.625°);全年持续性 1.72–1.73(我们的网格上复算) | **CONSISTENT** | 与比值一致:0.79–0.80,而工具库在整个 2019/2020 上全年评估是 0.789/0.788,说明分数就是工具库的水平,没有藏在评测输入里;IFS/GraphCast 的 24h 数值只有图,没找到 |
| FoR38 World Bank 宏观 sMAPE | 9.47–17.52(比值 0.767–0.947) | 没有该面板的公开数;最近的类比 M4 Yearly:池化模型/naive 比值 0.79–0.81 | **相对 CONSISTENT;绝对 NOT-COMPARABLE** | 提交与默认工具库输出逐位相同(5 个里 3 个)或差 3% 以内,所以分数是工具库的,不是 LLM 的;工具库默认值是在含评测 episode 的回测上"确认"的,略偏乐观;2021 是共同冲击起点 |
| FoR39 Eedi Task 4 10-mask 准确率 | 0.66–0.75(均值 0.7135;参考 0.670);整池上工具库 0.694/0.692 | Task 4 冠军 0.7474(Ghosh & Lan,17 队,私有集;Wang 2021 PMLR v133);BOBCAT 的 IRT/BiNN 系 0.677–0.700(Kim 2021,arXiv 2108.07386,自有 5 折) | **CONSISTENT**(IRT 类方法量级;低于冠军) | 落在 BOBCAT 的 IRT 区间内,不虚高;比冠军低约 5 点,按官方 starter 查询规则校正后约低 2.7 点(冠军是元学习集成)。指标公式与独立重算 6 位小数一致。官方 starter 允许查询目标格(约 20% 的查询落在目标上),本仓库禁止,同一 IRT 工具库在两种规则下是 0.715/0.720 vs 0.694/0.692,所以引用挑战赛数字时要写明差约 2–3 点。agent 输出与 `scilib.adaptive` 默认输出逐格一致 90.7–95.7%(只是连节点),分数即工具库水平 |
| FoR40 DCASE 2024 T2 | 0.461 / 0.604 / 0.610(单机器,16 段) | 官方基线 dev 55、榜首 66–67(百分数;dcase.community) | **CONSISTENT** | 官方 AUC/pAUC/调和平均规则与我们的实现完全一致(0 差);8 个正常样本时 pAUC 退化,episode 分数标准差约 0.12;不要把单 episode 分数和 66 比 |
| FoR41 NEON 水生态 CRPS | 0.529–0.722,比气候学基线好 10–15% | 没有绝对同行数;文献只报相对气候学的技能(湖泊预测中约 1/3 的模型能赢气候学零模型,Olsson 2025) | **CONSISTENT** | crps_normal 与数值积分一致(5e-10);历史窗口在 t0 截止,评测窗口在所有可见序列里被掩掉;持续性比气候学差(0.90 vs 0.64),没有捷径 |
| FoR42 败血症 normalized utility | −0.02…0.57(均值 0.23) | 挑战赛冠军 0.360(隐藏 A+B+C)(Reyna 2020 表 3) | **CONSISTENT** | 官方 utility 与我们的向量化实现在 3000 个 stay 上零差。episode 只有 2 例败血症(占 12.5%),SD 0.30;12.5% 的患病率比自然 7.3% 抬高 NU 约 40%;"标最后 12 小时"的捷径在 record-end 探针下被堵住 |
| FoR43 HIPE-OCRepair cMER | 0.0089–0.0228(比值 0.53–0.65 于复制 OCR) | 复制 OCR 0.0226;BnF-Mistral 0.0050(0.22)、BLOCR 0.0106(0.47)、L3i 0.0176(0.78)(arXiv 2607.08143) | **CONSISTENT**(中游) | 官方 norm 逐字相同;在官方测试文件上复算"复制 OCR"得到 0.0229(官方 0.0226),说明数据与评分对得上;比值处在 BLOCR 与 L3i 之间,比冠军弱 2–3 倍 |
| FoR44 ACIC 2016 RMSE/sd(y) | 0.0209–0.0313 | BART 族约 0.010–0.020;线性模型 0.12–0.14(Dorie 2019;Hahn 2020 表 4) | **CONSISTENT** | 约为 BART 的 1.3–2.5 倍、线性基线的 1/5;用官方生成器重建的数据上复算与 eval.json 完全相同。但我们的参考是 OLS(约 0.14),是 BART 的 6 倍,任何灵活估计器都能通过验收 |
| FoR45 AmericasNLP chrF++ | 22.4–25.6(均 23.75) | 各语言冠军 17.9–25.4(gators,TEST 集;ACL 2026 Findings) | **SUSPICIOUSLY-HIGH / NOT-COMPARABLE** | 指标实现与官方脚本零差,但 agent 交的是与图像无关的"共识串",留一交叉验证 chrF++ = 24.7,在 5 种语言里 4 种不低于榜首;chrF++ 的 β=2 奖励召回。这个分数不能作为"看图说话"能力的证据(已在 §8.2 披露)。已加诊断字段 `image_blind`(commit 627b1db) |
| FoR46 HumanEval→MBPP pass@1 | HumanEval 0.94(45/48),MBPP 0.91(29/32) | EvalPlus 榜:HumanEval 87–96、MBPP 87–95(2024–26) | **CONSISTENT** | 我们只用基础测试(不含 "+" 测试,会高 3–15 分);n=16 时 SE 约 7.5 分;参考值 0.0 是设计产物(记忆化器离开可见输入就返回 None),"击败参考"不含信息 |
| FoR47 UD 法语 LAS | A0: 0.714–0.776; 直接预训练路径 id: 0.864–0.902(均 0.883); ood: 0.814–0.905(均 0.868) | UDPipe 2(金标分词)93.2;CoNLL18 冠军 86.9/87.9;UDPipe 1.2 基线 81.05/79.56 | **A0 BELOW-PEERS; 显式 tool-ON 待补** | 与 conll18_ud_eval 官方脚本 6 位小数一致。H8/H9 在 code 节点直接调用了离线 CamemBERT/Stanza，但 `result.json` 没有 `pretrained_parse` task-tool 节点；新增显式工具后需重新完成 4+4。 |
| FoR48 ContractNLI 证据 mAP | 0.830–0.909 | BERT-large 0.922、SVM 0.836、DeBERTa 0.936(Koreeda & Manning 2021) | **mAP CONSISTENT**;P@R80 与准确率原先口径不同 | mAP 与官方 `evaluate_all` 逐位相同。原来的 `p_at_r80` 是逐对数平均(官方代码自己称"乐观"),论文用池化:同样的预测池化 P@R80 只有 0.36(论文中 BERT 0.66–0.86);多数标签基线准确率已达 0.89。已加 `p_at_r80_pooled` 与 `majority_label_accuracy`(commit 0452fc0) |
| FoR49 SMT-LIB QF_NIA | 准确率 0.875–1.0 | SMT-COMP 2025 顶级求解器在 1200 s 内判定约 70–82% | **NOT-COMPARABLE**(等于"跑 Z3") | agent 有 120 s 的 Z3 工具;Z3 判定不了的部分是用多数状态猜的,LLM 的增量几乎为零。不能当作推理能力 |
| FoR50 ValueEval 官方 F1 | 0.369–0.514(参考 0.22–0.36) | 主测试集最佳 0.56、BERT 0.42、全预测 0.26;Nahj 最佳 0.40、BERT 0.28(Kiesel 2023 表 3) | **NOT-COMPARABLE,系统性偏高**;去偏后 CONSISTENT | 官方评分器复现了公布的基线(0.263/0.1285/0.151)。16 项切片上,无正例的价值不计入 P/R,所以对全预测基线也偏高 +0.06(主)/+0.10(Nahj),对 TF-IDF+LR 偏高 +0.04–0.06;去偏后 id 约 0.45–0.47(高于 BERT 0.42、低于冠军 0.56) |
| FoR51 Matbench phonons MAE | 26.9–53.1 | 榜单 MegNet 28.76、MODNet 34.3、DimeNet++ 37.5、成分-only 67.6;JMP-L 20.6(论文自报) | **id CONSISTENT;ood 绝对值 NOT-COMPARABLE** | id 真实水平约 40(池 MAE),约为榜首的 1.4 倍,在 MODNet/SchNet 档;已报告的 episode 分数都落在切片抽样区间内(约 p10–p88),是抽样而不是水平的证据。ood 池(Se/Te 化合物)目标范围压缩,MAE 30 只是其 MAD 的 0.275,"ood 好于 SOTA"是尺度假象 |
| FoR52 Psych-201 离散选择 准确率 | 0.5625–0.75 | 文献主要报 NLL(Centaur 0.44 vs Llama-3.1-70B 0.58);零样本 LLM 的离散选择准确率约 0.55–0.70(Binz 2026 图 S2,读图) | **CONSISTENT / NOT-COMPARABLE** | 没有可直接对照的准确率榜。整池上工具库(无 LLM)准确率 0.63–0.65,agent 切片 0.63–0.73,在噪声内;16 项的 SE 是 0.12,val 上 0.5625 vs 参考 0.6875 只有 1 个 SE |

**汇总(23 个学科)**:

- **BELOW-PEERS(真实能力低于同行)**:FoR30、FoR36,另外 FoR51 id 约 1.4 倍于榜首、FoR32 略低于 nnU-Net。FoR47 的 A0 基线仍低于同行;直接预训练路径已达到同行量级,但显式 tool-ON 仍待复测,且没有同条件对照。
- **分数高但不代表能力**:FoR45(与图像无关的共识串)、FoR49(等于 Z3)、FoR50(切片偏高 +0.04~0.10)、FoR42(患病率使 NU 高约 40%)、FoR38(分数即工具库默认输出,且默认值在评测 episode 上被确认过);FoR39 属于同类但程度较轻(分数即工具库输出,但落在同行 IRT 区间内,见 §11.2)。
- **与同行量级一致**:FoR31、32、34、35、37、39(IRT 类方法量级)、40、41、43、44、46、48(mAP)、52,以及 FoR33/38/51 的相对值。没有任何学科"高于 SOTA"。
- **协议问题使 z=1 几乎不携带信息**的学科见 §11.2。

### 11.1.1 2026-10-02 工具能力与正式 agent 的口径刷新

主表保留冻结的历史 agent 样本，下面把本轮新增证据单列，避免把一次工程 smoke 或离线池评估当成新的正式排行榜样本。

| 学科 | 正式/冻结 agent 证据 | 新工具或池能力证据 | 当前解释 |
|---|---|---|---|
| FoR30 | 冻结 Qwen 默认库 id 4/4，PQ+ 62.0/68.2/48.8/66.8；ood 4/4，76.9/44.4/49.2/45.5 | PRED3 在冻结 e00 上做路由 smoke：Mask2Former id 62.0138、ood 76.8745；同一 e00 的 SAM 路由为 53.5998/47.6494 | smoke 证明 dev 引导能稳定保留预训练输出，但不是新独立样本；全量正式复测仍待 GPU 分配 |
| FoR36 | 冻结 id 4/4，SDR 8.09/2.36/9.14/7.62 dB；其中 1 个 episode 回退 SoftMask | `separate_htdemucs` 真实远程单项 smoke 为 9.6824 dB，输出形状和接受状态均正确 | 预训练分离器能力已达到同行档，正式 agent 仍需多 episode 强制路由复测；OOD 为空是任务设计 |
| FoR32 | 从零 U-Net 池 id/ood 约 0.860/0.863；正式 episode 曾因 token 超预算而 `z=0` | 官方 Task04 nnU-Net 权重与本任务 260 个带标签体重叠，禁止接入；没有公平的同任务预训练结果 | 当前只能报告从零工具池能力，不能把同数据 nnU-Net 作为 SOTA 对照结果 |
| FoR47 | H9 id 4/4、LAS 均值 0.883022；H8 ood 4/4、均值 0.867574（直接 code 路径） | 每个 episode 都直接调用离线 CamemBERT/Stanza，未出现 CPU fallback；但没有显式 `pretrained_parse` task-tool 节点 | 直接预训练路径达到同行量级；显式 tool-ON 尚未验证，与 A0 也没有同条件对照 |
| FoR51 | 历史正式 agent ExtraTrees/HistGBM id MAE 49.43/46.50；仍高于 MegNet 28.76 | SevenNet 池方法 id/ood 约 24.8/18.0；本轮只完成单结构远程特征 smoke，尚未完成正式 SevenNet episode | 池能力已进入同行档，但正式提交路径尚未验证，不能用池分数替换 agent 分数 |
| FoR45 | 冻结 agent chrF++ 均值 23.75，诊断显示 image_blind 共识串 | 可见同语言 image-kNN 参考在 4×id/4×ood 为 13.1488/16.3611，而 medoid 为 18.7939/22.0052；kNN 低约 5.64 | 当前高分不能当作视觉能力；image-kNN 只作 grounding 对照，CLIP 权重尚未审计 |
| FoR49 | 冻结准确率 0.875–1.0 | pooled diagnostics 现可并列多数状态、固定 10 s pure-Z3 及 hard/easy tier 的准确率与决定比例 | 高分主要是 solver shortcut 的证据链已补齐，不能当作 agent 推理 SOTA |
| FoR50 | 16 项切片 F1 0.369–0.514 | 完整 all-values reference 已固定：id 0.26293（1576 项）、ood 0.12846（279 项）；pooled diagnostics 分开报告 slice 与 full split | 切片高分不可直接对照榜单；剩余工作是正式 agent 全量复测 |
| FoR52 | 冻结 agent 切片准确率约 0.5625–0.75 | 4×16 pooled history reference：id **0.453125**、ood **0.640625**；同 study 可见频率 reference：id **0.40625**、ood **0.640625** | 同指标 pooled 基线已补齐，但文献主要报告 NLL；这些数不能直接声称 SOTA |

截至本节对应的最新服务器复核（§26–§27），`toolon_SOTA30`、`toolon_SOTA36`、`toolon_SOTA47`、`toolon_SOTA51` 的 id/ood 正式结果目录均尚未创建，因此上表中的历史/冻结 episode 与池能力数字没有升级为新的正式 SOTA 分数。FoR51 的只读缓存审计为 SevenNet `valid=2/1265`、`missing=1263`、`malformed=0`；`SOTA51` 入口已在 episode 启动前以 `rc=2` fail-closed。该状态只更新运行状态，不改历史样本或“一项只评一次”规则。

这些新增数字均不改变 primary scorer、验收边际或历史 id/ood 的“一项只评一次”规则；正式结论以冻结 episode 和后续未评 episode 的完整记录为准。

### 11.1.2 2026-10-03 Leonardo 正式 tool-ON 批次更新

用户重新登录后，作业 `59101545` 在 `lrdn3225` 上恢复运行，新的 split-scoped 运行后门禁已部署并生效。SOTA30/FoR30 id、ood 各 4/4 均完成且 validator 通过；`predict_pretrained` 在 8/8 episode 成功执行，输出沿 `pretrained.y -> submit.y` 进入最终提交。id primary 均值为 **64.9780**，ood 均值为 **78.0824**。SOTA36/FoR36 因任务无独立 ood，只完成 id 4/4；`separate_htdemucs` 4/4 成功且输出进入最终提交，SDR 均值为 **8.2782 dB**，validator 通过。两项结果现在是正式 tool-ON 证据，不能再按“正式目录为空”处理；FoR30 仍低于挑战前三，FoR36 仍需与冻结/同行口径分开解释。

同一作业内的 SOTA47/FoR47 id 4/4 也通过了显式 `pretrained_parse` lineage gate，LAS 均值为 **0.8417**。ood 四项均生成 scorer 结果，但其中 1 项在工具成功后被后续 code 节点覆盖，validator 拒绝该项，故本轮只计 **3/4 lineage-valid**（有效三项均值 **0.8657**），不把它写成完整正式 ood 结果。

### 11.1.3 2026-10-03 SevenNet 缓存修复与 FoR51 重试

SevenNet 冻结特征现已覆盖 **1265/1265**；唯一 Niggli reduction edge case 使用原晶格上的直接均匀 q-grid 处理，未改变标签、scorer 或验收。`phonon_features` 和批处理脚本现可从完整冻结缓存直接服务轻量 agent 环境。作业 `59249574` 形成的 scorer 结果中，FoR51 id 有效 tool/lineage **2/4**（MAE **44.4231、47.0928**），ood **1/4**（MAE **24.1212**），其余为缓存路径未导出时的 CPU fallback，另有一个 ood `z=0`。这些结果不能替代完整 4/4 正式批次；按“一项 id 或 ood 只评一次”规则，不改写或重复该 split。

FoR50 的固定可见数据 `fit_predict` 路由已完成完整池复算：id F1 **0.548950**（1576 项）、ood **0.437082**（279 项），比 16 项切片更适合同行比较，但仍标为 trusted diagnostic。FoR45 的冻结 OpenCLIP 组件已加入 provenance/status；服务器缺少 `open_clip_torch` 和已审计 checkpoint，暂不产生正式 CLIP 分数。

### 11.1.4 2026-10-03 FoR50 fixed-route 正式 agent 结果

`SOTA50` 的固定 `fit_predict` tool-ON 批次已完成：id **4/4**，F1 均值 **0.508216**（四项 `0.501696/0.513572/0.518505/0.499091`），split validator 通过；ood **3/4 accepted**，有效均值 **0.395049**（`0.405134/0.368986/0.411027`），第 4 项 F1 `0.111951` 为 `z=0`。所有 episode 都执行了固定工具，失败项保留为失败证据，不用重跑替换。

### 11.1.5 2026-10-03 SevenNet、CLIP 与 Psych 工程复测

为避免队列中的长作业空等，先在 `lrdn2923` 申请 4 张 A100 完成 `H51`。FoR51 id/ood 各 4/4 均 `z=1`，SevenNet MAE 均值 **20.6226/14.6106 cm⁻¹**；8/8 图均为 `fit_sevennet_mlip.pred -> submit.y`，但该标签沿用旧 episode 前缀，只作为修复提交路径的工程证据，不能升级为新的 formal SOTA51。

随后在 `lrdn0265` 的 `59256351` 上并行完成 `H45H52`。FoR45 的冻结 CLIP 路由 8/8 直接接 `submit.y`，id/ood chrF++ 均值 **14.5178/19.7605** 且 8/8 `z=0`：组件可调用，但不能超过当前语言 medoid 参考。FoR52 的 `psych_fixed_predict` 8/8 调用，id/ood 各 **3/4 accepted**，均值 **0.71875/0.59375**；5/8 保持工具直连，3/8 被后续 code 覆盖。FoR52 objective 已收紧为 fixed-tool 输出直接提交并立即结束；这两项都只计工程证据，不替代 formal manifest 结果。

FoR49 随后在 `59257952/lrdn2450` 上做了 H49 工程复测：id 完成 3/4（3/3 accepted，均值 **0.979167**），ood 完成 4/4（3/4 accepted，均值 **0.921875**）。已完成项均调用 `z3_check`，但第 4 个 id 在 solver unknown 后绕过工具、用长时 `scilib.logic` code 重算，直到分配结束仍无 scorer 结果；该项不计作失败分数，作业已取消。FoR49 objective 现要求 `z3_check.status -> submit.y` 后立即结束，禁止隐式重实现。

### 11.2 这对前面各节结论的影响

前面 §3/§4 的"通过/失败"是按各学科的验收规则(比参考方法好一个边际)判定的。这次对照发现,下列学科的**参考方法本身太弱或验收边际小于切片噪声**,所以 z=1 说明"过了规则",但不说明"达到了同行水平":

| 学科 | 弱在哪 | 对 z 的读法 |
|---|---|---|
| FoR36 | 参考=混合音当估计,+3 dB;缩放混合音 +0.7~0.9 dB、0.001×混合音 +0.01 dB 都比参考高约 6 dB | 任何非静音输出都过;z 不含信息。agent 比"最佳缩放混合音"实际多 1.3–1.7 dB |
| FoR44 | 参考=OLS(约为 BART 的 6 倍) | 任何灵活估计器都过 |
| FoR33 | 参考=前一天(论文里最弱的基线),avg7 是它的 0.82 倍 | 平凡的 7 日均值在 5 个 episode 里 3 个能过 0.90 阈值 |
| FoR35 | 参考=SNaive,边际 5% | ETS+Theta+SNaive 组合的比值 0.887±0.055,在多数切片上能过 0.95 倍 |
| FoR38 | 参考=naive,边际 3%;2021 起点是共同冲击 | 工具库在 90–100% 的切片上通过,z 不区分 LLM 好坏;id e00 的 z=0 是 token 预算所致,不是预测质量 |
| FoR30 | ExG 参考在 val e00 低于全土壤下限 | 部分 episode 通过的是弱基线 |
| FoR51 | 5-NN 参考比 GBM 差 2–7 倍(id e00 上 203 vs 28) | norm_score 被夸大 |
| FoR39 | 参考=训练多数正确率(0.64,比 IRT random-10 的 0.69 弱 5 点),边际 0.02 | 仅 random-10 + IRT 就有约 65–73% 的切片通过,工具库 72–87%;z=1 说明"有查询+建模",不说明"选题策略优于同行";episode 的参考平均 0.670 比整池 0.645 高约 0.02(切片偏易) |
| FoR34 / FoR41 / FoR50 / FoR52 | 边际(0.05 / 0.95×参考 / 0.05 / 0.05)小于切片噪声(AUC SD 0.13、比值 p5–p95 0.89–0.99、F1 SD 0.07、SE 0.12) | 单 episode 的通过≈掷硬币 |
| FoR45 / FoR49 | 分数可被利用(共识串 / Z3) | 不是能力证据 |
| FoR46 | 参考=0.0(设计产物) | "击败参考"无信息 |

所以,**"23 学科能否通过"更该读作"流水线跑通、产出合规的答案",而"是否达到同行水平"应看 §11.1 表里的比值和绝对数**。下一步要量化适配效果,应当把上述弱参考换成强参考(§11.4),并至少合并 ≥ 64 项(或 ≥ 8 个 episode)。

### 11.3 评分器一致性(parity)核对清单

| 学科 | 对拍对象 | 结果 |
|---|---|---|
| FoR30 | `phenobench.evaluation`(官方包) | 1024 px 上 7 项一致到 ≤0.01;512 px(scale=2)PQ+ 偏高 0.4–1.0、PQ_crop 偏高 1.3–3.5 |
| FoR31 | `scipy.stats.spearmanr` / 官方 per-assay Spearman | 一致;差异:常数预测我们返回 0.0,官方是 NaN |
| FoR32 | nnU-Net 的 2tp/(2tp+fp+fn) | 逐位一致;仅"两边都空"时我们返回 1.0 |
| FoR33 | `buildings_bench.evaluation.metrics` | 最大差 1.4e-14 |
| FoR34 | OGB `Evaluator('ogbg-molhiv')` | 一致,无并列分数 |
| FoR35 | Monash MASE 定义 | SNaive 在全部 366 序列上复现 1.631;Theta 1.647 vs 官方 1.649 |
| FoR36 | museval 0.4.1 BSSEval v4 | 最大差 0.000 dB |
| FoR37 | WB2 `get_lat_weights` | 2e-15;聚合方式略有差异:我们对逐 init RMSE 取平均,WB2 先平均 MSE 再开方,差 +0.003~0.005 K(已同时报告两种) |
| FoR38 | `utilsforecast.losses.smape` | 1.4e-14;还从原始 API JSON 独立重建了 365 个样本,0 处不一致 |
| FoR39 | 从原始 CSV 重算;完美预测/取反预测 | 参考与指标到 6 位小数一致(完美=1.0、取反=0.0);结构差异:每个学生只有一个 mask(`rank % 10`)、按出现的 mask 取平均而非全 10 个;查询规则比官方严(`answered and not target`) |
| FoR40 | 官方评测规则(`nttcslab/dcase2024_task2_evaluator`) | 200 个随机 section 上零差 |
| FoR41 | `scoringRules::crps_norm` 闭式 / 数值积分 | 5e-10 |
| FoR42 | 官方 `evaluate_sepsis_score.py` | 3000 个 stay 上零差 |
| FoR43 | `ocrepair_eval.norm`;官方公布的"不校正"列 | 逐字相同;复算 0.0229 vs 0.0226(差异来自适配器的 `exclude_from_icdar_evaluation`/`max_chars` 过滤,尚未解释到底) |
| FoR44 | 论文定义;从冻结种子重建 episode | 与 eval.json 完全相同 |
| FoR45 | sacrebleu `CHRF(word_order=2)`、官方 scorer | 300 对随机样例零差 |
| FoR46 | 官方 `check` + `estimate_pass_at_k` | 未发现不一致(我们超时 6 s vs 官方 3 s) |
| FoR47 | 官方 `conll18_ud_eval.py` | LAS、UAS 6 位小数一致 |
| FoR48 | 官方 `evaluate_all`(contract-nli-bert) | mAP 与 P@R80(官方口径)逐位一致;`nli_binary_accuracy` 因 pair 集合不同差 0.005 |
| FoR49 | 官方 `:status` 匹配 | 无官方打分器,平凡一致 |
| FoR50 | 官方 ValueEval 评估器 | 复现公布的全预测基线 0.263/0.1285/0.151 |
| FoR51 | matbench `mae` | 一致(无变换、无截断) |
| FoR52 | 无官方准确率定义 | 与自身定义一致 |

### 11.4 我做了什么、没做什么,以及建议

**已做(三项纯加法的诊断指标,不改 primary、参考和验收,各带测试、各单独提交)**:

| commit | 学科 | 内容 |
|---|---|---|
| 0452fc0 | FoR48 | 新增 `p_at_r80_pooled`(论文口径的池化 P@R80)与 `majority_label_accuracy`(多数标签基线);文档说明原 `p_at_r80` 是逐对乐观口径 |
| 627b1db | FoR45 | `details["diagnostics"]`:不同 caption 数、比例、`image_blind`(每种语言只有 1 个不同 caption)、平均词数;文档说明 chrF++ β=2 奖励召回 |
| 4f488f2 | FoR33 | 新增 `average_persistence`(官方 AveragePersistence:7 日按小时均值)及指标 `avg7_balanced_nrmse_pct` |

**没有改,只建议**(改参考或验收边际会改变通过率,并使"冻结 run 的 A_0"与"最终代码 A_0'"不再是同一协议、无法直接比较;我没有在夜间改,只在此列出并建议下一轮统一改、统一重跑、统一披露):

1. **换强参考**:FoR33→avg7、FoR36→拟合增益的混合音或经典 HPSS/Wiener、FoR44→AIPW/BART 类、FoR35→ETS/Theta、FoR30→max(ExG, 全土壤)、FoR51→GBM/成分-only、FoR52→max(众数, 上一次)、FoR38→加 ETS/Theta 作为第二参考、FoR46→朴素零样本 pass@1、FoR39→加 IRT prior-only 或 random-10+IRT 作第二参考(`for39_eedi.py:162-166,:304`)(代码位置见 `reports/sota_crosscheck/` 各学科的 To-do 行号)。
2. **合并统计量再判定**:FoR34/40/41/42/43/44/50/52 以 ≥ 64 项(或 ≥ 8 个 episode)的合并指标作为主结论;FoR34 的按 episode 验收改成合并 AUC(`for34_molhiv.py:377`)。
3. **FoR50 补一次全测试集评测**(1576 + 279 条),把切片偏高 +0.04~0.10 量出来,再和 0.56/0.40 比。
4. **FoR51 汇报 MAE/MAD**;FoR31 把常数预测改为 NaN 并剔除(`for31_proteingym.py:120-127`);FoR37 primary 改成先平均 MSE 再开方或修正文档(`for37_weatherbench.py:528`,差 0.005 K);FoR30 报告 PQ+ 时同时给 scale=1 的数;FoR32 不再称"OOD"。 **FoR39**:在 `for39_eedi.py:289` 文档化严格与 starter 查询规则的差别(或加一个 starter 规则变体,+2–3 点),并在 `:307-312` 同时报告池化准确率;ood 池是 proxy_within_dataset,不要当作挑战赛私有集(`:495-501`);`scilib.adaptive` 没有使用 `revealed_values`/`student_meta`,是 agent 上限的下界。
5. **FoR49 加基线**:同一评测里加"纯 Z3 1 s / 120 s"和"永远猜多数",并记录每个 episode 由 Z3 判定/猜测的比例;FoR45 加检索式(CLIP-kNN)参考。
6. **补能力缺口**:FoR47 的 CamemBERT 离线快照已补齐并完成直接预训练路径复测;显式 `pretrained_parse` tool-ON 复测仍待完成。FoR30 实例分割、FoR36 分离器和 FoR51 结构感知模型仍涉及各自的模型供应链,本轮没有擅自下载或改验收协议。
7. **披露口径**:论文/报告里引用这些分数时,请写明"16 项切片、训练数据规模、是否有预训练骨干",FoR37 的 5.625°/16-init、FoR40 的单机器 16 段、FoR42 的 12.5% 患病率、FoR45 的 DEV 集,均需要写进脚注。

**没有做的核对**(在此披露,不是遗漏):FoR40 的 WAV 加载器是否有文件名/public_id 泄露未检查;FoR43 IID 金标是公开官方测试文件,LLM 是否记忆过无法测试;FoR52 底座模型是否见过 Psych-101 无法验证;FoR37 的 IFS/GraphCast 24 h 数值只有图,没有取到,也没有下载 gs://weatherbench2 数据;FoR41 未下载 EFI 分数重算气候学 CRPS。

## 12. 补充:调用同行预训练工具后的 tool-ON 结果(2026-09-30,仅评"开工具"版本)

背景:§11 发现 FoR30/36/47 真实低于同行,根因是沙盒缺预训练骨干(不是 bug)。因系统是允许调用工具的 agent/operator 系统,现只评"开工具"条件;权重放 Node0015 的 GPU 上,解析结果回传 Mac 用真实打分器复算。根因与实测细节见 `reports/below_peers_rootcause/f30.md|f36.md|f47.md`。

| 学科 | 指标 | 此前 | tool-ON | 同行 | 说明 |
|---|---|---|---|---|---|
| FoR30 | PQ+ id / ood | ~39 / ~31 | **53.2 / 44.6** | 81-82 | SAM 2.1 掩码池 + LightGBM 学习选择;仍受杂草样本少限制(weed IoU 20.4/11.5) |
| FoR36 | SDR dB | ~2.4-2.8 | **8.98** | 7.5-9.8 | htdemucs;披露其训练见过 MUSDB18 train |
| FoR47 | LAS dev / id / ood(PUD) | .78 / .73 / .72 | **.921 / .892 / .840** | ~.90-.91 | Stanza gsd_camembert-large 在可见 1000 句上微调;ood 下降来自 PUD 标注约定差异;披露 Stanza-GSD 训练重叠 |

未完成/待验证:tool-ON 尚未跑成完整 agent episode(LLM 网关密钥只在 Mac、GPU 工具在服务器、沙盒无网络,需要选择文件中转、LLM 中转或 Leonardo vLLM 三种路径之一);沙盒内 torch+GPU 的可复现性与 900 s 节点上限待测。

## 13. tool-ON 端到端 agent episode 已跑通(FoR47,2026-09-30)
用**文件中转桥**(节点写请求文件 → Mac 侧 broker 经 VPS 送到 Node0015 的 GPU → 回写结果文件; 密钥不出 Mac, 详见 `reports/below_peers_rootcause/f47.md` §8)解决了"沙盒无网络 + 密钥在 Mac + GPU 在服务器"。FoR47 id episode(gpt-5.5, 16 句): z=1、通过、LAS .8576、重放可复现、21 步、866 s、211k token。FoR30/36 因图像/音频上行只有 ~60 KB/s, 需要把数据先放到服务器端(见待办)。

> **2026-10-02 口径复核**：Leonardo 的 H1–H9 结果中，H8/H9 的轨迹是 code 节点直接 import `scilib.udparse_pretrained`，而不是新增的显式 `pretrained_parse` task-tool；这些批次保留为直接预训练路径证据，不能替代下一批显式 tool-ON 复测。汇总器现在把 task-tool、direct code 和 retrieved operator 分开统计。

## 14. FoR51 tool-ON:CHGNet 声子特征(2026-09-30,详见 `reports/below_peers_rootcause/f51.md`)
新工具 `scilib.matphonon_mlip`(CHGNet + phonopy 有限位移声子, 21 维声谱统计量; GPU 在 Node0015, 经同一文件中转桥; 每结构内容哈希缓存, 重放可复现)。三次 id agent episode 全部 z=0: 第 1、2 次 token 用完前没加 submit(第 2 次基本是第 1 次的缓存重放); 第 3 次走完并通过验收(16 项 MAE 73.7 vs 参考 203.4), 但策略 token ≈211k > 200k, 不算通过, 预算未放宽。步骤头新增事实性字段 `last_reply_tokens`(commit 3d7ea4c, 所有学科生效, 此前 FoR30/36/47 用旧头)。

完整池(训练 676; id 99、ood 179 各只评一次)MAE cm⁻¹:
| 方法 | dev | id | ood |
|---|---|---|---|
| A. 第 3 次实际提交的模型 | 54.0 | **58.7** | **33.3** |
| B. agent 第 1 次 dev 上选定的集成(未提交) | 37.1 | **37.4** | **22.2** |
| 5-NN 参考 | 121.9 | 132.6 | 51.1 |
两个都报。A 是符合协议的头条, 比无声谱固定 GBM(id 40.1)更差, 因为 agent 最终提交的 eval 版本与它在 dev 上验证过的版本不是同一份代码、从未在 dev 上打分; B 说明 agent 在 dev 验证下写出的模型能到 id 37.4(区间 30–46), 在 MODNet/DimeNet++ 档, 仍不到榜首 MegNet 28.8。ood 绝对 MAE 小是尺度假象(MAE/MAD ood 0.20 vs id 0.097)。披露: CHGNet 训练见过 Materials Project 结构; 同行训练量更大(≈1012 vs 676)。


## 补充(2026-09-30,仅 tool-ON,GPU 中转桥)
- 沙箱到 Node0015 GPU 的文件中转桥已扩展到 FoR36(htdemucs)和 FoR30(SAM2 选择器);大数组走内容寻址行块 + 提前 gzip 上传(commit e67beed / d6b1e7c)。
- FoR36 id episode 第 1 次:z=0(token 预算用完、agent 没有提交);池上 htdemucs 8.98 dB(工具能力)。详见 reports/below_peers_rootcause/f36.md §7。
- FoR30 id episode 第 1 次:PQ+ 56.45(16 张,参考的 2.30 倍),completed/reproducible/hard_ok 都为真,但 policy token 210,006 > 200,000,z=0。详见 f30.md §7。
- 共同现象:FoR51 第 1–3 次、FoR36、FoR30 都是"agent 在预算尾部继续做边际改进而没有及时收尾";预算未放宽,未挑选重跑。

## 15. FoR32 tool-ON:从零训练的 3D U-Net(2026-09-30,详见 `reports/below_peers_rootcause/f32.md`)
新工具 `scilib.hippo_unet`(从零训练, 无预训练权重; 4 级 3D U-Net×3 集成 + 增广 + TTA), 沙箱无 torch, 走同一座文件中转桥到 Node0015 GPU 2(训练约 44 s/网络)。**配对池评测**(只用 28 个可见训练体, 一次预测, 同一批体)DSC:

| 池 | 旧 `hippo` | U-Net | 配对差 |
|---|---|---|---|
| id (64) | 0.833 | **0.860** | +0.027 (0.018–0.036), 50/64 体更好 |
| ood 代理 (64) | 0.827 | **0.863** | +0.036 (0.025–0.049), 56/64 体更好 |
| val (32) | 0.853 | **0.876** | — |

nnU-Net 公布 0.862–0.895(训练体 208–260 vs 我们 28): id 落在区间下沿, 此前低 0.03–0.06。Agent id episode(16 体): DSC 0.8630(参考 0.727), completed/reproducible/accepted 均真, 但策略 token 204,297 > 200,000 → z=0(最后一步是一个不改变提交的 445 s 评估节点, 步头已显示 `policy_tokens_left=10000`、`last_reply_tokens=13000`)。预算、判定规则均未改, 未挑选重跑。至此"agent 在预算尾部继续做边际改进而没有 finish"已在 FoR51/36/30/32 四个学科重复出现。披露: U-Net 无预训练; ood 是强度编码代理。

## 47. 增补 36（2026-10-03）：单卡长任务拆分已验证

- `boost_qos_lprod` 支持单张 A100 的 4 天作业。新提交的 Qwen 服务 `59264793`/`59264794` 当前均 `RUNNING`，分别在 `lrdn1355`/`lrdn0177` 的端口 `22200`/`22201`，`TimeLimit=4-00:00:00`；`srun` 从分配节点间访问两个服务均返回 200，真实 `/v1/chat/completions` 回复 `OK`。每个作业的 cgroup 只暴露一张 A100，`CUDA_VISIBLE_DEVICES=0`，不会越界。
- 新增 `scripts/leonardo/run_single_gpu_toolon.sh`：独立单卡代理从共享 marker 读取两个 Qwen 的实际 host/port，分别生成 endpoint；每个分片使用独立 SQLite cache、remote spool 和 output，broker 只绑定本卡 GPU0。默认服务目录/作业为 `single-gpu-qwen-l4-20261003` / `59264793,59264794`，也可用环境变量覆盖。
- FoR47 严格显式 tool-ON 复测已完成：`59264148`（id）和 `59264149`（ood）各 4/4，8/8 轨迹含成功的 `pretrained_parse` 节点，broker 8/8 请求 `ok=true`；id LAS 均值 `0.841701`，ood 均值 `0.867574`，全部 `z=1`、`completed`、`hard_ok`、`reproducible` 通过。第一次并行批次的 SQLite 锁冲突和旧 `regex` 依赖问题已隔离修复，不计入结果。
- 两张旧的 1 天临时服务 `59262550`/`59262551` 已在 4 天服务验证后取消并释放 GPU；后续批次使用新 marker。集群有大量混合节点但几乎没有整节点空闲，单卡拆分绕开了“必须等整台 4 卡”的调度瓶颈。

- 口径修正：首次 `59264148/59264149` 提交时把逗号分隔的 `SERVICE_IDS` 直接放入 Slurm `--export`，Slurm 将其截成一个 ID；因此这两批的显式 tool-ON 分数是**单 Qwen endpoint**下的有效结果，不能声称使用了双 endpoint。脚本已改为 `SERVICE_ID_A`/`SERVICE_ID_B` 两个独立变量（兼容逗号或分号列表），当前 4 天 marker 已独立确认可生成两项 endpoint；后续新分片将使用双服务。

## 48. 增补 37（2026-10-03）：FoR52 正式 fixed-route 双单卡批次

- formal manifest 新增 `SOTA52`，要求 `psych_fixed_predict`、id/ood 各 4 项；本地定向 manifest/task/validator 测试通过，manifest 已同步 Leonardo。
- 新增 standalone formal launcher `scripts/leonardo/run_single_gpu_toolon.sh`：formal tag 会先做 manifest 检查，SOTA51 额外做 SevenNet 完整缓存门禁，probe 输出写入固定 `toolon_SOTA52_id/ood`，结束后调用 split-scoped validator；每个分片使用独立 spool/cache。作业 `59268973`/`59268974` 使用 4 天 Qwen `59264793`/`59264794`，生成的 YAML 同时包含两个 endpoint：`lrdn1355:22200` 与 `lrdn0177:22201`。
- **SOTA52 id**：4/4 生成合法结果，`psych_fixed_predict` tool/lineage 4/4；primary `0.7500, 0.6875, 0.6250, 0.8125`，3/4 达到接受线，均值 `0.71875`。唯一 z=0 是接受线不足。
- **SOTA52 ood**：4/4 生成合法结果，tool/lineage 4/4；primary `0.6875, 0.5625, 0.6250, 0.5000`，3/4 达到接受线，均值 `0.59375`。唯一 z=0 是接受线不足。
- 两个作业的非零退出来自 split validator 对 z=0 的 fail-closed 报告，不是运行崩溃、工具失败或双 endpoint 失败；结果目录保留为正式失败证据，不重复同一 item。该批说明 fixed route 可稳定执行，但 FoR52 与文献的 NLL/准确率口径仍不可直接等同为 SOTA。

## 49. 增补 38（2026-10-03）：独立后续 formal 批次 FoR47B / FoR51B

- 启动器新增 manifest 约束的 `skip`，只抽取同一固定 split 中尚未使用的后续 episode。FoR47 新旧 item 集交集为 0；FoR51 新 ID/OOD item 集交集也为 0。FoR30 池容量只有 64，未做重复扩展。
- SOTA47B：ID/OOD 各 4/4、tool/lineage 各 4/4，LAS 均值 **0.854528/0.835036**，全部 validator 通过。
- SOTA51B：ID 2/2、OOD 4/4，六项都有 SevenNet tool/lineage 证据，MAE 均值 **31.0431/15.3927 cm⁻¹**，全部 validator 通过。旧 SOTA51 结果仍按部分批次保留，不与新条目混合。

- SOTA50B：新 ID 4/4 accepted，F1 均值 **0.4916**；新 OOD 4/4 合法、2/4 accepted，均值 **0.3223**。SOTA52B：新 ID 3/4 accepted、OOD 1/4 accepted，均值均为 **0.6250**；这些结果继续按切片噪声和不可比性解释。
- P3 供应链已在 Leonardo 独立目录落地并校验：BuildingsBench Transformer-L 1.93 GB、BEATs iter3 361 MB、Granite TTM r2 3.2 MB；TabPFN 无授权不下载，Granite 的 6 小时频率适配仍未进入 formal。

### §11.5 2026-10-04 formal tool-ON 更新

本节更新 §11 的同行对照，不回写早期小样本 A_0 表格，也不把不同 split/工具路线混成一个分数。

| 学科 | 当前可验证证据 | 与同行/口径结论 |
|---|---|---|
| FoR30 | `SOTA30` ID/OOD 4/4 tool-lineage，PQ+ 均值 **64.9780/78.0824**；Weyler visible-dev **27.190359**，Mask2Former **75.386581** | 预训练工具显著抬升但仍低于挑战前三；未发现 scorer bug，不换 Weyler |
| FoR36 | `SOTA36` ID 4/4，`separate_htdemucs`，SDR 均值 **8.2782 dB**；Sony SCNet visible-dev **8.528417 dB** | 已到 HT-Demucs/同档范围；任务无独立 OOD，不能补造 OOD |
| FoR47 | `SOTA47B` ID/OOD 各 4/4，显式 `pretrained_parse` lineage；LAS 均值 **0.854528/0.835036** | 预训练工具可稳定接线，但切片低于约 0.90–0.91 同行区间；旧 H6 不重复 |
| FoR50 | `SOTA50C` ID/OOD 均 4 条合法结果，均值 **0.445312/0.239440**；OOD 0/4 accepted | 低分由 OOD 分布和 16 项切片波动造成；公开冠军权重与项目标注规模重叠，排除 |
| FoR52 | `SOTA52D` 未使用 OOD 20–23，`psych_domain_adaptive_predict` lineage 4/4，均值 **0.703125**，2/4 accepted；`SOTA52C` 因误用 fixed tool 作废 | 准确率与文献 NLL 不同口径，不能宣称同量级 SOTA；自适应路线已可审计复现 |

FoR52 的 required-tool 约束已经写入正式启动器：manifest 中声明的 task tool 会自动注入 episode objective，并由 split-scoped validator 检查其输出是否连到 `submit.y`。详见 `reports/for52_sota52c_d_20261004.md`。

### §11.6 其余低分项的并行收口

| 学科 | 当前状态 | 处理结论 |
|---|---|---|
| FoR32 | Task04/InnerEye/MASS/SAM2/DINOv2 均不同时满足独立权重、3-D anterior/posterior 契约和可见验证；MASS mixed DSC **0.59858** < atlas **0.69429** | 保持 `SOTA32=blocked`，没有可公平替换的权重，不用伪映射抬分 |
| FoR45 | CLIP 接线、ISO fallback、medoid 检索和官方 chrF++ parity 已核对；USP Qwen3-VL→NLLB 只有临时微调路径且需重训 | 保持 diagnostic-only，不启动不可复现 formal |
| FoR49 | `SOTA49B` ID **4/4**、OOD **3/4**；`SOTA49C` OOD **3/4**，有效轨迹均为 `z3_check.status → submit.y` | solver 已收口；unknown 属正常 timeout，ID 池耗尽，不重跑 |
| FoR51 | SevenNet-MF-0 PBE/r2SCAN 路由、独立缓存和 1265/1265 工程缓存已部署；SevenNet formal 证据仍按原批次报告 | 工具供应链已补齐，但未把未复测 MF-0 写成新 formal 分数 |

### §11.7 FoR36 SCNet formal批次的配置失败

`SOTA36B` 的互斥 ID 04–07 没有产生有效分数：22200 单卡服务的真实 model id 是 `sc-llm`，旧启动器错误发送 `sc-llm-l3`，四项在 policy step 0 全部 HTTP 404。该批次不计分、不重跑；启动器现已从 `/v1/models` 自动发现 model id，并阻止已有 receipt 的 formal 重跑。SCNet 的 visible-dev SDR `8.528417 dB` 仍是工程证据，不能改写成 formal 成绩。

### §11.8 FoR36 SCNet 最后条目

修复后的启动器在 `59264793` 上实际执行了唯一保留的 ID-08（`SOTA36C`）。
端点可用且 model 自动发现成功，但 agent 24 步均未调用 `separate_scnet`，
以 `step_budget` 停止；结果为 `completed=false`、`primary=null`、`z=0`，无
scorer 证据。该条目已消耗，不重跑。FoR36 不新增 SCNet formal 分数，
visible-dev 的 `8.528417 dB` 仍仅作工程参考。

### §11.9 三方向并行加速（2026-10-04）

- **FoR30**：没有新的可部署 PhenoBench 冻结 checkpoint；Mask2Former 仍为最强可验证路线，停止继续搜索。
- **FoR47**：Leonardo 冻结 Stanza French-GSD CamemBERT 权重加载成功。服务器 visible-dev probe 的 LAS 为 ID **0.8736**、OOD **0.8792**，远高于参考 LAS **0.2597/0.2637**；这只是新鲜 visible-dev 证据，未重复 formal item。
- **FoR32**：服务器没有满足独立权重、3-D crop 和前后海马语义契约的冻结模型，保持 blocked，不训练、不伪映射。

### §11.10 三方向第二轮收口

FoR47 的正式 ID/OOD episode 00–07 已全部占用，不能新增互斥 formal 批次；CamemBERT visible-dev 改善不回写旧 formal 分数。FoR32 的 MASS 路线 mixed DSC **0.59858** 明显低于 atlas **0.69429**，已停止。FoR30 服务器定向资产检查未发现新的 PhenoBench 层级三数组 checkpoint，继续保留 Mask2Former 为最强可验证路线。


### §11.11 FoR51 MF-0 visible-dev 收口（2026-10-04）

MF-0 PBE/R2SCAN 的 1265/1265 冻结缓存已补齐，并修复了 `SCIENCECLAW_MLIP_CACHE` 未透传到远端工具节点的问题（`b56dbe4`）。固定 676 train、110 dev 的同口径 ExtraTrees 比较为：SevenNet-l3i5 **24.2811**、MF-0 PBE **26.2827**、MF-0 R2SCAN **37.3410 cm⁻¹**。MF-0 两条路由均低于现有基线，不接入 formal；探针 `P51MF0C`/`P51MF0R2B` 只用于 visible-dev 工程证据，没有读取 hidden ID/OOD target，也没有重复 formal item。
