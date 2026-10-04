# Tool-ON 持续优化记录（2026-10-02）

本轮把“低于同行、正式 agent 路由不稳定、或能力证据与正式提交脱节”的项目拆成并行工具线。验收器、验收边际、隐藏标签边界和数据切片均未修改；预训练组件只作为冻结推理或特征提取器使用，训练标签只来自任务明确暴露的可见训练池。

## 2026-10-03 新增可执行工具

- FoR52：`psych_fixed_predict` 已部署，固定 `scilib.psych` qlearn、seed=0 和可见 population prior，只接收 `load_train` 与无标签 `load_eval_inputs`；本地定向测试通过，尚未运行正式 episode。
- FoR45：官方 OpenAI CLIP ViT-B/32 已部署到 Leonardo，`open_clip_torch` 最小依赖通过 broker 远端 worker 提供；真实 smoke 返回 `(1,512)` 归一化 embedding。权重 SHA-256 为 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。

### FoR51：缓存完整后的真实 SevenNet 路径复测

`59254581` 在 `lrdn2923` 上以工程标签 `H51` 完成 FoR51 id/ood 各 4 项。8/8 episode 都是 `z=1`、accepted、hard_ok、reproducible；SevenNet MAE 均值 id **20.6226**、ood **14.6106 cm⁻¹**。每条图都验证 `fit_sevennet_mlip.pred -> submit.y`，provenance 为 `pretrained=true`，冻结缓存 `1265/1265`。旧 formal 结果的低分主要来自工具输出被 ExtraTrees/HistGBM 覆盖；H51 只作为提交路径修复后的工程证据，不能冒充同条件 SOTA51。

### FoR45/FoR52 并行工程批次

`59256351/H45H52` 在 `lrdn0265` 上完成两项的 id/ood 各 4 个 episode。FoR45 的 CLIP status + `clip_knn_reference` 全部成功，8/8 图为 CLIP captions 直接提交，但 id/ood chrF++ 均值 **14.5178/19.7605** 且全为 `z=0`；这确认组件可用，同时确认当前 caption 指标下 CLIP 近邻不胜语言 medoid。FoR52 的 `psych_fixed_predict` 8/8 调用，id/ood 各 **3/4 accepted**、均值 **0.71875/0.59375**；5/8 保持工具到提交的直接边，3/8 被后续 code 覆盖。FoR52 objective 已收紧为工具输出直接接 `submit.y` 并立即结束，待新配额再验证完整直连。

### FoR49 solver 路由复测

`H49/59257952` 在 `lrdn2450` 上完成 id 3/4、ood 4/4。已完成 7 项的 id 均值 **0.979167**（3/3 accepted），ood 均值 **0.921875**（3/4 accepted）；6/7 直接由 `z3_check.status` 接到 submit，1 项用显式 retry + merge。第 4 个 id 因 agent 绕过 `z3_check` 在 code 节点启动 600–800 秒的 `scilib.logic` 重求解而未产出结果，已取消作业并收紧 objective，后续禁止该隐式重实现。
- 可复现部署脚本：`scripts/leonardo/stage_clip.sh`；FoR45 仍需新的正式配额验证 `clip_knn_reference` 对 chrF++ 的实际收益。

## 已落地并验证

### FoR30：固定 Mask2Former 路由

- `predict_pretrained` 在 GPU/remote worker 可用时暴露，固定 PRBonn plants+leaves Mask2Former，仅读取 RGB，输出可直接提交的三数组，不拟合、不读取评测标注。
- `score_pretrained_dev` 在 trusted adapter 内返回同一模型的官方 dev 聚合指标，标注不出 adapter，也不进入 held-out 验收。
- Leonardo 的 PRED3 是沿用冻结 seed 的 e00 做路由工程 smoke（不是新的独立正式样本）：id PQ+ = **62.0138**、ood PQ+ = **76.8745**，均 `z=1`。轨迹先比较 dev PQ+ = **75.39**，因此保留 Mask2Former；没有把该结果写成从零 agent 学到的增益，也不覆盖既有冻结池结果。
- 远端 `PRED3_id/e00` 与 `PRED3_ood/e00` 的 `result.json` 复核显示 `uses=[]`、无 retrieved operators；这进一步确认它们是冻结 e00 的工程 smoke，而不是正式 Agent 调用 `phenoseg_m2f` 的全量 tool-ON 证据。
- Leonardo 的 `summarize.py` 现在同时审计轨迹中的内置 task-tool、代码节点直接调用和 `result.json` 的 retrieved operator。此前只看 `uses` 会误报：PRED3 的 id/ood 轨迹实际各有一个 `predict_pretrained` task-tool 节点（`tool 1/1`），但没有 retrieved operator；它仍是冻结 e00 smoke，不是新的独立正式样本。
- 旧路由在同样的冻结 e00 smoke 中改用 SAM 时只有 id **53.5998**、ood **47.6494**，说明新增 dev 工具修复的是可见路由稳定性，不能单独证明全量 OOD 已达到同行前三。
- 22:10 UTC 的服务器组件复核确认 `phenobench_m2f/hf_plants` 和 `hf_leaves` 的 safetensors 权重均存在，`phenoseg_m2f.available()`/`_local_ok()` 均为真；登录节点 64×64 CPU 推理超过 30 s，未把该超时当作分数，正式批次仍应走 GPU。

### FoR36：固定 htdemucs 路由

- `separate_htdemucs` 只接收 mixture，返回 vocals/drums/bass/other 四源估计；MUSDB18 训练重叠在工具说明中披露，目标 stems 不进入调用。
- Leonardo 真实远程 smoke：一项 id episode 返回 `(1, 4, 132300, 2)` 的 `float32` 输出，SDR **9.6824 dB**，`accepted=True, z=1`，远程耗时约 6.64 秒。
- 这解决了工具能力存在但 agent 偶尔退回 SoftMask 的路由问题；正式结论仍需在同一配置下完成多 episode pooled 复测。
- 同次服务器复核确认 `models/demucs/955717e8-8726e21a.th`（SHA-256 `8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4`）可读；登录节点 1×22050×2 CPU smoke 返回 `(1,4,22050,2)`、有限 `float32`，约 7.19 s。这只证明依赖和接口完整，不替代正式 agent 复测。

### FoR51：SevenNet 预训练声子特征

- 新增 `fit_sevennet_mlip`：SevenNet-l3i5 只做冻结结构声子特征，轻量 log-target 回归器只接收可见训练结构/标签；评测结构及其隐藏目标不进入拟合。
- Leonardo 的 `sv_pkgs` 已补齐 `sevenn`、`e3nn`、`torch_geometric`、`matscipy` 等依赖；单结构远程调用已返回 21 维有限特征。SevenNet 特征的全量 tool-ON/正式 episode 仍需在可用 GPU 分配上复测。
- 既有池评估的 SevenNet 方法约为 id MAE **24.8**、ood **18.0**；历史正式 agent 的 ExtraTrees/HistGBM id MAE 约 **49.43/46.50**，两者必须继续分开报告。
- `scripts/f51/prepopulate_sevennet.py` 已支持 `--start/--stop` 半开区间；可在多个分配上分别预热 `[0,400)`、`[400,800)`、`[800,1265)`，仍使用同一内容寻址缓存和参数，不读取评测目标。远端脚本已部署并通过 `py_compile`/`--help` 复核，SHA-256 为 `b79fabae…`。
- 预热脚本现在逐批检查返回特征的有限性；任何 NaN/失败行都会以非零状态退出并提示重跑同一分片，只有 `valid=n` 才打印 `completed`。提交 `da12fce` 已同步 Leonardo，避免把部分缓存误当作完整预热。
- 22:26 UTC 用新增的只读 `scripts/f51/audit_mlip_cache.py` 按结构 key 重新核查远端缓存：SevenNet 有效覆盖 `2/1265`，CHGNet `913/1265`；总计 2298 个 JSON 还包含其他参数/模型 key，不能直接当作 SevenNet 覆盖。有效行均为 21 维有限数组，正式 SevenNet episode 仍须等待完整预热。
- 同一脚本的 `--require-complete` 在当前远端覆盖上按预期返回 `rc=2`，因此不会把部分缓存误放行到正式批次；修正环境变量缺失后的远端脚本 SHA-256 为 `04cb01149f5418075f1f9e00de79b384a089e71b503efc44c08ee26747970aca`。
- `run_batch_leo.sh` 已把该审计接入 `SOTA51` 正式入口；Leonardo 实测 manifest 先通过，随后因 SevenNet `2/1265` 返回 `rc=2`，在 Qwen health/broker 启动前 fail-closed。最新入口脚本 SHA-256 为 `bc3c867afd0afbef276cda1b9556597ebcb0799f83d0fb743a400d20dee934d3`。
- `checkpoint_l3i5.pth` 的远端 SHA-256 为 `a7ff190d41efe5b8317a03d20a468e4036a1a3e64e0b14fdbab64ac3f4bfc09d`（14,345,754 bytes）；登录节点 CPU 对第 1 个结构的 SevenNet 调用返回 `(1,21)` finite、约 16.47 s。该 smoke 只证明权重/依赖完整，不替代全量缓存或正式 Agent 结果。

### FoR47：预训练解析器的显式工具封装与审计

- 远端 H8 OOD 与 H9 ID 的轨迹确实在代码节点中调用了 `scilib.udparse_pretrained.parse_gold_tokens`，且没有 `ResourcesFileNotFoundError` 或 CPU fallback；这些分数保留为“直接预训练路径”证据。
- 但 H1–H9 的 `result.json` 均为 `uses=[]`、`retrieved.operators=[]`，轨迹也没有 `pretrained_parse` 这样的 task-tool 节点。因此它们不能按当前严格口径写成“显式 tool-ON operator”结果。
- FoR47 现在新增主库 task-tool `pretrained_parse(sentences) -> parses`，让下一批正式 Agent 通过工具节点调用冻结解析器；必须看到 `tool 4/4`，再把 id/ood 记为正式 tool-ON。旧 H8/H9 的 LAS 仍单列，不能混入该批结果。

### FoR32：明确禁止同数据权重旁路

- 官方 nnU-Net Task004_Hippocampus checkpoint 已下载至 Leonardo `models/nnunet/Task004_Hippocampus.zip`，MD5 `44dad55102901e203f5cda68084cf0c5c` 与发布记录一致。
- 该 checkpoint 的训练元数据显示 `num_samples=260`，与本任务 260 个 `imagesTr` 标签全集重叠；因此没有接入正式 tool-ON，也没有把它当作公平分数。当前 FoR32 保持待补状态，等待独立数据或无重叠公共模型。
- 本轮进一步盘点了 Leonardo 的全部相关模型目录：SAM2/DINOv2 只有通用视觉权重，Hippodeep、InnerEye、HSF 等公开外部候选不在服务器且与本任务的 crop/前后类别接口不兼容；详细 provenance 与拒绝理由见 [`for32_fair_pretrained_audit_20261002.md`](for32_fair_pretrained_audit_20261002.md)。
- 在获得同时满足“逐病例无重叠、输入/输出契约匹配、预算内 smoke”的外部权重前，已有 28 个可见训练体从零拟合的 `hippo_unet` 只作为历史/协议基线，不作为本轮新的训练优化；若复核该基线，正式 Agent 提交后应立即结束，避免重复高成本搜索造成 `z=0`。

### FoR38：历史内滚动强参考与全量聚合

- 新增 trusted-side `damped_trend` 和 `history_backtest_forecast`，只用 origin 之前的可见历史选择随机游走或阻尼趋势；primary、验收边际和 agent 可见工具面保持不变。
- 4×16 个 id 项的 pooled mean sMAPE 从官方随机游走参考 **17.62823** 降到历史选择参考 **16.82989**；4 个 ood episode 则从 **14.52940** 变为 **15.25153**。因此该参考只在 IID 有帮助，OOD 反而退化，不能拿它替换正式参考或声称 SOTA。
- `Adapter.pooled_diagnostics` 现在能在 64 项全 split 上同时报告 primary、官方参考和强参考，避免把 16 项切片直接与完整同行榜单比较。

### FoR52：study-conditioned pooled reference

- 新增 trusted-side study-conditioned baseline：只统计 `load_train` 中其他参与者在同一 study 的可见 response；没有可用同 study 训练行时回退到原 participant-history reference。
- `pooled_diagnostics` 同时报告 agent micro accuracy、原 history reference、study-conditioned reference、episode 平均值和 pooled-vs-slice 差异；primary、acceptance、输入可见性和隐藏标签边界均未改变。
- 当前已完成单元/任务测试，尚未把该诊断替换为正式排行榜结果；需要后续对冻结的全量 id/ood payload 重新汇总。
- 新增 `full_split_reference(split)` 后，完整池诊断为：src `481` 项 participant/study=`0.58836/0.48025`，val `132` 项=`0.62121/0.50000`，id `263` 项=`0.52852/0.47909`，ood `544` 项=`0.60662/0.60662`。study-conditioned reference 没有改善完整 id，因此仍只作可比性诊断，不替换正式 reference。

### FoR45：可见图像参考与 image-blind 对照

- 新增 `image_knn_reference`：只从 `load_train` 的同语言、带图像行中检索可见 caption，使用已公开的 CPU 图像描述子；无图像时显式回退到可见 medoid。它不读取评测 caption、ID 或隐藏标签。
- trusted-side evaluator 记录 `chrf_knn_ref` 和 pooled image-kNN 诊断，primary、官方 medoid reference、acceptance 和冻结 episode 均未改变。
- 本地 4×id/4×ood 诊断中，medoid 为 **18.7939/22.0052**，image-kNN 为 **13.1488/16.3611**，均低约 **5.64 chrF++**。因此当前轻量视觉路线没有提分，作用是量化 image grounding 并揭示共识串分数的可利用性。
- 当前环境没有已审计的 CLIP/OpenCLIP 权重，未擅自下载；未来只有在服务器上取得可核验权重后，才可作为同样的 trusted-side 诊断加入。

### FoR49：Z3 捷径的 pooled 分层核查

- 新增 trusted-side `SmtAdapter.pooled_diagnostics`，在同一 pooled accuracy 口径下并列提交结果、训练集多数状态、固定 10 s pure-Z3，以及 hard/easy screen tier 的决定比例。
- `evaluate` 现在把 tier 和 pure-Z3 结果写入 `pooled_payload` 供审计；这些字段不进入工具输入、primary 或验收。旧 payload 没有可选字段时仍能计算普通 pooled/多数参考，pure-Z3 列保持 `None`。
- 这使“0.875–1.0 的准确率主要是 Z3 调用”可以在完整 episode 汇总中直接验证，但它仍是 shortcut audit，不能当作 agent 推理能力或 SOTA 证据。现有 per-episode 诊断与新增 pooled 单测均通过。
- 新增 trusted-side `full_split_reference(split)`：对每个 split 的全部 lineage representative 汇总多数状态准确率、hard/easy tier 数量及分层准确率，避免 16 项 episode 均值成为唯一参考。它只读取审计聚合所需的标签，不进入 `evaluate`、工具或 acceptance。
- 该方法同时从冻结 1 秒 screen 给出 pure-Z3 的保守边界：`easy` 是已决定且与公开 status 一致的已知下界，`hard` 是 screen 未决定项；不在全池上重新运行 Z3。输出 `pure_z3_decided_lower_bound/upper_bound` 与 `pure_z3_accuracy_lower_bound/upper_bound`，明确标注该边界不能替代 formal episode 的 10 秒 solver 诊断。
- Leonardo 远端复算（6683 lineage units；无 solver/label conflict）得到：src `1429` 项、多数 `0.70609`、Z3 已知下界 `1129/1429=0.79006`；val `397`、`0.71285`、`313/397=0.78841`；id `1017`、`0.69223`、`793/1017=0.77974`；ood `2595`、`0.71985`、`1909/2595=0.73565`。这些是 representative/screen 边界，不是正式 agent 分数。
- 服务器收尾：清理了绑定已取消作业 `59227175` 的孤儿 broker PID `2820409`（无端口监听、无现存锁文件，TERM 后已确认退出），避免它在下一次正式批次复用 spool 时抢消费请求；未触碰其他作业。

### FoR50：全量 ValueEval 口径已足够

- `ValueEvalAdapter.pooled_diagnostics` 和 `full_split_reference` 已覆盖完整 id/ood split，分别报告 pooled official macro-F1、16 项 slice 平均及 all-values 参考；primary、验收和标签边界未改变。
- 服务器重算的 all-values reference 是 id **0.26293**（1576 项）和 ood **0.12846**（279 项），说明 16 项切片的偏高不能拿来对照论文冠军。该项当前缺的是正式 agent 全量运行，而不是再加一个弱基线。
- 新增显式 `fit_predict` ToolSpec：输入仅为 `load_train` 的可见 argument table/labels 和 `load_eval_inputs` 的无标签 argument table，输出可直接提交的 `y` 与固定 provenance。参数锁定为 `seed=0, C=0.3, n_folds=5, k=16, decision=expected_f1, n_jobs=2, groups=conclusion`；工具拒绝评测表中的任何额外列，也拒绝非有限或非 0/1 的标签/预测，避免把隐藏标签或静默 cast 带入路线。
- 新增 trusted-side `full_split_diagnostics` 与 `scripts/f50/full_split_valueeval.py`。使用完整可见训练池拟合一次并在完整 split 内部聚合官方 F1；返回只含聚合指标、固定配置、数据 receipt/hash，不向 agent 暴露标签、item id 或预测矩阵。当前复算：id `1576` 项 F1 **0.548950**（P **0.498392**, R **0.610924**, all-values ref **0.262930**）；ood `279` 项 F1 **0.437082**（P **0.331837**, R **0.640093**, ref **0.128459**）。这两个数是 full-pool trusted diagnostic，不是正式 agent episode 或 leaderboard 分数。

## 尚未结案的项目（2026-10-02 快照）

- FoR30：需要按 PRED3 配置完成 id/ood 各 4 个 episode 的 pooled 复测，单 episode 结果不能代表全量。
- FoR36：需要至少 8 个同口径 episode，确认 htdemucs 选择不再回退到 SoftMask。
- FoR47：需要用新增的显式 `pretrained_parse` task-tool 完成 id/ood 各 4 个 episode；H8/H9 只算直接代码路径，不能替代该门槛。
- FoR38、FoR45、FoR52：继续修复绝对可比性、图像能力证明和同指标参考；FoR45 当前已完成轻量 image-kNN 对照，剩余是获得可审计视觉权重后再做对照，不能把现有 chrF++ 当作视觉 SOTA。
- FoR49 的捷径分层、FoR50 的完整 split 偏差已经有 trusted-side 诊断，剩余是正式 agent 的同口径复测。

## 下一阶段的可执行门槛

这些是复测的完成门槛，不是对 scorer 或验收边际的修改：

- **FoR30**：PRED3 在 id/ood 各完成 4 个正式 episode，全部提交 `z=1`，再用 pooled PQ+ 与 HAPT 约 **65.27** 及挑战前三约 **81–83** 的同口径数字比较。
- **FoR36**：至少 8 个同配置 episode，确认每项都调用 htdemucs；pooled SDR 以同行约 **7.5 dB** 下沿为风险线，任何 SoftMask 回退都单独记录。
- **FoR51**：先完成 SevenNet 冻结特征缓存，再用同一 `fit_sevennet_mlip` 路径完成正式 id/ood；MAE 与 MegNet 约 **28.76** 的参考必须在同一 split 和同一标签口径下比较。
- **FoR32**：只接受预算内、`z=1` 的正式提交；同任务 260 标签重叠的 nnU-Net 权重永远不进入公平分数。

FoR32 的下一次公平复测增加了独立执行配置 `configs/toolon_f32_budgeted.yaml`（`solver.max_steps=12`，仅
FoR32，id/ood 各 4 个 16-item episode）。任务 objective 要求固定默认 `hippo_unet` 一次并在第一份
合法 submit 后立即结束，避免历史 episode 在已经有 accepted 提交后继续调参而超过 token/墙钟预算。
`run_batch_leo.sh` 支持 `SCIENCECLAW_CONFIG_PATH` 显式选择该配置，默认行为不变。该护栏不改 scorer、
reference、acceptance 或标签边界，未来运行结果须以新的完整 episode 证据单独记录。
- **FoR47**：id/ood 各 4 个正式 episode 必须出现 `pretrained_parse` task-tool 调用并 `z=1`；直接在 code 节点 import 预训练模块的旧结果单独报告。
- **FoR38/45/49/50/52**：先交付 pooled 或强参考诊断，再决定是否值得加入新的预训练工具；诊断结果不自动升级为 SOTA 或正式 leaderboard 分数。
- FoR47：按交接文档恢复 Leonardo 登录后，继续 id/ood 各一次正式运行。

## 2026-10-03 服务器正式状态回写

本节覆盖上面快照中“正式目录为空/仍需首次复测”的表述，保留旧文字用于追溯当时的运行状态。

- Leonardo SSH 已恢复；作业 `59101545` 在 `lrdn3225` 运行。SOTA30 id/ood 各 4/4、SOTA36 id 4/4 均完成并通过 split-scoped validator。FoR30 id/ood 均值为 **64.9780/78.0824**，FoR36 id SDR 均值为 **8.2782 dB**。
- SOTA47 id 4/4 通过，LAS 均值 **0.8417**；ood 四项均有 scorer 结果，但一项因 `pretrained_parse` 成功后被后续 code 节点覆盖 `submit.y` 而被 lineage gate 拒绝，故当前只计 **3/4 lineage-valid**。FoR47 objective 已收紧为“工具输出直接接 submit 并立即 finish”，文件已部署并通过 focused test。
- FoR51 SevenNet 远端审计仍为 `valid=2/1265, missing=1263, malformed=0, complete=false`；正式入口继续 fail-closed。不要把 SevenNet 池 MAE 或部分缓存写成正式 agent 分数。
- FoR38、FoR45、FoR49、FoR50、FoR52 的强参考/全量诊断已完成；剩余问题分别是缺乏匿名 WDI 同口径外部 SOTA、缺少可审计 CLIP 权重、solver shortcut、完整 agent 配额和指标不可比。继续添加工具不会自动解决这些证据边界。
- 当前行动顺序：先完成 SevenNet 的长分配断点预热；获得新正式配额后补 FoR47 ood 的缺项；FoR30 继续优化到挑战前三量级；FoR32 只接受预算内正式提交。不得重复本轮已评 split，也不得放宽 lineage、预算或 acceptance 门禁。

## 服务器记录

- 代码同步目标：`/leonardo_scratch/fast/AIFAC_F02_774/rqian000/scienceclaw`。
- 预训练模型目录：`/leonardo_scratch/large/userexternal/rqian000/models`；FoR36 的静态 ffmpeg 由 `imageio-ffmpeg` 提供并由 `run_batch_leo.sh` 导出。
- 本轮工具桥验证复用了作业 59227175 的 2、3 号 GPU；该调试分配随后结束，未停止其他作业。下一次正式复测需要新的可用分配或恢复中的 lprod 分配。
- 最新按 1,265 个结构的内容哈希核查显示 SevenNet 缓存 **2/1265**、CHGNet 缓存 **913/1265**；SevenNet 仍需长分配断点预热。`scripts/f51/prepopulate_sevennet.py` 已可续跑，正式 episode 前不能把这 2 条命中当作完整缓存。
- SevenNet/CHGNet 缓存写入现在使用同目录临时文件加原子替换；分配被 wall-time 或节点故障中断时不会留下半个 JSON 行，后续续跑只会读取完整缓存。提交 `38da624` 已同步 Leonardo，并有本地回归测试覆盖。
- Leonardo 的 harness Python 加载 `SevenNetCalculator("7net-l3i5", device="cpu")` 已成功，`checkpoint_l3i5.pth` 可读；这只证明权重和依赖完整，不替代 GPU 预热或正式 Agent 结果。
- `check_avail.py` 已自动导出 Leonardo 上 `imageio-ffmpeg` 的静态 ffmpeg；重点四项复检为 `src/val/id=2/2/4/4`，FoR36 OOD=0 是任务设计限制。
- 修复后的统一预检已在 Leonardo 对 FoR30–FoR52 全 23 项运行成功（16 s）；所有项目 `src/val/id=2/2/4/4`，只有 FoR36/FoR40/FoR44 的 OOD 按任务设计为 0。
- 复核历史 Leonardo 日志时发现 id/ood 并发启动的 `pgrep` 竞态：同一节点曾出现两个 broker 同时消费同一 spool（同一请求键重复完成），并伴随过一次 CUDA OOM。提交 `4f2fe3a` 后，`run_batch_leo.sh` 与 `broker_step.sh` 共用 `$R/remote_spool/.broker.lock` 的 `flock`，并发锁回归只允许一个消费者；两份脚本已同步到 Leonardo 且 SHA-256 与本地一致。该修复不改变模型、数据、评分器或 episode 数。
- 正式运行门禁已补齐：`check_avail.py --strict --splits ... --n ...` 在登录节点先验证所需 episode 数；`leo_chain.sh` 在 `sbatch` 前逐行预检并跳过空行/注释；`leo_driver.sh` 对作业终态、端点等待和可选 pending 超时做明确退出；`run_batch_leo.sh` 校验 split/n/worker/GPU 参数、Qwen `/health` 和 FoR36 的 ffmpeg。Leonardo 实测 FoR30 严格预检通过，FoR36 OOD 严格预检以 `rc=2` 拒绝，未提交 GPU；不存在的 job 也会快速退出。
- 新增 `configs/formal_toolon_manifest.json`，把 FoR30/36/47/51 的正式批次、必需 ToolSpec、split/episode 数与 FoR36 无独立 OOD、FoR32 公平权重阻塞原因固定下来；它是运行前证据门禁，不改变 scorer、split 或验收。
- `check_formal_manifest.py` 已把该 manifest 接入三条启动入口；SOTA tag 的 discipline、split、n、blocked/unavailable 状态和 ToolSpec 注册不一致时 fail-closed，调试 tag 不受影响。门禁与汇总定向测试 4 项通过，且已同步到 Leonardo 的 `$F/sc-tools/`。
- 新增只读 `validate_formal_results.py`：正式运行后按 manifest 检查每个 episode 的结果旗标和成功 task-tool 轨迹，缺失或失败返回非零；它独立于 `summarize.py`，不改变 scorer、验收或历史计数。当前正式目录为空时会按预期拒绝，相关 9 项定向测试通过，脚本已同步到 `$F/sc-tools/`。
- SSH 恢复后的远端复核：manifest 与启动脚本哈希和本地一致，远端 Python 编译、shell 语法通过；`SOTA30` 启动门禁返回 0，`SOTA36/ood` 返回 2，空的 `SOTA30` 正式目录由运行后 validator 返回 2。

### FoR38：补齐完整 split 的强参考审计（本地数据复算）

- 新增 trusted-side `Adapter.full_split_reference(split)`，对 `src/val/id/ood` 全池逐项计算官方随机游走、历史阻尼趋势和滚动历史选择器的 sMAPE。它只读取隐藏目标生成审计数字，不进入 `evaluate`、acceptance 或任何 policy-visible tool。
- 全池规模与结果：`src` 128 项，官方/阻尼/历史选择器 = **15.1338 / 15.7870 / 14.5619**；`val` 32 项 = **12.4751 / 12.9207 / 11.9790**；`id` 64 项 = **17.6282 / 21.0853 / 16.8299**；`ood` 141 项 = **12.0508 / 13.7376 / 12.0392**（均为 mean sMAPE，越低越好）。滚动历史选择器相对官方参考改善 src **0.5719**、val **0.4961**、id **0.7983**，OOD 仅 **0.0116**，说明原有弱参考在 IID 上确有可量化偏差，但不能把该诊断替换为正式验收参考。
- `tests/test_task_for38.py` 增加全池规模、有限值和 primary 不变性检查；FoR38 定向测试 **13 passed**。该改动不改变 scorer、验收边际、分割或历史正式分数。

## 规则

不自行训练新的深度模型，不把同任务标签预训练权重接入正式评测，不泄露隐藏标签，不放宽验收条件；所有池能力、可见 dev 诊断和正式 agent episode 分开记录。

## 2026-10-03 约束与服务器更新

SevenNet 预热随后在 `59101545` 上完成到 **1264/1265**；唯一缺失的是 `mb-phonons-0915`（索引 914），phonopy/spglib 的 8×8×8 mesh 对其晶格报 `Niggli reduction failed`。该异常已保留为缓存缺口，未用旁路改写特征或正式门禁；FoR51 仍 fail-closed，下一步是设计并验证单结构的可审计几何处理修复。

FoR47 旧 ood 轨迹曾实际执行 `udparse_pretrained.finetune`，但该 episode 已被 lineage gate 拒绝；现在本地模块和远端 worker 都在任何后端操作前拒绝 finetune。FoR32 objective/配置也不再要求从零 U-Net，保持 manifest blocked，直到取得无重叠冻结模型。

## 2026-10-03 实际执行回写

- SevenNet 缓存已完成 `1265/1265`，并处理唯一的 phonopy Niggli edge case；完整缓存现在能被无 SevenNet 包的 `sc-run` 直接读取，批处理脚本也已导出缓存路径。
- FoR51 重试作业 `59249574` 不能算完整正式批次：id 只有 2/4 tool/lineage-valid，ood 只有 1/4，另有一个 ood `z=0`；其余项明确是 CPU fallback。不能把这些结果合并成 SevenNet agent 均值，也不重复同一评测项。
- FoR50 fixed `fit_predict` 与 FoR45 frozen CLIP 组件已完成代码和 focused tests；FoR50 full-pool 数字仍是 trusted diagnostic，FoR45 因服务器缺少 open_clip/checkpoint 尚未产生 formal score。

### FoR50 正式 fixed-route 结果（作业 59250225）

- id **4/4** tool/lineage-valid，F1 `0.501696/0.513572/0.518505/0.499091`，均值 **0.508216**；validator 通过。
- ood 4 项均调用 fixed `fit_predict`，其中 **3/4 accepted**，有效均值 **0.395049**（`0.405134/0.368986/0.411027`）；第 4 项 F1 `0.111951`、`z=0`，保留为失败项。
- 这是正式 agent 证据；完整池 trusted diagnostics（0.548950/0.437082）仍单独保留，不能相互替换。

### FoR51 当前部署状态（2026-10-03）

- Leonardo 只读审计已确认 SevenNet 冻结缓存完整：`valid=1265/1265, missing=0, malformed=0, complete=true`。缓存路径由正式入口显式导出，`SOTA51` 不会在覆盖不完整时启动。
- 目前没有可运行的 ScienceClaw allocation；`59142841` 仍处于 `PENDING/Priority`，`59250200` 属于其他项目且保持运行。旧 `59249574` 结果仍按部分 tool/lineage 证据保留，待新分配后才可依照 manifest 运行新的完整 id/ood 批次。

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

- `scripts/probe.py` 现在在 `--skip` 下先扩充对应 split 的 episode 计数，再选取未使用后缀；formal manifest 同时锁定 skip，防止把新标签误指向旧条目。远端验证的新旧 `item_ids` 交集为 0。
- `SOTA47B` 的两张单卡步骤在 `lrdn1355`/`lrdn0177` 上完成 ID/OOD 各 4/4；`pretrained_parse` 工具成功 8/8，LAS 均值 `0.854528/0.835036`。
- `SOTA51B` 完成 ID 2/2、OOD 4/4；SevenNet cache `1265/1265`，`fit_sevennet_mlip` 成功 6/6，MAE 均值 `31.0431/15.3927 cm⁻¹`。两个标签的 split-scoped validator 均返回成功。

- `SOTA50B` 完成新 ID/OOD 各 4 项：ID 4/4 accepted、F1 均值 `0.4916`；OOD 2/4 accepted、全量均值 `0.3223`。`SOTA52B` 的 fixed tool 在新 ID/OOD 各 4 项均被调用，accepted 为 `3/4` 与 `1/4`，两侧均值 `0.6250`；z=0 项保留为正式失败证据。
- P3 资产已放入 `$L/p3-stage-20261003` 并完成哈希校验；没有把未适配的权重伪装成 task-tool，也没有下载受限 TabPFN。
