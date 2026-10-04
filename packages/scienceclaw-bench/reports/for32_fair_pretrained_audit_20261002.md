# FoR32 公平预训练权重审计（2026-10-02）

## 结论

当前服务器上**没有可以直接接入 FoR32 正式 tool-ON 评分的公平预训练分割权重**。唯一任务相关的权重是官方 nnU-Net `Task004_Hippocampus` checkpoint；它就是同一 MSD Task04 的 260 个 `imagesTr` 训练病例，不能作为本任务的外部冻结模型。服务器上其他视觉权重（SAM2、DINOv2）没有 MRI 海马体的目标定义和输出头，也不是可直接调用的 FoR32 分割器。

因此本轮不新增一个看似 SOTA、但来源或接口不可核验的工具；已有的 `scilib.hippo_unet` 从零拟合路径只保留为历史/协议基线，不能作为本轮新的 SOTA 工具或“预训练能力”结论。当前遵守“不自行训练”的优化规则，不启动新的训练来抬高 FoR32 分数；若以后需要复核历史基线，也必须只用可见 28 个训练体、固定默认配置并在提交后立即停止搜索。这样保持不泄露标签、不使用外部同任务标签、也不放宽验收。

## 数据与任务边界证据

- 本地 `Task04_Hippocampus` 的 `imagesTr` 和 `labelsTr` 各有 **260** 个配对病例。数据 receipt 的解析检查记录 `numTraining=260`、`numTest=130`、单模态 MRI、标签 `0=background, 1=Anterior, 2=Posterior`。
- 服务器 receipt 是 `/leonardo_scratch/large/userexternal/rqian000/scienceclaw-data/datasets/for32-msd-hippocampus/receipt.json`；其来源为官方 MSD `Task04_Hippocampus.tar`，并记录了 260 个训练配对。
- 当前 task 的输入是围绕**一个**海马体的约 `32--40 x 48--57 x 30--40`、1 mm 体素 crop，输出是同一 crop 上的前部/后部两类。它不是完整脑 T1 输入，因此“完整脑模型”不能仅凭模型名称当作可用工具。

## Leonardo 权重盘点

审计时间为 2026-10-02 21:10 CEST；以下结论来自服务器文件、校验和及模型元数据。

| 权重/组件 | 服务器状态 | 可比性判断 | 处理 |
|---|---|---|---|
| `models/nnunet/Task004_Hippocampus.zip` | 存在；MD5 `44dad55102901e203f5cda68084cf0c5c`，官方 Zenodo 文件；53 个文件，含 2D/3D 五折 checkpoint | `debug.json` 写明 `dataset_directory=/media/fabian/nnunet/Task04_Hippocampus`、`classes=[1,2]`、`num_classes=3`、单输入通道；训练任务与本任务完全相同，训练病例重叠 260/260 | **永久禁止进入正式分数** |
| `models/sam2hf/`、`cache/hf/...facebook--sam2.1-hiera-large` | 存在 | 通用自然图像提示分割，未提供 T1 MRI 预处理或 anterior/posterior 类头；直接运行无法满足 FoR32 输入/输出契约 | 不接入；可保留为其他视觉任务组件 |
| `models/dinov2-base/`、`cache/hf/...facebook--dinov2-base`（另有 large） | 存在 | 通用图像特征编码器，没有 3D MRI 分割头；要变成 FoR32 工具仍需用可见标签训练适配器，尚无预算内、同口径的验证 | 不作为当前正式工具；若以后做诊断，必须明确为冻结特征 + 28 个可见训练体 |
| Hippodeep（公开 GitHub 权重） | 服务器未发现 | 文档称模型用多个外部 cohort 的全脑 T1，输出左右整个海马；FoR32 是 crop 且要求前/后两类，输入空间和类别都不兼容。训练 cohort 的逐病例清单也没有随 checkpoint 固化 | 仅列为候选诊断，未接入正式评分 |
| Microsoft InnerEye Hippocampus | 服务器未发现 | 模型卡说明用 ADNI 数据训练，输入全脑 MRI，输出左右海马；与本任务 crop/前后类别不匹配，且 ADNI 访问有数据使用协议 | 仅列为候选诊断，未接入正式评分 |
| HSF (Hippocampal Segmentation Factory) | 服务器未发现可核验 checkpoint | 公开说明为 700+ 手工标注海马体的 T1/T2 原始 MRI 模型，输出 subfields 或连续 head-to-tail 结果；不是本任务的 anterior/posterior crop 分割头，且具体模型卡/训练病例仍需核验 | 不接入；先做接口和来源审计再谈 |

外部候选的公开说明： [Hippodeep](https://github.com/bthyreau/hippodeep_pytorch)、[InnerEye Hippocampus model card](https://innereye-deeplearning.readthedocs.io/md/hippocampus_model.html)、[HSF](https://github.com/clementpoiret/HSF)。这些链接只证明候选的存在与接口/训练描述，不证明它们已经适配本任务，也不替代无重叠病例的逐项 provenance 证明。

## 2026-10-03 候选复核：存在的权重仍不满足 FoR32 契约

为避免把“公开仓库里有文件”误报成“可接入的公平工具”，本轮又核对了两个可直接下载的候选：

| 候选 | 可复现文件证据 | 仍不满足的条件 | 结论 |
|---|---|---|---|
| [schellm/hippodeep](https://github.com/schellm/hippodeep) | `torchparams/hippodeep.pt` 2,003,060 B，Git blob `95d0ce6f3e780311ece267724e8d5b1d7de36c4`；同仓库的 `params_head_00075_00000.pt` 和 `paramsaffineta_00079_00000.pt` 也公开。推理脚本先把**全脑** T1 配准到固定空间，再输出左/右整海马 mask。 | 本任务只向 agent 提供单海马 crop，要求同一 crop 上的 anterior/posterior 两类。仓库没有把全脑到 crop 的定位、左右到前后类别的无训练映射定义成可审计算子；把它硬接进去会改变输入/输出契约。 | 候选诊断；不进入工具池或 formal manifest。 |
| [TheoEst/hippocampus_registration](https://github.com/TheoEst/hippocampus_registration) | GitHub 中的 `save/models/*.pth.tar` 是 134 B 的 Git-LFS 指针，真实模型需另取 Zenodo；README 明确这是 Learn2Reg 配准模型，输出变形图/变形 mask。 | 这是配准网络而非 FoR32 的 anterior/posterior 分割器，且模型来源和病例清单不能证明与本任务接口无重叠。 | 排除。 |

这次复核没有下载或调用候选权重，也没有新增正式 episode；它只把“候选存在”与“满足契约”分开记录。只有同时满足逐病例无重叠、单海马 crop 输入、前后两类输出以及预算内 smoke 的冻结模型，才可解除 `SOTA32=blocked`。

## 同任务 nnU-Net 权重为什么必须拒绝

服务器 checkpoint 的摘要给出平均 Dice：2D `0.8693`、3D full-resolution `0.8891`、ensemble `0.8876`。这些数字是官方 Task004 训练/验证结果，不能与只给正式 Agent 28 个可见训练体的设定直接比较。更关键的是，checkpoint 的训练目录和标签定义与本任务的 260 个 `imagesTr` 完全相同；把它作为冻结工具会把评测标签通过权重旁路泄露给 id/ood。因此它只作为“同任务上限/泄漏警示”保存，不进入 scorer、tool pool 或正式 episode。

## 当前正式路径和预算门槛

`scilib.hippo_unet.fit_predict` 仍然是可用的公平路径：每次只接收 `load_train` 的 28 个可见图像/标签，从零训练 3 个 3-D U-Net；服务器实测约 44 秒/网络，默认拟合约 130 秒，整批预测低于 1 秒，低于 FoR32 的 2400 秒 episode 墙钟预算。此前 FoR32 的 `z=0` 来自 Agent 在已有提交后继续尝试高成本配置，导致 token/墙钟超限，不是把同任务 checkpoint 接入的理由。

下一次正式复测应固定以下顺序：

1. 只用 `fit_predict` 默认配置（`n_models=3, iters=2500, tta=True`）完成 dev 和一次 eval 提交；提交产生后立即 `finish`。
2. 不把 5000 迭代、5 网络等搜索结果写入正式分数；若要诊断，只在可见 dev 上单独记录。
3. 任何新外部模型先通过三项闸门：逐病例训练来源无 260 个 Task04 病例重叠、输入/输出能表示本任务 crop 与 anterior/posterior、单个 id/ood smoke 在预算内完成；三项未齐全时保持“候选”状态。

这条路径不改变 primary、reference、acceptance margin 或 OOD 定义；它只把可用能力与不可比的外部权重分开。
