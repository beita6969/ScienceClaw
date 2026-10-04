# FoR30 / FoR32 下一步供应链复核（2026-10-03）

这份记录只回答一个问题：在不训练、不读取评测标签、不重复正式 item、也不降低验收标准的条件下，服务器上是否已经有可以直接接入的新增冻结模型。检查在 Leonardo 登录节点完成，远端根目录仍为 `$F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000`、`$L=/leonardo_scratch/large/userexternal/rqian000`。

## FoR30

### 当前可用组件

- `$L/models/phenobench_m2f/hf_plants/model.safetensors` 与 `hf_leaves/model.safetensors` 均存在，哈希分别为 `15ba280977a809f8a21e16557198e940fd23bcae142a203eb06f0d3738e79e45` 与 `e6f0d1c4bfbd603b5b205b8361e1bc720d3df0877bae96c4730b5f68c12a92ef`。
- `scilib.phenoseg_m2f.predict_panoptic` 已把两个 PRBonn Mask2Former 检查点封装为 label-free、可直接提交的 `predict_pretrained` 工具；输入只包含评测 RGB 图像，输出为三组整数数组。
- 当前正式 SOTA30 已完成 id/ood 各 4/4，均通过 tool/lineage/acceptance；池均值为 id **64.9780**、ood **78.0824**。这证明部署与路由稳定，但 id 仍低于公开挑战前三约 81–83。

### 本轮没有安全的新增模型

服务器现有视觉权重只有上述 PhenoBench 检查点、通用 SAM2 和 DINOv2；没有发现新的 PhenoBench 专用模型、HAPT 可直接推理的 checkpoint 或 ReLeaf 权重。SAM2/DINOv2 的输出契约不能替换层级全景标签，不能把它们包装成同指标 SOTA。模型目录中的官方验证预测压缩包也只是已用检查点的离线输出，不能提供新的模型能力。

因此 FoR30 目前可执行的优化是已有 Mask2Former 路由的工程复用与以后取得独立模型后的并列 smoke；正式 id/ood 池已经用完，不能再通过重复 item 获得新 leaderboard 分数。没有新增权重时，不提交新的正式 FoR30 结果。

## FoR32

### 服务器上的精确模型

- `$L/models/nnunet/Task004_Hippocampus.zip`（SHA-256 `9b79415d303b38f66bd84c90281736c4ef7b5bb94ad6aa058f569a926cbd37fe`）及其 `$L/models/nnunet/Task004_extracted` 展开目录是唯一发现的精确 hippocampus segmentation checkpoint。
- 该官方 Task004 checkpoint 的训练病例与本项目的 260 个 `imagesTr/labelsTr` 病例重合，已有 `reports/for32_fair_pretrained_audit_20261002.md` 记录。因此它永久排除在公平正式路径之外，不能仅因分数接近同行而接入。
- Leonardo 的 `sc-harness`、`sc-run`、`p3-granite` 等环境均没有 MONAI、nnU-Net、TorchIO 或其他未登记的医学分割模型供应链；模型目录中也没有 Hippodeep、InnerEye、HSF、SwinUNETR 或 TotalSegmentator 的可用权重。

### 明确排除的伪候选

`$L/sc-tools/remote_models/*/unet.pt` 不是外部冻结模型。逐个读取 checkpoint 元数据得到：

| 目录 | `n_train` | `iters` | `fit_s` | SHA-256（截短） |
|---|---:|---:|---:|---|
| `5744557ac44301a85a431352` | 28 | 2500 | 167.3 s | `8c9edaea…` |
| `6f03ce880aafd4cecc5a3c15` | 28 | 4000 | 823.3 s | `1e674af4…` |
| `ecae9556f3a03ee032f95573` | 28 | 4000 | 544.9 s | `4483f5f4…` |

这些是本项目 visible `load_train` 数据上从零拟合的 U-Net 集成，违反当前“不自行训练”的边界；它们不得被注册成 pretrained 工具，也不得进入 `SOTA32`。其存在只作为历史工程缓存，不能作为新能力证据。

## 结论与可执行触发条件

本次服务器复核没有发现可以合规接入的 FoR30 新权重或 FoR32 公平冻结权重。现有代码和模型供应链已达到“可调用、可验证、可追溯”的状态；下一次实质性推进需要满足以下至少一项：

1. FoR30：取得不含本 benchmark 评测图像、且输出契约与层级 PQ+ 完全匹配的公开 checkpoint，先做本地/Leonardo GPU smoke，再与现有 Mask2Former 并列诊断；不重复已有正式 item。
2. FoR32：取得带公开训练集清单、逐病例与本项目 260 个病例不重叠、并能输出 0/1/2 三类 3-D mask 的冻结 checkpoint；在此之前 `SOTA32` 保持 blocked。

在触发条件出现以前，继续提交 FoR30/FoR32 正式批次只会重复池或引入训练/重叠污染，不能提高可信的同口径分数。

## FoR30 追加供应链复核（2026-10-03）

补查了两个公开的层级/全景候选：

- **PRBonn/PSPA** 提供 ERFNet 冻结模型及公开的 `phenobench_auxiliary/split.yaml`。逐文件名与本项目 `reconstructed_v2` 角色池比对，PSPA 训练清单与 `src/val/id/ood` 的交集分别为 **11/32、9/32、0/64、12/64**。它还只导出 semantics 与 plant instances，缺少 FoR30 所需的 crop-leaf instances。因此即使 `id` 没有交集，也不能作为同时覆盖 ID/OOD 且契约匹配的干净工具；不下载、不注册、不进入正式评分。
- **JunhaoXing/ZeroPlantSeg** 是 GPL-3.0 的公开代码，`ckpt_download.sh` 只下载通用 SAM/ GroundingDINO/OVSeg 基础权重，没有与本项目契约绑定的可复核冻结结果。它需要额外的 prompt/后处理链，仓库脚本输出 PNG 而不是三数组 `y`；当前没有完成契约适配或可复现 smoke，不能把论文中的 PhenoBench 数字当作本项目分数。

因此目前唯一已验证可直接接入的 FoR30 路由仍是 `predict_pretrained` 的 PRBonn Mask2Former（训练重叠风险已披露）。DINOv2+SAM2 现有 selector 的实测池均值仅 id 53.2、ood 44.6，不能替换它。没有新增合规权重，也没有证据支持重复正式 item 或修改 scorer；FoR30 的剩余差距归因于训练数据/模型能力，而非路由接线错误。

### HAPT checkpoint 状态更正

此前“没有发现 HAPT checkpoint”的描述已经过时。公开 HAPT checkpoint 已放入 `$L/models/hapt/hapt_model.ckpt` 并能严格加载，但上游 README/config 将训练数据写为 GrowliFlower。FoR30 visible-dev smoke 的 8 张图得到全背景预测，可信 `score_dev` 为 `PQ+=24.6095`（`IoU_soil=98.4379`，其余 crop/weed/leaf 指标为 0）。因此它只能作为跨域失败诊断，不能把 HAPT 声称为 PhenoBench specialist，也不加入正式工具或分数比较；`scilib/phenoseg_hapt.py` 的 provenance 说明已同步更正。

## 2026-10-03 供应补充

服务器复核后新增一个公开的 Microsoft InnerEye-HS v0.5 checkpoint 候选，已记录在 `reports/for32_innereeye_supply_20261003.md`。它有冻结 SHA 和公开训练说明，可做只读 binary smoke；但输出是 whole left/right hippocampus，而 FoR32 要求同一 crop 的 anterior/posterior 0/1/2 语义。没有契约匹配前后部映射，因此仍保持 `SOTA32=blocked`，不进入正式评分。

## 2026-10-03 外部候选补充：PlantInstGen 与 MASS

新增审计记录见 [`for30_for32_external_supply_20261003.md`](for30_for32_external_supply_20261003.md)。FoR30 的 PRBonn PlantInstGen 示例入口直接使用 `gt_sem`/`gt_inst`，只生成 plant instances，缺少 leaf-instance 输出和独立 frozen checkpoint，不能接入 RGB-only hierarchical PQ+。FoR32 的 Stanford AIMI MASS `mass_base.pth`（公开 SHA `8f27f19e5bd6013f792a2319207d6ec8652229cf4f89d1f29233ff12e9342b32`）加载和 `{0,1,2}` 输出契约 smoke 均通过；但用 visible train masks 作 reference、dev image 作 query 后，`score_dev` 的混合结果（前两例 MASS、其余 atlas）mean DSC **0.59858**，低于全 atlas **0.69429**。因此不注册 MASS formal ToolSpec，`SOTA32` 继续 blocked。
