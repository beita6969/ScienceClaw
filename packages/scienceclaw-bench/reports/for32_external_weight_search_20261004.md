# FoR32 外部 3-D 前后海马权重最后检索（2026-10-04）

本轮按“不自行训练、不读取 hidden labels、不重复 formal item”的边界，完成了对公开 3-D hippocampus anterior/posterior 候选的最后一次供应检索。目标是找到输入为 3-D hippocampus crop、输出语义为 `0=background, 1=anterior, 2=posterior`、且训练病例不与本项目 Task04 260 个标注病例重叠的冻结权重。没有把只有论文数字、只有训练代码或只有 whole-hippocampus/binary 输出的项目写成可用 SOTA。

## 候选结论

| 候选 | 公开证据 | 结论 |
|---|---|---|
| `czifan/PAM` | 官方 README 的数据表把 `MSD-Task04 Hippocampus` 列为 D32，目标明确写为 anterior + posterior；预训练 checkpoint 由外部 Drive/Baidu 提供。 | 训练数据包含本任务同一 MSD Task04，不能用于公平 formal；不下载、不接入。 |
| `Nicolik/HippocampusSegmentationMRI` | 公开仓库报告 V-Net 的 anterior/posterior Dice，但 README 说明数据为 Medical Segmentation Decathlon；仓库 `models/` 只有网络定义，没有冻结 checkpoint。 | 既没有独立权重，也没有病例隔离证明；不接入。 |
| `Mahdyy02/hippocampus_segmentation_msd_task4` | 公开代码和结果目录声称 3-D 模型输出三个类别，但训练/验证说明直接使用 MSD Task04；`results_hippocampus-3D/` 没有可下载 checkpoint。 | 同任务训练且无独立权重；不接入。 |
| Hippo-Net（Lei et al.） | 论文报告 260 个 T1w 数据集，前 200 做五折、后 60 hold-out；任务契约和病例数量与本地 Task04 260 pool 一致。 | 同数据来源，不能当作无重叠外部模型；不接入。 |
| Microsoft InnerEye-HS / Hippodeep / HSF | 公开冻结组件可以得到 whole hippocampus、left/right 或 subfield/head-to-tail 结构；HSF 明确不分配具体 head/tail 类。 | 不产生 FoR32 所需前/后 0/1/2 语义。先前 InnerEye 真实 crop smoke 还返回全背景；不做按轴硬切或语义伪映射。 |
| HippUnfold HCP-YA | HippUnfold 的 nnU-Net 权重由 HCP-YA 独立数据训练（训练数据说明和 Task101 HCP model URL 可追溯），并计算 intrinsic anterior-posterior 坐标。 | 输入流程要求 whole-brain T1/T2 → 0.3-mm coronal-oblique crop，模型输出 tissue/subfield，不是当前已裁 `(约35×51×N, 1-mm)` FoR32 crop 的 0/1/2 head-tail mask。把整脑流程强行套到 crop 需要未经验证的 registration、orientation 和语义阈值 adapter；不能在没有可见验证和契约证明时进入 formal。2.45-GB HCP checkpoint 不下载到本轮服务器目录。 |

## 服务器/仓库状态

- 已有的 `Task04` 官方 nnU-Net、`theme4-nnunet-hippocampus`、PAM D32 与 Hippo-Net 路线均被训练重叠或来源不可隔离排除；不修改 `configs/formal_toolon_manifest.json`，不重复 ID/OOD item。
- 之前的 MASS 只读验证仍保留：mixed mean DSC `0.59858`，低于 atlas `0.69429`，所以不把 MASS 包装成工具。
- 之前的 InnerEye 真实 crop smoke 仍保留：原始/pad/resize 三路均返回全背景；不做 left/right→anterior/posterior 映射。
- 当前 `SOTA32` 继续 `blocked` 的原因是**缺少同时满足独立训练病例、3-D crop 输入、anterior/posterior 输出契约和可见验证性能的冻结公开权重**，不是评分器或数据切分 bug。

## 下一次解除条件

只有发现可逐病例审计训练来源、明确未使用本地 Task04 260 病例、直接输出 `0/1/2` 的冻结模型，或任务方提供独立 OOD/外部权重后，才解除 `SOTA32=blocked`。届时先在 visible dev 做受控比较，再决定是否新增 formal manifest；本报告不启动新的 formal 作业。

参考：

- [PAM official README](https://github.com/czifan/PAM#download-pretrained-weights)（D32 Task04 listed in training data table）
- [Nicolik V-Net repository](https://github.com/Nicolik/HippocampusSegmentationMRI)
- [Mahdyy02 MSD Task04 repository](https://github.com/Mahdyy02/hippocampus_segmentation_msd_task4)
- [Hippo-Net paper](https://arxiv.org/abs/2306.08723)
- [HSF repository](https://github.com/clementpoiret/HSF)
- [HippUnfold HCP-YA training data](https://zenodo.org/record/7007362)
- [HippUnfold model configuration](https://github.com/khanlab/hippunfold/blob/master/hippunfold/config/snakebids.yml)
