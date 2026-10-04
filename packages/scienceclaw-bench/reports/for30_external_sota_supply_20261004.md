# FoR30 外部 SOTA 供应链追加复核（2026-10-04）

本轮针对当前可验证的 FoR30 hierarchical panoptic 低分，单独复查了公开 SOTA 的代码、冻结权重和输出契约。不重复已消耗的 ID/OOD item，不读取 hidden target，不修改 scorer 或 acceptance，也不从头训练。

## 1. 2024 boundary-loss Mask2Former

* 论文 [Exploiting Boundary Loss for the Hierarchical Panoptic Segmentation of Plants and Leaves](https://arxiv.org/abs/2501.00527) 报告标准训练设置 `PQ+ = 81.89`，并指向 [madeleinedarbyshire/HierarchicalMask2Former](https://github.com/madeleinedarbyshire/HierarchicalMask2Former)。
* 直接检查仓库 `main`（commit `29c891f8e589738cd4497922edc5dc9b107032fa`）的完整 tree 和源码：仓库包含训练代码、FoR30 配置、boundary/focal loss 实现和 ImageNet/Swin 初始化入口，但不包含 `*.pth`、`*.pkl`、`*.ckpt`、`*.safetensors` 等 PhenoBench 训练后 checkpoint，也没有 GitHub release asset 或可下载的训练后权重。
* 配置默认的数据根是 `/nvmedrive/PhenoBenchExtra`，训练/验证注册同时依赖 PhenoBench `train`、`val`，因此仅有代码不能在当前“不自行训练”的条件下得到可调用的 SOTA 路由；把其论文数字当作本项目可复现工具分数是不成立的。

## 2. CVPPA 2023 SAM/HQ-SAM 路线

* [Nguyen et al., A SAM-based Solution](https://arxiv.org/abs/2309.13578) 报告 `PQ+ = 81.33`，方法是 HQ-SAM + DINO/YOLOv8 提示框，并在 PhenoBench 标注上对 HQ-SAM 做 LoRA 微调。
* 论文明确写的是用 PhenoBench 训练图像的 ground-truth boxes 做 HQ-SAM 微调，再把检测器框作为推理提示；其可复现实验需要该微调 checkpoint 和 detector 配置/权重。公开论文页和作者可检索代码入口没有提供一个可直接加载、无需训练且输出 ScienceClaw 三数组的冻结 bundle。
* 服务器已经有通用 SAM2/DINOv2 组件，但它们没有同步的 crop/weed/leaf hierarchical output contract；之前 visible-dev pool 的 `phenoseg_sam` 结果为 id `53.2`、ood `44.6`，低于当前 Mask2Former 路由，不能用“换成通用 SAM/HQ-SAM”来推断安全增益。

## 3. 结论与行动边界

当前能直接部署并已做真实 visible-dev 对比的 FoR30 specialist 仍只有 PRBonn Mask2Former pair。PRBonn Weyler、HAPT、Panoptic-DeepLab 已有真实负面证据；boundary-loss 的 81.89 和 HQ-SAM 的 81.33 都依赖未公开的 PhenoBench 微调权重，不能在本项目约束下伪装成现成 SOTA 工具。若后续获得作者公开的冻结 bundle，可先对不重复的 visible-dev capture 做契约 smoke；在那之前不注册 ToolSpec、不重复 formal，不改默认 Mask2Former 阈值。

## 4. 其它公开 PhenoBench 代码仓库

* [chunbai1/cvppa2023-Crop-weed-Panoptic](https://github.com/chunbai1/cvppa2023-Crop-weed-Panoptic) 自报 CVPPA track-1 第 4 名、`PQ+ = 79.89`，但仓库只有 MMDetection 配置、数据转换和融合脚本；没有 release、checkpoint 文件或权重下载地址，配置仍以训练/加载 COCO 和 Swin 初始化权重为起点。
* 可检索的 YOLOv9/语义分割 PhenoBench 仓库同样只包含 notebook、训练命令或通用初始化权重；它们不提供 hierarchical `semantics + plant_instances + leaf_instances` 冻结输出，不能直接接入本任务。

这类代码可以作为未来取得权重后的实现参考，但在当前“不自行训练、只调用可追溯冻结权重”的边界内，不能带来新的可部署分数。
