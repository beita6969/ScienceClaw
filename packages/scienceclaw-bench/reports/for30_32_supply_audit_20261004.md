# FoR30 / FoR32 公开预训练供给审计（2026-10-04）

本轮只做服务器上的冻结权重、真实 visible-dev 与来源核对，不重复已消耗的 ID/OOD item，不读取 hidden target，不改 scorer、acceptance 或 formal manifest。

## FoR30

Leonardo 作业 `59277346`（`lrdn2979`，A100，`sc-harness`）加载了官方 PRBonn Weyler checkpoint：

- 路径：`$L/models/phenobench_weyler/weyler_checkpoint_0381.pth`
- SHA-256：`ef78b9b9c1a9ac56e4359ac847038e3901dfd2ddff093bfcdf5cd3e44a116e67`
- 推理：`FoR30-id-s5-e00` 的 8 张 visible dev RGB crop，使用 `predict_weyler`，不向工具传目标。
- trusted `score_dev`：`PQ+ = 27.190359`，`IoU_soil = 99.332481`，`IoU_weed = 0`，`PQ_crop = 0`，`PQ_leaf = 9.428956`。

同一 episode、同一 scorer 上，现有 PRBonn Mask2Former pair 为 `PQ+ = 75.386581`（`IoU_soil=99.737882`、`IoU_weed=71.997157`、`PQ_crop=73.991112`、`PQ_leaf=55.820172`）。Weyler 输出虽然权重严格加载且有实例前景，但其可见 crop/weed 语义与本交付域不匹配，不能替换 Mask2Former 或作为层级后处理。

另外在同一 visible dev slice 上做了不改变模型的阈值诊断：M2F `(plant_threshold, leaf_threshold)=(0.4,0.4),(0.6,0.6),(0.7,0.7),(0.8,0.8),(0.9,0.9)` 的 `PQ+` 分别为 `75.263918, 75.370730, 75.370730, 75.386581, 75.406332`；混合 `(0.6,0.8),(0.8,0.6),(0.7,0.8),(0.8,0.7)` 为 `75.386581,75.370730,75.386581,75.386581`。`0.9/0.9` 只在一个 8-image dev capture 上高 `0.0198`，没有跨 capture 证据，不能据此把默认阈值改成 dev-overfit 版本，也没有新增 ToolSpec。

上游 Weyler 复核确认预处理不是当前低分的接线错误：官方 `ToTensor` 使用 `torchvision.F.to_tensor`（RGB 除以 255），推理分辨率为 1024，`Cluster` 参数为 `sigma/alpha=11`、parts `32/0.7`、objects `64/0.7`；wrapper 使用了相同的缩放和聚类参数。

官方供给仍只有已记录的 Mask2Former、Weyler、HAPT、Panoptic-DeepLab。HAPT 和 Panoptic-DeepLab 的真实 visible 结果均为跨域/全背景低分；通用 SAM/DINO 没有 hierarchical 三数组契约。因此 FoR30 当前最强可验证路由仍是 Mask2Former，后续只有拿到独立且同契约的公开 checkpoint 才值得新部署。

## FoR32

Leonardo `$L/models` 当前与 FoR32 相关的冻结资产为：

- `nnunet/Task004_Hippocampus.zip`：精确 3-D anterior/posterior 输出，但训练病例与本地 Task04 260/260 重叠，永久排除。
- `innereye/hippocampus_segmentation_model.zip`：公开 ADNI whole-left/right binary 模型；真实 FoR32 crop 只读 smoke 全背景，且不存在合法 anterior/posterior 映射。
- `mass/mass_base.pth`：独立、冻结、reference-mask 多类接口；此前真实 crop 只读比较中，全 visible atlas `mean DSC=0.69429`，把两个 query 替换为 MASS 后 mixed `mean DSC=0.59858`（anterior `0.61817`、posterior `0.57899`），不能替换 atlas 或解除 blocked。
- `sam2*`、`dinov2-*`：通用视觉基础模型，没有 3-D MRI anterior/posterior head；BiomedParse 仍是 gated 2-D 输入，未取得可审计 checkpoint。

本轮没有找到同时满足“公开冻结、无 260 个 Task04 病例重叠、3-D crop 输入、输出 `{0,1,2}` anterior/posterior、visible-dev 有意义得分”的新路线。因此不新增 FoR32 wrapper、不做语义伪映射、不重复 formal item，`SOTA32=blocked` 保持不变。

## 结论

本轮没有可安全接入的新 FoR30/FoR32 预训练路线。FoR30 的 Weyler 和 FoR32 的现有外部候选均已有真实服务器证据，低分来自模型/契约迁移而非评分器接线错误；现有代码继续保留，等待独立合规权重或新不重复配额。

来源：PRBonn [phenobench-baselines](https://github.com/PRBonn/phenobench-baselines)（Weyler/Mask2Former 官方实现和 checkpoint 入口）；FoR32 候选边界见 `reports/for32_external_candidates_20261004.md`、`reports/for30_for32_external_supply_20261003.md`。
