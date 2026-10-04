# FoR30 / FoR32 外部供应链追加复核（2026-10-03）

本轮继续只接入公开、冻结、可追溯且不读取 hidden target 的组件。正式 item 不重复；没有把论文分数当作本项目分数。

## FoR30：PlantInstGen 不满足工具契约

补查了 PRBonn 的 [PlantInstGen](https://github.com/PRBonn/PlantInstGen)（论文标题为 *Tailored Refinement of Vision-Language Models for Plant Instance Segmentation*）。仓库 README 明确说明测试数据是 PhenoBench 子集，下载脚本只获取 `instance_samples.zip`，并没有发布一个可独立加载的冻结网络 checkpoint。仓库的 `test.py` 的推理路径读取样本中的 `gt_sem` 和 `gt_inst`，再用 `utils.instance.fine_instances(rgb, gt_sem, coarse_inst, ...)` 做 plant-instance refinement；这不是 RGB-only 的端到端预测。

这与 FoR30 的工具边界有三处硬冲突：

1. 评测图像的 semantic/instance labels 对 agent/tool 不可见；PlantInstGen 的示例入口直接把 `gt_sem` 和 `gt_inst` 作为输入。
2. 它只生成 plant instances，FoR30 hierarchical PQ+ 还需要 semantics、plant instances 和 crop-leaf instances 三个整数数组；仓库没有 leaf-instance 输出接口。
3. 没有一个可核对训练清单和 checkpoint 文件可供建立独立 frozen lineage；README 的 `samples.zip` 是 PhenoBench 的样例数据，不是独立外部权重。

因此不下载、不注册 ToolSpec、不运行 formal item。现有 `phenoseg_m2f` 仍是 FoR30 唯一可直接接线的 specialist；PlantInstGen 不能作为其替代或后处理器。

## FoR32：MASS 是合规候选，但先保持 smoke-only

补查了 Stanford AIMI 的 [MASS](https://github.com/Stanford-AIMI/MASS)（CVPR 2026）。官方仓库和 model card 描述 `mass_base.pth` 为 Iris in-context segmentation checkpoint：预训练只使用自动生成的 class-agnostic masks，明确没有使用 expert-labelled ground truth；推理输入是 reference image + reference mask 与 query image，支持每个非零 reference label 的多类输出。这一契约原则上可以由 FoR32 的 28 个 visible `train` image/mask 作为 references，query 只传 `dev`/`id`/`ood` images，且不需要自行训练或读取 query labels。

服务器已将上游仓库固定到 `$F/p3/mass`，并将公开权重放在 `$L/models/mass/mass_base.pth`：

```text
bytes  967276762 (HEAD content-length; staged file 967276762 bytes)
sha256 8f27f19e5bd6013f792a2319207d6ec8652229cf4f89d1f29233ff12e9342b32
```

该文件的 checkpoint metadata 为：

```text
checkpoint_format: mass_release_v1
architecture: Iris in-context segmentation
pretraining_objective: MASS mask-guided self-supervised learning
supervision: auto-generated class-agnostic masks only; no expert labels during pretraining
config target_spacing: [1.5, 1.5, 1.5]
config training_size: [128, 128, 128]
```

这些声明和指纹只证明它是独立、冻结的公开预训练组件，不证明它在本 FoR32 crop 上有可用性能。MASS 官方 raw-NIfTI runner 依赖 SimpleITK；本轮没有把它直接接入正式适配器，而是用仅依赖 nibabel/scipy 的只读 probe 做契约 smoke，也没有做任何标签映射。候选验证 `59293404` 首次因临时脚本路径失败，随后在已有服务分配 `59277346` 上完成 CPU overlap；脚本仅读取 4 个 visible train image/mask 作为 reference，读取 2 个 dev image 作为 query，输出写入 `$L/p3-stage-20261003/mass`，不读取 dev/id/ood labels，不调用 scorer，不写 `SOTA32` manifest。只有当该验证同时满足：模型正常加载、输出 shape 可逆回原 crop、值域 `{0,1,2}`、并由任务的独立 `score_dev` 评估后显示有意义的 DSC，才考虑新增 `scilib.mass_hippo` wrapper 和正式预算批次；在此之前 `SOTA32=blocked`。

## 当前边界

- FoR30：本轮没有新增合规、契约匹配且独立的冻结权重；不重复正式池。
- FoR32：MASS 是目前第一个同时具备“公开冻结权重、无 expert labels 预训练、reference-mask 多类接口”的可试候选，但仍是候选，不把论文/README 叙述写成分数。
- Existing Task04 nnU-Net checkpoint remains excluded because its fold train/validation IDs overlap all 260 local labeled cases; InnerEye-HS/Hippodeep remain whole-hippocampus or left/right binary and cannot be relabeled as anterior/posterior.

## MASS 只读 smoke 结果（非 formal）

第一次 `59293404` 失败是批处理脚本把 probe 放在计算节点本地 `/tmp`（ExitCode 2，`can't open file '/tmp/mass_probe.py'`），不是模型失败。修正为共享路径 `$F/p3/mass/mass_probe.py` 后在已有服务分配 `59277346` 上用 CPU overlap 完成只读前向（不启动 broker、不占用 Qwen 的 GPU）。权重加载结果为 `missing=0, unexpected=0`，模型配置为 Iris `in_ch=1`, `channels=[32,64,128,256,512]`, `training_size=[128,128,128]`。

probe 使用 4 个 visible FoR32 `train` case 的 image/mask（`hippocampus_325`, `_326`, `_389`, `_390`）生成 label 1/2 reference task embeddings；目标只读取两个 `dev` image（`_015`, `_355`），没有打开它们的 `labelsTr`，也没有调用 scorer。输出恢复到原始数组形状且值域正确：

```text
hippocampus_015  shape=(42,51,28), unique={0,1,2}, counts=[59399,343,234]
hippocampus_355  shape=(33,47,38), unique={0,1,2}, counts=[58499,25,414]
```

随后通过 FoR32 episode 的 `score_dev` ToolSpec（不在 probe 中读取或打印任何 dev labels）做了只读比较。8 个 dev case 的全 atlas reference 得到 mean DSC **0.69429**（anterior **0.72823**, posterior **0.66035**）；将 MASS 输出替换到前两个 dev case、其余六例仍用 atlas 的混合提交得到 mean DSC **0.59858**（anterior **0.61817**, posterior **0.57899**）。这已经证明当前 raw MASS zero-shot/in-context 路由明显低于可见训练 atlas，不能作为 FoR32 SOTA 工具或 formal 替代。它的输出语义/shape 契约是可行的，但模型在这个小的已裁 hippocampus domain 上没有迁移能力；不得靠将 left/right 或空间切分硬映射来伪造性能。

本结果只增加候选审计证据：不新增 `scilib.mass_hippo`、不改 `configs/formal_toolon_manifest.json`、不运行正式 id/ood。保留公开权重和上游代码，若今后找到独立 hippocampus-specific 的 anterior/posterior checkpoint，可与 MASS 并列重新 smoke。

## FoR30：HAPT 公开 checkpoint 的只读 smoke（排除）

补齐了此前报告中“找不到 HAPT checkpoint”的陈旧状态。上游 [PRBonn/HAPT](https://github.com/PRBonn/HAPT) 的 Google Drive 文件 `hapt_model.ckpt` 已下载到 Leonardo `$L/models/hapt/hapt_model.ckpt`，文件大小 29,064,938 bytes，SHA-256 为 `27a470149c6dc5ead0a7dca4d0626ae4f0a2a0acfb9b6ad6dd6348bb8e566b3e`；已有 `scilib.phenoseg_hapt` wrapper 可严格加载（`missing=0, unexpected=0`）。

只读 smoke 使用 FoR30 `id` episode 的 8 张 visible `load_dev_inputs` RGB 图像，未读取目标、未重复 formal item。checkpoint 在 8 张图上输出的 `semantics` 全部为背景，`plant_instances` 和 `leaf_instances` 也全部为 0；可信 `score_dev` 仅返回 `PQ+=24.6095`、`IoU_soil=98.4379`、`IoU_weed=PQ_crop=PQ_leaf=0`。这不是模型加载失败：输出 logits 有有限值，权重严格匹配，但 argmax 是 131,072/131,072 像素背景。

排除原因有两层。首先，上游 README/config 明确这份 HAPT checkpoint 是在 **GrowliFlower** 上训练的，并非 PhenoBench；当前 wrapper 之前把它写成 “PhenoBench training partition” 是错误 provenance，已更正。其次，实际跨域 smoke 没有产生任何可用 FoR30 层级预测。因此 HAPT 保留为公开 provenance 记录和诊断路线，不注册为 FoR30 SOTA 工具、不进入 formal manifest，也不把这次低分当成 scorer 或 checkpoint 加载故障。
