# FoR32 InnerEye-HS 外部权重补充记录（2026-10-03）

这份补充记录更正并扩展 `reports/for30_for32_next_options_20261003.md`：服务器复核之后找到一个可公开追溯的医学分割 checkpoint，但它只能用于只读 binary smoke，不能直接变成 FoR32 formal 工具。

## 来源与文件指纹

- Microsoft InnerEye-DeepLearning v0.5 release：<https://github.com/microsoft/InnerEye-DeepLearning/releases/tag/v0.5>。
- 模型说明：<https://innereye-deeplearning.readthedocs.io/md/hippocampus_model.html>。
- 下载地址：<https://github.com/microsoft/InnerEye-DeepLearning/releases/download/v0.5/hippocampus_segmentation_model.zip>。
- 许可证：release model card 标注 MIT。
- Leonardo 存储：`$L/models/innereye/hippocampus_segmentation_model.zip`，外层 ZIP SHA-256 为 `256dd69d88b8c3ee6a4117868ad7fe68ece8fc6f0b952d315cf2b0b71d20da81`。
- 内层 `Hippocampus_123.zip`（`$L/models/innereye/extracted/hippocampus_segmentation_model/Hippocampus_123.zip`）SHA-256 为 `5c00ffeab96affb2ad3d7c7f4d63a023a0bae6ea0f7195e459d8d1494cfc7e38`。
- 展开后的 checkpoint 路径为 `$L/models/innereye/extracted/nested/Hippocampus_123/MODEL/final_ensemble_model/checkpoints/last.ckpt`；五个 fold checkpoint 的文件大小均为 97,121,589 bytes。

## 训练集与公平性证据

InnerEye v0.5 文档明确写的是五折 3-D U-Net、998 个 ADNI 图像/标注对。该公开 model card 和 release 文件中没有 MSD Task04 或 Medical Segmentation Decathlon 的训练集声明；目前没有发现它读取本项目 260 个 FoR32 `imagesTr/labelsTr` 病例的证据。因此它是一个有来源和冻结指纹的外部候选，但“无 MSD Task04 重叠”仍以公开训练说明为证据边界，不扩写成逐病例证明。

## 输出契约不匹配

InnerEye-HS 的输出是 whole left/right hippocampus（两个侧别的全结构分割）。FoR32 的 formal 契约要求同一个 hippocampus crop 内的 `1=anterior`、`2=posterior` 两个标签。侧别不能当作前后部，任何把 left/right 映射成 anterior/posterior 的代码都会制造标签语义，因此本候选保持 `smoke_only_unintegrated`，不注册进 `SOTA32`，不运行正式 item，也不改变验收阈值。

## 只读 smoke 边界

新增 `scilib/hippo_innereeye.py` 和 `tests/test_scilib_hippo_innereeye.py`。wrapper 只从已展开路径加载官方 checkpoint，以随机体积验证 forward，输出 binary whole-hippocampus union；它不下载、不训练、不读取 target、不调用 scorer，也不做 anterior/posterior 映射。`configs/p3_asset_manifest.json` 保存来源、外层/内层 SHA 与状态；`configs/formal_toolon_manifest.json` 保留 FoR32 blocked，并在原因中说明该契约差异。任何真实 FoR32 formal 接入仍需一个输出 0/1/2 且逐病例可审计的独立 checkpoint。

### Leonardo 只读 smoke 结果

2026-10-03 在 Leonardo `sc-harness` 上用已核对的 `last.ckpt` 和随机 `float32` 体积 `(1,16,16,16)` 执行一次 CPU forward：`available=True`，输出 shape `(1,16,16,16)`、`uint8`、取值 `[0]`、前景和为 `0`，耗时约 `2.77 s`。该命令只加载官方权重和随机体积，没有读取 FoR32 target、运行 scorer 或进入 formal；结果仅证明 wrapper/权重可加载并满足 binary 输出边界。

### 真实 FoR32 crop smoke（排除直接替换）

为避免把一个可加载的 binary checkpoint 误报成可用的 FoR32 SOTA，本轮在 Leonardo 已有 A100 分配上直接对三种真实编码的 FoR32 `imagesTr` crop 做了只读前向。输入分别是 `hippocampus_325`（IID，`35x51x40`）、`hippocampus_057`（IID，`35x51x34`）和 `hippocampus_217`（OOD，`38x53x27`）；没有读取对应 `labelsTr`、没有调用 scorer，也没有写入 formal manifest。InnerEye 原始 checkpoint 期望 whole-brain ADNI crop `(128,176,176)`，而 FoR32 item 是已经裁好的小体积且要求前/后两个语义类。直接 pad、固定尺寸三线性 resize，以及官方 MRI-window 的只读输入都返回全背景（前景体素数分别为 `0/0/0`；单次 A100 前向约 `4.6 s`，模型随后驻留缓存）。

这说明 InnerEye-HS 的“whole left/right hippocampus”二类输出不仅语义不匹配，在本任务的 crop 输入契约上也没有可用的零样本行为。任何把 binary mask 按轴硬切成 anterior/posterior 的做法都会制造标签语义，而且当前 smoke 没有前景可切，因此**不新增 split wrapper、不接入 ToolSpec、不运行 formal**。该 checkpoint 仍保留为可追溯的 external binary smoke 资产；FoR32 `SOTA32` 继续 blocked，直到出现输出 0/1/2 且输入 crop 契约匹配、训练病例可审计的冻结模型。
