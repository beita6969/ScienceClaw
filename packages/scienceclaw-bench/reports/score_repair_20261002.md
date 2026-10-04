# Score repair and comparability pass — 2026-10-02

本轮目标是把低于同行、协议不可比或容易被工具捷径抬高的分数，改成可复现、可解释、能与同行对照的结果。没有修改 primary scorer、验收边际或隐藏标签边界。

## 已落地的组件

- **FoR51** 新增 `scilib.matphonon_regression` 的两种 CPU 结构回归器：log-target ExtraTrees 和 log-target HistGradientBoosting。FoR51 工具面板新增 `fit_log_extra_trees`、`fit_hist_gradient_boosting`，两者只接收 `load_train` 的结构和标签以及评测结构特征；不读取评测目标。返回 provenance，交叉验证只在训练行内进行。
- **FoR40** 增加强 k-NN log-mel 参考和 pooled diagnostics。原 1-NN 参考、agent primary 和强参考分别记录，诊断不参与验收。
- **FoR49** 增加纯 Z3、多数猜测和提交结果的分层诊断，记录 hard/easy tier 与 Z3 决定比例；primary 和验收保持不变。
- **FoR50** 增加 pooled diagnostics 和完整 split 的 all-values reference，量化 16 项切片相对完整测试集的偏差；不改变 official F1、primary 或验收。

## 服务器复测

代码已同步到 Leonardo 的 `$F/scienceclaw`，数据入口全部可用。

- **FoR51**：4 个 id episode 使用 ExtraTrees，MAE 为 57.26、47.99、49.70、42.77，均值 **49.43**；同 episode 的 5-NN 参考均值 **145.59**。4 个代理 OOD episode 的 MAE 均值 **31.01**，参考均值 **55.11**。HistGradientBoosting 在独立 id episode 上 MAE **46.50**，参考 **158.82**，验收通过。MegNet 榜单约 28.76，因此能力缺口缩小但尚未消除。
- **FoR30**：4 个 id episode 的 PQ+ 为 62.01、68.23、62.82、66.85，均值 **64.98**；4 个 id 轨迹都调用了 `phenoseg_m2f`。同行前三约 81–83，差距仍来自杂草/叶片实例分割。4 个 OOD episode 为 36.40、44.37、49.24、77.34，均值 **51.84**；其中 3 个没有调用 Mask2Former，说明 OOD 差距仍主要是工具选择和跨域泛化问题。
- **FoR36**：4 个 id SDR 为 8.09、2.36、9.14、7.62 dB；3 个轨迹调用 `htdemucs`，1 个仍走 SoftMask。工具调用时已经达到 7.5–9.8 dB 同行区间，低分来自 agent 没有稳定选用分离器。
- **FoR50**：完整 split 的 all-values reference 已在服务器重算：id 1576 项 F1=0.26293，ood 279 项 F1=0.12846；这给后续 full-test agent 评测提供了同口径基线。

## 验证

本地完整 `pytest -q` 通过；新增模块、FoR40/49/50/51 针对性测试均通过。新增诊断字段只写入 trusted-side details，不进入 agent 可见输入，也不改变历史 primary。

## 下一步

1. 将 FoR30 的 4+4 结果写入 overnight 对照表，并把 OOD 的工具选择/跨域差距作为单独诊断项。
2. 用 FoR51 两个回归器做 8 episode pooled 评测，并报告 MAE/MAD；若仍显著高于 MegNet，再考虑引入结构图模型供应链。
3. 在 FoR49 报告纯 Z3/多数猜测占比，在 FoR40 报告 full-cohort pooled score；FoR50 补 agent full-test 运行，不能再拿 16 项 F1 直接对照榜单。
