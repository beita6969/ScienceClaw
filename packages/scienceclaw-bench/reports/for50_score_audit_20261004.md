# FoR50 ValueEval 低分与可比性核查（2026-10-04）

本次只核查 FoR50 的切分、标签列、官方评分和可见数据路线；没有重复正式 item、修改 scorer/验收，或把评测标签提供给工具。

## 结果

- Leonardo 当前数据完整且列顺序正确：training/validation/test/Nahj 去表头分别为 `5393/1896/1576/279` 行；标签表的 20 列与 `VALUES` 顺序一致。总记录数 9144，未发现跨文件重复 Argument ID。
- adapter 的全池划分为 src/val/id/ood `1424/472/1576/279` 条。可见 training 与四个评测池没有完全相同的 `(conclusion, stance, premise)` 记录；因此没有发现训练标签泄漏或 split 过滤错误。重复 conclusion 很多是官方按 conclusion 分组的协议，不是重复抽样 bug。
- `ValueEvalAdapter.official_f1` 与随数据提供的上游 evaluator 公式一致：只统计有 gold positive 的 value，分别求 macro precision/recall，再取 harmonic mean；空预测和无支持类别的边界也有 focused test 覆盖。远端 `tests/test_task_for50.py` 全部通过（10 passed）。
- 固定 visible-train-only `fit_predict` 全池诊断为 ID **0.548950**（1576 条）和 OOD **0.437082**（279 条），明显高于 all-values reference **0.262930/0.128459**。因此当前低分不是 scorer 被压低；16 条 episode 的波动来自类别支持和样本量，而不是切片计算 bug。
- 现有三批互斥 agent 证据：`SOTA50` ID 均值 0.508216、OOD 接受项均值 0.395049；`SOTA50B` ID 均值约 0.4916、OOD 全量均值约 0.3223；`H50C` ID 均值 0.4735、OOD 均值 0.3196。它们都直接调用固定 `fit_predict`；OOD 未通过的条目保留，不重复运行。

## 外部权重与处理决定

公开 `tum-nlp/Deberta_Human_Value_Detector`（单模型约 0.55，冠军集成约 0.56）声明的标注规模正好覆盖本项目公开的 training、validation、ID、Nahj、Zhihu 和 NYT 文件总量 9324。其训练边界无法证明排除了本项目评测标签，因此不能作为公平冻结工具接入；排除证据见 `reports/for50_pretrained_supply_20261004.md`。

此前 BGE-large 可见训练诊断为 ID/OOD `0.54849/0.36655`，没有优于 TF-IDF 固定路线；不注册新 alias。当前没有安全、同口径、独立训练的冻结 checkpoint 可替换现有路线。

## 结论

FoR50 的“低分”主要是 16-item episode 与完整 ValueEval test 的统计口径差异，以及 Nahj OOD 的真实分布偏移；没有发现评分、切分、标签列或工具覆盖 bug。保留固定 `fit_predict` 和 full-pool trusted diagnostic，继续禁止重跑已消耗 item；只有取得明确 train-only/独立语料 provenance 的模型，才可用新工具名和新互斥池复核。
