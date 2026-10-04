# FoR52 pooled same-metric reference audit — 2026-10-02

本轮只补充 trusted-side 诊断，没有修改 FoR52 的 primary accuracy、验收边际（`+0.05`）、episode 抽样、隐藏标签或 agent 可见工具。

## 新增内容

`Psych201Adapter.pooled_diagnostics()` 现在同时汇报：

- 全部 episode 合并后的 agent accuracy；
- 现有 participant-history reference（每个目标 participant 的历史多数响应）；
- 新的 `visible_study_prediction` reference：只统计 `load_train` 中其他 participant、同 study 的已见响应，在当前 item 的 legal options 中选频数最高者，平票遵循 item 的 option 顺序；没有可用响应时回退到原 history reference。

`y_study_ref` 只写入 evaluator 的 `pooled_payload`，不会出现在 `load_train`、`load_dev_inputs` 或 `load_eval_inputs`，也不参与 acceptance。无效提交仍按官方 pooled 约定计为全错。

## 可复现本地审计

使用当前冻结分片、4 个 id episode（seed 502）和 4 个 ood episode（seed 503），每项 16 个：

| split | items | pooled history reference | pooled study reference |
|---|---:|---:|---:|
| id | 64 | 0.453125 | 0.40625 |
| ood | 64 | 0.640625 | 0.640625 |

这次运行提交的是原有 history reference，用来固定同指标基线；study reference 只用于比较。它说明 FoR52 的 pooled accuracy 不能只用某一批 16 项 episode 的均值解释，且可见同 study 频率并没有自动带来提升。上述数字不是正式 agent SOTA 结果，也不改变排行榜分数。

## 验证

`tests/test_task_FoR52.py`：10 passed。新增测试覆盖可见 study 频率、无可用行时的回退、pooled accuracy 与原 `pooled_metric` 一致，以及 diagnostic-only 标记。
