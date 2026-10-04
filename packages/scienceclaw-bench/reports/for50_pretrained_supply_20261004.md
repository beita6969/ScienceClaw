# FoR50 公开冠军权重供应链核查（2026-10-04）

## 候选

SemEval-2023 Task 4（ValueEval）的公开冠军单模型是
[`tum-nlp/Deberta_Human_Value_Detector`](https://huggingface.co/tum-nlp/Deberta_Human_Value_Detector)，配套代码在
[`danielschroter/human_value_detector`](https://github.com/danielschroter/human_value_detector)。论文报告的冠军系统使用 DeBERTa/RoBERTa 集成，官方 test F1 约 0.56；Hugging Face 单模型说明约 0.55。

## 当前数据边界核对

Leonardo 上当前 FoR50 原始文件行数（含表头）为：

| 文件 | 行数 |
|---|---:|
| arguments-training.tsv | 5394 |
| arguments-validation.tsv | 1897 |
| arguments-test.tsv | 1577 |
| arguments-test-nahjalbalagha.tsv | 280 |
| arguments-validation-zhihu.tsv | 101 |
| labels-test-nyt.tsv | 81 |

去掉各文件表头后合计 **9324** 条标注记录，和该公开模型卡声明的训练/测试数据规模完全相同。项目当前 FoR50 的正式边界包含 training/validation、ValueEval ID test、Nahj al-Balagha OOD；NYT 标签也不在可见输入中。

## 决策

该 checkpoint 的公开说明没有给出一个只使用本项目可见 training split、且排除 ID/OOD/Nahj/Zhihu/NYT 的冻结版本。其 9324 条规模与本地完整标注供应直接相等，不能证明没有使用本项目评测标签。因此：

- 不下载或接入该 checkpoint 的预测作为 FoR50 formal/tool-on 结果；
- 不把冠军 0.55–0.56 与当前 fixed-route 分数拼接比较；
- 保留现有只用 `load_train` 可见标签的 `fit_predict` 工具和全池诊断；
- 若以后取得明确只在独立训练语料或严格 train-only split 上训练的 checkpoint，再以新工具名、独立 item 和完整 provenance 复核。

这是一条供应链排除证据，不是 FoR50 分数或 scorer 修改；没有读取 hidden labels，也没有重复正式 episode。
