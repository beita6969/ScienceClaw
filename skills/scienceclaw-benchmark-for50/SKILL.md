---
name: scienceclaw-benchmark-for50
description: "Use when a task gives arguments (conclusion, stance, premise) with a training set labelled for the 20 human-value categories of the ValueEval taxonomy (SemEval-2023 Task 4 / Touche23-ValueEval) and asks for a 0/1 multi-label matrix (arguments by 20 values), judged by the official F1 built from macro precision and macro recall. Corresponds to FoR50 of the companion ScienceClaw-Eval benchmark."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Human-value detection in arguments (ValueEval)

## Task
- Inputs: tables with `conclusion`, `stance` (`in favor of` / `against`) and `premise`; training arguments with an (n, 20) 0/1 label matrix; the value names (and taxonomy) that fix the column order.
- Deliverable: a 0/1 integer matrix, rows = target arguments in input order, columns = value names in the given order.

## Quality
Official F1 (ValueEval 2023): over the categories with at least one gold positive in the scored set, macro-average precision (0 when nothing is predicted) and recall separately, then F1 = harmonic mean of the two (not the mean of per-category F1s); higher is better. Compare against the all-ones prediction: because precision and recall are averaged separately and categories without a gold positive are skipped, it is a strong baseline, especially on small batches.

## Tools
- `scilib.valueeval.fit_predict(train_df, Y, [target_df, ...], n_folds=5, C=0.3, k=16, seed=0, groups=None, decision="expected_f1")` returns one 0/1 matrix per target table (also `.probs`, `.scores`, `.counts`, `.oof_f1`). It uses word and character TF-IDF (`TextFeatures`), one-vs-rest logistic regression, out-of-fold scores with folds grouped by conclusion, per-label Platt calibration, then a decision that picks how many top-ranked rows to mark per label to maximise the expected official F1. `decision="threshold"` applies one global cut-off instead. Typed operators `human_value_multilabel_predict` and `value_detection_official_f1`.
- Building blocks: `official_f1`, `official_f1_report`, `grouped_oof_scores`, `column_calibration`, `calibrate`, `expected_f1_decision`, `episode_f1`, `episode_threshold`.
- Optional extra features: sentence embeddings from `scilib.textenc.embed` (asset `bge_large_en`; check `scienceclaw_tools(operation=weights)`). Its public pretraining text may overlap the arguments; disclose that.

## Rules
- Keep validation folds grouped by conclusion (arguments sharing a conclusion are near duplicates); never calibrate or choose thresholds on the target rows.
- The decision depends on how many rows are scored together and on this metric's averaging, so pass the whole target table in one call and say that the gain is specific to the metric definition.
- Keep shape, column order and binary integer entries; evaluate on a held-out source (different corpus, culture or translation) when one exists, since reliability drops under such shift.
