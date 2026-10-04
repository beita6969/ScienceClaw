# FoR50 Philosophy and religious studies — SemEval-2023 Task 4 ValueEval (official F1, higher is better)

Adapter: `scienceclaw/bench/tasks/for50_valueeval.py` (`Adapter`; shared helpers in `_adapter_utils_for43_45_47_48_50.py`).
Status: **available** (all argument/label files of the Zenodo deposit present).

## Dataset
| field | value |
|---|---|
| name / version | Touché23-ValueEval, version 2024-01-24, Zenodo record 10564870 |
| URL / DOI | https://doi.org/10.5281/zenodo.10564870 |
| license | CC BY 4.0 |
| local path | `<DATA_ROOT>/for50-valueeval-2023/raw/{arguments,labels}-<split>.tsv`, `value-categories.json` |
| official scorer | touche-code `semeval23/human-value-detection/evaluator/evaluator.py` @ `f6d34abc` (version 2023-08-13; copy in `reconstructed_v1/evaluator/upstream/`) |
| population | main dataset training 5,393 / validation 1,896 / test 1,576; supplementary test-nahjalbalagha 279, validation-zhihu 100, test-nyt 80 (labels only) |

## Items, pools, splits
* **Item** = one argument (Conclusion, Stance, Premise); label = 0/1 vector over the 20 level-2 value categories
  (official column order). Item id `valueeval/<Argument ID>`.
* **IID** = main dataset: src / val from the official **validation** split, partitioned once **by conclusion** (the
  official splits keep a conclusion's arguments together; 75/25 %, `partition_seed = 20260928`): 1,424 / 472
  arguments; id = official **test** split (1,576).
* **OOD** = official supplementary test set **Nahj al-Balagha** (279 arguments from and based on Islamic religious texts,
  translated from Farsi) — a different source dataset of the same task (`ood_kind = "cross_dataset"`). The NYT test set
  is not usable: its argument texts are not in the public deposit (labels only). Zhihu is not used.
* Episodes: 16 arguments drawn prefix-stably from a seeded permutation.

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | 2,000 labelled arguments of the official training split (table + (n, 20) label matrix + value names) |
| `load_value_taxonomy` | official taxonomy: 20 categories → level-1 values → example effects |
| `load_dev_inputs` | 64 further training arguments (labels withheld) |
| `score_dev(dev_predictions)` | official F1, macro P, macro R on the dev slice + F1 of predicting everything (visible dev signal; `_dev_evaluate` is None) |
| `load_eval_inputs` | the 16 evaluation arguments (table) + value names (column order of y) |

OOD episodes also train on the main training split (as in the shared task).

## Required output
`y`: 0/1 matrix of shape (16, 20), rows = evaluation arguments, columns = `value_names`.

## Metric, reference, acceptance (D_V)
* **Official F1** of the ValueEval'23 evaluator: over the categories with ≥ 1 gold-positive argument in the episode,
  precision_c = TP/(TP+FP) (0 when nothing is predicted) and recall_c = TP/(TP+FN) are macro-averaged; `F1 = 2PR/(P+R)`
  (harmonic mean of macro P and macro R — *not* the mean of per-category F1s, which is reported as
  `mean_category_f1`). Zero guard: F1 = 0 when P = R = 0 (the upstream script divides by zero; data-team compat copy).
  `tests/test_task_for50.py::test_matches_official_evaluator_script` runs the unchanged official script in a subprocess
  and matches its F1 to 1e-12. Pooled metric: official F1 over all arguments of the given episodes.
* **Reference** = predict all 20 categories (the shared task's "1-baseline"). Default plan pooled reference: src 0.280,
  val 0.313, id 0.263 (full official test: 0.2585, leaderboard 1-baseline 0.26), ood 0.148; per episode 0.20–0.36 (the
  all-ones baseline is strong on 16-item episodes because unsupported categories are skipped).
* Calibration (sanity check, not a research result): one-vs-rest TF-IDF logistic regression on `load_train` with a
  0.2 probability threshold beats the reference by −0.04…+0.10 per IID episode (about half pass the margin) and is at
  or below the reference on OOD.
* Toolkit effect size (`scilib/valueeval.py`, 16-item pseudo-episodes drawn from val/test/Nahj pools, model fit on 2,000
  training arguments): with a single global probability cut-off the mean gain over the all-ones baseline is about
  +0.07…+0.09 (per-episode pass rate about 0.6–0.8 IID, 0.25–0.4 OOD). `fit_predict(decision="expected_f1")` (per-column
  Platt calibration on conclusion-grouped out-of-fold scores + a plug-in expected-F1 choice of how many rows to mark per
  category) lifts this to about +0.155…+0.185 (pass rate about 0.94–0.98 IID, 0.67–0.73 OOD); on real src episodes 6/6
  accepted vs 2/6 for the cut-off. Because the official F1 averages precision and recall over categories separately, the
  gain comes from marking all rows for some categories and only the top-ranked rows for others; the estimate depends on
  that metric definition. Model variants (conclusion features, ridge/SVM blends, stacking, other C, isotonic or shared-slope
  calibration, prior-shift adaptation) did not change the result beyond noise.
* **Acceptance**: `F1 >= reference + 0.05` (`margin` configurable).
* `details`: `reference`, `norm_score` (F1/reference clipped to [0,10]), `pooled_payload` (`y_true`, `y_pred`, `y_ref`).
* **Hard constraints**: `output_shape` ((16, 20)), `binary` (entries 0/1).

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 600, max_llm_items 4 × max(items, dev) = 256`.

## Known deviations
* Exact historical sample ids were **not** recovered: rebuilt splits.
* The NYT supplementary test set cannot be used (texts not public); OOD = Nahj al-Balagha only.
