# FoR52 Psychology — Psych-201 discrete sequential choice prediction (micro accuracy, higher is better)

Adapter: `scienceclaw/bench/tasks/for52_psych201.py` (`Adapter = Psych201Adapter`). Status: **available**.
Tests: `tests/test_task_FoR52.py` (~2 s with the cached index).

## Dataset
| field | value |
|---|---|
| name / version | Hugging Face `marcelbinz/Psych-201-discrete`, commit `060062064d00766ea0b5666c73268329d0f556b6` (single official split `train`, 4 parquet shards, 131,834 participant sessions, 103 studies) |
| URL / license | https://huggingface.co/datasets/marcelbinz/Psych-201-discrete — Apache-2.0 |
| local path | `<DATA_ROOT>/for52-psych201-discrete/reconstructed_v1/population/train-0000{0..3}-of-00004.parquet` (LFS sha256 verified by the data team) and `population-index-and-exclusions.parquet` (row index: shard, row, study, participant, `in_eval`, `is_psych101_test`, participant group id) |
| format | one row = one participant session in natural language; every human response is written `<<KEY>>` |

## Items, pools, splits
* **Item** = one response marker of one session. Input = the whole session text before that marker's `<<`
  (instructions, all earlier trials, the participant's own earlier responses and outcomes: teacher-forced
  history, as in the Psych-101/Centaur evaluation), the study id and the legal response keys. Target = the recorded
  human response (one uppercase key).
* **Legal keys** are parsed only from text before the marker with the data team's explicit option templates
  (`scripts/psych201_strict_adapter.py`, rule `strict_single_key_complete_group_v1`, ported to
  `legal_options`): dynamic lines ("You can choose between option K and option W. You press"), current
  lottery descriptions, study-specific fixed key maps from the instructions. A marker is eligible if its target
  is a single uppercase key inside the parsed options.
* **Units / leakage control.** Per study, participant groups (study + participant) are ranked by
  `sha256("FoR52|20260928|cand|<study>|<group>")`; the first 72 are parsed (first row of the group = its
  session) and up to 48 qualifying sessions are kept (≥ 3 valid markers and ≥ 1 eligible target with ≥ 2 earlier
  responses and ≤ 16,000 characters of history). One target marker per session is fixed by a seed-independent hash.
  Each participant contributes at most one unit anywhere in the benchmark, so no item's history can contain
  another item's target. The parsed index is cached in `cache/tasks/FoR52/index_v1.json.gz` (built in ~15 s).
* **Studies:** 36 studies with ≥ 16 qualified sessions. Ranked by `sha256("FoR52|20260928|studies|<study>")`,
  every third study is **OOD** (12 studies: anllo2024weird, bavard2018magnitude, binz2022heuristics,
  dubois2022value, enkavi2019adaptivenback, garcia2023experiential, guenther2020TS, palminteri2017confirmation,
  ruggeri2022globalizability, rutledge2023happiness, steingroever2015data, thoma2025problearn); the other 24 are
  IID. `lineage.ood_kind = "proxy_within_dataset"` (same corpus, held-out experiments; no second behavioural
  dataset exists locally).
* **IID roles** per study (hash order): first 6 non-flagged sessions → visible train, next 2 → dev, the rest →
  src / val / id (55/15/30 %). Groups flagged `in_eval` / `is_psych101_test` upstream are never visible data.
  Pool sizes: train 144, dev 48, src 481, val 132, id 263, ood 544 units.
* Episode = 16 items drawn block-wise from a seeded permutation of the split pool (prefix-stable, item-disjoint).
  Capacity (16 items): src 30, val 8, id 16, ood 34 episodes.

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | 32 complete sessions of other participants from IID studies (for IID episodes one per item study first, then round-robin over studies); dicts `study`, `text` (truncated at a line boundary after 24,000 characters), `responses` (`pos`, `response`, `options`) |
| `load_dev_inputs` | 48 dev items (the whole IID dev pool, 2 participants per IID study; other participants than the evaluation items), same format as the evaluation items |
| `score_dev(dev_pred)` | `dev_accuracy`, `dev_reference_accuracy`, `dev_invalid`, `n_dev` for `dev_pred[i]` = key predicted for `dev_items[i]` of `load_dev_inputs` (visible dev signal; `_dev_evaluate` is None). Its standard error is ~0.07 at 48 items (~0.12 at the earlier 16), against the 0.05 acceptance margin of the 16-item evaluation. |
| `load_eval_inputs` | 16 items: `study`, `history`, `options`, `options_text` (in output order) |

## Domain library
`scilib/psych.py` (docstring shown through `scilib.describe("psych")` at the end of the objective; importable in code nodes):
participant-mode reference / accuracy helpers with the evaluator's semantics, a text parser for the `<<KEY>>` markers and
their outcomes, history features (last choices, choice frequency and kernels, delta-rule Q-values, win/lose-conditioned
stay), maximum-likelihood fits of small choice models per item (`kernel`, `qlearn` = Q-learning + softmax +
perseveration, `wsls`), a LightGBM candidate-level model trained on the visible sessions and histories, the single entry
point `fit_predict(items, train_sessions, method, extra_items)` and `cross_validate` (pseudo-items cut from complete
visible sessions, folds grouped by participant). It reads no files and no hidden labels.

Language-model route (`build_prompts`, `parse_answers`, `fit_predict(..., llm_answers=, llm_weight=)`): `build_prompts`
turns items into one text prompt each (task instructions + the latest trials of the transcript, legal keys, "which key does
the participant press next"; styles `transcript` / `long` / `model`, the last adds the probabilities of the fitted
`personal` model); the prompts of the evaluation and the dev items (16 + 48 <= 96) go through one `llm` node (template
`{item}`, `parse: text`, small `max_tokens`), `parse_answers` turns the replies into keys, and `fit_predict` mixes the
answer shares into the history-model probabilities (`(1 - w) * p + w * share`, default `w = 0.35`). The language model reads
the instruction text of the experiment, which the history models cannot; the history models supply a per-participant
confidence. Offline (30 seed-20260928 src episodes, 480 items, executor 27B, temperature 0, no thinking): participant-mode
reference 0.588, `auto` 0.652, LLM alone 0.704 (`transcript`) / 0.706 (`long`), `auto` + LLM at w = 0.35: 0.710 / 0.713
(gain over the reference +2.0 items per 16 vs +1.0 for `auto`; share of episodes with at least +1 item: 0.80 / 0.73 vs 0.57
for `auto`, standard error about 0.07). Numbers are documentation for maintainers; the policy-visible
docstring states no accuracies.

## Metric, reference, acceptance (D_V)
* **Micro accuracy** = correct / items; a prediction outside the legal keys is wrong (data-team scorer
  `score_response`) and additionally violates the `allowed_labels` hard constraint. Pooled metric = micro
  accuracy over all items of the episodes (an invalid output counts as 16 wrong).
* **Reference baseline** (visible input only): the participant's most frequent earlier response among the legal
  keys (ties → most recent; no earlier response → first key). Mean reference accuracy on seed-11 episodes: src
  0.60, val 0.66, id 0.53, ood 0.66.
* **Acceptance:** `accuracy >= reference + 0.05` (≥ 1 more correct item than the reference out of 16).
* `details`: `reference`, `norm_score` (= accuracy / reference, clipped [0, 10]), `pooled_payload`
  (`item_ids`, `y_true`, `y_pred`, `y_ref`, `studies`).
* **Hard constraints:** `output_format` (list of 16 strings), `allowed_labels` (y[i] ∈ options[i]).

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 300, max_llm_items 96 (= 3 × (16 eval + 16), independent of the 48 dev items, i.e. unchanged when the dev slice was enlarged from 16 to 48)`.
Median history length 6,255 characters (max 16,000): an LLM pass over the 16 items is ≈ 30–60k prompt tokens.

## Deviations from the historical protocol
* Exact historical sample ids, qualification and split rules were **not recovered**; these are rebuilt splits.
  Historical protocol: 64 IID + 64 OOD items; here id = 4×16 = 64 and ood = 4×16 = 64 items.
* Item-level qualification (one eligible target marker per session) instead of the data team's whole-group
  qualification; the option templates are the data team's.
* Histories are limited to 16,000 characters (targets are chosen among early-enough markers) to bound LLM cost.
* OOD = held-out studies inside Psych-201 (not a different dataset).
