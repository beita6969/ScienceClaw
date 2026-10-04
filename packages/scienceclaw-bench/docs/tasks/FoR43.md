# FoR43 History, heritage and archaeology — HIPE-OCRepair-2026 OCR post-correction (weighted cMER-micro, lower is better)

Adapter: `scienceclaw/bench/tasks/for43_hipe.py` (`Adapter`; shared helpers in `_adapter_utils_for43_45_47_48_50.py`).
Status: **available** (checks the IID/OOD/train files and the `jiwer` package).

## Dataset
| field | value |
|---|---|
| name / version | HIPE-OCRepair-Bench, release **v0.9.5** (final competition release with unmasked test files), commit `bd463f7e340280473d9e5bab5e231f4922ea1a3b` |
| URL | https://github.com/hipe-eval/HIPE-OCRepair-2026-data (archive `raw/HIPE-OCRepair-2026-v0.9.5.tar.gz`, sha256 `8573f473…4563`) |
| license | per corpus (participation guidelines): icdar2017 CC BY-NC-SA 4.0, impresso-snippets CC BY-NC-SA 4.0, dta19 CC BY-NC 4.0 (DTA pages CC BY-SA 4.0), overproof research use only, impresso-nzz CC BY-NC 4.0 |
| local path | `<DATA_ROOT>/for43-hipe-ocrepair-2026/upstream/data/v0.9/<corpus>/<lang>/*.jsonl` |
| official scorer | HIPE-OCRepair-scorer commit `d1e76e447629ea9cf8dead32ae3c44b0d48d77b3` (MIT), `hipe_ocrepair_scorer/ocrepair_eval.py`, dependency `jiwer` 4.0.0 |

## Items, pools, splits
* **Item** = one transcription unit (newspaper chunk / paragraph / article or book page). Input = raw OCR text +
  metadata (dataset, language, test_set, document_type, date, publication_title). Label = verified transcription.
  The OCR `quality_report` (CER/WER computed *against the ground truth*) is never exposed.
  Item id `hipe/<dataset>/<lang>/<file split>/<document_id>`.
* **Test set** (= stratum) = `<dataset>/<language>` as in the official ranking.
* **IID pool** = official labelled test files of the ranking newspaper test sets `icdar2017/en`, `icdar2017/fr`,
  `impresso-snippets/de|en|fr` (501 units). Partitioned once (`partition_seed = 20260928`) by **natural document**
  (ICDAR periodical document, Impresso page; data team grouping rules) into src / val / id = 55 / 15 / 30 %,
  stratified by test set: src 277, val 76, id 148 units.
* **OOD pool** = corpora from other collections / document types never used in IID: `dta19` (19th-century German
  books from the Deutsches Textarchiv with synthetic OCR degradation; the official `dev` split, because the DTA test
  gold is masked in v0.9.5) with **one noise level per page** (hash-chosen; the three levels of a page share the gold)
  → `dta19-l0/de` 39, `dta19-l1/de` 37, `dta19-l2/de` 34 pages; and `overproof-combined/en` (Trove + Chronicling
  America articles, official test + dev, 62). `lineage.ood_kind = "cross_corpus_within_benchmark"` — a different
  primary dataset inside the HIPE-OCRepair-Bench umbrella (no other OCR post-correction benchmark is available locally).
* Episodes: 16 units, balanced over the pool's test sets with a rotating remainder (IID 5 strata → 4/3/3/3/3; OOD 4
  strata → 4 each), drawn prefix-stably from a seeded permutation; items never repeat inside a split.

## Visible data and tools (D_E)
Visible pool = official **train/dev** partitions of the pool's corpora (IID: icdar2017 train/dev, impresso-snippets
train/dev; OOD: dta19 train levels 0-2, overproof train) **minus every natural document containing an evaluation unit
of any split** (719 records removed: the official ICDAR test chunks come from periodicals whose other chunks are in the
official train set). Per episode a seeded sample (balanced over test sets):

| tool | returns |
|---|---|
| `load_train` | 96 labelled units: metadata + `ocr_text` + `gt_text` |
| `load_dev_inputs` | 16 further units (ground truth withheld) |
| `score_dev(dev_predictions)` | weighted cMER-micro of the corrections, of the uncorrected OCR, and per test set — the visible dev signal (`Episode._dev_evaluate` is None) |
| `load_eval_inputs` | the 16 evaluation units (no gold) in output order |
| `text_mer(texts_a, texts_b)` | per-pair character MER after the official normalisation (domain tool) |

## Required output
`y`: list of 16 strings, `y[i]` = corrected transcription of `items[i]`.

## Metric, reference, acceptance (D_V)
* **Weighted cMER-micro** exactly as the official ranking (guidelines §5.5): texts normalised with the scorer's `norm()`
  (lower-case; ß→ss, ꝛ→r, œ→oe, æ→ae, aͤ/oͤ/uͤ→ä/ö/ü; `—\n` and `¬\n` removed; non-word characters and `_` → space;
  whitespace collapsed); per unit the alignment counts (H, S, D, I) of `jiwer.process_characters(gt, hyp)`; per test
  set `cMER_micro = ΣS+ΣD+ΣI / ΣH+ΣS+ΣD+ΣI`; score = weighted mean over the test sets present in the episode with the
  official weights (`dta19-l0/l1/l2` 1/3 each, all others 1). `tests/test_task_for43.py::test_matches_official_scorer`
  checks per-test-set cMER-micro against the unchanged official `Evaluation.score_over_datasets(normalize=True)`
  (agreement to 1e-12). Auxiliary: unweighted cMER-micro, cMER-macro, pref_score_cmer_macro (vs. OCR).
* **Pooled metric**: the same weighted cMER-micro over all units of the given episodes (sums of counts per test set).
* **Reference** = copy the OCR input (official no-edit baseline). Default plan (bench seed 20260928), pooled reference:
  src 0.0238, val 0.0253, id 0.0196, ood 0.0443; per episode 0.016–0.052.
* **Acceptance**: `score <= 0.95 × reference` (≥ 5 % relative error reduction; `rel_margin` configurable).
* `details`: `reference`, `norm_score` (= reference/score, clipped to [0, 10]), `per_test_set`, `pooled_payload`
  (`test_sets`, `counts_pred`, `counts_ref`). Invalid outputs: `primary=None`, pooled with the reference counts.
* **Hard constraints** (visible): `output_length` (16 entries), `strings`, `length_plausible` (normalised length within
  [0.5, 2.0] × normalised OCR length — no truncation / run-away generation).

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 3600, max_node_s 900, max_llm_items 128 (8 × items)`.

## Domain library (`scilib/ocrfix.py`)
The objective ends with `scilib.describe("ocrfix")` (module docstring = interface documentation; sandboxed code nodes can
import `scilib.ocrfix`). The library packages what the HIPE-OCRepair / BLOCR post-correction solutions do around the LLM
call; the `llm` node itself stays a node of the agent's graph (plan -> `llm` node -> merge).
* **Metric**: `norm`, `char_counts`, `cmer`, `edit_rate`, `official_weight`, `score` (weighted cMER-micro; identical to the
  adapter's `hipe_norm` / `char_counts` / `weighted_cmer_micro`, checked in `tests/test_scilib_ocrfix.py`).
* **Training-data statistics**: `noise_profile(train)` per test set (OCR cMER, per-unit quantiles, gold/OCR length ratio,
  end-of-line hyphens, gain of the hyphen rule). `join_line_hyphens` writes the gold convention (`-\n` between letters before
  a lower-case line start -> `¬\n`, which `norm` removes).
* **Chunking and prompts**: `split_chunks` (balanced pieces cut at line/sentence ends, so that a chunk stays far below the
  2000-token executor cap), `plan(units, train, ...)` builds one llm item per chunk (instruction from the organisers'
  guidelines / BLOCR error classes: over-/under-segmentation, misrecognised and missing characters, per-language hint, and
  `n_examples` aligned OCR/gold excerpts of the same test set from `load_train`); `max_items` enlarges the chunks when a plan
  would exceed the 128 items of an llm node.
* **Guarded merge**: `merge(plan, outputs, ...)` accepts a chunk reply only inside a length band (0.95-1.05) and an
  edit-rate cap (0.15), then word-level: a changed passage is taken if it differs by at most `max_hunk_edits` (4) normalised
  characters (the visible training pairs: 91 % of the OCR->gold passage changes need at most 4). Otherwise the OCR text
  is kept; the report counts rejected chunks and hunks. Defaults come from visible training statistics only; no
  evaluation-item statistics are used anywhere in the library.
* **Measured on visible training pairs** (dev-style sample of 108 units, 9 test sets, real 27B `llm` node, thinking off):
  the raw reply of the built-in prompt changes weighted cMER by about -46 % (0 examples) / -49 % (4 examples); with the merge
  guard (length band, edit-rate cap 0.15, hunk filter 4) about -42 % / -44 %; on a separate 54-unit sample a deliberately
  loose "improve the text" prompt is +13 % without the hunk filter and -10 % with it. Chunk sizes 600-2200 characters and whole units differ by a few points; 0-6 examples were explored.
* **Steps**: 3 nodes (code, llm, code) are enough; dev and evaluation units can go through one plan (about 40-60 items).

## Deviations from the historical protocol
* Exact historical sample ids / IID-OOD split were **not** recovered: these are rebuilt splits (new seeded selection).
* The official 8-test-set ranking cannot be reproduced (DTA test and all masked-test gold are unavailable); the IID
  metric uses the 5 labelled newspaper test sets, the OOD metric the DTA dev split and Overproof.
* The OOD is a cross-corpus shift inside HIPE-OCRepair-Bench, not a separate benchmark.
* `impresso-nzz` (very long Fraktur pages, 17 test units) is not used for evaluation.
