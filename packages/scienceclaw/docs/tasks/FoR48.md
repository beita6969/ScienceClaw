# FoR48 Law and legal studies — ContractNLI evidence identification (mAP, higher is better)

Adapter: `scienceclaw/bench/tasks/for48_contractnli.py` (`Adapter`; shared helpers in `_adapter_utils_for43_45_47_48_50.py`).
Status: **available** (official train/dev/test JSON present).

## Dataset
| field | value |
|---|---|
| name / version | ContractNLI (Koreeda & Manning, Findings of EMNLP 2021), initial release 2021-10-05 |
| URL | https://stanfordnlp.github.io/contract-nli/ (`contract-nli.zip`, sha256 `e03fc77b…287b`) |
| license | CC BY 4.0 (see `upstream/contract-nli/LICENSE`, `TERMS`) |
| local path | `<DATA_ROOT>/for48-contractnli/upstream/contract-nli/{train,dev,test}.json` |
| official scorer | `contract_nli/evaluation.py` of stanfordnlp/contract-nli-bert @ `058c56fd` (Apache-2.0; copy in `reconstructed_v1/evaluator/upstream/`) |
| population | 607 NDAs (train 423 / dev 61 / test 123), 17 fixed hypotheses; document types search-pdf 375, sec-text 119, sec-html 113 |

## Items, pools, splits
* **Item** = one (contract, hypothesis) pair whose gold label is **Entailment or Contradiction** with a non-empty
  evidence set — exactly the pairs on which the official evidence metrics are defined (gold NotMentioned pairs are
  dropped by the official scorer). Input: contract text, candidate spans (char offsets + texts), hypothesis. Labels:
  evidence span set and NLI label. Item id `contractnli/<doc id>/<hypothesis key>`.
* **Proxy OOD** (no second contract-NLI dataset locally): **document source**. IID = web-search PDFs (`search-pdf`);
  OOD = SEC EDGAR filings (`sec-text`, `sec-html`). Evaluation documents come from the official dev + test partitions.
  `ood_kind = "proxy_within_dataset"`.
* IID documents are partitioned once **by contract** (`partition_seed = 20260928`) into src / val / id (50/15/35 % of
  pairs): 540 / 162 / 378 pairs; OOD pool 722 pairs. Splits are contract-disjoint.
* Episodes: 16 pairs drawn prefix-stably from a seeded permutation (typically 10–14 contracts per episode).

## Visible data and tools (D_E)
Official **train** contracts of the same source type (IID 261, OOD 162); per episode a seeded sample:

| tool | returns |
|---|---|
| `load_train` | 40 contracts (`train_documents`: doc_id, text, spans, span_texts) + `train_annotations` (all 17 hypotheses: label incl. NotMentioned, evidence span indices) |
| `load_hypotheses` | the 17 hypotheses (text + short description) |
| `load_dev_inputs` | ≤ 32 E/C pairs from 8 further train contracts (annotations withheld) + those contracts |
| `score_dev(dev_predictions)` | dev mAP, P@R80, binary NLI accuracy, reference mAP (visible dev signal; `_dev_evaluate` is None) |
| `load_eval_inputs` | the 16 evaluation pairs (doc_id, hypothesis_key, hypothesis, n_spans) + their contracts |

## Required output
`y`: list of 16 dicts `{"label": "Entailment" | "Contradiction", "span_scores": [float]*n_spans}` (one score in [0, 1]
per candidate span of the pair's contract, span order).

## Metric, reference, acceptance (D_V)
* **mAP** = mean over the episode's pairs of `sklearn.metrics.average_precision_score(gold_evidence, span_scores)` —
  the official `micro_label_macro_doc.span.map` aggregation (AP per document–hypothesis pair, averaged). Auxiliary:
  official precision@recall 80 % (same aggregation; the official function's source is executed in
  `tests/test_task_for48.py::test_precision_at_recall_matches_official` and agrees exactly), `macro_label_micro_doc` mAP
  (AP pooled per hypothesis, averaged over hypotheses) and the binary NLI accuracy (official `class_binary` accuracy on
  E/C pairs). Pooled metric: mean per-pair AP over all pairs of the given episodes.
* The historical paper's choice among the four official span-mAP aggregations was **not recovered**; the per-pair
  aggregation is used because it is defined per item and decomposes over episodes.
* **Reference** = lexical overlap (fraction of the hypothesis' content-word stems present in the span) as span scores
  + per-hypothesis majority label of the episode's training contracts. Default plan pooled reference mAP: src 0.344,
  val 0.421, id 0.409, ood 0.357. Calibration (sanity check, not a research result): a per-hypothesis TF-IDF logistic
  span classifier fit on `load_train` reaches mAP 0.77–0.88 on the id/ood episodes.
* **Acceptance**: `mAP >= reference + 0.10` (`margin` configurable).
* `details`: `reference`, `norm_score` (mAP/reference clipped to [0,10]), `pooled_payload` (`ap_pred`, `ap_ref`).
* **Hard constraints**: `output_length`, `span_scores_valid` (length = #spans of the contract, finite, in [0, 1]),
  `labels_valid` (Entailment / Contradiction).

## Domain toolkit (`scilib/contracts.py`)
The objective ends with `scilib.describe("contracts")` (module docstring = factual interface; no reference or margin is
stated). Code nodes import it from the repo root. Entry point `fit_predict(train_documents, train_annotations, hypotheses,
items, documents)` returns the deliverable format; `cross_validate` gives a contract-grouped estimate on the visible
training pairs; metric helpers are the adapter's definitions (per-pair AP, official P@R80, binary accuracy).
* Span scorer: per-hypothesis class-balanced logistic regression on word 1-2-gram TF-IDF of the span + half-weighted
  neighbouring spans, contract-grouped out-of-fold scores, then one LightGBM over all hypotheses on hypothesis similarity
  (TF-IDF cosine, BM25, overlap and their within-contract ranks), structure/position features, stage-1 scores of the span and
  its neighbours and kNN similarity to the hypothesis' training evidence; hypotheses without training pairs fall back to a
  hypothesis-independent LightGBM.
* Optional pretrained-text features, `ContractModel(plm=True)` (or a subset of `PLM_GROUPS`): `scilib.textenc` (BGE-large-en-v1.5 sentence
  embeddings, BGE-reranker-large relevance logits, DeBERTa-v3-large NLI log-probabilities; weights on the GPU host, reached through the remote
  file-spool bridge when the sandbox has no torch) feeds stage 2 with a relevance, an NLI and an embedding-cosine block per span, an
  out-of-fold embedding logistic regression per hypothesis and the cosine to the hypothesis' training evidence spans. The objective shows
  `scilib.describe_extra("textenc")` only when `scilib.textenc.available()`. Through the bridge a 40-contract fit takes about 160 s and a
  prediction for 16 pairs about 120 s (one idle GPU).
* Label: per-hypothesis logistic regression on the text of the top-scoring span(s) (n-grams + negation/condition cues);
  majority label when a hypothesis has (almost) one label in the training data.
* Measured with `scripts/dev/run_node_on_episodes.py --disc FoR48` (hand-written node calling `ContractModel().fit(...)`
  then `.predict(...)` for dev and eval; 20-36 s wall in the sandbox under load 70): src episodes 0.756 / 0.864 / 0.743 vs
  reference 0.175 / 0.313 / 0.386; ood episodes 0.850 / 0.893 / 0.837 vs reference 0.214 / 0.355 / 0.287 (all accepted).
  Train-holdout ablations (40 fit contracts, 30 held-out train contracts, 4 trials): stage 2 mAP 0.875, stage 1 alone 0.857,
  no neighbouring-span context 0.866, character n-grams / NotMentioned pairs / hypothesis index / C change it by <= 0.003.
  Binary label accuracy on gold evidence text 0.91-0.94 (per-hypothesis majority 0.88-0.89).

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 2400, max_node_s 600, max_llm_items 8 × max(items, dev) = 256`.

## Known deviations
* Exact historical sample ids were **not** recovered: rebuilt splits.
* OOD is a source-type proxy inside ContractNLI.
* NotMentioned pairs are not evaluated (as in the official evidence metric); the NLI label is binary auxiliary only.
