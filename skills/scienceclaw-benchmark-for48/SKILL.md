---
name: scienceclaw-benchmark-for48
description: "Use when a task asks to locate the clauses of contracts or NDAs (split into candidate spans) that support or contradict fixed natural-language hypotheses, ContractNLI-style: inputs are contract spans, hypotheses and annotated training contracts; the deliverable is, per (contract, hypothesis) pair, an Entailment or Contradiction label plus one relevance score per span, judged by mean average precision. Corresponds to FoR48 of the companion ScienceClaw-Eval benchmark."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Contract evidence identification (ContractNLI)

## Task
- Inputs: contracts as ordered candidate spans (`doc_id`, `span_texts`, or text plus character spans); a fixed set of hypotheses (key, text, short description); training contracts with annotations per (contract, hypothesis): label Entailment / Contradiction / NotMentioned and evidence-span indices.
- Deliverable: one dict per requested pair, in the given order: `{"label": "Entailment" | "Contradiction", "span_scores": [float in [0, 1]] * n_spans}`, scores in span order. The label is binary (requested pairs are ones the contract mentions); never output NotMentioned.

## Quality
Mean over pairs of the average precision of the span scores against the gold evidence spans (mAP, higher is better). Secondary: precision at 80 % recall and label accuracy. A baseline: lexical overlap of the hypothesis' content words with the span, plus the per-hypothesis majority label.

## Tools
- `scilib.contracts.fit_predict(train_documents, train_annotations, hypotheses, items, documents, seed=0)`: per-hypothesis TF-IDF logistic regression over spans (neighbour context, contract-grouped out-of-fold scores), then a LightGBM re-ranker on similarity (cosine, BM25, overlap), position and nearest-training-evidence features; a label model reads the top spans. Typed operator `contract_evidence_identification`.
- `scilib.contracts.cross_validate(...)`: contract-grouped k-fold estimate (`map`, `p_at_r80`, `nli_binary_accuracy`, `per_hypothesis_map`); `annotation_pairs`, `evaluate_predictions`, `average_precision`, `precision_at_recall` for metrics (operator `contract_evidence_metrics`).
- Optional `ContractModel(plm=True)` (or a subset of `PLM_GROUPS`: rerank, nli, cos, emb_lr, emb_knn) adds pretrained features through `scilib.textenc`: BGE-large sentence embeddings, BGE-reranker-large relevance, DeBERTa-v3-large NLI (weight ids `bge_large_en`, `bge_reranker_large`, `nli_deberta_v3_large`). Check `scilib.textenc.available()` or `scienceclaw_tools(operation=weights)` first; it is costly (every span against every hypothesis; GPU or remote worker).

## Rules
- Never fit or tune on the pairs being predicted; split validation folds by contract, not by pair or span.
- `span_scores` must be finite, in [0, 1], one per span; `label` only Entailment or Contradiction.
- Check `per_hypothesis_map`: hypotheses with few training pairs fall back to a hypothesis-independent model and are weaker.
- The pretrained encoders were not trained on this task's labels, but their public pretraining text may overlap contract text; disclose that when they are used.
