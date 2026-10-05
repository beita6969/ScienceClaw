---
name: scienceclaw-benchmark-for43
description: "Use when a task gives noisy OCR text of historical newspapers, periodicals or books (several languages, with metadata and a set of labelled OCR/ground-truth pairs) and asks for corrected transcriptions, one string per unit, scored by character match error rate against the gold text (HIPE-OCRepair OCR post-correction; corresponds to FoR43 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# OCR post-correction of historical text

**Task.** Input: text units (raw OCR plus language, test set / corpus, document type, date) and labelled pairs (`ocr_text`, `gt_text`) from the same collections. Deliverable: a list of strings, one corrected transcription per unit, same order, no truncation or run-away generation.

**Quality.** Weighted cMER-micro, lower is better. After normalisation (lower-case, ß->ss, ligature mappings, soft hyphen + newline removed, non-word characters -> space, whitespace collapsed), per test set `(S+D+I) / (H+S+D+I)` of the character alignment summed over units; test sets are averaged with weight 1/3 for `dta19-l0/l1/l2`, 1 otherwise. Baseline: the OCR text unchanged.

**Tools (`from scilib import ocrfix`, no pretrained weights).**
- Metric: `norm`, `char_counts`, `cmer`, `edit_rate`, `score(test_sets, golds, hypotheses)` (operator `ocr_character_error_rate`).
- `noise_profile(train)` per test set (operator `ocr_noise_profile`); `join_line_hyphens`.
- `plan(units, train, max_chars, skip_below, n_examples)` chunks units and builds one prompt per chunk with aligned OCR/gold excerpts of the same test set (operator `ocr_correction_plan`); `merge(plan, outputs)` reassembles with a length band, edit-rate cap and word-level hunk filter (operator `ocr_guarded_merge`).
- Workflow: code node (`plan`) -> `llm` node (items = the plan's prompts, template `{prompt}`, `config.parse = "text"`) -> code node (`merge`).

**Routes.** Baseline: copy OCR. Main: chunked LLM correction with the guarded merge; compare chunk size, number of examples and merge caps with `ocrfix.score` on labelled pairs not shown in the prompt.

**Rules.**
- An unconstrained "improve the text" prompt rewrites correct passages and can raise the error: keep names, numbers, punctuation, line breaks and historical spelling; reply with the text only.
- When a reply drifts in length or edit rate, keep the OCR text; skip test sets that are already near error-free.
- Apply the end-of-line hyphen convention only where the training pairs show a gain.
- Tune on labelled pairs only; do not pool statistics across test sets (language, era and noise differ). Replies are token-limited and an `llm` node takes a bounded number of items, so chunk long units.
