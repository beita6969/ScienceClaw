---
name: scienceclaw-benchmark-for47
description: "Use when a task gives a Universal Dependencies treebank (annotated training sentences with upos, head, deprel) and tokenised target sentences, and asks for a dependency tree (head and relation per word) for every target sentence, scored by labeled attachment score LAS (CoNLL 2018 UD parsing with gold tokenization, French GSD/PUD; corresponds to FoR47 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Universal Dependencies parsing with gold tokenization

**Task.** Input: annotated sentences `{"words": [{"form", "upos", "feats", "head", "deprel", ...}]}` (head 1-based, 0 = root) and target sentences with `id` and `form` only. Deliverable: one dict per target sentence, `{"head": [int] * n_words, "deprel": [str] * n_words}`, in word order. Each tree needs exactly one root, heads in range, no cycle, and relations from the 37 UD v2 relations (`udparse.UD_RELATIONS`).

**Quality.** LAS (CoNLL 2018 evaluation, gold tokenization), higher is better: share of words with correct head and universal relation (`deprel` before `:`), punctuation included, micro over all words; UAS is auxiliary. Weak baseline: right-branching chain with each word's most frequent relation.

**Tools.**
- `from scilib import udparse`: `UDParser(decode="eisner"|"cle", epochs, n_models, jackknife_folds, use_morph, seed).fit(train)` with `.tag`, `.parse`, `.arc_scores` (operator `ud_parser_train_parse`); `fit_predict(train, [targets_a, targets_b])` fits once, parses every list; `cross_validate`; `las_uas(gold, parses)` (`ud_attachment_scores`); `check_parse(parse, n_words)` (empty = valid); `eisner`, `chu_liu_edmonds` (`dependency_tree_decode`). Needs only target word forms; ~20-30 CPU s to fit 1000 sentences.
- Pretrained French: `fit_predict(train, targets, pretrained=True)` or `udparse_pretrained.parse_gold_tokens(sentences)` (`french_dependency_parse_stanza`; Stanza French-GSD on a CamemBERT-large encoder; `train` unused). Weights `camembert_large` and `stanza_fr`; needs torch, transformers, stanza or a GPU worker. Check `udparse_pretrained.available()` and `scienceclaw_tools(operation=weights)`; nothing downloads implicitly.

**Routes.** Trained parser on the visible treebank (any language), selected on held-out annotated sentences or `cross_validate`. For French the pretrained parser is an alternative; compare on held-out sentences outside its training data.

**Rules.**
- Disclose overlap: the pretrained models were trained on the UD French-GSD train file and selected on its dev file (other treebanks, e.g. French-PUD, unseen); do not judge them on those sentences. Their scheme follows a newer UD release than 2.2 in places.
- Use the given words: no re-tokenisation, no multiword splitting.
- Pretrained labels may carry subtypes (`nsubj:pass`); strip them if only universal relations are allowed.
- Run `check_parse` on every output; never train on target trees.
