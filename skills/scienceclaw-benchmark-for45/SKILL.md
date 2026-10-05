---
name: scienceclaw-benchmark-for45
description: "Use when a task gives small culturally specific images with language metadata (Indigenous languages of the Americas) and a few captioned training examples per language, and asks for one caption string in the item's own language per image, scored by mean sentence chrF++ against the reference captions (AmericasNLP 2026 cultural image captioning; corresponds to FoR45 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Image captioning for low-resource Indigenous languages

**Task.** Input: evaluation items (language, ISO 639-3 code, culture) with uint8 RGB images (e.g. 64x64x3), and visible training rows (`caption`, `iso_lang`, `has_image`, images; some rows are caption-only). Deliverable: a list of non-empty strings, one per item, in that item's language (`captions.fit_predict` caps strings at 1000 characters; follow what the task declares).

**Quality.** Mean sentence chrF++, higher is better: `sacrebleu CHRF(word_order=2)` (character 1-6-grams, word 1-2-grams, beta 2, case-sensitive, 0-100) against the reference caption. Baseline: the per-language medoid of the visible training captions.

**Tools.**
- `from scilib import captions`: `chrf`, `mean_chrf` (operator `caption_chrf_score`), `medoid` (`caption_medoid`), `consensus_caption` (`caption_consensus_string`), `mbr_caption` (`caption_mbr_clauses`), `by_language`, `fit_predict(train_rows, items, method="consensus"|"mbr"|"medoid")` (one string per language, from that language's captions only). Estimate a strategy on unseen captions with `loo_score` / `loo_compare` / `loo_lengths` (operator `caption_holdout_chrf_estimate`), setting `n_train` to the real training size.
- `from scilib import clip_retrieval`: `available()`, `provenance()`, `encode_images` (`image_embedding_clip`), `retrieve_captions(train_rows, train_images, eval_items, eval_images, k=1, selection="nearest"|"caption_medoid", fallback=...)` (`image_caption_retrieval_clip`): same-language cosine nearest-neighbour caption lookup; caption-only rows are skipped.
- Weights `open_clip_vit_b32` (OpenAI CLIP ViT-B/32 via `open_clip`; needs torch). Check with `scienceclaw_tools(operation=weights)` and `clip_retrieval.available()`; nothing downloads implicitly; verify the licence. Disclose its use: it is an external pretrained encoder that sees no captions or labels.

**Routes.** (1) Language-level caption: medoid, or a consensus/MBR string maximising mean chrF++ against the language's pool. (2) Image-conditioned retrieval, optionally `k>1` with `selection="caption_medoid"`. With few captions per language, retrieval can score below the medoid; choose by leave-one-out chrF++ on the training rows.

**Rules.**
- Never pool captions across languages. `retrieve_captions` returns `""` for a language without a captioned image, so pass `fallback=` (e.g. that language's medoid).
- Use no target captions. Consensus and MBR strings optimise n-gram overlap and are not grammatical; say so in the report.
- One string per item, input order. Do not claim image understanding from pixel statistics.
