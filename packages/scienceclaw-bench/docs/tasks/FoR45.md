# FoR45 Indigenous studies — AmericasNLP 2026 cultural image captioning (mean sentence chrF++, higher is better)

Adapter: `scienceclaw/bench/tasks/for45_americasnlp.py` (`Adapter`; shared helpers in `_adapter_utils_for43_45_47_48_50.py`).
Status: **available** (210 of 250 labelled dev rows have a local image; Pillow + sacrebleu installed).

## Dataset
| field | value |
|---|---|
| name / version | AmericasNLP 2026 Shared Task: Cultural Image Captioning for Indigenous Languages, commit `7ff73013eca06cc54ac1bdaa4e6edbd4154d08f0` |
| URL | https://github.com/AmericasNLP/americasnlp2026 |
| license | CC BY-NC 4.0 |
| local path | annotations `<DATA_ROOT>/for45-americasnlp-2026/reconstructed_v1/upstream/data/dev/<lang>/<lang>.jsonl` (fallback `upstream/data/dev/...`); images in `.../dev/<lang>/images/` of either tree (resolved by file name) |
| official scorer | `baseline/eval.py` of the pinned commit (copy in `for45-americasnlp-2026/official_scorer/eval.py`): `CHRF(word_order=2).sentence_score(generated, [target])`, mean over captions |
| what is missing | official **test** captions are not public (981 test rows without `target_caption`); 40 dev images were not downloaded (caption-only rows); pilot images (Wixárika, 20 rows) not downloaded |

## Items, pools, splits
* **Item** = one image (64×64 RGB thumbnail, centre square crop, bicubic) + language / ISO 639-3 / culture;
  label = target caption in that language. Item id `americasnlp/<iso>/<id>`.
* **Proxy OOD** (no second Indigenous captioning dataset locally): **held-out language family**. IID = Bribri (bzd,
  Chibchan), Guaraní (grn, Tupian), Yucatec Maya (yua, Mayan); OOD = Central/Orizaba Nahuatl (nlv, the task's
  surprise language) and Wixárika (hch) — both Uto-Aztecan, never used in src/val/id. `ood_kind = "proxy_within_dataset"`.
* Per language the image rows are partitioned once (`partition_seed = 20260928`) by image content (sha256 of the
  file; duplicates stay together): IID languages src/val/id/train = 19/6/11/6 of 42 rows; OOD languages ood/train =
  40/60 % (17/26 nlv, 16/25 hch). Caption-only rows (8 per IID language, 7 nlv, 9 hch) and the 20 Wixárika pilot
  captions are visible training rows only.
* **Episode size = min(items_per_episode, 8)**: 250 labelled captions cannot supply 16-item episodes for 7 src + 2 val +
  4 id + 4 ood episodes with disjoint visible data. ID and OOD hold 32 items each (historical: 64). `lineage` records
  `items_requested` and `items_per_episode_capped`. Episodes are balanced over the pool's languages (IID 3/3/2
  rotating, OOD 4/4).

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | the pool's training rows (IID 36: 6 image + 8 caption-only per language minus the dev slice; OOD 81 incl. pilot): id, language, iso_lang, culture, caption, has_image + `train_images` (n,64,64,3) uint8 (zeros when no image) |
| `load_dev_inputs` | 6 training-partition images (2 per IID language / 3 per OOD language) with captions withheld |
| `score_dev(dev_captions)` | mean sentence chrF++ of the dev captions, of the reference captions, and per item (visible dev signal; `_dev_evaluate` is None) |
| `load_eval_inputs` | evaluation items (language metadata, no ids, no captions) + `images` (k,64,64,3) |
| `image_features(images)` | CPU descriptors: 4×4×4 RGB histogram, 8×8 grey thumbnail, HSV / edge statistics (`kind` config) |
| `image_knn_reference` | same-language cosine nearest neighbour over visible training images, returning its caption (caption-only rows are skipped; falls back to the visible medoid) |

## Required output
`y`: list of k (=8) strings; `y[i]` = caption in the language of `items[i]`.

## Metric, reference, acceptance (D_V)
* **Mean sentence chrF++** = mean over items of `sacrebleu.metrics.CHRF(word_order=2).sentence_score(y[i].strip(),
  [target]).score` (char 6-grams, word 2-grams, β = 2, case-sensitive; 0–100) — identical to the official `eval.py`.
  Pooled metric: mean over all items of the given episodes.
* **Reference** = per-language *medoid caption* of the episode's visible training captions (highest mean sentence
  chrF++ against the other captions of that language), predicted for every item of that language. Default plan pooled
  reference: src 18.6, val 20.0, id 18.8, ood 22.0 chrF++ (per episode 17.0–24.7). Calibration (sanity check, not a
  research result): nearest-neighbour caption retrieval with `image_features` scores 0–10 points *below* the reference;
  concatenating the three most central captions per language scores −3.8…+4.5 around it.
* **Acceptance**: `score >= reference + 1.0` chrF++ point (`margin` configurable).
* `details`: `reference`, `norm_score` (score/reference clipped to [0, 10]), `per_language`, `pooled_payload`
  (`chrf_pred`, `chrf_ref`, `chrf_knn_ref`, `languages`), and trusted-side visual diagnostics. The
  `image_knn_reference_mean_chrf_pp` diagnostic uses only visible image-caption pairs and hidden targets inside the
  trusted scorer; it does not replace the medoid reference or affect acceptance. It uses the lightweight CPU
  descriptors exposed by `image_features`, not a downloaded CLIP checkpoint.
* **Hard constraints**: `output_length`, `captions_valid` (non-empty strings ≤ 1000 characters).

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 600, max_llm_items 8 × max(items, dev)`.

## Known deviations / limitations
* Exact historical sample ids were **not** recovered: rebuilt splits from the public dev partition only.
* ID/OOD have 32 items each (8-item episodes) instead of 64; the OOD is a held-out-language proxy.
* The policy/executor LLM of this code base is text-only: image content is available only as pixels and simple
  descriptors. The image-kNN tool is an interpretable reference route; a vision-capable executor or CLIP checkpoint
  would require a separately audited dependency and is not silently substituted here.
* The data team's `reconstructed_v1` 80/40/80 image sample (all five languages, no OOD) is not used; this adapter
  partitions the same population differently to obtain a language-level OOD.
