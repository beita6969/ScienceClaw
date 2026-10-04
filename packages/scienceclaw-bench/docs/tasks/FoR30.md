# FoR30 Agricultural, veterinary and food sciences — PhenoBench hierarchical panoptic segmentation (PQ+, higher is better)

Adapter: `scienceclaw/bench/tasks/for30_phenobench.py` (`Adapter`; shared helpers in `_life_health_common.py`, delivery
resolution in `_delivery_roles.py`).
Status: **available** on the data team's `reconstructed_v2` delivery (design `FoR30.v2`); tests `tests/test_task_for30.py`.

## Dataset
| field | value |
|---|---|
| name / version | PhenoBench v1.1.0 (UAV RGB images of sugar-beet fields, Univ. Bonn) |
| URL | https://www.phenobench.org/dataset.html — archive https://www.phenobench.org/data/PhenoBench-v110.zip (7,630,658,167 B, published md5 `5168bba762053725890478432cdbdb1d`) |
| retrieval | data team extracted only the selected ZIP members via HTTP Range; every PNG CRC32-verified, SHA-256 recorded; whole-archive checksum *not* verified (archive not downloaded) |
| license | CC BY-SA 4.0 (official devkit README FAQ; dataset page) |
| delivery | catalog `configs/data-delivery-v1/catalog.json` entry `FoR30` -> `data_root = .../datasets/for30-phenobench/reconstructed_v2` (`resolve_delivery` follows the catalog; a newer `reconstructed_v*` directory that the catalog does not bind is only listed in `lineage.newer_reconstructed_dirs_on_disk`) |
| role files | `configs/episode-designs/episodes/FoR30-v2/{source,val,id,ood}.jsonl` (sha256 `7227bf31…`, `718c96a9…`, `3c2e08f4…`, `654f30bf…`, checked against the catalog on every load); one row per image: `unit_id` = `<official split>/<file name>`, `lineage_group` = capture id (P-number), `crop_lineage_ids`, six files (`images`, `semantics`, `plant_instances`, `leaf_instances`, `plant_visibility`, `leaf_visibility`) with path + sha256, `input_to_agent = ["images"]`, `evaluator_only` = all labels |
| files | the annotation PNGs are under `reconstructed_v2/candidate_labels/PhenoBench/...`; 80 of the 1,152 file paths in the role rows still point into `reconstructed_v1/data/PhenoBench/...` (the paths are used as delivered; every file is sha256-verified against its row when a derived array is built, `Adapter.verify_files()` checks all 1,152 in ~0.5 s) |
| official evaluator | PRBonn/phenobench @ `0edc128ef7f67c8c6577554c7d1a2e382e2ea81f` (pinned source in the delivery, installed by the data team in `environments/vision` with torch 2.6 + torchmetrics 0.10.3) |

Data-team policy carried into the adapter: the agent sees an opaque image handle + RGB only; the fit uses only the source
role; every unselected image of the official population is quarantined and unreachable here.

## Items, pools, splits
* **Item** = one image with its official annotations. Default resolution 512x512: the official 1024x1024 image is
  downsampled by 2 (RGB: 2x2 area mean; labels and visibility: nearest neighbour, i.e. every second pixel).
  `Adapter(scale=1)` gives native resolution (tool outputs then ~4x larger). Item id = the row's `unit_id`.
* **Lineage** = the image, its capture id (P-number) and the persistent crop-lineage ids. The pools are exactly the four
  frozen role files, and `check_roles` (run whenever the roles are loaded; `Adapter.verify_disjoint()` repeats it)
  re-verifies that no image, no capture id and no crop id occurs in two roles and that the OOD acquisition date differs
  from every IID date.
* **IID** = capture date 05-15 (official train and val). **OOD** = date shift 05-15 -> 06-05 (later growth stage) *plus*
  held-out crop lineages; `lineage.ood_kind = "proxy_within_dataset"` — no second panoptic dataset is available locally.

| pool | images | captures (images each) | dates | official split |
|---|---|---|---|---|
| src | 32 | 3 (13 / 11 / 8) | 05-15 | train |
| val | 32 | 2 (17 / 15) | 05-15 | train |
| id | 64 | 5 (36 / 18 / 8 / 1 / 1) | 05-15 | val |
| ood | 64 | 3 (28 / 27 / 9) | 06-05 | train + val |

* **Held-out episodes** (val / id / ood): `items_per_episode` images (16) drawn from the role by a seeded permutation,
  prefix-stable in `n`, never reused across episodes. When `n * items_per_episode` exceeds the pool the episodes shrink
  to `pool // n` images (id with n = 5 -> 12 images) and record `lineage.items_requested = 16`. At the default plan
  (val 2, id 4, ood 4) every episode has the full 16 images (v1: OOD 14).
* **src / rep episodes** hold the images of *one* source capture, cycled over a seeded capture order (13 / 8 / 11 images
  at the default plan); they repeat images across episodes (7 x 16 > 32; SplitPlan warns
  `FoR30/src: 32 item ids reused ...`, expected) and are flagged with `lineage.reused_items`.

## Visible data and tools (D_E)
The labelled visible data are episode-relative and come from the **source role only**: `load_train` returns the labelled
images of the source captures that hold none of the episode's images (held-out episodes: all 3 source captures = 24
train + 8 dev images; a src episode never sees labels of its own capture). Val / ID / OOD annotations are never visible.
Image names are opaque handles (`ph-<sha256[:8]>`); the objective contains neither dates nor capture ids.

| tool | returns |
|---|---|
| `load_train` | the labelled source images: `images` uint8 (n,512,512,3), `semantics` (0 soil, 1 crop, 2 weed, 3 partial crop, 4 partial weed), `plant_instances`, `leaf_instances`, `plant_visibility` / `leaf_visibility` (float 0..1), `names` |
| `load_eval_inputs` | the episode's RGB images (k,512,512,3) and `names`, in output order — no annotations |
| `load_dev_inputs` | RGB images of the smallest remaining source capture (8 for held-out episodes), annotations withheld |
| `score_dev(prediction)` | official hierarchical metrics on the dev images: `pq_plus`, `iou_soil`, `iou_weed`, `pq_crop`, `pq_leaf` (%) — the visible dev signal (`Episode._dev_evaluate` is None) |

`Adapter(expose_source_labels=False)` gives the strictest reading of the data team's "RGB only" agent view: the episode then
has only `load_eval_inputs`. (Default `True`, because the design fixes the fit to the source role and a supervised
segmentation task without any labelled data is not meaningful; `lineage.source_labels_exposed` records the choice.)

## Domain toolkit (`scilib.phenoseg`)
The objective ends with the interface text of `scilib/phenoseg.py` (numpy / scipy / lightgbm; importable from code nodes,
reads no dataset path and no hidden data; it works on the arrays returned by `load_*`). It offers 35 per-pixel colour /
texture / vegetation-density features, a class-balanced LightGBM pixel classifier (`fit_pixel_classifier`, `predict_probs`,
`oof_probs` with capture-blocked folds, a `budget_s` time knob), `growth_scale` (crop-mask distance-transform size relative
to the labelled plant size), `panoptic_from_probs` / `split_instances` (crop mask, marker watershed for plants and leaves,
plants as connected components when the plants are large), `pq_plus` (the objective's metric, equal to the adapter's, so
labelled train images can be scored without `score_dev`) and `fit_predict`, which returns prediction dicts in the layout of
`y`. Fit on 24 images ~30 s, prediction ~2-3 s per image (16 images + fit ~70 s on an idle core pair; node limit 900 s).
`tests/test_scilib_phenoseg.py` checks the metric against the adapter on synthetic maps and runs the pipeline on synthetic scenes.

## Required output
`y = {"semantics", "plant_instances", "leaf_instances"}`, each an integer array (k, 512, 512) in item order; semantics
0/1/2 (3/4 accepted and read as partial crop/weed exactly like the official scorer), instance ids >= 0 (0 = none).

## Metric, reference, acceptance (D_V)
* **PQ+** of the official `--task hierarchical`: `PQ+ = (IoU_soil + IoU_weed + PQ_crop + PQ_leaf) / 4` (percent).
  * IoU: 3-class confusion matrix accumulated over the episode's images after mapping 3->1, 4->2 in prediction and GT
    (torchmetrics `MulticlassJaccardIndex(num_classes=3, average=None)`; a class absent from both gives 0 — verified
    against torchmetrics 0.10.3).
  * PQ_crop: official `evaluate_plant_instances` — per image, GT plants with visibility <= 0.5 are removed from the GT
    semantics and predicted plants lying > 50 % inside such a partial GT plant are removed from the prediction
    (`filter_partial_masks`); per class PQ = sum IoU / (TP + FP/2 + FN/2) with greedy IoU > 0.5 matching in ascending
    predicted-id order; PQ_crop = mean over images that contain GT crops (raw semantics, class 1).
  * PQ_leaf: official `evaluate_leaf_instances` (semantics = instance > 0, leaf visibility filter).
  * The official scorer is torch code; the adapter uses an exact numpy port (unchanged from v1). **Validation**: on 6 images
    (3 IID + 3 OOD) at 512x512 with three prediction sets (ExG reference; a perturbed ground truth with relabelled/removed
    plants, merged/shifted leaves and partial labels; the oracle) the pinned official scorer printed IoU_soil / IoU_weed /
    PQ_crop / PQ_leaf / PQ+ = 98.57/0.0/40.95/3.68/35.80, 99.91/52.9/83.8/34.14/67.69 and 100/100/100/100/100; the port
    gave identical values to all printed decimals. `tests/test_task_for30.py::test_matches_official_scorer` repeats a
    comparison on v2 images (opt-in: `SCIENCECLAW_OFFICIAL_SCORERS=1`). Components are not rounded to 2 decimals as in the
    official printout.
  * The primary value is the **cohort PQ+** of the episode's images (confusion matrices and PQ sums over all images).
    The data team's per-image convention ("scorer reset for each image", mean of per-image PQ+) is reported as
    `metrics.mean_image_pq_plus`; it is lower than the cohort value even for the oracle (98.4) because an absent class
    scores IoU 0 in a single-image scorer.
  * Pooled metric: sums of confusion matrices and of per-image PQ values over all images of the given episodes.
* **Reference** (deterministic, same episode): excess-green index 2g-r-b on chromatic coordinates, Otsu threshold per
  image, 8-connected components >= 16 px (64 px at native scale) = plant instances = leaf instances, all vegetation
  labelled crop. Reference PQ+ at the default plan (bench seed 20260928; per episode; measured on v2): src 23.1 / 22.9 /
  38.1 / 23.1 / 22.9 / 38.1 / 23.1 (mean 27.3), val 19.9 / 9.8, id 24.6 / 26.5 / 20.9 / 29.2 (mean 25.3), ood 33.2 / 30.1 /
  31.9 / 30.0 (mean 31.3).
* **Acceptance**: `PQ+ > PQ+_ref + 2.0` points. The reference itself is never accepted; the oracle scores 100 (accepted).
* `details`: `reference`, `norm_score` (PQ+/PQ+_ref clipped to [0,10]), `pooled_payload` (`confusion`,
  `plant_pq_sum/count` per class, `leaf_pq_sum/count`, `items`), `n_components`.
* Invalid outputs: `primary=None`, `norm_score=0`; the pooled payload is computed for an all-soil prediction.
* **Hard constraints**: `output_structure` (dict with the 3 keys, shape (k,512,512)), `label_values` (finite integers,
  semantics in {0..4}, ids in [0, 2^31)), `declared_unit` ("1").

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 3600, max_node_s 900, max_llm_items 4 x items`. Tool outputs:
`load_train` (24 images) ~120 MB of arrays, evaluation images 12 MB. Evaluation of one episode takes ~0.5-1.5 s (reference
ExG, 16 images ~1 s); the whole default plan (val + id + ood reference) ~ 10 s.

## Caches
`<repo>/cache/tasks/FoR30/scale2/*.npz` (decoded, downsampled arrays; entries carry a fingerprint of the role row hashes
+ `CACHE_VERSION`; an unreadable, v1-style or mismatching entry is rebuilt from the sha256-verified PNGs, files are
written atomically). Building all 192 images takes ~12 s.

## Changes from v1 (`reconstructed_v1`)
* Pools: v1 rebuilt them from a 192-image seeded sample (id 64, val 33, train 16, dev 4, src 19, ood 56 images, 14 per OOD
  episode); v2 takes the four frozen role files (src 32, val 32, id 64, ood 64 images) and the capture/crop lineage from
  the rows. v1's fixed `train` / `dev` pools are gone; the labelled visible data are now drawn from the source role,
  episode-relatively, and are disjoint from the episode in image *and* capture.
* OOD = date shift + crop-lineage isolation (v1: date only); 64 distinct images, full 16-image episodes.
* Item ids changed from `<split>/<file>` of the raw folders to the row `unit_id`; agent-visible names are opaque handles.
* Role files (catalog sha256), PNGs (row sha256) and the newest-delivery lookup are verified, and `available()` reports
  the reason for any missing / mismatching file.
* Metric, reference, acceptance margin and the numpy port are unchanged.

## Deviations from the historical protocol and open issues
* Exact historical sample ids / split rules were **not recovered** (`historical_ids_recovered = false` in the lineage);
  full independence of the upstream population components is not claimed.
* OOD is a within-dataset date / growth-stage proxy, not a different dataset.
* Default resolution is 512x512 (downsampled); PQ+ at native resolution can differ.
* The design lists all labels as evaluator-only; the default adapter nevertheless exposes the *source-role* labels as
  training data (see above; `expose_source_labels=False` for RGB only). Val / ID / OOD labels are never exposed.
* Only 32 source images in 3 captures: src episodes repeat images (each holds one capture) and the visible training set is
  at most 24 images. **Data-team request**: a larger source role (>= 100 images) would remove the repetition.
* 80 role-row file paths still point into `reconstructed_v1/data/...`; if the data team removes that directory the rows
  must be re-issued (sha256 checks will report it).

## Pretrained PhenoBench Mask2Former (`scilib/phenoseg_m2f.py`, added 2026-10-01)
`scilib.describe_extra("phenoseg_m2f")` is appended to the objective only when `phenoseg_m2f.available()` (local torch + the two converted checkpoints, or the GPU bridge). The tool runs the two Mask2Former (ResNet-50) checkpoints released by the PhenoBench authors (PRBonn; `plants` = panoptic, `leaves` = leaf instances; weights `https://www.ipb.uni-bonn.de/html/projects/phenobench/{panoptic_segmentation,leaf_instance_segmentation}/Mask2former/model.pth`, 176,462,331 B each; sha256 `8c24e0a2…6939d1` (plants) and `70bb98c2…352814` (leaves)) converted to Hugging Face format with `reports/below_peers_rootcause/scripts/f30_m2f/convert_phenobench_m2f.py`. Input normalisation is the detectron2 default that the PRBonn config leaves active (RGB minus 103.53 / 116.28 / 123.675, no division, no /255) — the HF default ImageNet normalisation makes the model predict "no object" everywhere.
* Conversion check [measured]: on three official-val images the converted models agree with the released val predictions to 99.8-99.96 % of the semantic pixels (plant instances 6/6, 9-11/10-12, 11/11, 48/48, 50/52 leaves).
* Result (pre-declared defaults: thresholds 0.8 / 0.8, nothing fitted; id evaluated once with the adapter's own `evaluate`, 512 px): **id (official val = not in the checkpoints' training data) pooled PQ+ 64.96**, episodes 62.01 / 68.23 / 62.82 / 66.85 (references 20.9-29.2); weed IoU 42-54, PQ_leaf 44-54. Before: agent mean 38.7; SAM 2.1 + DINOv2 selector 56.45 (one episode). Published: challenge top-3 81-83 at full resolution, HAPT 65.3.
* **Contamination**: the checkpoints were trained on the official PhenoBench `train` partition (1,407 images). Pools src / val (official train) are therefore seen by the model; the ood pool (06-05 date, 28+27+9 images from train+val) is mostly seen — an ood run gave PQ+ 76.9-79.6 and must not be read as evidence. Only the id pool is clean. Training used 1,407 labelled images versus 24 in an episode.
* Open: the end-to-end agent run has not been done (the agent has to call `phenoseg_m2f.predict_panoptic`); whether to add a pointer/example in the `phenoseg` module docstring (the experience with FoR33/35/40/48: the agent only uses an option it sees in a call example or when its first attempt misses the dev threshold).
