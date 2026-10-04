# FoR32 Biomedical and clinical sciences — MSD Task04 Hippocampus (DSC, higher is better)

Adapter: `scienceclaw/bench/tasks/for32_msd_hippocampus.py` (`Adapter`; shared helpers in `_life_health_common.py`).
Status: **available**.

## Dataset
| field | value |
|---|---|
| name / version | Medical Segmentation Decathlon, Task04_Hippocampus (release 1.0, 04/05/2018; Vanderbilt University Medical Center) |
| URL | http://medicaldecathlon.com/ — archive https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar |
| sha256 (tar) | `282d808a3e84e5a52f090d9dd4c0b0057b94a6bd51ad41569aef5ff303287771` (data-team receipt) |
| license | CC-BY-SA 4.0 (`dataset.json`) |
| local path | `<DATA_ROOT>/for32-msd-hippocampus/data/Task04_Hippocampus/{imagesTr,labelsTr}/hippocampus_XXX.nii.gz` (260 labelled volumes; `imagesTs` 130 unlabelled, unused; macOS `._*` files ignored) |
| content | T1-weighted MRI crops around one hippocampus, 1 mm isotropic, shapes ~(31-43)x(40-59)x(24-47); labels 0 background, 1 anterior, 2 posterior |

## Items, pools, splits
* **Item** = one labelled volume (image + mask). Item id = `hippocampus_XXX`.
* **Lineage group** = subject: volumes `2k-1` and `2k` are the left/right crops of the same scan (verified on the data:
  74 % of odd-start neighbour pairs have near-identical intensity distributions and encodings vs 10 % of even-start
  pairs). All pools consist of whole subjects `S = (id + 1) // 2`.
* The volumes come with three intensity encodings: float32 with ~1e3 range (172 labelled), uint8 0-255 (55) and
  float32 with ~1e5 range (33). **IID** = the standard float ~1e3 encoding. **OOD** = the two non-standard encodings
  (different export / scaling pipelines), `lineage.ood_kind = "proxy_within_dataset"`,
  `lineage.ood_rule = "non-standard intensity encoding"`. No second hippocampus dataset is available locally.
* Pools (fixed by `pool_seed = 20260928`, independent of the SplitPlan's per-split seeds; seeded group order, first
  pool with room):

| pool | volumes | subjects | encodings |
|---|---|---|---|
| id | 64 | 38 | float ~1e3 |
| val | 32 | 22 | float ~1e3 |
| dev (inputs visible, masks withheld) | 8 | 5 | float ~1e3 |
| train (visible, labelled) | 28 | 19 | float ~1e3 |
| src | 40 | 26 | float ~1e3 (source episodes reuse volumes: 7 x 16 > 40) |
| ood | 64 | 43 | 43 uint8 + 21 float ~1e5 |

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | 28 labelled volumes: `images` (list of float32 3-D arrays, scanner units), `labels` (uint8 masks), `case_ids`, `spacing_mm` |
| `load_eval_inputs` | the 16 evaluation images (no masks), `case_ids`, `spacing_mm`, in output order |
| `load_dev_inputs` | 8 dev images (masks withheld) |
| `score_dev(predictions)` | mean / anterior / posterior DSC on the dev volumes (the visible dev signal; `Episode._dev_evaluate` is None) |

## Metric, reference, acceptance (D_V)
* Required output: list of 16 integer arrays, array i with exactly the shape of image i, values in {0, 1, 2}.
* **Metric**: Dice similarity coefficient per case and label, `DSC = 2|P∩G| / (|P| + |G|)` (1 if both empty), the
  primary MSD metric (Antonelli et al., Nat. Commun. 2022); primary = mean over cases of the mean over labels 1 and 2.
  Secondary (reported only): normalized surface Dice at 1 mm (`mean_nsd_1mm`; surface voxels counted, not surface-area
  weighted as in the DeepMind surface-distance library — an approximation). Pooled = mean over all cases.
* **Reference**: location-only probabilistic atlas — the 28 training masks are mapped to a normalized 48x64x48 grid,
  class frequencies averaged, argmax mapped back to each evaluation grid (ignores intensities). The crops are centred
  on the hippocampus, so this is already strong: pooled DSC src 0.717, val 0.696, id 0.701, ood 0.692 at the default
  plan (bench seed 20260928).
* **Acceptance**: `mean DSC > reference + 0.03`.
* `details`: `reference`, `norm_score` (primary/reference clipped to [0,10]), `pooled_payload` (`case_ids`, per-case
  `[DSC_anterior, DSC_posterior]`).
* Invalid outputs: `primary=None`, `norm_score=0`; pooled payload uses an empty prediction (DSC 0).
* **Hard constraints**: `output_structure` (16 arrays with the input shapes), `label_values` (finite integers in
  {0,1,2}), `declared_unit` ("1").
* **Calibration** (adapter sanity check, one seed): a voxel HistGradientBoosting classifier on per-volume normalized
  intensity, Gaussian-smoothed intensities, atlas prior and normalized coordinates scored 0.784/0.735/0.763/0.796 on id
  vs reference 0.714/0.666/0.689/0.734 and 0.781/0.752/0.771/0.770 on OOD vs 0.700/0.683/0.697/0.686 (accepted 8/8).
  With per-volume intensity normalization the encoding proxy is a **weak** shift (no ID->OOD drop for this method).

## Domain library (`scilib/hippo.py`)
The episode objective ends with `scilib.describe("hippo")`. The module gives code nodes the task metric (`dsc`,
`case_dsc`, `mean_dsc`), per-volume intensity normalization (robust / rank, independent of the encoding), a location
atlas built from the visible masks, patch-based multi-atlas label fusion with integer-translation alignment
(`label_fusion`), voxel features (`voxel_features`), a LightGBM voxel classifier restricted to the neighbourhood of the
atlas support (`HippocampusSegmenter`), post-processing (`postprocess`: probability smoothing, background weight, largest
components, hole filling), the entry point `fit_predict(train_images, train_labels, eval_images, train_ids, ...)` and a
subject-grouped `cross_validate`. It reads no files (everything comes from the arrays passed in) and needs about 25 s CPU
for the fit on the 28 visible volumes plus ~1.5 s per predicted volume. Library-level DSC of `fit_predict` trained on
the 28 visible volumes (default parameters, 16 volumes per pool): dev 0.84, val 0.85, src 0.85, ood 0.83 against the
reference 0.70 (location-only atlas). Real episodes through a hand-written node (src n=3: 0.843/0.839/0.846 vs reference
0.718/0.721/0.718; ood n=2: 0.844/0.802 vs 0.700/0.683). 27B agent A_0, 2 src episodes: fail (0.268, norm 0.37, 19 steps,
120k tokens) -> pass on both (0.846 / 0.838, norm 1.17 / 1.20). The tests are `tests/test_scilib_hippo.py` (synthetic volumes).

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 2400, max_node_s 600, max_llm_items 64`.

## Caches
`<repo>/cache/tasks/FoR32/index_<hash>.json` (subject, dtype, 99th percentile, shape, encoding per case).

## Deviations from the historical protocol and open issues
* Exact historical sample ids / split rules were **not recovered**; these are rebuilt splits. Historical protocol:
  64 IID + 64 OOD items; here id = 64 and ood = 64 volumes.
* OOD is a within-dataset intensity-encoding proxy (weak for normalized methods). A genuinely different hippocampus
  dataset (e.g. another site / scanner with hippocampal subfield or head/body labels) would be a better OOD source;
  the data team would need to fetch one with compatible labels.
* NSD is an approximation and not used for acceptance.
