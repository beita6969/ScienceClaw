---
name: scienceclaw-benchmark-for32
description: "Use when segmenting the hippocampus into anterior and posterior parts in small 3-D T1-weighted brain MRI crops (Medical Segmentation Decathlon Task04) from a set of labelled volumes, delivering one integer label volume per image scored by Dice. Corresponds to FoR32 of the companion ScienceClaw-Eval benchmark."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Hippocampus segmentation in 3-D MRI crops (MSD Task04)

**Task.** Input: 3-D T1-weighted MRI crops around one hippocampus (about 1 mm voxels, shapes roughly 31-43 x 40-59 x 24-47) and labelled volumes (0 background, 1 anterior, 2 posterior). Deliverable: a list with one uint8 label volume per image, exactly the image's shape, values in {0, 1, 2}.

**Quality.** Dice per case and label, `2|P and G| / (|P| + |G|)` (1 if both empty), averaged over labels 1 and 2, then over cases; higher is better. Locally: `scilib.hippo.dsc`, `case_dsc`, `mean_dsc`.

**Library** (check `scienceclaw_tools(operation=show|status)`):
- `scilib.hippo` (CPU): `fit_predict(train_images, train_labels, eval_images, train_ids=...)`: location atlas, patch-based multi-atlas `label_fusion`, LightGBM voxel classifier (`HippocampusSegmenter`), `postprocess`. Also `LocationAtlas`, `normalize_intensity`, `subject_groups`, subject-grouped `cross_validate`.
- `scilib.hippo_unet.fit_predict`: 3-D U-Net ensemble trained from scratch (GPU or remote worker; guard with `hippo_unet.available()`).
- Operators: `hippocampus_segmentation_atlas_fusion_lgbm`, `multi_atlas_label_fusion_patch`, `location_prior_atlas_from_masks`, `hippocampus_segmentation_unet3d`, `segmentation_dice_two_labels`.
- No pretrained weights are needed. `hippo_innereeye.predict_binary` (asset `innereye_hippocampus`, ADNI whole-head scans) gives only a binary left/right union and cannot give anterior/posterior; never invent a class mapping. Asset `mass_base` has no wrapper and transferred worse than a location atlas on these crops.

**Routes.** Baseline: location-only atlas (class frequencies of the labelled masks on a normalised grid); crops are centred, so it is already strong. Better: label fusion, `hippo.fit_predict`, or the U-Net ensemble with a GPU. Compare by subject-grouped cross-validation.

**Rules.**
- Volumes `2k-1` and `2k` are the left/right crops of one scan (subject = (id + 1) // 2): keep them in one fold and pass `train_ids` so fusion skips same-subject atlases.
- Intensity encodings differ between volumes (float around 1e3, uint8, float around 1e5): normalise per volume (`normalize_intensity`, "robust" or "rank").
- Keep shapes and dtypes exactly; never read held-out labels.
