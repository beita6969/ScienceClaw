---
name: scienceclaw-benchmark-for30
description: "Use when segmenting top-down field or UAV RGB images of crops (sugar beet, PhenoBench-style) into soil / crop / weed semantics plus crop-plant and crop-leaf instances (hierarchical panoptic segmentation), from a few labelled images, delivering per-image integer label maps scored by PQ+. Corresponds to FoR30 of the companion ScienceClaw-Eval benchmark."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Hierarchical panoptic segmentation of field images (PhenoBench)

**Task.** Input: uint8 RGB images (n, H, W, 3), usually plus a few labelled images (semantics, plant and leaf instances, optional visibility maps). Deliverable: a dict of integer arrays (n, H, W) at the input size: `semantics` (0 soil, 1 crop, 2 weed; labels 3/4 count as 1/2), `plant_instances`, `leaf_instances` (ids >= 1, 0 = none).

**Quality.** PQ+ in percent, higher is better: mean of IoU soil, IoU weed, PQ of crop plants and PQ of crop leaves (heavily occluded instances are filtered). `scilib.phenoseg.pq_plus(pred, gt)` scores labelled images locally.

**Library** (check `scienceclaw_tools(operation=show|weights)`):
- CPU: `scilib.phenoseg.fit_predict(train_images, train_semantics, targets)`: LightGBM pixel classifier plus watershed instances (parts: `fit_pixel_classifier`, `predict_probs`, `oof_probs`, `panoptic_from_probs`, `growth_scale`). Variants: `phenoseg_deep.fit_predict_deep` (DINOv2, assets `dinov2_base` / `dinov2_large`); `phenoseg_sam.fit_sam_selector` + `predict_sam_panoptic` (SAM 2.1, asset `sam2_hf`, GPU).
- Pretrained, guard with `available()`: `phenoseg_m2f.predict_panoptic` (Mask2Former, asset `phenobench_m2f`); `phenobench_weyler.predict_panoptic` (asset `phenobench_weyler`; crop plants and leaves only, no weeds). `phenoseg_hapt.predict_panoptic` (asset `hapt`) was trained on cauliflower: a cross-domain diagnostic only.
- Operators: `field_panoptic_segmentation_pixel_lgbm`, `field_panoptic_segmentation_m2f`, `field_panoptic_score_pq_plus`.

**Routes.** Baseline: excess-green threshold with connected components. Then pixel classifier, DINOv2 / SAM variants, and pretrained Mask2Former when staged. Compare on held-out labelled images with `pq_plus`.

**Rules.**
- Validate with folds blocked by capture or field (`oof_probs(..., groups=...)`), never by random pixels; never tune on the images being predicted.
- The released PhenoBench checkpoints were trained on PhenoBench itself (Mask2Former: official train partition, 1,407 images): images from it are not held-out evidence. Disclose the overlap.
- Instance sizes depend on growth stage and resolution; take them from labelled data (`model.plant_size`, `growth_scale`).
- Keep shapes, integer dtypes and label values exactly as above.
