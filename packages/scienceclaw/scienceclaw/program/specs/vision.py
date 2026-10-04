"""Typed operators for the image and volume segmentation / embedding tools of the library (field panoptic segmentation, hippocampus
MRI segmentation, CLIP image retrieval)."""
from __future__ import annotations

from typing import Any

from scienceclaw.program.specs._common import p

_IMG = ("n", "H", "W", 3)
_LAB = ("n", "H", "W")

_SEMANTICS = "semantic labels 0 soil, 1 crop, 2 weed, 3 partial crop, 4 partial weed"
_PANOPTIC = ("dict of integer arrays (n, H, W): 'semantics' (0 soil, 1 crop, 2 weed), 'plant_instances' (ids >= 1 per plant, 0 = none) "
             "and 'leaf_instances' (ids >= 1 per crop leaf, 0 = none)")
_FIELD_TAGS = ["agriculture", "field images", "segmentation", "panoptic", "plants", "leaves", "phenotyping", "images"]

# id, tool, description, inputs, outputs, body of run(), pre, post, applicability tags
SPECS: list[dict[str, Any]] = [
    dict(id="field_panoptic_segmentation_pixel_lgbm", tool="phenoseg.fit_predict",
         description="Hierarchical panoptic segmentation of top-down field images (soil / crop / weed semantics, crop-plant instances, crop-leaf instances) "
                     "with a LightGBM pixel classifier fitted on the labelled images (colour, vegetation-index, texture and vegetation-density features) "
                     "followed by distance-transform watershed splitting; CPU only, no pretrained weights.",
         inputs={"train_images": p("array", "uint8 RGB labelled field images", shape=("n_train", "H", "W", 3), dtype="int"),
                 "train_semantics": p("array", _SEMANTICS, shape=("n_train", "H", "W"), dtype="int"),
                 "images": p("array", "uint8 RGB field images to segment", shape=_IMG, dtype="int")},
         outputs={"panoptic": p("dict", _PANOPTIC)},
         code="from scilib import phenoseg\nreturn {'panoptic': phenoseg.fit_predict(inputs['train_images'], inputs['train_semantics'], inputs['images'])}",
         pre=[{"port": "train_images", "check": "nonempty"}, {"port": "images", "check": "nonempty"}], post=[],
         tags=_FIELD_TAGS + ["lightgbm", "pixel classifier", "watershed", "crop weed soil"]),
    dict(id="field_class_probabilities_pixel_lgbm", tool="phenoseg.predict_probs",
         description="Per-pixel probabilities of soil, crop and weed for top-down field images from a LightGBM classifier fitted on labelled images "
                     "(35 colour / vegetation-index / texture features per pixel).",
         inputs={"train_images": p("array", "uint8 RGB labelled field images", shape=("n_train", "H", "W", 3), dtype="int"),
                 "train_semantics": p("array", _SEMANTICS, shape=("n_train", "H", "W"), dtype="int"),
                 "images": p("array", "uint8 RGB field images to classify", shape=_IMG, dtype="int")},
         outputs={"probabilities": p("array", "class probabilities, channels (soil, crop, weed)", shape=("n", "H", "W", 3), dtype="float")},
         code="from scilib import phenoseg\nmodel = phenoseg.fit_pixel_classifier(inputs['train_images'], inputs['train_semantics'])\n"
              "return {'probabilities': phenoseg.predict_probs(model, inputs['images'])}",
         pre=[{"port": "train_images", "check": "nonempty"}, {"port": "images", "check": "nonempty"}],
         post=[{"port": "probabilities", "check": "range", "value": [0.0, 1.0]}],
         tags=_FIELD_TAGS + ["lightgbm", "pixel classifier", "class probabilities", "crop weed soil"]),
    dict(id="field_panoptic_from_class_probabilities", tool="phenoseg.panoptic_from_probs",
         description="Panoptic prediction (soil / crop / weed semantics, crop-plant and crop-leaf instances) from per-pixel class probabilities of field "
                     "images: vegetation and crop thresholds, plant-size-adaptive watershed or connected-component instances.",
         inputs={"probabilities": p("array", "class probabilities, channels (soil, crop, weed)", shape=("n", "H", "W", 3), dtype="float")},
         outputs={"panoptic": p("dict", _PANOPTIC)},
         code="from scilib import phenoseg\nreturn {'panoptic': phenoseg.panoptic_from_probs(inputs['probabilities'])}",
         pre=[{"port": "probabilities", "check": "finite"}], post=[],
         tags=_FIELD_TAGS + ["post-processing", "watershed", "instances"]),
    dict(id="field_panoptic_score_pq_plus", tool="phenoseg.pq_plus",
         description="PhenoBench hierarchical metric PQ+ (mean of IoU soil, IoU weed, PQ crop plants, PQ crop leaves, in percent) of a field panoptic "
                     "prediction against annotated semantics, plant instances and leaf instances, with the visibility filter of the benchmark.",
         inputs={"prediction": p("dict", "arrays (n, H, W): semantics, plant_instances, leaf_instances"),
                 "ground_truth": p("dict", "arrays (n, H, W): semantics, plant_instances, leaf_instances and optionally plant_visibility / leaf_visibility (0..1)")},
         outputs={"scores": p("dict", "pq_plus, iou_soil, iou_crop, iou_weed, pq_crop, pq_leaf in percent (None when the class is absent)")},
         code="from scilib import phenoseg\nreturn {'scores': phenoseg.pq_plus(inputs['prediction'], inputs['ground_truth'])}",
         pre=[{"port": "prediction", "check": "type", "value": "dict"}, {"port": "ground_truth", "check": "type", "value": "dict"}], post=[],
         tags=["agriculture", "segmentation", "panoptic", "metric", "evaluation", "PQ", "IoU", "phenobench"]),
    dict(id="binary_mask_instance_split_watershed", tool="phenoseg.split_instances",
         description="Split a binary mask into instances (touching plants, leaves, cells): Gaussian-smoothed distance transform, local maxima as markers, "
                     "marker watershed, removal of small fragments.",
         inputs={"mask": p("array", "boolean foreground mask", shape=("H", "W"), dtype="int"),
                 "sigma": p("number", "Gaussian smoothing of the distance transform (0 = none)", unit="px"),
                 "min_dist": p("number", "minimum distance between two instance centres", unit="px"),
                 "min_area": p("number", "instances smaller than this are removed", unit="px")},
         outputs={"labels": p("array", "instance ids, 0 = background", shape=("H", "W"), dtype="int")},
         code="from scilib import phenoseg\nreturn {'labels': phenoseg.split_instances(inputs['mask'], float(inputs['sigma']), float(inputs['min_dist']), float(inputs['min_area']))}",
         pre=[{"port": "mask", "check": "nonempty"}], post=[{"port": "labels", "check": "range", "value": [0, None]}],
         tags=["segmentation", "instances", "watershed", "distance transform", "morphology", "images"]),
    dict(id="hippocampus_segmentation_atlas_fusion_lgbm", tool="hippo.fit_predict",
         description="Voxel-wise segmentation of the hippocampus into anterior (1) and posterior (2) parts in small 3-D T1-weighted MRI crops: location "
                     "atlas, patch-based multi-atlas label fusion and a LightGBM voxel classifier fitted on labelled volumes; CPU only, no pretrained weights.",
         inputs={"train_images": p("list", "list of labelled 3-D MRI volumes (float arrays, any shape and intensity scale)"),
                 "train_labels": p("list", "list of uint8 label volumes shaped like the images: 0 background, 1 anterior, 2 posterior"),
                 "eval_images": p("list", "list of 3-D MRI volumes to segment"),
                 "train_ids": p("list", "case ids of the training volumes, strings ending in an integer such as 'hippocampus_123' (volumes 2k-1 and 2k are one subject "
                                        "and are left out of each other's label fusion)")},
         outputs={"segmentations": p("list", "one uint8 label volume (0, 1, 2) per evaluation image, shaped like the image")},
         code="from scilib import hippo\nreturn {'segmentations': hippo.fit_predict(inputs['train_images'], inputs['train_labels'], inputs['eval_images'], train_ids=inputs['train_ids'])}",
         pre=[{"port": "train_images", "check": "nonempty"}, {"port": "eval_images", "check": "nonempty"},
              {"port": "train_ids", "check": "len_eq_port", "value": "train_images"}, {"port": "train_labels", "check": "len_eq_port", "value": "train_images"}],
         post=[{"port": "segmentations", "check": "nonempty"}],
         tags=["medical imaging", "MRI", "brain", "hippocampus", "segmentation", "volume", "3D", "multi-atlas", "label fusion", "lightgbm", "dice"]),
    dict(id="segmentation_dice_two_labels", tool="hippo.mean_dsc",
         description="Dice similarity coefficient of segmentation label volumes with labels 1 and 2 (hippocampus anterior / posterior) against reference label "
                     "volumes: per-case Dice of each label and the mean over cases of the mean of both labels (1.0 when both masks are empty).",
         inputs={"predictions": p("list", "list of integer label volumes"),
                 "references": p("list", "list of reference label volumes, same shapes and order")},
         outputs={"mean_dice": p("number", "mean over cases of the mean Dice of label 1 and label 2", unit="1"),
                  "per_case": p("array", "Dice of label 1 and label 2 for every case", shape=("n", 2), dtype="float")},
         code="import numpy as np\nfrom scilib import hippo\n"
              "per_case = np.array([hippo.case_dsc(p, g) for p, g in zip(inputs['predictions'], inputs['references'])], dtype=float)\n"
              "return {'mean_dice': hippo.mean_dsc(inputs['predictions'], inputs['references']), 'per_case': per_case}",
         pre=[{"port": "predictions", "check": "nonempty"}, {"port": "references", "check": "len_eq_port", "value": "predictions"}],
         post=[{"port": "mean_dice", "check": "range", "value": [0.0, 1.0]}],
         tags=["medical imaging", "segmentation", "metric", "evaluation", "dice", "DSC", "hippocampus", "volume"]),
    dict(id="multi_atlas_label_fusion_patch", tool="hippo.label_fusion",
         description="Patch-based multi-atlas label fusion for 3-D MRI: every labelled atlas volume is translated onto the target volume, the k best-matching "
                     "atlases vote voxel by voxel with patch-similarity weights, giving class probabilities (background, anterior, posterior) around the atlas support.",
         inputs={"image": p("array", "target 3-D MRI volume", shape=("X", "Y", "Z"), dtype="float"),
                 "atlas_images": p("list", "list of labelled 3-D MRI volumes"),
                 "atlas_labels": p("list", "list of uint8 label volumes (0, 1, 2) shaped like the atlas images"),
                 "k": p("number", "number of best-matching atlases that vote", unit="1")},
         outputs={"probabilities": p("array", "class probabilities (background, label 1, label 2)", shape=(3, "X", "Y", "Z"), dtype="float")},
         code="from scilib import hippo\nreturn {'probabilities': hippo.label_fusion(inputs['image'], inputs['atlas_images'], inputs['atlas_labels'], k=int(inputs['k']))}",
         pre=[{"port": "atlas_images", "check": "nonempty"}, {"port": "atlas_labels", "check": "len_eq_port", "value": "atlas_images"}],
         post=[{"port": "probabilities", "check": "range", "value": [0.0, 1.0]}],
         tags=["medical imaging", "MRI", "segmentation", "volume", "3D", "multi-atlas", "label fusion", "hippocampus", "atlas"]),
    dict(id="location_prior_atlas_from_masks", tool="hippo.LocationAtlas",
         description="Location-only probabilistic atlas of labelled masks: class frequencies of the training masks on a normalised grid, evaluated as the prior "
                     "probability of every class at every voxel of a volume of the given shape; its argmax is the location-only reference segmentation.",
         inputs={"labels": p("list", "list of integer label volumes (0 background, 1, 2) of any shape"),
                 "image": p("array", "volume whose shape the prior is evaluated on", shape=("X", "Y", "Z"), dtype="float")},
         outputs={"priors": p("array", "prior probabilities (background, label 1, label 2)", shape=(3, "X", "Y", "Z"), dtype="float")},
         code="from scilib import hippo\natlas = hippo.LocationAtlas().fit(inputs['labels'])\nreturn {'priors': atlas.prob(inputs['image'].shape)}",
         pre=[{"port": "labels", "check": "nonempty"}], post=[{"port": "priors", "check": "range", "value": [0.0, 1.0]}],
         tags=["medical imaging", "segmentation", "volume", "3D", "atlas", "prior", "hippocampus", "baseline"]),
]
