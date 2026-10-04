# FoR30 Panoptic-DeepLab frozen-candidate audit (2026-10-04)

## Decision

The official PRBonn PhenoBench Panoptic-DeepLab checkpoint was staged on Leonardo and exercised on a visible FoR30 dev slice. It is **not a safe replacement for the current Mask2Former route** and is not registered as a FoR30 tool or formal route. The candidate was evaluated only on the visible dev capture; no ID/OOD formal item was repeated.

## Provenance and staging

- Upstream implementation: [PRBonn/phenobench-baselines](https://github.com/PRBonn/phenobench-baselines), pinned at commit `78db625441e54c5b64efabeb0d886d020961665b`.
- Public checkpoint URL: `https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/PanopticDeeplab/model.pth`.
- Leonardo path: `$L/models/phenobench_panopticdeeplab/model.pth`.
- Size: `31,232,501` bytes; SHA-256: `1f1f1be69560551478fc58f5c06709ff24efe0572622a6d4a8dc1d3d5553fd4d`.
- The checkpoint loads into the released MobileNetV2 + Panoptic-DeepLab plants architecture with `missing=0`, `unexpected=0` under the pinned source.

The model reads RGB only and produces semantic logits plus center/offset plant instances. It does **not** produce crop-leaf instances. A contract-complete hybrid was therefore also checked by retaining its semantic/plant arrays and taking leaf instances from the already released PhenoBench Mask2Former leaf model. This was only a visible diagnostic; it was not registered as a new tool because the plant/semantic branch already failed.

## Visible score comparison

The slice is the first `FoR30-val-s20260928-e00` visible dev capture (`P0030855`, eight images). Images were loaded through `load_dev_inputs`; predictions were scored through the task's `score_dev` tool, which keeps annotations inside the trusted scorer. The model runner did not read labels.

| route | IoU soil | IoU weed | PQ crop | PQ leaf | PQ+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Released Mask2Former plants + leaves | 99.7509 | 71.8429 | 72.7730 | 55.9824 | **75.0873** |
| Panoptic-DeepLab plants + Mask2Former leaves | 98.4379 | 0.0000 | 0.0000 | 0.0000 | **24.6095** |
| Panoptic-DeepLab semantic/plant + raw Mask2Former leaves before crop gating | 98.4379 | 0.0000 | 0.0000 | 55.9824 | **38.6051** |

The Panoptic-DeepLab output on all eight images was background-only after the released post-processing (`semantics={0}`, `plant_instances={0}`). Consequently, it cannot improve the semantic or plant components, and blindly attaching its output to the leaf model would lower PQ+ substantially. The score gap is a model/route failure, not a scorer discrepancy: the same `score_dev` call gives the Mask2Former baseline `75.0873` and the candidate `24.6095`.

## Boundary

- No `predict_panoptic_deeplab` wrapper, ToolSpec, or formal manifest entry was added.
- No ID/OOD item was rerun; the one-item-per-pool rule is unchanged.
- The checkpoint and upstream repository remain available on Leonardo for future debugging, but this candidate is excluded from the SOTA tool pool until a separately verified preprocessing/checkpoint pairing produces non-background predictions and a visible score above the existing Mask2Former route.
