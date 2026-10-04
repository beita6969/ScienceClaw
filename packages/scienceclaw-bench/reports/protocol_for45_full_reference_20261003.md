# FoR45 full-pool visual reference diagnostic (2026-10-03)

This pass adds `AmericasNLPAdapter.full_split_reference(split)` and the reproducible driver
`scripts/f45_full_split_reference.py`. It evaluates the complete image-backed pool for each split using the
complete visible training partition for that pool. It reports the existing per-language caption medoid and the
visible-image nearest-neighbour route on the same items. The method is trusted-side only: target captions are read
for aggregation, no captions or labels are returned to the agent, and `evaluate`, `pooled_metric`, the acceptance
margin, and the formal item allocations are unchanged.

## Full-pool results

| split | pool | items | visible train (image / caption-only) | medoid chrF++ | image-kNN chrF++ | kNN - medoid |
|---|---:|---:|---:|---:|---:|---:|
| src | iid | 57 | 18 / 24 | 18.6669 | 13.8897 | -4.7772 |
| val | iid | 18 | 18 / 24 | 20.2403 | 12.1054 | -8.1349 |
| id | iid | 33 | 18 / 24 | 18.3885 | 13.5398 | -4.8487 |
| ood | ood | 33 | 51 / 36 | 21.2421 | 16.2846 | -4.9575 |

The aggregate uses the same sentence-level chrF++ implementation as the formal scorer. The split and training-pool
receipts are recorded by the driver (`split_ids_sha256` and `train_ids_sha256`) so a later run can detect data or
partition drift. The image-kNN reference is lower than the language medoid on every full pool, including OOD. Thus
the earlier low image-kNN values are not caused by a 16-item episode accident; this lightweight descriptor route is
not a stronger reference and should not be used to claim visual SOTA.

The result also makes the visual evidence boundary explicit: an image-blind language medoid reaches 18.39--21.24
chrF++ on complete pools, while the available frozen pixel-neighbour route reaches 12.11--16.28. Any future CLIP
route must be audited on these same full pools with its checkpoint provenance before being compared with the medoid.

## Follow-up route ready for the next formal allocation

`scilib.clip_retrieval.retrieve_captions` now supports `selection="caption_medoid"` when `k > 1`. It selects a
deterministic character-ngram medoid from the visible captions of the top CLIP image neighbours; it never reads
evaluation captions or labels. The FoR45 `clip_knn_reference` tool exposes this as an explicit configuration while
keeping the historical `selection="nearest"`, `k=1` default unchanged. This is an engineering route awaiting a
fresh, disjoint formal batch; no score is attributed to it yet. The next run should record both `k=1,nearest` and
`k>=3,caption_medoid` on the same new items, together with the frozen-checkpoint provenance, before deciding whether
the visual reference is materially stronger.

For reproducible tool use, `clip_knn_reference_medoid` is now also exposed as a fixed alias: it always requests
`k=3` and `selection="caption_medoid"` (using all available visible image rows when a language has fewer than
three). Config values cannot change those two fields, and the tool returns them in its receipt metadata. This alias
still has no formal score; it only removes agent-side configuration ambiguity for the next disjoint engineering run.
