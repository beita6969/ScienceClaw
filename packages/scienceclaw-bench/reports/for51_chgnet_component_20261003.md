# FoR51 CHGNet frozen route boundary (2026-10-03)

## Route

FoR51 now exposes `fit_chgnet_mlip` as an explicit ToolSpec. It forces
`model="chgnet"`, uses the frozen CHGNet 0.3.0 potential only as a feature
extractor, and fits the documented lightweight regressor on visible training
targets. The existing `fit_sevennet_mlip` route and its formal results are
unchanged.

## Server evidence

- Leonardo `sc-harness` loaded CHGNet 0.3.0 on an A100 and produced a finite
  `(1,21)` phonon feature vector for a Si smoke structure.
- A standard agent probe on the independent `src` pool completed with finite
  predictions, but its provenance was `model=sevennet`; it is not evidence for
  the CHGNet route and is not counted as a CHGNet result.
- An explicit CHGNet feature extraction attempt over the full 676-row visible
  training pool plus a 16-item source evaluation pool was stopped by the
  300-second wall-time limit while running phonopy/CHGNet. No evaluation
  target was read and no formal ID/OOD item was touched.

## Decision

Keep `fit_chgnet_mlip` staged as a frozen engineering option, but do not claim
an FoR51 score or replace SevenNet until a longer, cached engineering run
produces a provenance record with `model=chgnet`. The current formal ID/OOD
capacity is exhausted, so no duplicate formal run is started.

## Repair update (2026-10-04)

The earlier `1239/1265` result was a GPU-capacity artifact: Qwen vLLM occupied the same A100 and all 26 failures were `torch.OutOfMemoryError`. A CPU-only continuation on the existing Leonardo allocation (`CUDA_VISIBLE_DEVICES=""`, batch 1) recomputed the full cache and completed `1265/1265` structures in 29.6 seconds. The read-only audit now reports `valid=1265, missing=0, malformed=0, complete=true`. The explicit `fit_chgnet_mlip` smoke remains finite for `FoR51-src-01352840-02` with `model=chgnet, pretrained=true, fit_targets=676`; this is engineering evidence only and does not create a formal ID/OOD score.
