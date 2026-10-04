# FoR38 TimesFM 2.5 frozen-route audit (2026-10-04)

This is a trusted-side, label-free route comparison. It does not change the FoR38 scorer,
acceptance margin, split, item allocation, or any formal result. No formal item was rerun.

## Deployment

On Leonardo, the official Google TimesFM 2.5 package (`timesfm==2.0.2`) was installed in
an isolated directory:

- code: `$F/packages/for38-timesfm`
- cache: `$F/models/hf/hub/models--google--timesfm-2.5-200m-pytorch`
- checkpoint: `model.safetensors`, 925,181,104 bytes
- checkpoint SHA-256: `2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8`
- model: `google/timesfm-2.5-200m-pytorch`, 200M parameters

The model loaded strictly with `torch_compile=False`, compiled with `max_context=32`,
`max_horizon=128`, `per_core_batch_size=32`, and returned finite four-step forecasts for
the complete trusted-side ID+OOD pool. The run used only histories ending at the fixed
2021 origin. The first run was CPU-only on the Leonardo login environment; this did not
produce a formal score or consume an evaluation item.

Official sources: [TimesFM source](https://github.com/google-research/timesfm) and
[TimesFM 2.5 checkpoint](https://huggingface.co/google/timesfm-2.5-200m-pytorch).

## Complete-pool comparison

The pool contains 64 ID and 141 OOD items. Forecasts were clipped with the same physical
bounds as FoR38 before applying the adapter's M4 sMAPE implementation.

| route | ID (64) | OOD (141) | pooled (205) |
|---|---:|---:|---:|
| fixed macro (`ridge,huber,lgbm_core,robdrift`, equal) | 13.6593 | 10.2101 | 11.2862 |
| TimesFM 2.5 raw point forecast | 15.12694 | 11.26845 | 12.47305 |
| TimesFM 2.5 log-value input / exp output | 14.74305 | 11.59567 | 12.57827 |
| TimesFM 2.5 log-relative-to-last input | 14.83465 | 11.59567 | 12.60686 |
| TimesFM 2.5 relative-to-last input | 15.12693 | 11.26845 | 12.47305 |
| naive random walk | 17.6282 | 12.0508 | 13.7904 |

The raw point route is worse than the deployed macro ensemble on both complete pools;
log and relative transforms do not recover the gap. The model is therefore recorded as a
reproducible pretrained comparison and is **not** registered as a FoR38 tool alias.

## Contract and metric checks

- The TimesFM runner passed only each item's history and a fixed four-step horizon; no
  target array, scorer object, or post-origin field was passed to the model.
- The adapter's sMAPE is the M4 formula `200/H * sum(abs(y-p)/(abs(y)+abs(p)))`.
  Existing parity checks against `utilsforecast.losses.smape` remain exact to numerical
  precision; no metric or aggregation bug was found in this audit.
- The existing `macro_fixed_predict` route remains the strongest frozen route observed on
  the complete trusted-side pools. No scorer, acceptance, split, or reference change is
  justified by this candidate.

The 925 MB checkpoint and isolated package are retained on Leonardo for reproducibility;
they are not copied into the policy-visible tool pool.
