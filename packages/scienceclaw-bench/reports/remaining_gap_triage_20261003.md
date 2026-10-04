# Remaining low-score route triage (2026-10-03)

This note records a targeted route decision for FoR38/FoR45/FoR49/FoR50/FoR52. It does not alter acceptance, the scorer, item allocation, or target visibility.

## FoR38

The complete trusted-side pools were evaluated with the existing visible-data `scilib.macro` implementation. The frozen equal-weight route (`ridge,huber,lgbm_core,robdrift`, `n_backtest=0`) produced mean sMAPE **13.6593** on ID and **10.2101** on OOD. A second, non-policy diagnostic with `weights=auto,n_backtest=8` produced **13.6169** on ID and **10.4402** on OOD. The tiny ID gain is outweighed by the OOD regression; the all-member combination (`MEMBERS`) was worse (**14.9566/11.1035**). No single replacement is supported by these split-level results, so the frozen equal route remains the reproducible route.

## FoR45

The frozen OpenCLIP ViT-B/32 route and its fixed top-three caption-medoid alias are already implemented. Existing evidence shows the image-conditioned route below the visible language-medoid reference, and each ID/OOD pool has already consumed its four disjoint 8-item episodes. There is no remaining mutually exclusive episode on which to claim a new formal score; no rerun is performed.

## FoR49

The fixed `z3_check -> submit.y` route is already the strongest available engineering route. The trusted full-pool pure-Z3 screen is a lower-bound diagnostic only; it cannot be promoted to an agent score, and the incomplete timeout item is retained. No new solver or acceptance shortcut is added.

## FoR50

The fixed visible-data `scilib.valueeval.fit_predict` route scores **0.54895** on the complete ID pool and **0.43708** on the complete OOD pool. These exceed the complete-pool all-values reference (**0.26293/0.12846**), while the 16-item episode scores remain noisy. A new pretrained text route would require a separately staged frozen checkpoint and a disjoint evaluation pool; neither is available in this checkout, so no incomparable score is introduced.

## FoR52

The fixed and adaptive visible-data routes are deployed and deterministic. Their accuracy metric has no directly matching published NLL reference, so a higher accuracy value cannot be relabelled as same-metric SOTA. Existing ID/OOD episode allocations are exhausted for the current pools; no duplicate run is made.

**Decision:** no safe non-duplicate SOTA component was identified in this targeted pass. The existing routes and explicit comparability boundaries are retained. This is a route decision, not a new formal result.

## FoR50 pretrained text check (Leonardo A100)

As a targeted engineering check, the already-staged BGE-large-en-v1.5 encoder was run through the same grouped OOF calibration and expected-F1 decision logic on the complete visible training table. The diagnostic full-pool result was **0.54849 ID / 0.36655 OOD**, versus the frozen TF-IDF route's **0.54895 / 0.43708**. The embedding route therefore does not improve either split and is not integrated as a policy-visible alias.
