---
name: scienceclaw-benchmark-for31
description: "Use when ranking or predicting the fitness effect of single amino-acid substitutions in a protein (deep mutational scanning, ProteinGym-style assays) from labelled variants of the same assay and the wild-type sequence, delivering one score per query variant evaluated by per-assay Spearman correlation. Corresponds to FoR31 of the companion ScienceClaw-Eval benchmark."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Variant-effect ranking for single substitutions (ProteinGym)

**Task.** Per assay (one protein): wild-type sequence, labelled single substitutions (`position` 1-based, `wt_aa`, `mut_aa`, `DMS_score`, higher = fitter) and query variants without scores. Deliverable: a 1-D float array, one score per query row in the query order (higher = fitter); only the ranking within an assay matters.

**Quality.** Spearman rank correlation per assay (a constant prediction scores 0), averaged over assays; higher is better. Locally: `scilib.proteinfit.spearman`, `mean_spearman(pred, truth, groups)`.

**Library** (check `scienceclaw_tools(operation=show|weights)`):
- `scilib.proteinfit.fit_predict(train, wild_type, query)`: BLOSUM62 and amino-acid property features, sequence context and position statistics of the training scores, rank-averaged ridge / LightGBM / extra trees. `fit_predict_assays` for several assays; also `variant_features`, `cross_validate(..., by="variant"|"position")`.
- Zero-shot ESM-2 650M (asset `esm2_650m`, GPU or remote worker; guard with `scilib.proteinplm.available()`): `proteinplm.plm_features(table, wild_type)` gives log-likelihood-ratio columns; pass them via `extra_train` / `extra_query` or `fit_predict_assays(..., plm=True)`. CPU runs are slow beyond roughly 120 residues.
- Operators: `dms_variant_effect_supervised`, `dms_variant_effect_supervised_with_features`, `dms_cross_validated_spearman`, `protein_variant_effect_esm2`.

**Routes.** Baseline: mean training score at the variant's position (shrunk towards neighbouring positions, assay mean if unseen). Default: `fit_predict`; add ESM-2 columns when available (BLOSUM62 alone is weak). Select with `cross_validate`, using `by="position"` when query variants lie at unseen positions.

**Rules.**
- Use only labelled variants of the same assay; never use query scores or columns derived from them (for example a binarised score).
- Position statistics of a training row must exclude that row (`exclude_self=True`; `fit_predict` does this).
- `position` is 1-based and `wt_aa` must match the sequence; keep the query order; output finite; do not compare raw scores across assays.
- ESM-2 was pretrained on UniRef50 sequences, so homologs of the assayed protein are likely in it. Disclose this overlap.
