# Full-pool comparability refresh (2026-10-03)

This is a trusted-side refresh for the five datasets whose sampled episode scores are easy to misread against a
published number. Every value below comes from the current adapter and complete rebuilt split on the local data
receipt. These diagnostics read held-out targets only inside the adapter, return aggregates and receipts, and do not
change formal agent episodes, acceptance margins, or policy-visible tools.

## FoR38 — macro forecasting

| split | items | no-change reference sMAPE | damped trend sMAPE | history-backtest sMAPE | history gain over reference |
|---|---:|---:|---:|---:|---:|
| src | 128 | 15.1338 | 15.7870 | 14.5619 | +0.5719 pp |
| val | 32 | 12.4751 | 12.9207 | 11.9790 | +0.4961 pp |
| id | 64 | 17.6282 | 21.0853 | 16.8299 | +0.7983 pp |
| ood | 141 | 12.0508 | 13.7376 | 12.0392 | +0.0116 pp |

The history-only route provides only a small OOD gain, so the low agent score is not explained by an obviously
broken no-change reference. The OOD proxy remains a regional shift within the same WDI vintage and should not be
presented as an independent external benchmark.

## FoR45 — image captioning

The new full-pool visual diagnostic is recorded in
`reports/protocol_for45_full_reference_20261003.md`. Across src/val/id/ood, the language medoid scores
18.6669/20.2403/18.3885/21.2421 chrF++, while the lightweight image-kNN route scores
13.8897/12.1054/13.5398/16.2846. kNN is lower by 4.78--8.13 points on every pool, so this route is not a strong
visual reference. A future frozen CLIP route must be scored on the same full pools before it can support a visual
SOTA comparison.

## FoR49 — SMT solver shortcut

The full lineage-representative pools contain 1,429/397/1,017/2,595 items for src/val/id/ood. The all-`sat`
majority reference accuracies are 0.7061/0.7128/0.6922/0.7198. The frozen 1-second screen proves 1,129/313/793/1,909
easy items are solver-decided and status-correct; hard items remain unsolved by that screen, giving conservative
pure-solver lower bounds 0.7901/0.7884/0.7797/0.7356. These bounds are not formal agent scores and reinforce that
the observed high FoR49 episodes are solver-shortcut evidence, not general reasoning SOTA.

## FoR50 — ValueEval

The fixed visible-data route was refit once on all 5,393 training arguments and evaluated on complete pools:

| split | items | fixed-route F1 | all-values reference F1 | delta |
|---|---:|---:|---:|---:|
| src | 1,424 | 0.523088 | 0.291183 | +0.231905 |
| val | 472 | 0.505852 | 0.275169 | +0.230683 |
| id | 1,576 | 0.548950 | 0.262930 | +0.286020 |
| ood | 279 | 0.437082 | 0.128459 | +0.308623 |

These are trusted full-pool diagnostics, not formal 16-item agent results. They confirm that the earlier slice means
were not a scorer bug, but that the slice and full-pool F1 values must be reported separately because category
support and argument counts differ substantially.

## FoR52 — Psych-201

The participant-history reference over complete pools is 0.5884/0.6212/0.5285/0.6066 for src/val/id/ood. A
visible same-study reference is lower on IID (0.4802/0.5000/0.4791) and identical on OOD (0.6066), because OOD
studies have no visible same-study sessions and fall back to participant history. The participant-history baseline is
therefore retained for formal scoring; the study-conditioned route is diagnostic only and cannot explain away the
low ID/OOD episode results.

## Reproducibility

FoR45 can be rerun with:

```text
PYTHONPATH=. python scripts/f45_full_split_reference.py --split src val id ood
```

The other four adapters expose `full_split_reference` or `full_split_diagnostics` directly for the same trusted-side
refresh. Their current outputs are recorded above; no item was reused or reallocated in this pass.
