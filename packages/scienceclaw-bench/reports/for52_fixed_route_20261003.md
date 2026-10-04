# FoR52 fixed psychometric route engineering batch C (2026-10-03)

- Runner: `H52C` through the existing one-card Qwen service `59277346:37346`.
- Episode selection: `skip=8`, ID/OOD episodes `08–11`; `SOTA52` and `SOTA52B` consumed `00–07`, so this batch is disjoint.
- Tool: fixed `psych_fixed_predict` route from `scilib.psych`; it receives visible training sessions and label-free evaluation inputs, with fixed seed `0` and provenance.

| split | accuracy per episode | mean | accepted | direct tool-to-submit |
|---|---|---:|---:|---:|
| ID | 0.6250, 0.7500, 0.6875, 0.6875 | 0.6875 | 4/4 | 4/4 |
| OOD | 0.6875, 0.6875, 0.7500, 0.6250 | 0.6875 | 2/4 | 4/4 |

The two OOD failures are ordinary acceptance failures and are retained without
rerun. This is engineering evidence; FoR52's accuracy is not directly
comparable to literature NLL results, so the formal manifest and SOTA status do
not change.

## Fixed route engineering batch D

- Runner: the same `59277346:37346` single-card service, with a per-tag cache.
- Episode selection: ID/OOD `12–15`; H52C and the formal SOTA52/SOTA52B batches
  consumed `00–11`, so this batch is disjoint.

| split | accuracy per episode | mean | accepted | direct tool-to-submit |
|---|---|---:|---:|---:|
| ID | 0.5000, 0.5625, 0.5625, 0.6250 | 0.5625 | 3/4 | 4/4 |
| OOD | 0.5000, 0.4375, 0.7500, 0.6250 | 0.5781 | 2/4 | 4/4 |

All eight graphs use `psych_fixed_predict` (the OOD-14 node is named `pred`)
and connect its `y` directly to `submit.y`. The two ID/OOD failures are normal
acceptance-line failures; no item is rerun.
