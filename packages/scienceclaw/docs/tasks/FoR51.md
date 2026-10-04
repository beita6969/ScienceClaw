# FoR51 Physical sciences — Matbench v0.1 matbench_phonons (MAE in cm^-1, lower is better)

Adapter: `scienceclaw/bench/tasks/for51_matbench.py` (`Adapter = MatbenchPhononsAdapter`). Status: **available**.

## Dataset
| field | value |
|---|---|
| name / version | Matbench v0.1 `matbench_phonons` (matminer metadata commit `8ddb18c7`) |
| URL | https://ml.materialsproject.org/projects/matbench_phonons.json.gz |
| sha256 | `4db551f21ec5f577e6202725f10e34dfc509aa7df3a6bdaac497da7f6dbbb9b3` (matches publisher hash) |
| license | Matbench MIT (https://github.com/materialsproject/matbench/blob/main/LICENSE); data Petretto et al., Sci. Data 5:180065 (2018) |
| local path | `<DATA_ROOT>/for51-matbench-phonons/raw/matbench_phonons.json.gz` |
| size | 1,265 relaxed crystal structures (2–48 sites, ordered, no partial occupancy); target `last phdos peak` 59.6–3643.7 cm^-1 |

## Items, pools, splits
* **Item** = one crystal structure (pymatgen dict parsed without pymatgen into lattice, species, fractional coordinates) with its target in cm^-1. Id `mb-phonons-<row:04d>` (Matbench naming, dataset order kept).
* **OOD pool** (179 items) = every compound containing **Se or Te** — leave-element-out chemistry shift; these elements occur in no visible or IID item. `lineage.ood_kind = "proxy_within_dataset"` (no second phonon dataset locally).
* **IID pool** (1,086 items) cut once (stratified by target decile, `partition_seed = 20260928`) into **train 676 / dev 110 / src 161 / val 40 / id 99**.
* **Episode** = 16 structures, block `e` of a seeded permutation of the split's sub-pool. Capacity: src 10, val 2, id 6, ood 11 episodes (default plan 7/2/4/4 fits; val is exactly 2).

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | the fixed 676 training structures + targets (cm^-1) |
| `load_dev_inputs` | 110 dev structures (targets withheld) |
| `score_dev(dev_pred)` | `dev_mae`, `dev_reference_mae` (cm^-1) — visible dev signal (`_dev_evaluate` is None) |
| `load_eval_inputs` | the 16 evaluation structures (no targets) |
| `featurize_structures(structures)` | 35 descriptors: site-weighted mean/min/max/std of Z, mass (amu), Pauling electronegativity, covalent radius (Å), group, period, outer electrons (RDKit periodic table + embedded Pauling table); n_sites, n_elements, volume/atom (Å^3), density (g/cm^3), packing fraction, min/mean nearest-neighbour distance (Å, 27 neighbouring cells) |

Structure dict: `{"formula", "lattice" (3×3 row vectors, Å), "species", "frac_coords"}`.

## Metric, reference, acceptance (D_V)
* Metric: MAE = mean |y − y_true| in cm^-1 (Matbench's primary regression metric). Pooled = MAE over all items (invalid episodes use the reference predictions).
* Reference: 5-nearest-neighbour regression (uniform, Euclidean) on the standardized `featurize_structures` descriptors of the 676 training crystals.
* Acceptance: `MAE <= 0.8 × MAE_ref`.
* Calibration (adapter check, not a research result; seed-dependent): reference MAE ≈ 92 (src), 93 (val), 142 (id), 55 (ood) cm^-1; an ExtraTrees(500) on the descriptors (log target) reaches ≈ 36/41/37/27 and is always accepted; ridge on the log target is accepted on ~50–85 %; plain ridge on raw targets (MAE ≈ 150) fails. A 3-feature ridge reference (MAE ≈ 220) was rejected because every ML pipeline beat it by > 5×.
* Hard constraints: `output_shape`, `finite`, `plausible_frequency_cm-1` (every value in [10, 5000] cm^-1; catches THz outputs). The required unit `cm^-1` is declared in `required_output`, so the canvas compat rule forces an explicit conversion edge for any other declared unit.

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 300, max_llm_items 64`. Featurising all 1,265 structures takes < 1 s.

## Deviations from the historical protocol
* Exact historical sample ids / split rules were **not recovered**; rebuilt splits. Historical 64 IID + 64 OOD items; here id 64 + ood 64.
* Matbench's official protocol is 5-fold nested CV over all 1,265 entries; here a fixed train/dev/eval partition is used so that episodes are item-disjoint and training data never overlaps evaluation items.
* OOD is a within-dataset leave-element-out shift, not a second dataset.
* The data team's `reconstructed_v1` sample (official Matbench fold 0: 128 source / 64 validation from the fold-0 training part, 128 evaluation from the fold-0 test part) is not used; the adapter defines its own leave-element-out OOD and fixed train/dev/eval partition over the same 1,265 official entries.
