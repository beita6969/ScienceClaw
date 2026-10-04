# FoR34 Chemical sciences — OGB ogbg-molhiv (ROC-AUC, higher is better)

Adapter: `scienceclaw/bench/tasks/for34_molhiv.py` (`Adapter = MolhivAdapter`). Status: **available**.

## Dataset
| field | value |
|---|---|
| name / version | OGB `ogbg-molhiv`, OGB registry v1 (`hiv.zip`, Last-Modified 2020-05-04) |
| URL | https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip (docs: https://ogb.stanford.edu/docs/graphprop/) |
| sha256 (zip) | `47d747664b9e1653de5aac99bf26c015d88a3daa474f5a32f3fe5c014111375b` (data-team receipt) |
| license | OGB code MIT; labels from MoleculeNet HIV / NCI DTP AIDS Antiviral Screen |
| local path | `<DATA_ROOT>/for34-ogbg-molhiv/extracted/hiv/` (`mapping/mol.csv.gz`, `split/scaffold/*.csv.gz`, `raw/*.csv.gz`) |
| size | 41,127 molecules; official scaffold split train 32,901 (1,232 active) / valid 4,113 (81) / test 4,113 (130) |

## Items, pools, splits
* **Item** = one molecule (SMILES) with label 1 = confirmed/moderately active (HIV replication inhibition), 0 = inactive. Id `ogbg-molhiv/<row>`.
* **IID pool** = official scaffold `valid` partition, cut once into src / val / id (55/15/30 %, stratified by label, ranked by `sha256("FoR34|<partition_seed>|iid|<label>|<id>")`, `partition_seed = 20260928`). Sizes (inactive/active): src 2218/45, val 605/12, id 1209/24.
* **OOD pool** = official scaffold `test` partition (3983/130). `lineage.ood_kind = "proxy_within_dataset"`: no second HIV-type MoleculeNet set exists locally; the shift is the official *scaffold* shift (test holds the rarest Bemis–Murcko scaffolds, disjoint from valid and train scaffolds).
* **Episode** = 16 molecules: `round(0.25*16) = 4` actives + 12 inactives, so every episode has both classes (active rate enriched from ~2 % — ROC-AUC is prevalence invariant). Episode `e` of a split takes block `e` of a seeded permutation of each class (seed = SplitPlan's per-split seed), so episodes are prefix-stable and item-disjoint; sub-pool partitions do not depend on the per-split seed, so splits are disjoint for any seeds.
* Capacity (16 items): src 11, val 3, id 6, ood 32 episodes (default plan 7/2/4/4 fits).

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | 4,000 labelled molecules (`smiles` list, `labels` 0/1) — per-episode deterministic, label-stratified sample of the official scaffold **train** partition (scaffold-disjoint from all evaluation molecules) |
| `load_dev_inputs` | 1,000 further train-partition molecules (disjoint from `load_train`), labels withheld |
| `score_dev(dev_scores)` | `dev_roc_auc`, `dev_reference_roc_auc`, `dev_n_active` — the visible dev signal (`Episode._dev_evaluate` is None) |
| `load_eval_inputs` | the 16 evaluation SMILES (no labels), in output order |
| `featurize_molecules(smiles)` | RDKit features: `morgan` (default r=2, 2048 bits), `morgan_counts`, `maccs`, `descriptors` (24 RDKit descriptors); `valid` mask (7 of 41,127 SMILES only parse unsanitized; unparsable rows are NaN) |

## Domain library (`scilib.molecules`)
The objective ends with `scilib.describe("molecules")`; code nodes can `from scilib.molecules import ...` (repo root on the worker `PYTHONPATH`). It offers `featurize` (Morgan counts, atom-pair counts, MACCS, all 217 RDKit 2D descriptors; unsanitisable SMILES parsed unsanitised, unparsable ones flagged in a `valid` mask), `scaffold_groups` (Bemis-Murcko ids), `TreeEnsemble` / `fit_predict` (class-balanced entropy RandomForest + ExtraTrees, mean probability; unparsable query SMILES get their set's median score, so scores are always finite in [0, 1]) and `grouped_cv_auc` (StratifiedGroupKFold over scaffolds or given ids). `load_dev_inputs` molecules are training-partition molecules: about half share a Bemis-Murcko scaffold with `load_train` (the objective says so), whereas no evaluation molecule shares a scaffold with `load_train`. CPU cost on one thread: featurising 1,000 molecules ~11 s, fitting 4,000 x 4,480 with 250 trees per model ~22 s.

## Metric, reference, acceptance (D_V)
* Metric: `sklearn.metrics.roc_auc_score(y_true, y)` — identical to the OGB `Evaluator("ogbg-molhiv")` computation for its single task. Pooled metric = ROC-AUC over all items of all episodes (invalid episodes contribute the reference scores).
* Reference: L2 logistic regression (C=1, `class_weight="balanced"`) on 11 trivial graph-count features from the OGB raw files (log atoms/bonds/rings, C/N/O/S/halogen/other fractions, aromatic and ring-atom fractions), fit on the episode's `load_train` rows.
* Acceptance: `AUC >= AUC_ref + 0.05`.
* `details`: `reference`, `norm_score` (= AUC/AUC_ref clipped to [0,10]), `pooled_payload` (`item_ids`, `y_true`, `y_score`, `y_ref`).
* Hard constraints: `output_shape` (1-D, length 16), `finite`, `probability_range` ([0, 1]).
* Calibration (adapter check on one seed, not a research result): a generic Morgan-2048 + 300-tree random forest was accepted on 6/7 src, 1/2 val, 3/4 id and 1/4 ood episodes with the 0.05 margin; mean AUC src/val/id/ood 0.85/0.91/0.68/0.54 vs reference 0.63/0.83/0.55/0.66. Per-episode AUC over 4×12 pairs is noisy (resolution 1/48).
* Effect-size / noise analysis (improver pass, 2026-09-29, paired simulation: the default `scilib.molecules.fit_predict` recipe and the reference trained on four real adapter training draws, scored on every molecule of the src / val / id / ood pools plus a scaffold-disjoint train-partition proxy, then 4000 sampled 4-active + 12-inactive episodes per pool). Pooled AUC of the default recipe vs the reference: iid (src+val+id) 0.81 vs 0.69, ood 0.78 vs 0.76, train-partition proxy 0.75 vs 0.71. Per-episode acceptance probability (AUC >= ref + 0.05): about 0.70 on src/iid, 0.40 on ood, 0.48 on the proxy. Episode-level AUC has SE 0.07-0.11 and exact ties with the reference (e.g. 38/48 pairs both) are ordinary; the frozen-run `val` failure (0.7917 = reference 0.7917) is such a tie, produced by `fit_predict` defaults (the ranking correlates 0.75 with the trivial-count model, dev AUC 0.80 vs 0.66). Variants tried without a robust paired gain (all within simulation noise or worse on iid/ood): single feature blocks, extra FCFP / Avalon / ErG / MQN / EState / topological-torsion / RDKit-FP blocks, scaffold-size sample weights, 1000 trees, min_samples_leaf 1/3, max_features 0.05, LightGBM, L2 logistic regression on fingerprints or descriptors, rank-average blends of these with the forest, Tanimoto kNN / kernel-vote blends. Expected pass rate per episode is therefore ~0.5-0.7 (higher on iid, lower on ood); getting it substantially higher needs larger episodes or a larger margin-to-noise ratio, which are acceptance/episode-design decisions and were not changed.

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 300, max_llm_items 64`. RDKit featurisation of 5,000 molecules takes a few seconds on the laptop CPU.

## Deviations from the historical protocol
* Exact historical sample ids / split rules were **not recovered**; these are rebuilt splits (new seeded sampling from the official partitions). Historical protocol: 64 IID + 64 OOD items; here id = 4×16 = 64, ood = 4×16 = 64 items.
* Episodes are class-stratified (25 % actives) instead of natural prevalence, so per-episode ROC-AUC is defined.
* OOD is the official scaffold shift inside ogbg-molhiv (no second dataset available locally).
* The data team's `reconstructed_v1/episodes_v1` (32-graph cohorts with 1 active each; its ID cohorts are drawn from the official *train* partition) is a different design and is not used: this adapter reads the official OGB files directly, keeps the train partition as visible training data and evaluates on valid (IID) / test (OOD).
