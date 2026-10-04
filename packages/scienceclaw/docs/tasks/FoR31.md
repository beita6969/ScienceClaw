# FoR31 Biological sciences — ProteinGym v1.3 DMS substitutions (mean Spearman, higher is better)

Adapter: `scienceclaw/bench/tasks/for31_proteingym.py` (`Adapter`; shared helpers in `_life_health_common.py`).
Status: **available**.

## Dataset
| field | value |
|---|---|
| name / version | ProteinGym DMS substitution benchmark, v1.3 (historical project version unconfirmed) |
| URL | https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip (repo https://github.com/OATML-Markslab/ProteinGym) |
| sha256 (zip) | `3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921` (data-team receipt; no published checksum) |
| license | ProteinGym repository MIT (https://github.com/OATML-Markslab/ProteinGym/blob/main/LICENSE); the DMS measurements come from the original studies |
| local path | `<DATA_ROOT>/for31-proteingym-substitutions/data/DMS_ProteinGym_substitutions/*.csv` (217 assays, 2,465,767 rows; columns `mutant, mutated_sequence, DMS_score, DMS_score_bin`) |
| not available | the ProteinGym reference file (`DMS_substitutions.csv` with selection type / taxon) — not needed by the adapter |

## Items, pools, splits
* **Item** = one DMS assay restricted to single amino-acid substitutions (`^[A-Z]\d+[A-Z]$`, 20 standard residues,
  duplicates and NaN scores dropped, rows consistent with the reconstructed wild type). Per assay, fixed by
  `pool_seed = 20260928`: up to 128 **query** variants (40 % cap), up to 32 **dev** variants (10 % cap) and up to 1,024
  **visible training** variants, all disjoint. An assay is eligible with >= 160 singles (212 of 217; excluded:
  F7YBW8_MESOW_Aakre_2015, F7YBW8_MESOW_Ding_2023, GCN4_YEAST_Staller_2018, NPC1_HUMAN_Erwood_2022_RPE1,
  SPG1_STRSG_Wu_2016).
* **Why an item is an assay** (not a single variant): Spearman is defined per assay, and ProteinGym's headline number
  is the Spearman per assay averaged over assays. An episode = 16 assays; id = 4 x 16 = 64 assays, ood = 64 assays.
* Wild type: reconstructed from `mutated_sequence` of the first single mutant (mutation reverted) and verified.
* **Lineage group** = UniProt entry name (leading all-caps tokens of the assay id, e.g. `BLAT_ECOLX`); proteins never
  straddle pools.
* **IID** = the 148 eligible functional assays (activity, binding, expression, organismal fitness; many labs).
  **OOD** = the 64 Tsuboyama et al. 2023 mega-scale cDNA-display proteolysis *stability* assays — a different source
  study and assay technology on 64 small domains, no UniProt entry shared with IID
  (`lineage.ood_kind = "cross_source_within_benchmark"`).
* Pools: id 64 assays (55 proteins), val 32 (24), src 52 (41; source episodes reuse assays: 7 x 16 > 52), ood 64 (64).

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | `table` (item, assay_id, mutant, position, wt_aa, mut_aa, DMS_score) — visible labelled training singles of the 16 assays; `wild_type` {assay_id: sequence} |
| `load_eval_inputs` | `table` of query variants (no scores) — its row order is the order of `y`; `wild_type` |
| `load_dev_inputs` | dev variants of the same assays (scores withheld) |
| `score_dev(predictions)` | mean and per-item Spearman on the dev variants (the visible dev signal; `Episode._dev_evaluate` is None) |
| `substitution_features(table)` | per-row BLOSUM62, Kyte-Doolittle hydropathy (wt, mut, delta), side-chain volume delta (Zamyatnin, A^3), formal charge delta, polarity change, to-proline, from-glycine |

## Metric, reference, acceptance (D_V)
* Required output: 1-D float array, one score per query row (length = sum of query sizes, 1,956-2,048), higher = fitter.
* **Metric**: `scipy.stats.spearmanr` per assay on its query variants (average ranks for ties; constant predictions
  score 0), episode primary = mean over its 16 assays. Pooled = mean over assays (values of an assay seen in several
  episodes are averaged first). ProteinGym's leaderboard additionally averages by UniProt id and by functional
  category; that aggregation is not used here (no reference file locally).
* **Reference**: site-mean — each query variant gets the mean training DMS score of its position (assay training mean
  if the position has no training variant). Reference at the default plan (bench seed 20260928), pooled: src 0.533,
  val 0.552, id 0.518, ood 0.726.
* **Acceptance**: `mean Spearman > reference + 0.03`.
* `details`: `reference`, `norm_score` (primary/reference clipped to [0,10]), `pooled_payload` (`assays`, `spearman`,
  `n_query`), `per_item_spearman`, `reference_per_item`.
* Invalid outputs: `primary=None`, `norm_score=0`; pooled payload uses Spearman 0 for the episode's assays.
* **Hard constraints**: `output_shape` (1-D, exact length), `finite_values`, `declared_unit` ("1").
* **Calibration** (adapter sanity check, one seed): a per-assay ridge regression on (leave-one-out site mean,
  substitution descriptors, mutant-residue one-hot) scored 0.588/0.579/0.586/0.636 on id vs reference
  0.482/0.537/0.510/0.545 and 0.808/0.824/0.799/0.780 on OOD vs 0.724/0.752/0.736/0.691 (accepted 8/8); zero-shot
  BLOSUM62 alone scores 0.12-0.26 (rejected).

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 300, max_llm_items 64`.

## Caches
`<repo>/cache/tasks/FoR31/index_<hash>.json` (per-assay single counts / protein / WT length) and
`cache/tasks/FoR31/assays/<assay>.{parquet,json}` (parsed singles + wild type, keyed by source byte count). ~5 s to build.

## Deviations from the historical protocol and open issues
* Exact historical sample ids / split rules were **not recovered**; these are rebuilt splits. Historical protocol:
  64 IID + 64 OOD items; here 64 IID assays + 64 OOD assays.
* Supervised within-assay setting (visible labelled variants of the same assay). The historical paper numbers
  (~0.37-0.42) look like a lower-data or zero-shot regime; the absolute scale is therefore not comparable.
* Only single substitutions are used (multi-mutants are ignored).
