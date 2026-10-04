# FoR42 Health sciences — PhysioNet/CinC Challenge 2019 early sepsis prediction (normalized clinical utility, higher is better)

Adapter: `scienceclaw/bench/tasks/for42_sepsis.py` (`Adapter`; shared helpers in `_life_health_common.py`).
Status: **available**.

## Dataset
| field | value |
|---|---|
| name / version | PhysioNet/CinC Challenge 2019 "Early Prediction of Sepsis from Clinical Data", v1.0.0, public training sets A and B |
| URL | https://physionet.org/content/challenge-2019/1.0.0/ (files from https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training/) |
| license | Open Data Commons Open Database License v1.0 (`raw/LICENSE.txt`) |
| population | 40,336 stays (A 20,336, B 20,000); one pipe-separated file per ICU stay, hourly rows, 40 variables + `SepsisLabel` |
| local files | `raw/training_setA` (641 files, prefix p000001-), `raw/training_setB` (64, p100001-), `reconstructed_v1/patients/training_set{A,B}` (192 + 128 seeded random stays, SHA-256 verified); union by file name (5 A stays in both copies; reconstructed copy preferred): **A 828 stays (66 septic), B 192 stays (8 septic)** |
| note | `EtCO2` is never measured in hospital A (all NaN in the training stays) |

## Items, pools, splits
* **Item** = one complete ICU stay (all hourly rows of one patient file). Item id = patient id (`p010139`).
* **IID** = hospital A. **OOD** = hospital B, the challenge's own cross-hospital shift
  (`lineage.ood_kind = "cross_hospital_within_dataset"`; not a different dataset). Visible training data are always
  hospital A only.
* Pools (fixed by `pool_seed = 20260928`, stratified by outcome, independent of the SplitPlan's per-split seeds):

| pool | stays | septic | source |
|---|---|---|---|
| id | 64 | 8 | A |
| val | 32 | 4 | A |
| src | 128 | 16 | A |
| dev (inputs visible, labels withheld) | 80 | 8 | A |
| train (visible, labelled) | 524 | 30 | A (all remaining A stays) |
| ood | 64 | 8 | B (all 8 locally available septic B stays + 56 random non-septic) |

* **Episode** = 16 stays with exactly 2 septic stays (12.5 %; challenge prevalence 7.3 %), so the utility
  normalization (best - inaction > 0) is always defined. Episode j takes block j of seeded per-stratum permutations
  (prefix-stable, item-disjoint within held-out splits).

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | long table (patient_id, hour, 40 variables, SepsisLabel) of the 524 training stays (~20k rows) |
| `load_eval_inputs` | long table of the 16 evaluation stays without SepsisLabel, `patient_ids` (output order), `n_hours` |
| `load_dev_inputs` | the 80 dev stays without labels |
| `score_dev(predictions)` | normalized utility of binary hourly predictions on the dev stays (the visible dev signal; `Episode._dev_evaluate` is None) |

## Metric, reference, acceptance (D_V)
* Required output: list of 16 integer arrays (stay order of `patient_ids`), array i of length `n_hours[i]`, values 0/1.
  Interface rule stated in the objective: the prediction for hour t may only use rows 0..t of that stay (plus the
  training data); later rows, whole-stay statistics and the stay length (`n_hours[i]` only sets the length of output
  array i) must not enter it. Enforcement: see "Causality" below.
* **Metric**: normalized clinical utility exactly as in the official `evaluate_sepsis_score.py`: dt_early = -12,
  dt_optimal = -6, dt_late = 3, max_u_tp = 1, min_u_fn = -2, u_fp = -0.05, u_tn = 0;
  `t_sepsis = argmax(SepsisLabel) - dt_optimal`; best predictions = 1 on [t_sepsis - 12, t_sepsis + 3];
  `NU = (U_obs - U_inaction) / (U_best - U_inaction)` with utilities summed over the episode's stays. The official
  per-stay loop is ported line by line (`compute_prediction_utility_official`) and the vectorized form used by the
  adapter is tested equal to it on 300 random cases. Pooled = the same formula with the per-stay utilities of all
  stays of the given episodes (`pooled_payload`: `observed`, `best`, `inaction` per stay).
* **Reference**: class-balanced L2 logistic regression (C = 1) on trivial causal features — last-observation-carried-
  forward HR, O2Sat, Temp, SBP, MAP, Resp plus Age, Gender, ICULOS, HospAdmTime; training-median imputation; z-scoring —
  fitted on the visible training stays; threshold chosen on the training stays to maximize NU (train NU 0.318).
  Reference NU at the default plan (bench seed 20260928), pooled: src 0.293, val 0.279, id 0.305, ood 0.363; per
  episode -0.13 .. 0.83 (two septic stays make per-episode values noisy).
* **Acceptance**: `NU > max(NU_ref, 0) + 0.05` (`ACCEPT_FLOOR = 0`, LEAK-1). The floor matters because two septic
  stays make the per-episode reference noisy and sometimes negative (id e01: -0.129, i.e. a logistic regression that
  does worse than never alarming); without the floor the all-zero output (NU = 0) "beat" such a reference. The reported
  `details['reference']` and `norm_score` still use the true reference; `details['acceptance_floor']` records the floor.
* `details`: `reference`, `norm_score` (NU/NU_ref clipped to [0,10]; if NU_ref <= 0 the documented shift
  `1 + NU - NU_ref` is used), `pooled_payload`.
* Invalid outputs: `primary=None`, `norm_score=0`; pooled payload uses all-zero (inaction) predictions.
* **Hard constraints**: `output_structure` (16 one-dimensional arrays with the right lengths), `binary_values`
  (finite 0/1), `not_positional_only` (visible, see below), `causal_prefix` (**hidden**, see below), `declared_unit`
  ("1").

### Causality (LEAK-1)
Whole stays are given to the agent, and the data have a strong end-of-record shortcut: septic records end 8-10 h after
the first positive label, non-septic local stays are 9-59 h (median 39), so "flag the last 12 hours of every stay"
reaches NU 0.68-0.71 on full stays against a reference of ~0.23-0.35, without reading a single vital sign. The stated
interface rule alone does not stop that, so it is enforced in three layers:
1. `not_positional_only` (visible tripwire, checkable on `y` alone): fails when the output is a function of position
   alone, in the frame "hour index from the start" or "distance to the end of the record" (`positional_only`): some
   position shared by >= 4 stays is flagged in every stay and no shared position is split between stays. A model that
   reads the vitals splits positions between septic and non-septic stays; all-zero output is not positional.
   Because it can only see *shared* positions, a length rule that mixes patient-specific information with the record
   end is not caught here, which is why layer 2 exists.
2. `causal_prefix` (hidden): a **truncation re-run**. `Episode.run_probes(y, trace, runner)` (generic Probe API,
   DESIGN 8.6) re-executes the *same graph* on a derived episode whose `load_eval_inputs` returns every stay cut to a
   hidden number of rows (40-80 % of the stay, >= 4 rows, always strictly shorter; deterministic in `pool_seed` and the
   stay id, `probe_keeps`). Causal predictions for hours 0..t cannot change when later rows are removed, so the check
   compares the prefix predictions with the full-stay predictions: fail when more than 1 % of the compared hours
   changed (`CAUSAL_TOL`). A graph that crashes or returns nothing on the shorter stays fails the probe. The cut lengths
   are not shown to the policy (the probe episode is never exposed). **The verdict is only computed when the runner
   calls `run_probes` before `Episode.evaluate`; until it is wired into `Solver._replay_eval` the constraint reports
   "not probed" and passes** (the runtime and solver are outside this adapter's remit; recipe in DESIGN 8.6).
3. Acceptance floor (above): the all-zero output can no longer be accepted.

The visible tool still exposes `n_hours` and the per-stay row counts. They are needed to shape `y` (array i has
`n_hours[i]` entries) and are derivable from the table anyway, so hiding them is not possible; the objective forbids
using them for the prediction and the two constraints above detect the practical ways of doing so. A length feature that
is combined with vitals is a genuine causal violation only when it uses the *total* length, which is exactly what the
truncation probe changes.
* **Calibration** (adapter sanity check, one seed): a regularized HistGradientBoosting model on LOCF values, 6-h deltas,
  6-h maxima and measurement counts, threshold tuned on a held-out quarter of the training stays: pooled NU id 0.509,
  ood 0.498 (reference 0.305 / 0.363); accepted id 4/4, ood 2/4, src 4/7, val 1/2. An unregularized class-balanced
  GBM scored *below* the reference (pooled id 0.09), so the reference is not trivial.

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 300, max_llm_items 64`.

## Caches
`<repo>/cache/tasks/FoR42/stays_<hash>.parquet` (all local stays parsed into one table; keyed by file names and sizes).

## Deviations from the historical protocol and open issues
* Exact historical sample ids / split rules were **not recovered**; these are rebuilt splits. Historical protocol:
  64 IID + 64 OOD items; here id = 64 stays, ood = 64 stays.
* Episodes are outcome-stratified (2 septic / 16) — enriched vs. the natural prevalence so that per-episode utility is
  defined; OOD uses all 8 septic hospital-B stays that are available locally.
* **Causality is enforced by a truncation re-run + a positional tripwire** (see "Causality"); the re-run needs the
  solver to call `Episode.run_probes` before `evaluate` (not wired yet: the hidden constraint passes as "not probed").
  Residual gaps: a small look-ahead (for example one hour ahead) flips only the last prefix hour of a few stays and
  stays below the 1 % tolerance, so it is not detected (its utility gain is correspondingly small); a graph with
  unseeded randomness can fail the probe spuriously (the tolerance absorbs a handful of flips).
* **A prefix / onset-anchored redesign was rejected**: cutting stays at onset-relative hours leaves an "end of prefix"
  shortcut and uniform cuts leave NU undefined for episodes with a non-septic-only prefix.
* **Data-team request** (optional): more hospital-B stays, in particular septic ones (training_setB has ~1,100 septic
  stays), would allow natural-prevalence OOD episodes and more than 8 distinct septic OOD stays.
