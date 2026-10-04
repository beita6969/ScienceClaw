# FoR44 Human society — ACIC 2016 causal inference challenge (response-SD normalized RMSE, lower is better)

Adapter: `scienceclaw/bench/tasks/for44_acic.py` (`Adapter = ACIC2016Adapter`).
Status: **available with a reduced plan on the data team's delivered roles** (2026-09-28). The splits are the data
team's `source` / `validation` / `evaluation` role folders (64 whole simulations each); `ood` is **empty by default**
(see "OOD decision"). `available()` requires only one full episode (16 simulations) per non-empty split plus the dev
sub-pool, and prints the exact shortfall otherwise. The adapter is cross-checked against the data team's R outputs
(OLS coefficient equal to R `lm(y ~ .)` within 1e-8; true ATE and response SD identical; see the test suite).

## Dataset
| field | value |
|---|---|
| name / version | ACIC 2016 (`aciccomp2016` R package, github vdorie/aciccomp commit `282d26659b2d3d6fd060dde6d32feeb8f1e8ab5a`), simulations from the unmodified `dgp_2016` under R 4.4.3 (data-team environment `environments/for44-r`) |
| URL / license | https://github.com/vdorie/aciccomp — GPL (>= 2) per `2016/DESCRIPTION` |
| local path | `<DATA_ROOT>/for44-acic-2016/reconstructed_v1/` (`{source,validation,evaluation}/acic2016-pPP-rRRR/{observed,oracle}.csv.gz`, `covariates.csv.gz`, `selection.tsv`, `baseline-per-simulation.csv`); upstream package in `upstream/2016/` |
| size | 192 simulations x 4,802 subjects (covariates 4,802 x 58 shared by design); 71 of 77 scenarios represented |
| note | the data team reports that 3 of the 20 historical `testData` fixtures (scenarios 13/16/19) differ from the current generator output; the current pinned generator is the declared data definition |

The adapter also accepts the official competition release (`<p>/zymu_<s>.csv` + `x.csv`) and `p<p>_s<s>.csv` exports
with `mu.0`/`mu.1` columns (no role folders; whole settings are then hashed to roles in the data team's 39:19:19
proportions). No code change is needed to add further simulations.

## Items, pools, splits
* **Item** = one whole simulated dataset (scenario p, replication r), id `acic2016-pPP-rRRR`. Estimand:
  `SATT = mean over treated units of (mu1 − mu0)` (conditional on the sample's covariates), the estimand targeted
  in the ACIC 2016 competition (Dorie et al., 2019).
* **Episode** = 16 whole simulations (block `e` of a seeded permutation of the split's pool). Items never repeat inside
  a split, so an item belongs to at most one episode and one split.
* **Role -> split mapping** (design of the data team: `source` = development/source pool, `validation` = model
  selection, `evaluation` = held-out IID test):

| data-team role | split | default pool (items) | 16-item episodes |
|---|---|---|---|
| `source` | `src` (+ `dev`) | 48 (+ 16 dev) | 3 |
| `validation` | `val` | 64 | 4 |
| `evaluation` | `id` | 64 | 4 |
| — | `ood` | 0 (empty) | 0 |

  The roles are **setting-disjoint** (whole scenario p per role: 33 / 19 / 19 settings actually sampled), so val and id
  simulations come from data-generating settings never seen in src/dev. The adapter verifies this at `available()`
  time and refuses (with a message) roles that share a setting.
* **Dev sub-pool** (visible dev signal via `score_dev`, `_dev_evaluate` is None): the `n_dev` (default 16) source
  simulations with the smallest salted hash rank (`partition_seed = 20260928`), removed from `src`. Dev is
  item-disjoint from every episode of every split and setting-disjoint from val/id; it shares 6 settings with the
  remaining src items (replicates of the same settings).
* The 77-row `parameters_2016` table is embedded in the adapter (verified identical to the RData).
* If a plan asks for more episodes than the pool holds, `draw_episodes` raises `PoolExhausted` (no cycling, so an
  item can never reappear). At paper scale (7 source rounds x 16 = 112 items) `src` is exhausted (48 items -> 3
  episodes); use fewer rounds / fewer items per episode, or add simulations.

## OOD decision: `ood` is empty by default
* The delivery defines **no OOD role**, and the data team's manifest states that the setting-held-out partition is
  "not cross-dataset OOD": every setting uses the same 4,802 covariate rows, so there is no covariate, population or
  dataset shift. The data team's `DATASETS.md` (data root) says items without a verifiable OOD split keep OOD empty.
* The `step` treatment-assignment scenarios (9, 17, 48–77) would be the only DGP-side shift, but the delivered roles put
  step simulations in all three roles (source 28, validation 27, evaluation 23 of 64), so under the delivered roles a
  step-assignment shift is **not** held out; deriving OOD from the evaluation role would be arbitrary, and it is also
  not the paper's design.
* Consequently `build_episodes("ood", ...)` returns `[]`; `SplitPlan` reports the empty OOD split as a shortfall
  warning. Paper claims about OOD for FoR44 can therefore not be reproduced on this delivery.
* **Opt-in proxy** (`Adapter(ood_mode="step_assignment")`, off by default and not a delivered role): all 78 step
  simulations of every role form the `ood` pool and are removed from src/val/id/dev, so the assignment mechanism is not
  seen elsewhere; `lineage.ood_kind = "proxy_within_dataset"`. Sizes on the delivery: dev 16 / src 20 / val 37 / id 41 /
  ood 78 (episodes: src 1, val 2, id 2, ood 4). This overrides the data team's role assignment for those simulations
  and is documented as an adapter-chosen proxy only; it needs the owner's sign-off before any paper claim uses it.

## Leakage risks (read before using the splits)
1. Replicates of one setting share their DGP and the covariates, so simulations inside one split (and inside one
   episode) are not independent samples; only **settings** are disjoint across src/val/id, and items are disjoint
   everywhere (`SplitPlan` checks lineage overlap).
2. Dev and src come from the same source settings (dev is a visible dev signal for the source distribution only).
3. All settings share the 4,802 covariate rows. A solver that memorizes the shared covariate matrix or learns a
   covariate -> effect map from src carries that knowledge unchanged to val/id; held-out settings measure
   generalization to nearby DGPs (ACIC settings are neighbours: same response model with different overlap or
   alignment), not to a different dataset.
4. `oracle.csv.gz` (mu0/mu1) is never exposed as a tool; only z and y are visible. The reference (OLS) only uses
   visible data.
5. With `ood_mode="step_assignment"` the opt-in OOD pool is a proxy within one dataset, not cross-dataset OOD.

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_covariates` | the shared 4,802 × 58 covariate table |
| `load_eval_inputs` | `treatment` (n×4802, 0/1) and `outcome` (n×4802) of the episode's datasets |
| `load_dev_inputs` / `score_dev(dev_estimates)` | z/y of the dev datasets (default 16) and the normalized RMSE of SATT estimates for them + the reference's value (visible dev signal; `_dev_evaluate` is None) |

## Metric, reference, acceptance (D_V)
* Metric: `sqrt(mean_items(((tau_hat − SATT) / sd(y))^2))`, `sd(y)` = sample SD (ddof 1) of the observed outcome of that dataset. Pooled over all items of all episodes (invalid outputs use the reference estimates).
* Reference: OLS `y ~ 1 + z + X` (categoricals one-hot), coefficient of `z` (the `dgp_2016` documentation's own example).
* Acceptance: `RMSE <= 0.9 × RMSE_ref`.
* Hard constraints: `output_shape` (1-D length = items), `finite`, `effect_scale` (`|tau_hat_j| <= 5 sd(y_j)`).
* Sanity values on the delivery (16 items/episode, seed 3): OLS reference RMSE 0.097 (src) / 0.103 (val) /
  0.146 (id), normalized score 1.0; naive difference of means 0.163 / 0.205 / 0.193 (normalized score 0.60 / 0.50 /
  0.76); all-zero estimates ≈ 0.69–0.76 (score 0.14–0.19); the hidden truth scores 0 (score clipped to 10, accepted).

## Domain library (`scilib/causal.py`)
The episode objective ends with `scilib.describe("causal")`. The module gives code nodes: `design_matrix` (mixed-type
covariate table -> float matrix; strings/categories one-hot, NaN filled, constants dropped, z-scored; robust to the
pandas 3 `str` dtype), outcome regression estimators of the SATT/ATE (`regression_adjustment`, `outcome_imputation`,
`t_learner`, `x_learner` with ridge / HistGradientBoosting / LightGBM / forest learners or a callable), propensity models
(`propensity_scores`, cross-fitted, clipped) with `ipw_effect` and cross-fitted `aipw_effect` (influence-function SEs),
`overlap_report`, `get_method`/`METHODS` (name -> estimator), the single entry point
`estimate_effects(cov, treatment, outcome, methods=..., ...)` (one estimate per dataset, several methods combined by
mean/median, non-finite values fall back to OLS, clipped to +-4 sd(y)), and two validation helpers that use only the
visible data: `semi_synthetic_check` (plug-in simulation from one dataset with a known effect) and `validate_methods`
(semi-synthetic RMSE plus a permuted-treatment placebo per method). It reads no files, needs no network, is deterministic
(HistGradientBoosting fits run on one OpenMP thread: identical predictions, but no thread-barrier stalls on a shared CPU)
and costs roughly 0.3-1.5 s per method and dataset on an idle machine. Defaults of `estimate_effects`:
`methods=("impute_lgbm", "xlearner_lgbm")`, `estimand="att"`, `cap=4`, `combine="mean"`. The interface text names no
reference or margin (F5). Tests: `tests/test_scilib_causal.py` (synthetic confounded data) and the objective assertion in
`tests/test_task_FoR44.py`.

* **Library-level accuracy** (64 source-role simulations = the 48 src + 16 dev ones; the library never reads the truth,
  only the measurement below does; RMSE of the SATT error in units of sd(y), pooled): difference of means 0.217,
  OLS `y ~ z + X` 0.117, ridge imputation 0.088, extra-trees imputation 0.082, random-forest imputation 0.050,
  HistGradientBoosting imputation 0.025, LightGBM imputation 0.026, LightGBM T-learner 0.026, LightGBM X-learner 0.018,
  LightGBM AIPW 0.069, logistic IPW 0.225 (max |err| 1.5). Default mean of LightGBM imputation and X-learner: 0.020. The
  boosted outcome-model estimators are 4-6 times more accurate than OLS; propensity-only weighting is not competitive on
  this benchmark. This selection used src/dev only; val/id truth was not used for any choice.
* **BART / BCF (stochtree 0.4.5, `bart_effect` / `bcf_effect`, registered as `bart` / `bcf` when `stochtree` is importable)**:
  the BART authors' sampler, fitted per dataset on its own observed data only (no pretrained weights, no external data).
  A properly converged chain is required (`num_gfr=0`, burn-in 300, 1,000 draws): a GFR warm start with 10-20 iterations
  or a 100-draw chain is unconverged and is several times worse (first 12 source-role datasets: 0.070 vs 0.008). Cost on
  one thread: about 12 s (BART) / 30 s (BCF) per 4,802 x 80 dataset, so a 16-dataset node needs ~8 minutes for BCF and
  must be split under the 600 s node limit. RMSE/sd(y): dev+src (64; selection) BCF 0.0143, joint BART 0.0161, default
  0.0198; id (64; one look) BCF 0.0193, joint BART 0.0207, default 0.0227 (paired bootstrap on id: BCF - default
  -0.0035, 95% CI [-0.0090, +0.0027]; BCF + X-learner -0.0035 [-0.0065, -0.0001]). Details: `reports/below_peers_rootcause/f44.md`.
* **Real episodes through a hand-written node** that only calls `causal.estimate_effects` (src n=3, 16 items each; the
  reference is the OLS above): primary 0.0174 / 0.0211 / 0.0185 against the
  reference 0.1036 / 0.129 / 0.0971 (accepted 3/3, norm 5.96 / 6.12 / 5.24; dev 0.022 vs reference 0.132), about 60 s per
  episode for 32 datasets (eval + dev) on a loaded machine.
* **27B agent A_0** (2 src episodes): stock (no library) `probe_local27b_a0` 1 src episode: pass
  (0.038, norm 3.32, 11 steps, 58k tokens, stopped on a policy error); `probe_pass10_n3` 3 src episodes: 2/3 pass (0.043 norm 1.86, 9 steps,
  40k tokens; 0.023 norm 5.25, 10 steps, 43k; the third failed at 1.04, norm 0.12, wall budget, 132k tokens) -> with
  the library (`runs/probe_for44_scilib`, same two episodes 776400c7-00/-01) pass on both: 0.0192 (norm 6.52, 8 steps, 43k tokens) and 0.0361
  (norm 2.24, 8 steps, 34k tokens); both graphs call `causal.estimate_effects(..., estimand="att", combine="median")`
  with a list of methods. Sample size is small (2 episodes), so the end-to-end pass-rate claim is only indicative.
* Wall-clock is dominated by machine load (development machine load 15-80). An early version whose HistGradientBoosting and
  LightGBM fits used 2 OpenMP threads took 300-500 s per episode (one node hit the 600 s limit) under that load; the fits
  now run on one thread (same predictions), which cut this to about 60 s.

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 600 (16 fits on 4,802 × ~60), max_llm_items 32`.

## Deviations / notes
* Exact historical simulation ids were **not recovered**; rebuilt splits. The paper's metric name "response-SD normalized RMSE" is implemented as the RMSE over datasets of SATT errors divided by each dataset's outcome SD. Its scale is consistent with the paper's values (0.018–0.021; the OLS reference gives ≈ 0.08–0.09 normalized absolute ATE error on the data team's simulations), whereas the data team's alternative per-subject ITE-RMSE / SD is ≈ 0.7–0.8 for OLS, an order of magnitude above the paper's range. The exact historical definition could not be verified.
* Using `mu` (CATT-style SATT) rather than realized `y1 − y0`; both are available in the counterfactual files and the choice is a one-line change in `_Sim.satt` if the team prefers the realized version.
* Reduced plan: the paper-scale FoR44 plan is not reachable with 64 simulations per role; `available()` only requires one full 16-item episode per non-empty split plus the dev sub-pool. The `MIN_EPISODES` constant in the adapter holds the reduced minimum.
