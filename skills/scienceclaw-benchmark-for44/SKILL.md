---
name: scienceclaw-benchmark-for44
description: "Use when a task gives a covariate table plus, for each of many simulated observational datasets, a binary treatment vector and an outcome vector, and asks for one average treatment effect on the treated (SATT) per dataset without access to the counterfactuals, scored by RMSE relative to the outcome standard deviation (ACIC 2016 causal inference challenge; corresponds to FoR44 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Treatment-effect estimation on ACIC-2016-style data

**Task.** Input: a covariate table (mixed numeric and categorical; the same units in every dataset) and, per dataset j, a 0/1 treatment `z_j` and outcome `y_j` (arrays `(n_datasets, n_units)`). Deliverable: a finite 1-D array, one estimate per dataset, in outcome units. The estimand is SATT, the mean of `mu1 - mu0` over that dataset's treated units; counterfactuals are never given.

**Quality.** `sqrt(mean_j(((tau_hat_j - SATT_j) / sd(y_j))^2))` with `sd(y_j)` the sample SD (ddof 1) of the observed outcome; lower is better. Baseline: the OLS coefficient of `z` in `y ~ 1 + z + X`; the raw difference in means is confounded.

**Tools (`from scilib import causal`, no pretrained weights).**
- `estimate_effects(cov, treatment, outcome, methods=("impute_lgbm", "xlearner_lgbm"), estimand="att", cap=4.0, combine="mean")`: one estimate per dataset; non-finite values fall back to OLS, estimates are clipped to +-`cap`*sd(y); `combine` is `mean` or `median`.
- `causal.METHODS`: `diff_means`, `regression_adjustment`, `<impute|tlearner|xlearner|aipw>_<ridge|hgb|hgb3|lgbm|rf|extra|ridge_hgb>`, `ipw_logit`, `ipw_hgb`, plus `bart`/`bcf` when `stochtree` is installed (slow: tens of seconds per dataset of ~4,800 units x 80 columns, so split across nodes; keep the default chain, burn-in 300 / draws 1000, since much shorter chains can be unconverged). Operators `causal_effect_estimate`, `causal_effect_bayesian_forest`.
- `design_matrix(cov)`, `overlap_report(X, z)`, `get_method(name, estimand)(X, z, y)` returning `Effect(tau, se)`.
- Visible-data validation: `validate_methods(cov, treatment, outcome, methods)` / `semi_synthetic_check`: plug-in simulation with a known effect plus a permuted-treatment placebo. It favours methods close to its surface learner; use it to rank, not to certify.

**Routes.** Outcome-model estimators (imputation or X-learner with boosted trees; BART/BCF) are the usual strong choice on ACIC-style data; a mean or median of two is robust. Propensity-only weighting is typically weak.

**Rules.**
- Use only covariates, `z` and `y`; never read oracle, `mu0`/`mu1` or truth files.
- Match the estimand (`att`). Check overlap before trusting weights; an estimate of several sd(y) signals failure.
- Datasets share units and covariates, so they are not independent samples.
- Output: 1-D, length = number of datasets, finite.
