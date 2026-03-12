---
name: biostatistics
description: Performs biostatistical analyses specialized for clinical and biomedical research including survival analysis, Kaplan-Meier estimation, Cox proportional hazards regression, longitudinal data modeling, and diagnostic test evaluation; trigger when users discuss clinical outcomes, survival curves, or biomedical study statistics.
---

## When to Trigger

Activate this skill when the user mentions:
- Survival analysis, time-to-event, censoring
- Kaplan-Meier curves, log-rank test, median survival
- Cox regression, proportional hazards, hazard ratio
- Longitudinal data, mixed-effects models, GEE
- Diagnostic accuracy, sensitivity, specificity, ROC/AUC
- Competing risks, Fine-Gray model, cumulative incidence
- Sample size for clinical endpoints, multiplicity adjustment
- Missing data in clinical studies, multiple imputation, MCAR/MAR/MNAR

## Step-by-Step Methodology

1. **Study design assessment** - Confirm study type (cohort, case-control, cross-sectional, RCT). Identify primary endpoint type (continuous, binary, time-to-event, count, ordinal). Determine if data is clustered or longitudinal.
2. **Survival analysis** - Define time origin, event definition, and censoring mechanism. Verify censoring is non-informative. Estimate survival curves with Kaplan-Meier method. Compare groups with log-rank test (or weighted variants: Wilcoxon, Tarone-Ware for non-proportional hazards).
3. **Cox regression** - Check proportional hazards assumption (Schoenfeld residuals, log-log plots). If violated, use time-varying coefficients, stratified Cox, or restricted mean survival time (RMST). Report hazard ratios with 95% CIs. Handle multiple covariates with purposeful selection or penalized regression.
4. **Competing risks** - When multiple event types exist, use cumulative incidence functions (not 1-KM). Apply Fine-Gray subdistribution hazard model or cause-specific hazard models. Report cumulative incidence at clinically relevant timepoints.
5. **Longitudinal analysis** - For repeated measures: linear or generalized mixed-effects models (random intercepts/slopes). Choose appropriate correlation structure. Handle dropout with pattern mixture models or joint models for longitudinal and survival data.
6. **Diagnostic test evaluation** - Compute sensitivity, specificity, PPV, NPV at defined cutoffs. Generate ROC curve and compute AUC with DeLong confidence intervals. For biomarker discovery, apply cross-validation to avoid overoptimism.
7. **Missing data handling** - Classify missingness mechanism (MCAR, MAR, MNAR). For MAR: multiple imputation (m >= 20 imputations, Rubin's rules for pooling). Conduct sensitivity analysis under MNAR assumptions.

## Key Databases and Tools

- **R survival / survminer** - Survival analysis packages
- **SAS PROC PHREG / LIFETEST** - Clinical biostatistics standard
- **STATA stcox / stcurve** - Survival modeling
- **R mice / Amelia** - Multiple imputation
- **pROC / cutpointr** - ROC analysis

## Output Format

- Kaplan-Meier curves with number-at-risk table, median survival with 95% CI.
- Cox model results as a table: variable, HR, 95% CI, p-value, with PH assumption test.
- Cumulative incidence curves for competing risks with event-specific estimates.
- ROC curves with AUC, optimal cutpoint, and sensitivity/specificity at that point.
- Missing data report: pattern, mechanism assessment, imputation method, sensitivity results.

## Quality Checklist

- [ ] Time origin and event definition clearly specified
- [ ] Censoring mechanism described and non-informative assumption justified
- [ ] Proportional hazards assumption tested and result reported
- [ ] Competing risks handled appropriately (not ignored)
- [ ] Multiple comparisons adjustment applied when needed
- [ ] Missing data mechanism assessed and appropriate method used
- [ ] Sample size adequate for number of covariates (EPV >= 10 for Cox)
- [ ] Effect estimates reported with confidence intervals, not just p-values
- [ ] Sensitivity analyses performed for key assumptions
