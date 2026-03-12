---
name: clinical
description: # Clinical Research
---

# Clinical Research

## Overview
Clinical study design, statistical analysis, and regulatory compliance for medical research.

## Study Designs
| Design | Level of Evidence | Best For |
|--------|------------------|----------|
| RCT | I | Treatment efficacy |
| Cohort (prospective) | II | Risk factors, prognosis |
| Cohort (retrospective) | III | Exposure-outcome associations |
| Case-control | III | Rare diseases, risk factors |
| Cross-sectional | IV | Prevalence, correlations |
| Case report/series | V | Novel observations |

## Common Analyses
- **Survival analysis**: Kaplan-Meier curves, Log-rank test, Cox regression
- **Diagnostic accuracy**: Sensitivity, specificity, ROC curve, AUC
- **Meta-analysis**: Fixed/random effects, forest plot, heterogeneity (I-squared)
- **Propensity score matching**: For observational study confounding
- **Nomogram**: Predictive model visualization for clinical use

## Sample Size Calculation
- Define: alpha (usually 0.05), power (usually 0.80), effect size, outcome type
- Tools: G*Power, R `pwr` package, Python `statsmodels`
- Report: formula used, assumptions, expected dropout rate

## Regulatory Compliance
- **IRB/Ethics committee**: Required for all human subjects research
- **Informed consent**: Written, voluntary, comprehensive
- **ClinicalTrials.gov**: Registration before enrollment (ICMJE requirement)
- **GDPR/HIPAA**: Data privacy for patient information
- **GCP (Good Clinical Practice)**: ICH E6 guidelines

## Reporting Guidelines
| Guideline | Study Type |
|-----------|-----------|
| CONSORT | Randomized controlled trials |
| STROBE | Observational studies |
| PRISMA | Systematic reviews / meta-analyses |
| STARD | Diagnostic accuracy |
| TRIPOD | Prediction models |
| SPIRIT | Study protocols |
