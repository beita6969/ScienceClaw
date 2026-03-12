---
name: statistics
description: # Statistical Analysis & Quality Control
---

# Statistical Analysis & Quality Control

## Overview
Comprehensive statistical methodology for scientific research. Covers test selection, assumption verification, power analysis, effect size reporting, and reporting standards.

## Test Selection Guide

| Data Type | Groups | Paired? | Normal? | Recommended Test |
|-----------|--------|---------|---------|-----------------|
| Continuous | 2 | No | Yes | Independent t-test |
| Continuous | 2 | No | No | Mann-Whitney U |
| Continuous | 2 | Yes | Yes | Paired t-test |
| Continuous | 2 | Yes | No | Wilcoxon signed-rank |
| Continuous | 3+ | No | Yes | One-way ANOVA + post hoc |
| Continuous | 3+ | No | No | Kruskal-Wallis + Dunn |
| Continuous | 3+ | Yes | Yes | Repeated measures ANOVA |
| Categorical | 2x2 | — | — | Chi-square / Fisher's exact |
| Time-to-event | 2+ | — | — | Log-rank + KM curves |
| Time-to-event | Adjusted | — | — | Cox proportional hazards |
| Continuous | Prediction | — | — | Linear/logistic regression |

## Assumption Checks
- **Normality**: Shapiro-Wilk (n < 50), Kolmogorov-Smirnov (n > 50), Q-Q plot visual
- **Homoscedasticity**: Levene's test, Bartlett's test
- **Independence**: study design review (not a statistical test)
- **Proportional hazards**: Schoenfeld residuals, log-log plot

## Multiple Comparison Correction
- **Bonferroni**: conservative, for few comparisons
- **Holm-Bonferroni**: step-down, less conservative than Bonferroni
- **FDR (Benjamini-Hochberg)**: for many comparisons (e.g., genomics)
- **Tukey HSD**: for all pairwise comparisons after ANOVA

## Effect Size Guidelines
| Measure | Small | Medium | Large |
|---------|-------|--------|-------|
| Cohen's d | 0.2 | 0.5 | 0.8 |
| Pearson r | 0.1 | 0.3 | 0.5 |
| Odds Ratio | 1.5 | 2.5 | 4.3 |
| R-squared | 0.02 | 0.13 | 0.26 |

## Reporting Standards
- Always report: test statistic, degrees of freedom, exact p-value, effect size, 95% CI
- Follow STROBE (observational), CONSORT (RCTs), PRISMA (reviews), ARRIVE (animal)
- Never say "trend toward significance" for p > 0.05
- Report non-significant results honestly
