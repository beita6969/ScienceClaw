---
name: statistical-testing
description: Advanced statistical testing including hypothesis testing, Bayesian analysis, survival analysis, time series, multivariate methods, and meta-analysis. Use when user needs specific statistical tests beyond basic EDA, power analysis, Bayesian inference, survival curves, time series forecasting, or meta-analysis. Triggers on "hypothesis test", "Bayesian", "survival analysis", "time series", "meta-analysis", "bootstrap", "permutation test", "mixed model", "structural equation".
---

# Statistical Testing

Advanced statistical methods for scientific research. Venv: `source /Users/zhangmingda/clawd/.venv/bin/activate`

## Bayesian Analysis

```python
# Simple Bayesian estimation (conjugate priors)
import numpy as np
from scipy import stats

# Beta-Binomial (proportions)
# Prior: Beta(alpha_prior, beta_prior), Data: k successes in n trials
alpha_prior, beta_prior = 1, 1  # uniform prior
k, n = 45, 100
alpha_post = alpha_prior + k
beta_post = beta_prior + (n - k)
posterior = stats.beta(alpha_post, beta_post)
print(f"Posterior mean: {posterior.mean():.3f}")
print(f"95% credible interval: {posterior.ppf([0.025, 0.975])}")

# Bayes Factor (BF10) approximation for t-test
# Use the BayesFactor approach or JZS prior
```

## Survival Analysis

```python
# Kaplan-Meier and Cox regression
# pip install lifelines
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

kmf = KaplanMeierFitter()
kmf.fit(durations=df['time'], event_observed=df['event'], label='Overall')
kmf.plot_survival_function()

# Compare groups
results = logrank_test(df[df['group']==0]['time'], df[df['group']==1]['time'],
                       df[df['group']==0]['event'], df[df['group']==1]['event'])
print(f"Log-rank test: χ²={results.test_statistic:.2f}, p={results.p_value:.4f}")

# Cox proportional hazards
cph = CoxPHFitter()
cph.fit(df[['time', 'event', 'age', 'treatment']], 'time', 'event')
cph.print_summary()
```

## Time Series

```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA

# Stationarity test
result = adfuller(ts)
print(f"ADF statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")

# ARIMA
model = ARIMA(ts, order=(p, d, q)).fit()
forecast = model.forecast(steps=12)

# Seasonal decomposition
decomp = sm.tsa.seasonal_decompose(ts, period=12)
decomp.plot()
```

## Bootstrap & Permutation

```python
# Bootstrap confidence interval
def bootstrap_ci(data, stat_func=np.mean, n_boot=10000, ci=0.95):
    boot_stats = [stat_func(np.random.choice(data, size=len(data), replace=True))
                  for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return np.percentile(boot_stats, [alpha*100, (1-alpha)*100])

# Permutation test
def permutation_test(group1, group2, n_perm=10000):
    observed = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:len(group1)]) - np.mean(combined[len(group1):])
        if abs(perm_diff) >= abs(observed):
            count += 1
    return count / n_perm
```

## Multiple Comparison Corrections

```python
from statsmodels.stats.multitest import multipletests

# Methods: bonferroni, holm, fdr_bh (Benjamini-Hochberg), fdr_by
reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
```

## Effect Sizes

| Test | Effect Size | Small | Medium | Large |
|------|------------|-------|--------|-------|
| t-test | Cohen's d | 0.2 | 0.5 | 0.8 |
| ANOVA | η² (eta-squared) | 0.01 | 0.06 | 0.14 |
| Correlation | r | 0.1 | 0.3 | 0.5 |
| Chi-square | Cramér's V | 0.1 | 0.3 | 0.5 |
| Regression | R² | 0.02 | 0.13 | 0.26 |

## Meta-Analysis

```python
# Fixed-effects and random-effects meta-analysis
# Inverse-variance weighted
def meta_analysis(effects, variances, method='random'):
    weights = 1 / np.array(variances)
    pooled_fixed = np.sum(weights * effects) / np.sum(weights)
    
    if method == 'random':
        Q = np.sum(weights * (effects - pooled_fixed)**2)
        k = len(effects)
        C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
        tau2 = max(0, (Q - (k-1)) / C)
        weights_re = 1 / (np.array(variances) + tau2)
        pooled = np.sum(weights_re * effects) / np.sum(weights_re)
        se = np.sqrt(1 / np.sum(weights_re))
    else:
        pooled = pooled_fixed
        se = np.sqrt(1 / np.sum(weights))
    
    ci = (pooled - 1.96*se, pooled + 1.96*se)
    return pooled, se, ci
```

## Reporting Standards
- APA: F(df1, df2) = X.XX, p = .XXX, η² = .XX
- Always include: test statistic, df, p-value, effect size, CI
- Use exact p-values (not p < .05) unless p < .001
- Report Bayesian results as BF₁₀ with interpretation scale
