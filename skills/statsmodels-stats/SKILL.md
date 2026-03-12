---
name: statsmodels-stats
description: "Statistical analysis via statsmodels. Use when: user asks for regression, hypothesis testing, or time series analysis. NOT for: machine learning models or deep learning."
metadata: { "openclaw": { "emoji": "📉", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-statsmodels", "kind": "uv", "package": "statsmodels pandas" }] } }
---

# Statsmodels Statistical Analysis

Statistical modeling, hypothesis testing, and time series analysis using statsmodels and pandas.

## When to Use

- Linear regression (OLS, GLS, WLS, robust)
- Logistic regression and generalized linear models
- Hypothesis testing (t-tests, ANOVA, chi-squared)
- Time series analysis (ARIMA, VAR, seasonal decomposition)
- Survival analysis and diagnostic plots

## When NOT to Use

- Machine learning classification/regression (use scikit-learn)
- Deep learning or neural networks (use PyTorch/TensorFlow)
- Simple descriptive statistics only (use scipy-analysis)

## OLS / GLS / WLS Regression

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.ols('y ~ x1 + x2 + x1:x2', data=df).fit()
print(model.summary())

# Matrix interface
X = sm.add_constant(df[['x1', 'x2']])
model = sm.OLS(df['y'], X).fit()

# WLS for heteroscedasticity
model_wls = sm.WLS(df['y'], X, weights=1.0/df['variance']).fit()

# Robust regression
model_rlm = sm.RLM(df['y'], X, M=sm.robust.norms.HuberT()).fit()
```

## Logistic Regression

```python
model = smf.logit('outcome ~ age + treatment', data=df).fit()
print(model.summary())
odds_ratios = np.exp(model.params)
conf_int = np.exp(model.conf_int())
mfx = model.get_margeff()
print(mfx.summary())
```

## Hypothesis Testing

```python
from statsmodels.stats.anova import anova_lm
from statsmodels.stats import weightstats, proportion

# t-test
t_stat, p_val, df_val = weightstats.ttest_ind(group_a, group_b)

# One-way ANOVA
model = smf.ols('value ~ C(group)', data=df).fit()
print(anova_lm(model, typ=2))

# Two-way ANOVA
model = smf.ols('value ~ C(factor_a) * C(factor_b)', data=df).fit()
print(anova_lm(model, typ=2))

# Proportion z-test
z_stat, p_val = proportion.proportions_ztest(count=[45, 60], nobs=[100, 120])
```

## Time Series Analysis

```python
# ARIMA
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(series, order=(p, d, q)).fit()
forecast = model.forecast(steps=12)

# Seasonal ARIMA (SARIMAX)
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
forecast = model.get_forecast(steps=24)

# VAR (vector autoregression)
from statsmodels.tsa.api import VAR
model = VAR(multivariate_df).fit(maxlags=5, ic='aic')

# Stationarity tests
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(series)
print(f'ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}')
```

## Survival Analysis

```python
from statsmodels.duration.hazard_regression import PHReg
model = PHReg(df['time'], df[['age', 'treatment']], status=df['event']).fit()
print(model.summary())
```

## Diagnostic Plots

```python
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig = sm.qqplot(model.resid, line='45')
fig.savefig('qqplot.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots()
sm.graphics.influence_plot(model, ax=ax)
fig.savefig('influence.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Heteroscedasticity test
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
```

## Statistical Rigor Standards

**Every statistical result MUST include:**
1. Test name and type
2. Test statistic value
3. p-value (exact, not "p < 0.05")
4. Effect size (Cohen's d, odds ratio, R-squared, etc.)
5. 95% confidence interval
6. Sample size (n per group)

**Before interpreting any test:**
- Verify assumptions (normality: Shapiro-Wilk; homoscedasticity: Levene/Breusch-Pagan; independence)
- If assumptions violated, use non-parametric alternatives or robust methods
- For multiple comparisons, apply FDR (Benjamini-Hochberg) or Bonferroni correction

**Reporting standards:**
- A significant p-value with a tiny effect size is NOT meaningful — always report both
- Distinguish correlation from causation explicitly
- Report negative results honestly — absence of effect is a finding, not a failure
- Never report p = 0.000; use scientific notation (e.g., p = 2.3e-7)

```python
# Template for proper statistical reporting
def report_ttest(group_a, group_b, label_a="Group A", label_b="Group B"):
    from scipy import stats
    import numpy as np
    t, p = stats.ttest_ind(group_a, group_b)
    d = (np.mean(group_a) - np.mean(group_b)) / np.sqrt((np.std(group_a)**2 + np.std(group_b)**2) / 2)
    ci = stats.t.interval(0.95, len(group_a)+len(group_b)-2,
                          loc=np.mean(group_a)-np.mean(group_b),
                          scale=stats.sem(np.concatenate([group_a, group_b])))
    print(f"Independent t-test: t({len(group_a)+len(group_b)-2}) = {t:.3f}, "
          f"p = {p:.2e}, Cohen's d = {d:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], "
          f"n = {len(group_a)} vs {len(group_b)}")
```

## Best Practices

1. Always check model assumptions before interpreting results.
2. Use `model.summary()` for comprehensive fit statistics.
3. Report confidence intervals alongside point estimates.
4. For time series, verify stationarity (ADF/KPSS) before fitting ARIMA.
5. Use information criteria (AIC/BIC) for model selection.
6. Use robust standard errors (`model.get_robustcov_results()`) when heteroscedasticity is present.
7. **NEVER fabricate statistical results.** Every number must come from actual computation on real data.
