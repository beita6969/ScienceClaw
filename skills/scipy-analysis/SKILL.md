---
name: scipy-analysis
description: "Scientific computing and statistical analysis with SciPy, NumPy, and pandas. Use when: (1) statistical hypothesis testing, (2) optimization problems, (3) signal processing, (4) numerical integration, (5) data manipulation and analysis. NOT for: symbolic math (use sympy-math), machine learning (use sklearn directly), or visualization (use matplotlib-viz)."
metadata: { "openclaw": { "emoji": "📊", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-scipy", "kind": "uv", "package": "scipy numpy pandas" }] } }
---

# SciPy Analysis

Scientific computing and statistical analysis using SciPy, NumPy, and pandas.

## Statistical Hypothesis Testing

```python
from scipy import stats
import numpy as np

# Two-sample t-test (Welch's)
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

# Paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

# One-way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3)

# Chi-square test of independence
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Mann-Whitney U (non-parametric)
u_stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')

# Correlation: Pearson, Spearman, Kendall
r, p = stats.pearsonr(x, y)
rho, p = stats.spearmanr(x, y)
tau, p = stats.kendalltau(x, y)

# Normality: Shapiro-Wilk (small samples) or KS test
w_stat, p_value = stats.shapiro(data)
ks_stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
```

## Pandas Data Analysis

```python
import pandas as pd
df.describe()                                                    # summary stats
grouped = df.groupby('category')['value'].agg(['mean', 'std', 'count'])
pivot = pd.pivot_table(df, values='measurement', index='group',
                       columns='condition', aggfunc='mean')
```

## NumPy Operations

```python
import numpy as np
eigenvalues, eigenvectors = np.linalg.eig(A)
solution = np.linalg.solve(A, b)
mean = np.mean(arr, axis=0)
std = np.std(arr, ddof=1)  # sample std dev
percentiles = np.percentile(arr, [25, 50, 75])
```

## Optimization

```python
from scipy.optimize import minimize, curve_fit, brentq, fsolve

# Function minimization
result = minimize(lambda x: (x[0]-1)**2 + (x[1]-2.5)**2, x0=[0,0], method='Nelder-Mead')

# Curve fitting
def model(x, a, b, c): return a * np.exp(-b * x) + c
popt, pcov = curve_fit(model, xdata, ydata, p0=[1, 0.1, 0])
perr = np.sqrt(np.diag(pcov))  # parameter standard errors

# Root finding
root = brentq(lambda x: x**3 - 2*x - 5, 1, 3)
solution = fsolve(lambda v: [v[0]+v[1]-4, v[0]*v[1]-3], [1, 1])
```

## Signal Processing

```python
from scipy import signal
b, a = signal.butter(N=4, Wn=[0.1, 0.4], btype='band')
filtered = signal.filtfilt(b, a, data)
freqs, psd = signal.welch(data, fs=sampling_rate, nperseg=256)
peaks, props = signal.find_peaks(data, height=0.5, distance=10)
```

## Numerical Integration and ODEs

```python
from scipy import integrate
result, error = integrate.quad(lambda x: np.exp(-x**2), 0, np.inf)
result, error = integrate.dblquad(lambda y, x: x*y, 0, 1, 0, 1)

def dydt(t, y): return -0.5 * y
sol = integrate.solve_ivp(dydt, [0, 10], [1.0], t_eval=np.linspace(0, 10, 100))
```

## Data Cleaning

```python
df = df.dropna(subset=['key_column'])
df['value'] = df['value'].fillna(df['value'].median())
# Outlier removal (IQR)
Q1, Q3 = df['value'].quantile(0.25), df['value'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df['value'] >= Q1 - 1.5*IQR) & (df['value'] <= Q3 + 1.5*IQR)]
```

## Best Practices

1. Set random seeds (`np.random.seed(42)`) for reproducibility.
2. Use `ddof=1` for sample standard deviation.
3. Check normality assumptions before parametric tests.
4. Report effect sizes alongside p-values.
5. Prefer vectorized NumPy operations over Python loops.
6. Use `float64` for numerical stability.
