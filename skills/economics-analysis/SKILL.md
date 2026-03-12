---
name: economics-analysis
description: Economic analysis including econometrics, causal inference, time series economics, game theory, welfare analysis, and economic modeling. Use when user works with economic data, regression analysis, instrumental variables, difference-in-differences, RDD, panel data, or economic theory. Triggers on "econometrics", "regression", "causal inference", "instrumental variable", "difference-in-differences", "panel data", "game theory", "supply demand", "GDP", "inflation", "economic model".
---

# Economics Analysis

Econometrics and economic modeling. Venv: `source /Users/zhangmingda/clawd/.venv/bin/activate`

## Causal Inference Methods

### Selection Guide
| Method | When to Use | Key Assumption |
|--------|------------|----------------|
| RCT | Can randomize treatment | Random assignment |
| IV (2SLS) | Endogeneity, have instrument | Exclusion restriction |
| DiD | Policy change, panel data | Parallel trends |
| RDD | Treatment at threshold | Continuity at cutoff |
| Matching/PSM | Observational, rich covariates | Selection on observables |
| Synthetic Control | Aggregate intervention, few treated | Parallel trends (weighted) |

### Difference-in-Differences
```python
import statsmodels.formula.api as smf

# Basic DiD
model = smf.ols('outcome ~ treated * post + C(unit) + C(time)', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['unit']})
print(model.summary())
# DiD estimate = coefficient on treated:post interaction
```

### Instrumental Variables (2SLS)
```python
from linearmodels.iv import IV2SLS

# Y = β₀ + β₁X + ε, where X is endogenous
# Z is the instrument
model = IV2SLS.from_formula('outcome ~ 1 + controls + [endogenous ~ instrument]', data=df)
result = model.fit(cov_type='robust')
print(result.summary)
```

### Regression Discontinuity
```python
# Local linear regression around cutoff
from sklearn.linear_model import LinearRegression

bandwidth = 5  # choose appropriately
cutoff = 0
left = df[(df['running'] >= cutoff - bandwidth) & (df['running'] < cutoff)]
right = df[(df['running'] >= cutoff) & (df['running'] <= cutoff + bandwidth)]

# Fit separate regressions
model_left = LinearRegression().fit(left[['running']], left['outcome'])
model_right = LinearRegression().fit(right[['running']], right['outcome'])

# RDD estimate
rdd_effect = model_right.predict([[cutoff]])[0] - model_left.predict([[cutoff]])[0]
```

## Panel Data

```python
from linearmodels.panel import PanelOLS, RandomEffects, BetweenOLS

df = df.set_index(['entity', 'time'])

# Fixed effects
fe = PanelOLS.from_formula('y ~ x1 + x2 + EntityEffects + TimeEffects', data=df)
fe_result = fe.fit(cov_type='clustered', cluster_entity=True)

# Random effects
re = RandomEffects.from_formula('y ~ x1 + x2', data=df)
re_result = re.fit()

# Hausman test: FE vs RE
# If significant → use FE
```

## Game Theory

```python
import numpy as np
from scipy.optimize import linprog

# Nash equilibrium (2-player, finite)
def find_nash_pure(payoff_A, payoff_B):
    """Find pure strategy Nash equilibria"""
    nash = []
    rows, cols = payoff_A.shape
    for i in range(rows):
        for j in range(cols):
            # Check if i is best response to j, and j is best response to i
            if payoff_A[i,j] == max(payoff_A[:,j]) and payoff_B[i,j] == max(payoff_B[i,:]):
                nash.append((i, j))
    return nash

# Example: Prisoner's Dilemma
A = np.array([[-1, -3], [0, -2]])  # Row player payoffs
B = np.array([[-1, 0], [-3, -2]])  # Column player payoffs
print(f"Nash equilibria: {find_nash_pure(A, B)}")
```

## Economic Data Sources

| Source | Data | Access |
|--------|------|--------|
| FRED (St. Louis Fed) | US macro data | `https://api.stlouisfed.org/fred/` |
| World Bank | Global development | `https://api.worldbank.org/v2/` |
| IMF | International finance | REST API |
| BLS | US labor statistics | REST API |
| OECD | OECD country data | REST API |
| Penn World Table | Cross-country GDP | Download |
| CNKI/CSMAR | Chinese economic data | Institutional access |

### FRED API
```bash
# Get GDP data (need API key)
curl -s "https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key=YOUR_KEY&file_type=json"
```

### World Bank API
```bash
curl -s "https://api.worldbank.org/v2/country/CHN/indicator/NY.GDP.MKTP.CD?format=json&per_page=20"
```

## Tips
- Always cluster standard errors at the treatment level
- Test parallel trends assumption for DiD
- Report first-stage F-statistic for IV (F > 10 rule of thumb)
- Use robust standard errors by default
- For Chinese economic research, consider CSMAR and CNKI databases
- Report economic significance alongside statistical significance
