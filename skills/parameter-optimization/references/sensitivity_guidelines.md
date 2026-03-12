# Sensitivity Analysis Guidelines

## Overview

Sensitivity analysis quantifies how input parameter variations affect simulation outputs. It helps identify the most influential parameters for calibration and uncertainty reduction.

## Method Categories

| Category | Purpose | Computational Cost |
|----------|---------|-------------------|
| Local (OAT) | Derivative at a point | Low (d+1 evals) |
| Screening (Morris) | Rank parameters | Medium (r*(d+1) evals) |
| Global (Sobol) | Variance decomposition | High (N*(d+2) evals) |

---

## Local Sensitivity (One-At-a-Time)

### How It Works

Vary one parameter while holding others fixed at nominal values. Compute partial derivatives:

```
S_i = (dy/dx_i) * (x_i / y)  [normalized]
```

### When to Use

- Quick screening
- Nearly linear response
- Local behavior near operating point

### Limitations

- Misses parameter interactions
- Depends on chosen nominal point
- Invalid for nonlinear responses

---

## Morris Method (Elementary Effects)

### How It Works

Compute elementary effects by stepping through parameter space along random trajectories:

```
EE_i = [y(x + delta*e_i) - y(x)] / delta
```

Statistics computed:
- `mu*` (mean absolute EE): Overall importance
- `sigma` (std of EE): Nonlinearity/interactions

### Interpretation

| mu* | sigma | Interpretation |
|-----|-------|----------------|
| High | Low | Important, linear effect |
| High | High | Important, nonlinear or interacting |
| Low | Low | Unimportant |
| Low | High | Nonlinear but weak |

### Sample Requirements

- Trajectories `r`: 10-50 (typically 20)
- Levels `p`: 4-8
- Total evaluations: `r * (d + 1)`

### When to Use

- Moderate budgets
- Screening before detailed analysis
- Want to detect interactions

---

## Sobol Indices (Variance-Based)

### How It Works

Decompose output variance into contributions from each parameter and their interactions:

```
V(Y) = sum(V_i) + sum(V_ij) + ... + V_12...d
```

Indices:
- `S_i` (first-order): Main effect of parameter i
- `S_Ti` (total): Main + all interactions involving i

### Interpretation

| S_i | S_Ti | Interpretation |
|-----|------|----------------|
| ~0 | ~0 | Not influential |
| High | ~S_i | Mainly additive (linear) |
| Low | High | Important via interactions |

### Sample Requirements

For Saltelli estimator:
- Base samples `N`: 512, 1024, 2048
- Total evaluations: `N * (d + 2)`

| Dimension | N | Total Evals |
|-----------|---|-------------|
| 3 | 512 | 2560 |
| 5 | 1024 | 7168 |
| 10 | 2048 | 24576 |

### When to Use

- Quantitative variance attribution needed
- Sufficient budget for global analysis
- Nonlinear, interacting models

---

## Interpreting Rankings

### Score Thresholds

| Score Range | Interpretation |
|-------------|----------------|
| > 0.5 | Dominant parameter |
| 0.2 - 0.5 | Important parameter |
| 0.05 - 0.2 | Moderate influence |
| < 0.05 | Negligible |

### When Rankings Are Close

If top parameters have similar scores:
1. Check for interactions (S_Ti >> S_i)
2. Consider fixing less important parameters
3. Use higher sample sizes for better precision

### Red Flags

| Observation | Possible Cause |
|-------------|----------------|
| All scores near zero | Wrong output metric or insensitive region |
| Sum of S_i > 1 | Numerical error or strong negative correlations |
| S_Ti << S_i | Estimation error (impossible theoretically) |

---

## Visualization Recommendations

### Bar Charts

Plot parameters sorted by sensitivity score with confidence intervals.

```
kappa       ████████████████████ 0.52
mobility    ███████████ 0.28
W           ██████ 0.15
rho         ██ 0.05
```

### Interaction Heatmaps

For second-order indices S_ij, use heatmap with parameters on both axes.

### Scatter Plots

Plot output vs each input to visually confirm sensitivity rankings.

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Too few samples | Increase N; check confidence intervals |
| Ignoring interactions | Use total indices S_Ti, not just S_i |
| Wrong parameter ranges | Match realistic physical bounds |
| Correlated inputs | Use methods that handle correlations |
| Discrete parameters | Use Morris or specialized methods |

---

## Implementation Notes

The `sensitivity_summary.py` script in this skill:
- Takes pre-computed sensitivity scores as input
- Ranks parameters from most to least influential
- Flags if all sensitivities are very low
- Returns structured JSON for downstream use

For computing Sobol indices, use external tools:
- `SALib` (Python, comprehensive)
- `sensitivity` (R package)
- `UQlab` (MATLAB)
