# Aggregation Methods

Techniques for combining and analyzing results from simulation campaigns.

## Basic Statistics

### Scalar Metrics

For a single output metric across all runs:

| Statistic | Formula | Use Case |
|-----------|---------|----------|
| Min | `min(x)` | Best case (if minimizing) |
| Max | `max(x)` | Best case (if maximizing) |
| Mean | `sum(x)/n` | Expected value |
| Std | `sqrt(var(x))` | Spread/uncertainty |
| Median | `sorted(x)[n/2]` | Robust central tendency |

### Percentiles

For uncertainty quantification:
- P5: 5th percentile (lower bound)
- P50: Median
- P95: 95th percentile (upper bound)

```python
import numpy as np
p5, p50, p95 = np.percentile(values, [5, 50, 95])
```

## Multi-Objective Analysis

### Pareto Frontier

When optimizing multiple objectives (e.g., accuracy vs cost):

1. Identify non-dominated solutions
2. Plot Pareto frontier
3. Select based on trade-off preference

```python
def is_pareto_optimal(costs):
    """Return mask of Pareto-optimal points."""
    n = len(costs)
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if all(costs[j] <= costs[i]) and any(costs[j] < costs[i]):
                    is_optimal[i] = False
                    break
    return is_optimal
```

### Weighted Sum

Combine objectives into single scalar:
```
score = w1 * obj1 + w2 * obj2 + ...
```

Requires choosing weights (subjective).

## Sensitivity Analysis

### One-at-a-Time (OAT)

Simple but limited:
1. Vary one parameter, hold others at baseline
2. Compute output change
3. Repeat for each parameter

**Limitation**: Misses interactions.

### Sobol Indices

Global sensitivity analysis:
- First-order (Si): Main effect of parameter i
- Total-order (STi): Main + all interactions involving i

```
Si = Var(E[Y|Xi]) / Var(Y)
STi = 1 - Var(E[Y|X~i]) / Var(Y)
```

### Morris Screening

Efficient for many parameters:
1. Random trajectories through parameter space
2. Compute elementary effects (EE)
3. Mean(|EE|) indicates importance
4. Std(EE) indicates nonlinearity/interaction

## Interpolation and Surrogate Models

### Grid Data Interpolation

For grid sweeps, interpolate between points:

```python
from scipy.interpolate import RegularGridInterpolator

interp = RegularGridInterpolator((x_grid, y_grid), values)
result = interp([x_new, y_new])
```

### Scattered Data Interpolation

For LHS or irregular samples:

```python
from scipy.interpolate import RBFInterpolator

rbf = RBFInterpolator(points, values)
result = rbf(new_points)
```

### Gaussian Process Regression

Provides uncertainty estimates:

```python
from sklearn.gaussian_process import GaussianProcessRegressor

gp = GaussianProcessRegressor()
gp.fit(X, y)
y_pred, y_std = gp.predict(X_new, return_std=True)
```

## Result Validation

### Consistency Checks

Before aggregation, verify:
- [ ] All results loaded successfully
- [ ] No NaN/Inf in metrics
- [ ] Values within expected ranges
- [ ] Sufficient completed runs

### Outlier Detection

Identify and handle anomalous results:

```python
# IQR method
q1, q3 = np.percentile(values, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = (values < lower) | (values > upper)
```

**Options**:
1. Remove outliers from analysis
2. Investigate (may indicate bugs)
3. Use robust statistics (median, MAD)

## Export Formats

### CSV Table

Human-readable, Excel-compatible:
```csv
job_id,dt,kappa,objective
job_0000,0.001,0.1,0.532
job_0001,0.001,0.2,0.487
```

### JSON Summary

Machine-readable, structured:
```json
{
  "statistics": {"mean": 0.51, "std": 0.05},
  "best_run": {"job_id": "job_0015", "value": 0.41}
}
```

### HDF5 Dataset

For large numerical data:
- Efficient storage and access
- Supports compression
- Self-describing metadata

## Visualization Suggestions

| Data Type | Plot Type |
|-----------|-----------|
| 1D sweep | Line plot |
| 2D sweep | Heatmap/contour |
| 3D+ sweep | Parallel coordinates |
| Time series | Multi-line plot |
| Distribution | Histogram/violin |
| Correlation | Scatter matrix |
