---
name: code-execution
description: "Execute scientific Python code for computation, data analysis, simulation, and verification. Use when: (1) running statistical analyses, (2) numerical computation, (3) data processing pipelines, (4) verifying calculations, (5) running simulations. NOT for: literature search (use literature-search), writing papers (use paper-writing), or non-computational tasks."
metadata: { "openclaw": { "emoji": "💻", "requires": { "bins": ["python3"] } } }
---

# Code Execution (Meta Skill)

Execute scientific Python code for computation, analysis, simulation, and
verification of results.

## Common Imports

```python
import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate, signal
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, csv, sys
from collections import Counter, defaultdict
```

## Pattern 1: Statistical Analysis

```python
import numpy as np
from scipy import stats

data_a, data_b = np.array([...]), np.array([...])
print(f"Group A: mean={np.mean(data_a):.4f}, std={np.std(data_a, ddof=1):.4f}, n={len(data_a)}")
print(f"Group B: mean={np.mean(data_b):.4f}, std={np.std(data_b, ddof=1):.4f}, n={len(data_b)}")

t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
print(f"Welch's t-test: t={t_stat:.4f}, p={p_value:.6f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((np.std(data_a, ddof=1)**2 + np.std(data_b, ddof=1)**2) / 2)
print(f"Cohen's d: {(np.mean(data_a) - np.mean(data_b)) / pooled_std:.4f}")
```

## Pattern 2: Numerical Computation

```python
from scipy import integrate, optimize

result, error = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
print(f"Integral result: {result:.6f} (error: {error:.2e})")

solution = optimize.fsolve(lambda v: [v[0]**2+v[1]**2-4, v[0]-v[1]-1], [1, 0])
print(f"Solution: x={solution[0]:.4f}, y={solution[1]:.4f}")
```

## Pattern 3: Data Processing

```python
import pandas as pd
from io import StringIO

df = pd.read_csv(StringIO("col1,col2\n1,2\n3,4"))
df = df.dropna()
df['computed'] = df['col1'] * df['col2']
print(df.groupby('col1').agg({'col2': ['mean', 'std', 'count']}).round(4).to_string())
```

## Pattern 4: Monte Carlo Simulation

```python
np.random.seed(42)
n = 100000
x, y = np.random.uniform(-1, 1, n), np.random.uniform(-1, 1, n)
pi_est = 4 * np.sum(x**2 + y**2 <= 1) / n
print(f"Pi estimate: {pi_est:.6f} (error: {abs(pi_est - np.pi):.6f})")
```

## Pattern 5: Verification

```python
# Verify claimed results against raw data
actual_mean = np.mean(data)
se = np.std(data, ddof=1) / np.sqrt(len(data))
ci = (actual_mean - 1.96*se, actual_mean + 1.96*se)
print(f"Mean: {actual_mean:.2f}, 95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
print(f"Verification: {'PASS' if abs(claimed - actual_mean) < 0.5 else 'FAIL'}")
```

## Error Handling

```python
try:
    result = perform_computation(data)
except (ValueError, np.linalg.LinAlgError) as e:
    print(f"Error: {e}")
except MemoryError:
    print("Data too large. Consider chunked processing.")
```

## Output Formatting

```python
print("=" * 50)
print(f"RESULTS | n={n} | mean={mean:.4f} | p={p:.6f}")
print("=" * 50)
```

## Sandbox Constraints

1. No network access -- data must be provided inline or on disk.
2. No persistent state -- each execution is independent.
3. Memory limits -- use chunked processing for large data.
4. Time limits -- reduce iterations for long simulations.

## Best Practices

1. Set random seeds and report library versions for reproducibility.
2. Use `np.float64` and `np.log1p` for numerical stability.
3. Prefer vectorized NumPy over Python loops.
4. Validate input shapes, types, and ranges before computation.
5. Report appropriate significant figures; do not over-report precision.
6. Check statistical test assumptions before applying parametric methods.
7. Apply Bonferroni/FDR correction for multiple comparisons.
