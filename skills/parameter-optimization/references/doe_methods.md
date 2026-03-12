# Design of Experiments (DOE) Methods

## Overview

Design of Experiments (DOE) creates sample points in parameter space to efficiently explore simulation behavior. The goal is to maximize information gained per simulation run.

## Method Comparison

| Method | Sample Count | Space Coverage | Best Dimension | Deterministic |
|--------|--------------|----------------|----------------|---------------|
| LHS | User-defined | Good | 3-20 | No (random) |
| Sobol | User-defined | Excellent | 2-15 | Yes |
| Factorial | k^d (levels^dim) | Complete | 1-3 | Yes |

---

## Latin Hypercube Sampling (LHS)

### How It Works

LHS divides each parameter range into `n` equal intervals, then places exactly one sample in each interval per dimension. This ensures no two samples share the same row or column in any 2D projection.

```
Example: 5 samples in 2D

    1.0 |     x     |     |     | x   |
    0.8 |  x  |     |     |     |     |
    0.6 |     |     |     | x   |     |
    0.4 |     |     | x   |     |     |
    0.2 |     | x   |     |     |     |
        +-----+-----+-----+-----+-----+
          0.2   0.4   0.6   0.8   1.0
```

### When to Use

- General parameter exploration
- Moderate dimensions (3-20 parameters)
- Unknown response surface shape
- Limited simulation budget

### When to Avoid

- Need exact corner/edge coverage
- Very low dimensions where factorial is feasible
- Reproducibility required (use fixed seed)

### Sample Size Recommendations

| Dimension | Minimum Samples | Recommended |
|-----------|-----------------|-------------|
| 2-3 | 10 | 20-30 |
| 4-6 | 20 | 40-60 |
| 7-10 | 30 | 60-100 |
| 11-20 | 50 | 100-200 |

---

## Sobol Sequences (Quasi-Random)

### How It Works

Sobol sequences are low-discrepancy sequences that fill space more uniformly than random sampling. Points are generated deterministically using bit operations on direction numbers.

```
Example: Sobol vs Random in 2D (16 points)

Sobol (uniform fill)        Random (clusters/gaps)
+---+---+---+---+          +---+---+---+---+
| x |   | x |   |          |   | x |   | x |
+---+---+---+---+          +---+---+---+---+
|   | x |   | x |          | x |   |   |   |
+---+---+---+---+          +---+---+---+---+
| x |   | x |   |          |   | x | x | x |
+---+---+---+---+          +---+---+---+---+
|   | x |   | x |          |   |   | x |   |
+---+---+---+---+          +---+---+---+---+
```

### When to Use

- Sensitivity analysis (Sobol indices)
- Need uniform coverage guarantees
- Reproducible experiments
- Sequential sampling (can add points incrementally)

### When to Avoid

- Very high dimensions (>15, curse of dimensionality)
- Need stratified random sampling

### Sample Size Recommendations

For Sobol sensitivity analysis, use `N * (d + 2)` samples where:
- `N` = base sample size (64, 128, 256, 512, 1024)
- `d` = number of parameters

| Dimension | Base N | Total Samples |
|-----------|--------|---------------|
| 3 | 64 | 320 |
| 5 | 128 | 896 |
| 10 | 256 | 3072 |

---

## Full Factorial Design

### How It Works

Factorial designs test all combinations of discrete parameter levels. For `k` levels across `d` dimensions, this produces `k^d` samples.

```
Example: 3 levels, 2 dimensions = 9 samples

    High  | x   x   x |
          |           |
    Med   | x   x   x |
          |           |
    Low   | x   x   x |
          +-----------+
            L   M   H
```

### When to Use

- Low dimensions (1-3 parameters)
- Need exact corner coverage
- Testing parameter interactions
- Screening designs

### When to Avoid

- High dimensions (exponential growth)
- Continuous parameters with smooth response
- Limited budget

### Sample Count Growth

| Dimension | 2 Levels | 3 Levels | 5 Levels |
|-----------|----------|----------|----------|
| 2 | 4 | 9 | 25 |
| 3 | 8 | 27 | 125 |
| 4 | 16 | 81 | 625 |
| 5 | 32 | 243 | 3125 |

---

## Decision Flowchart

```
START
  |
  v
Is d <= 3 AND need corner coverage?
  |
  +-- YES --> FACTORIAL
  |
  +-- NO --> Is sensitivity analysis the goal?
               |
               +-- YES --> SOBOL
               |
               +-- NO --> LHS
```

## Implementation Notes

The `doe_generator.py` script in this skill:
- Uses standard library only (no scipy/numpy required)
- LHS: True Latin Hypercube with random permutations
- Sobol: Simplified quasi-random (for full Sobol, use scipy.stats.qmc)
- Factorial: Full grid with level interpolation
