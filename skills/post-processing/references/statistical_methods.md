# Statistical Methods Reference

## Basic Statistics

### Computed Quantities

| Statistic | Formula | Notes |
|-----------|---------|-------|
| Mean | μ = Σx / n | Arithmetic average |
| Variance | σ² = Σ(x-μ)² / (n-1) | Sample variance (Bessel's correction) |
| Std Dev | σ = √variance | Sample standard deviation |
| Min/Max | min(x), max(x) | Extreme values |
| Range | max - min | Spread of data |

### Percentiles

Computed using linear interpolation:
- P0 (min), P25 (Q1), P50 (median), P75 (Q3), P100 (max)
- Custom percentiles available via `--percentiles`

### Higher Moments

**Skewness (γ₁):**
- γ₁ = E[(X-μ)³] / σ³
- γ₁ > 0: Right-skewed (tail extends right)
- γ₁ < 0: Left-skewed (tail extends left)
- γ₁ ≈ 0: Symmetric

**Excess Kurtosis (γ₂):**
- γ₂ = E[(X-μ)⁴] / σ⁴ - 3
- γ₂ > 0: Heavy tails (leptokurtic)
- γ₂ < 0: Light tails (platykurtic)
- γ₂ ≈ 0: Normal-like tails

## Histogram Analysis

### Binning

Default: 50 bins with automatic range detection.

**Bin width:** (max - min) / n_bins

**Density:** count / (total × bin_width)
- Integrates to 1 over the range
- Useful for comparing distributions

### Distribution Detection

The `statistical_analyzer.py` attempts to classify distributions:

| Type | Characteristics | Typical Cause |
|------|-----------------|---------------|
| Uniform | Flat distribution | Random initial conditions |
| Unimodal symmetric | Single peak, centered | Normal distribution |
| Unimodal skewed | Single peak, off-center | Bounded quantities |
| Bimodal | Two peaks | Two-phase systems |
| Multimodal | Multiple peaks | Multi-phase systems |

## Time Series Analysis

### Trend Detection

**Monotonicity check:**
- Count increasing vs decreasing steps
- Report dominant trend direction
- Count violations (direction changes)

**Steady State Detection:**
- Window-based analysis (default: 10 points)
- Relative variation: max|x - mean| / |mean|
- Threshold: typically 1e-6 for convergence

### Convergence Rate

For sequences converging to a target:

1. Compute errors: eᵢ = |xᵢ - target|
2. Fit log(e) vs iteration (linear regression)
3. Rate = exp(slope)
4. Iterations per decade = -1 / log₁₀(rate)

**Classification:**
| Rate | Type | Interpretation |
|------|------|----------------|
| > 0.99 | Stalled | Not converging |
| 0.9-0.99 | Slow | May need tuning |
| 0.5-0.9 | Linear | Normal convergence |
| < 0.5 | Fast | Superlinear convergence |

### Oscillation Detection

- Zero crossings of detrended signal
- Amplitude estimation
- Frequency estimate: crossings / (2 × length)

## Spatial Statistics

### Variation Along Axes

For 2D fields:
- Row-wise means: average along x for each y
- Column-wise means: average along y for each x
- Variation: std of these means

**Uniform criteria:** std < 0.01 × |mean|

## Interpretation Guidelines

### Phase-Field Systems

| Distribution | Physical Meaning |
|--------------|------------------|
| Bimodal near 0,1 | Two-phase equilibrium |
| Unimodal near 0.5 | Mixed/interfacial region |
| Uniform | Disordered/initial state |

### Conservation Checks

For conserved quantities (e.g., mass):
- Compute mean over time
- Check std/mean < tolerance (e.g., 1e-10)
- Report any drift

### Residual Analysis

For iterative solvers:
- Should decrease monotonically
- Rate indicates convergence speed
- Plateau indicates stalled solver
