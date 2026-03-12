# Comparison Metrics Reference

## Error Metrics

### L1 Error (Relative Mean Absolute Error)

**Formula:** L1 = Σ|sᵢ - rᵢ| / Σ|rᵢ|

**Properties:**
- Robust to outliers (compared to L2)
- Equal weighting to all points
- Range: [0, ∞), 0 is perfect match

**Use when:**
- Outliers should not dominate
- All errors are equally important
- Data may have occasional spikes

### L2 Error (Relative Root Mean Square Error)

**Formula:** L2 = √[Σ(sᵢ - rᵢ)² / Σrᵢ²]

**Properties:**
- Standard norm for numerical analysis
- Penalizes large errors more than small ones
- Corresponds to energy-like measures

**Use when:**
- Large errors are more significant
- Smoothness is important
- Standard in numerical convergence studies

### L∞ Error (Maximum Relative Error)

**Formula:** L∞ = max|sᵢ - rᵢ| / max|rᵢ|

**Properties:**
- Captures worst-case error
- Very sensitive to outliers
- Strict accuracy measure

**Use when:**
- Maximum error must be bounded
- Any large error is unacceptable
- Strict tolerance requirements

### RMSE (Root Mean Square Error)

**Formula:** RMSE = √[Σ(sᵢ - rᵢ)² / n]

**Properties:**
- Absolute error measure (same units as data)
- Standard for regression analysis
- Not normalized by reference

**Use when:**
- Absolute error magnitude matters
- Comparing models on same dataset
- Physical interpretation needed

### MAE (Mean Absolute Error)

**Formula:** MAE = Σ|sᵢ - rᵢ| / n

**Properties:**
- Absolute average error
- Robust to outliers (vs RMSE)
- Linear weighting

**Use when:**
- Simple average error wanted
- Outliers should not dominate
- Easy interpretation needed

### Maximum Difference

**Formula:** max_diff = max|sᵢ - rᵢ|

**Properties:**
- Absolute worst-case error
- In original data units
- Very conservative metric

**Use when:**
- Need to know worst case
- Tolerance checking
- Strict validation requirements

## Correlation Metrics

### Pearson Correlation

**Formula:** r = Σ[(sᵢ - μs)(rᵢ - μr)] / (σs × σr × n)

**Properties:**
- Range: [-1, 1]
- r = 1: Perfect positive correlation
- r = 0: No linear correlation
- r = -1: Perfect negative correlation

**Use when:**
- Testing linear relationship
- Scale-independent comparison
- Trend matching is important

### R² (Coefficient of Determination)

**Formula:** R² = 1 - Σ(rᵢ - sᵢ)² / Σ(rᵢ - μr)²

**Properties:**
- Range: (-∞, 1], 1 is perfect
- Fraction of variance explained
- Negative means worse than mean prediction

**Use when:**
- Assessing predictive power
- Comparing model quality
- Regression analysis

## Interpretation Guidelines

### Error Thresholds

| Metric | Excellent | Good | Moderate | Poor |
|--------|-----------|------|----------|------|
| L1, L2, L∞ | < 1% | 1-5% | 5-10% | > 10% |
| RMSE, MAE | Context-dependent | | | |
| Correlation | > 0.99 | 0.95-0.99 | 0.9-0.95 | < 0.9 |
| R² | > 0.99 | 0.95-0.99 | 0.9-0.95 | < 0.9 |

### Metric Selection Guide

| Comparison Type | Recommended Metric |
|-----------------|-------------------|
| Numerical convergence | L2 error |
| Experimental validation | RMSE or MAE |
| Maximum tolerance | L∞ error or max_diff |
| Trend agreement | Correlation |
| Model quality | R² |
| Robust comparison | L1 error |

## Interpolation

When simulation and reference have different coordinates:

**Linear interpolation:**
- Enabled with `--interpolate` flag
- Reference interpolated to simulation coordinates
- Extrapolation uses boundary values

**Caution:**
- Interpolation introduces error
- Best when simulation has finer resolution
- May smooth sharp features

## Common Issues

### Length Mismatch
- Scripts truncate to shorter length
- Use `--interpolate` for different grids
- Warning issued if mismatch detected

### Scale Differences
- L1, L2, L∞ normalize by reference
- RMSE, MAE are absolute
- Check data ranges before comparing

### Zero Reference Values
- Division issues for relative metrics
- Script handles gracefully
- Consider absolute metrics instead

### Noisy Data
- L1 error more robust than L2
- Consider smoothing before comparison
- Report uncertainty in interpretation
