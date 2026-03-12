# Grid Convergence Index (GCI) Guidelines

## Overview

The Grid Convergence Index (GCI) is a standardized method for reporting
discretization uncertainty in CFD and numerical simulations. It was developed by
Patrick Roache and is recommended by ASME V&V 20 and AIAA standards.

## Three-Grid GCI Procedure

### Requirements

- Three systematically refined grids with spacings h1 < h2 < h3
- Constant or near-constant refinement ratios r21 = h2/h1, r32 = h3/h2
- Recommended: r >= 1.3 (to ensure measurable differences)
- Monotone convergence (no oscillation in solution values)

### Step-by-Step Calculation

1. **Compute refinement ratios**:

       r21 = h2 / h1
       r32 = h3 / h2

2. **Compute observed order** p:

       p = |ln|e32/e21|| / ln(r21)

   where e32 = f3 - f2 and e21 = f2 - f1.

   For non-uniform ratios, an iterative procedure using:

       p = |ln|e32/e21| + ln((r21^p - s) / (r32^p - s))| / ln(r21)

   where s = sign(e32/e21). This reduces to the simple formula when r21 = r32.

3. **Compute GCI for fine grid**:

       GCI_fine = Fs * |e21/f1| / (r21^p - 1)

   where Fs is the safety factor.

4. **Compute GCI for coarse grid**:

       GCI_coarse = Fs * |e32/f2| / (r32^p - 1)

5. **Check asymptotic ratio**:

       AR = GCI_coarse / (r21^p * GCI_fine)

   If AR is approximately 1.0 (within 10%), the grids are in the asymptotic
   range and the GCI is reliable.

6. **Richardson extrapolated value**:

       f_extrap = f1 + (f1 - f2) / (r21^p - 1)

## Safety Factors

| Scenario | Safety Factor Fs | Rationale |
|----------|-----------------|-----------|
| 3+ grids with observed order | 1.25 | Order verified, lower uncertainty |
| 2 grids with assumed order | 3.0 | Order not verified, higher uncertainty |
| Oscillatory convergence | N/A | GCI not applicable |

The factor of 1.25 is analogous to a 95% confidence interval for well-behaved
convergence data.

## ASME V&V 20 Standard

The ASME V&V 20 standard (Verification and Validation in Computational Fluid
Dynamics and Heat Transfer) recommends:

- Using at least 3 systematically refined grids
- Reporting GCI with the fine-grid solution
- Checking the asymptotic ratio
- Documenting the observed convergence order
- Reporting the Richardson-extrapolated value as the best estimate

## Practical Guidelines

### Grid Design

- Use constant refinement ratio across all directions
- Recommended ratio: r = 1.5 to 2.0
- Avoid r < 1.3 (differences may be in round-off noise)
- Avoid r > 3.0 (large jumps may skip pre-asymptotic behavior)

### Common Issues

1. **Observed order much higher than expected**: Possible superconvergence or
   error cancellation. Verify with more grid levels.

2. **Observed order much lower than expected**: Solution may not be in the
   asymptotic range. Use finer grids.

3. **Negative observed order**: Solution is diverging. Check for coding errors,
   boundary condition issues, or inadequate resolution.

4. **Oscillatory convergence**: The GCI method does not apply. Consider using
   bounding approaches or the range of solutions as the uncertainty.

5. **Very small GCI (< 0.1%)**: Solution may be grid-independent already, or
   the quantity of interest is insensitive to grid refinement.

### Reporting

When reporting GCI results, include:

- The three grid spacings and corresponding solution values
- The observed convergence order
- The GCI value for the fine grid (as a percentage)
- The Richardson-extrapolated value
- The asymptotic ratio
- Whether the solution is in the asymptotic range

### Example Report Format

    Grid spacings: h1=0.01, h2=0.02, h3=0.04
    Refinement ratio: r = 2.0
    Solution values: f1=1.0008, f2=1.0032, f3=1.0128
    Observed order: p = 2.00
    GCI_fine = 0.027%
    Richardson extrapolated value: 1.00000
    Asymptotic ratio: 1.000
    Conclusion: Solution is in asymptotic range; GCI is reliable.
