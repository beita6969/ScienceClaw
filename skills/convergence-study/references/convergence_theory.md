# Convergence Theory

## Formal Convergence Order

A numerical method has **order p** if the discretization error satisfies:

    e(h) = f(h) - f_exact = C * h^p + O(h^{p+1})

where h is the grid spacing (or timestep), C is a constant, and f_exact is the
exact solution.

## Observed Convergence Order

Given solutions on three grids with spacings h1 < h2 < h3 and uniform
refinement ratio r = h2/h1 = h3/h2, the observed order is:

    p_obs = log(|f3 - f2| / |f2 - f1|) / log(r)

For non-uniform refinement ratios, the general formula uses consecutive pairs
and the local ratio.

### Log-Log Analysis

On a log-log plot of error vs. spacing, a convergent solution shows a straight
line with slope equal to the convergence order. Deviations indicate:

- **Steeper slope at coarse grids**: Pre-asymptotic regime (higher-order terms
  still contribute)
- **Flattening at fine grids**: Round-off error or iterative error dominance
- **Irregular slope**: Possible implementation error or singularity

## Asymptotic Range

A solution is in the **asymptotic range** when the leading-order error term
dominates:

    e(h) ~ C * h^p

Indicators of being in the asymptotic range:

1. Observed order matches expected order (within ~10%)
2. Consecutive observed-order estimates are consistent
3. Richardson extrapolation gives consistent results from different grid pairs

### Pre-Asymptotic Behavior

When the solution is NOT in the asymptotic range:

- Higher-order terms C2*h^{p+1} + C3*h^{p+2} + ... are significant
- Observed order varies between consecutive grid pairs (>50% variation)
- Richardson extrapolation may be unreliable
- Resolution: use finer grids until asymptotic behavior is observed

## Richardson Extrapolation

Given two solutions f1 (fine, spacing h1) and f2 (coarse, spacing h2) with
refinement ratio r = h2/h1 and assumed order p:

    f_extrapolated = f1 + (f1 - f2) / (r^p - 1)

This removes the leading-order error term, giving an estimate with error
O(h^{p+1}).

### Error Estimate

The discretization error estimate for the fine-grid solution is:

    error ~ |f1 - f2| / (r^p - 1)

### Reliability

Richardson extrapolation is most reliable when:

- The solution is in the asymptotic range
- The assumed order matches the actual convergence order
- The refinement ratio is moderate (r = 1.5 to 3.0)

It becomes unreliable when:

- Too few grid levels to verify the order
- Oscillatory convergence (sign changes in differences)
- Observed order differs significantly from assumed order

## Monotone vs. Oscillatory Convergence

**Monotone convergence**: Solution values approach the exact solution from one
side. Differences (f3-f2) and (f2-f1) have the same sign.

**Oscillatory convergence**: Solution values alternate above and below the exact
solution. Differences have opposite signs. Standard Richardson extrapolation and
GCI do not apply.

## Manufactured Solutions

For verification, the Method of Manufactured Solutions (MMS) provides:

1. Choose an exact solution f_exact(x,t)
2. Substitute into the PDE to compute the source term
3. Solve numerically with the computed source
4. Compare numerical and exact solutions

This guarantees the exact solution is known, enabling precise convergence
studies without approximation of the reference solution.
