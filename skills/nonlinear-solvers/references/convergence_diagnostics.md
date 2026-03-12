# Convergence Diagnostics

Guide to analyzing and diagnosing convergence behavior of nonlinear solvers.

## Convergence Types

### Quadratic Convergence

**Definition:**
```
||e_{k+1}|| ≤ C ||e_k||²
```

**Log-linear plot behavior:**
```
log||r_k|| vs k:

    |
 0  |*
    | *
-2  |  *
    |   *
-4  |    *
    |     **
-8  |       ****
    +------------- k
    0  2  4  6  8

Residual drops faster each iteration
```

**Characteristics:**
- Number of correct digits doubles each iteration
- Typical of Newton's method near solution
- Requires good initial guess and accurate Jacobian

**Expected behavior:**
| Iteration | Residual | Digits |
|-----------|----------|--------|
| 0 | 10⁻¹ | 1 |
| 1 | 10⁻² | 2 |
| 2 | 10⁻⁴ | 4 |
| 3 | 10⁻⁸ | 8 |
| 4 | 10⁻¹⁶ | 16 |

### Superlinear Convergence

**Definition:**
```
||e_{k+1}|| ≤ C_k ||e_k||  where C_k → 0
```

**Log-linear plot behavior:**
```
log||r_k|| vs k:

    |
 0  |*
    | *
-2  |  *
    |   *
-4  |    *
    |     *
-6  |      *
    |        *
-9  |          *
    +------------- k
    0  2  4  6  8

Steady acceleration
```

**Characteristics:**
- Faster than linear, slower than quadratic
- Typical of quasi-Newton methods (BFGS, Broyden)
- Usually order 1 < p < 2

### Linear Convergence

**Definition:**
```
||e_{k+1}|| ≤ ρ ||e_k||  where 0 < ρ < 1 constant
```

**Log-linear plot behavior:**
```
log||r_k|| vs k:

    |
 0  |*
    | *
-1  |  *
    |   *
-2  |    *
    |     *
-3  |      *
    |       *
-4  |        *
    +------------- k
    0  2  4  6  8

Constant slope (straight line on log scale)
```

**Rate classification:**
| Rate ρ | Quality | Iterations for 10⁻⁸ |
|--------|---------|---------------------|
| < 0.1 | Fast | < 10 |
| 0.1 - 0.5 | Good | 10-40 |
| 0.5 - 0.9 | Slow | 40-200 |
| > 0.9 | Very slow | > 200 |

### Sublinear Convergence

**Definition:**
```
||e_{k+1}|| ≤ ||e_k|| / k^α  for some α > 0
```

**Log-linear plot behavior:**
```
log||r_k|| vs k:

    |
 0  |*
    |*
-0.5| *
    |  *
-1  |   *
    |    **
-1.5|      ***
    |         ****
-2  |             *****
    +------------------- k
    0    10   20   30

Very slow, diminishing returns
```

**Characteristics:**
- Common in ill-conditioned problems
- May indicate near-singularity
- Often needs reformulation or preconditioning

### Stagnation

**Pattern:**
```
log||r_k|| vs k:

    |
 0  |*
    | *
-2  |  *
    |   *
-3  |    ***************
    |
    |
    +------------------- k
    0    10   20   30

Residual stops decreasing
```

**Common causes:**
1. Tolerance of inner solve too loose
2. Near-singular Jacobian
3. Loss of precision (round-off)
4. Wrong solution branch
5. Inconsistent constraints

### Divergence

**Pattern:**
```
log||r_k|| vs k:

    |              *
    |            *
    |          *
    |        *
 0  |*     *
    | *  *
-2  |  **
    |
    +------------------- k
    0    5    10

Residual increases
```

**Common causes:**
1. Step too large (need line search/trust region)
2. Bad initial guess
3. Jacobian inaccurate or wrong sign
4. Problem has no solution
5. Numerical overflow

## Diagnostic Procedures

### Step 1: Plot Residual History

Always start by plotting log||r_k|| vs iteration k.

**What to look for:**
```
Pattern          → Diagnosis
─────────────────────────────
Straight line    → Linear convergence (estimate ρ from slope)
Curving down     → Superlinear/quadratic
Curving up       → Sublinear or approaching stagnation
Flat             → Stagnated
Increasing       → Diverging
Oscillating      → Step size issues or saddle point
```

### Step 2: Compute Convergence Rate

**For linear convergence:**
```python
# Estimate linear convergence rate
rates = [r[k+1] / r[k] for k in range(len(r)-1)]
avg_rate = geometric_mean(rates)
```

**For superlinear/quadratic:**
```python
# Look at log-log behavior
log_r = [log(r_k) for r_k in r]
# For quadratic: log(r_{k+1}) ≈ 2 * log(r_k) + const
ratios = [log_r[k+1] / log_r[k] for k in range(len(r)-2)]
# ratios ≈ 2 suggests quadratic
```

### Step 3: Check for Common Issues

**Oscillation detection:**
```python
# Count sign changes in residual differences
oscillations = sum(1 for k in range(len(r)-2)
                   if (r[k+1]-r[k]) * (r[k+2]-r[k+1]) < 0)
if oscillations > len(r) / 3:
    print("Oscillating - check step size or Jacobian")
```

**Stagnation detection:**
```python
# Check if recent residuals are nearly constant
recent = r[-5:]
rel_change = (max(recent) - min(recent)) / max(recent)
if rel_change < 0.01:
    print("Stagnated - check preconditioner or Jacobian accuracy")
```

## Rate Estimation Methods

### Ratio Method (Linear Rate)

```
ρ ≈ r_{k+1} / r_k
```

Average over several iterations:
```
ρ ≈ (r_n / r_0)^(1/n)
```

### Log-Log Method (Order Estimation)

For convergence order p where ||e_{k+1}|| ≈ C ||e_k||^p:

```
log||e_{k+1}|| ≈ log(C) + p * log||e_k||
```

Estimate p from slope of log||e_{k+1}|| vs log||e_k||:
```
p ≈ (log||e_{k+1}|| - log||e_k||) / (log||e_k|| - log||e_{k-1}||)
```

### Using Residuals as Error Proxy

When exact error unknown, use residual ratios:
```
p ≈ log(r_{k+1}/r_k) / log(r_k/r_{k-1})
```

Note: This assumes ||r|| ~ ||e||, which holds near the solution.

## Troubleshooting Table

### Slow Linear Convergence (ρ > 0.5)

| Cause | Indicator | Fix |
|-------|-----------|-----|
| Poor preconditioner | High Krylov iterations | Better preconditioner |
| Ill-conditioned problem | Large condition number | Scaling, better formulation |
| Inexact Newton too loose | Inner residual high | Tighten forcing sequence |
| Far from solution | Large initial residual | Better initial guess |

### Stagnation

| Cause | Indicator | Fix |
|-------|-----------|-----|
| Inner solve tolerance | Residual stuck at ηr_0 | Decrease η |
| Near-singular Jacobian | Condition number huge | Regularization |
| Round-off limitation | Residual ≈ machine ε × scale | Problem solved |
| Wrong formulation | Constraints violated | Check problem setup |

### Divergence

| Cause | Indicator | Fix |
|-------|-----------|-----|
| No globalization | First step increases r | Add line search |
| Bad Jacobian | Compare to finite diff | Fix Jacobian code |
| Wrong sign | Jacobian has wrong sign | Check derivatives |
| No solution | Physical insight | Reformulate problem |

### Oscillation

| Cause | Indicator | Fix |
|-------|-----------|-----|
| Overshoot | Step size > optimal | Reduce step, add damping |
| Saddle point | Jacobian indefinite | Trust region, different direction |
| Coupled oscillation | Physics coupling | Under-relaxation |

## Practical Convergence Criteria

### Standard Criteria

```
Absolute: ||f(x_k)|| < τ_a
Relative: ||f(x_k)|| < τ_r ||f(x_0)||
Step:     ||x_{k+1} - x_k|| < τ_x (1 + ||x_k||)
Combined: Absolute OR Relative
```

### Recommended Tolerances

| Problem Type | τ_a | τ_r |
|--------------|-----|-----|
| Engineering | 1e-6 | 1e-4 |
| Scientific | 1e-10 | 1e-8 |
| Inner solve | 1e-3 × outer | 1e-2 |

### Safeguards

```
Max iterations: Prevent infinite loops
Min step:       ||δ|| > 1e-14 × ||x|| (progress check)
Function value: For optimization, F decreases
Gradient:       For optimization, ||∇F|| small
```

## Convergence Monitoring Example

### Healthy Newton Convergence

```
Iter    ||r||       ||r||/||r_0||   Rate
----    -----       -------------   ----
 0      1.0e+00     1.0e+00         -
 1      2.3e-01     2.3e-01         0.23
 2      1.1e-02     1.1e-02         0.05
 3      3.2e-05     3.2e-05         0.003
 4      8.1e-11     8.1e-11         3e-6
 5      1.2e-16     1.2e-16         converged

Analysis: Quadratic convergence established after iteration 2
```

### Problematic Convergence

```
Iter    ||r||       ||r||/||r_0||   Rate    Alert
----    -----       -------------   ----    -----
 0      1.0e+00     1.0e+00         -
 1      8.5e-01     8.5e-01         0.85    slow
 2      7.3e-01     7.3e-01         0.86    slow
 3      6.2e-01     6.2e-01         0.85    STAGNATING
 4      5.4e-01     5.4e-01         0.87
 5      4.6e-01     4.6e-01         0.85

Analysis: Linear convergence with rate ≈ 0.85
Action: Improve preconditioner, check Jacobian accuracy
```

### Divergent Case

```
Iter    ||r||       ||r||/||r_0||   Alert
----    -----       -------------   -----
 0      1.0e+00     1.0e+00
 1      3.2e+00     3.2e+00         INCREASING
 2      1.5e+01     1.5e+01         DIVERGING
 3      8.7e+02     8.7e+02         ABORT

Analysis: Divergence from iteration 1
Action: Add line search, check Jacobian, try smaller step
```
