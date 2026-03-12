# Nonlinear Solver Method Catalog

Comprehensive catalog of nonlinear solver methods with parameters and guidance.

## Newton-Type Methods

### Newton's Method (Full Newton)

**Update formula:**
```
x_{k+1} = x_k - J(x_k)^{-1} f(x_k)
```

**Requirements:**
- Jacobian J(x) = ∂f/∂x available
- J(x) nonsingular

**Convergence:**
- Quadratic near solution: ||e_{k+1}|| ≤ C ||e_k||²
- Requires good initial guess

**Parameters:**
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| max_iter | 50-100 | Depends on problem |
| tol_abs | 1e-10 | Absolute residual tolerance |
| tol_rel | 1e-8 | Relative tolerance |

**When to use:**
- Small to medium problems (n < 5000)
- Jacobian cheap to compute and factor
- Quadratic convergence needed

**When to avoid:**
- Large problems (use Newton-Krylov)
- Expensive Jacobian (use modified Newton)
- Far from solution (add globalization)

### Modified Newton

**Update formula:**
```
x_{k+1} = x_k - J(x_0)^{-1} f(x_k)   (frozen Jacobian)
x_{k+1} = x_k - J(x_m)^{-1} f(x_k)   (recompute every m steps)
```

**Convergence:**
- Linear when Jacobian frozen
- Faster than Picard if Jacobian is good

**Parameters:**
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| update_frequency | 3-5 | Iterations between Jacobian updates |
| max_frozen_steps | 10 | Force update after this many |

**When to use:**
- Jacobian expensive to compute
- Jacobian changes slowly
- Initial iterations where exact Jacobian less critical

### Inexact Newton

**Idea:** Solve J(x_k) δ = -f(x_k) approximately

**Inner solve criterion:**
```
||J(x_k) δ_k + f(x_k)|| ≤ η_k ||f(x_k)||
```

**Forcing sequence η_k:**
| Choice | η_k | Convergence |
|--------|-----|-------------|
| Constant | η = 0.1 | Linear |
| Type 1 | min(0.5, ||f_k||) | Superlinear |
| Type 2 | min(0.5, ||f_k||/||f_{k-1}||) | Quadratic (if appropriate) |
| Eisenstat-Walker | Safeguarded version | Practical choice |

**When to use:**
- Large problems where exact solve too expensive
- Combined with Krylov inner solver (Newton-Krylov)

### Newton-Krylov Methods

**Idea:** Use Krylov method (GMRES, BiCGSTAB) for inner linear solve

**Key methods:**
| Inner solver | Best for |
|--------------|----------|
| GMRES | General nonsymmetric Jacobian |
| BiCGSTAB | When GMRES memory is issue |
| CG | Symmetric positive definite Jacobian |
| MINRES | Symmetric indefinite |

**Parameters:**
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| inner_tol | 1e-3 to 1e-6 | Decreases as outer converges |
| max_krylov | 30-100 | GMRES restart for memory |
| preconditioner | ILU, AMG | Critical for performance |

**Jacobian-free variant:**
```
J(x) v ≈ (f(x + εv) - f(x)) / ε
```
where ε ≈ √(machine_eps) × ||x|| / ||v||

**When to use:**
- Large-scale problems (n > 10000)
- Jacobian too expensive to form/store
- Sparse structure can be exploited

## Quasi-Newton Methods

### BFGS (Broyden-Fletcher-Goldfarb-Shanno)

**Update formula (for B ≈ Hessian):**
```
s_k = x_{k+1} - x_k
y_k = ∇F(x_{k+1}) - ∇F(x_k)
B_{k+1} = B_k - (B_k s_k)(B_k s_k)^T / (s_k^T B_k s_k) + y_k y_k^T / (y_k^T s_k)
```

**Properties:**
- Maintains positive definiteness if B_0 is PD and y_k^T s_k > 0
- Superlinear convergence on smooth convex problems
- Self-correcting

**Parameters:**
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| B_0 | Identity | Initial Hessian approximation |
| skip_update | y^T s < 1e-10 | Skip if curvature info poor |
| damped_update | true | Modify for robustness |

**When to use:**
- Optimization when Hessian unavailable
- Moderate problem size (n < 1000)
- Smooth, convex objectives

### L-BFGS (Limited-memory BFGS)

**Idea:** Store only last m pairs (s_k, y_k), reconstruct H_k v implicitly

**Two-loop recursion:**
```
Algorithm for computing H_k ∇F(x_k):
1. q = ∇F
2. For i = k-1, ..., k-m: α_i = ρ_i s_i^T q; q = q - α_i y_i
3. r = H_0 q
4. For i = k-m, ..., k-1: β = ρ_i y_i^T r; r = r + (α_i - β) s_i
5. Return r
```

**Parameters:**
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| m | 5-20 | Memory pairs stored |
| H_0 | γ_k I | Scale: γ_k = y_{k-1}^T s_{k-1} / y_{k-1}^T y_{k-1} |

**Memory:** O(mn) vs O(n²) for full BFGS

**When to use:**
- Large-scale optimization (n > 1000)
- Memory constrained
- Smooth objectives

### Broyden's Method (Good and Bad)

**Broyden's "good" update (for J ≈ Jacobian):**
```
J_{k+1} = J_k + (y_k - J_k s_k) s_k^T / (s_k^T s_k)
```

**Broyden's "bad" update:**
```
J_{k+1}^{-1} = J_k^{-1} + (s_k - J_k^{-1} y_k) y_k^T / (y_k^T y_k)
```

**Properties:**
- Superlinear convergence for root-finding
- Does not maintain symmetry or positive definiteness
- "Good" is better for most problems

**When to use:**
- Root-finding without exact Jacobian
- Moderate problem size
- Jacobian is nearly constant

### SR1 (Symmetric Rank-1)

**Update formula:**
```
B_{k+1} = B_k + (y_k - B_k s_k)(y_k - B_k s_k)^T / ((y_k - B_k s_k)^T s_k)
```

**Properties:**
- Maintains symmetry
- Does NOT maintain positive definiteness
- Can capture indefinite Hessian

**When to use:**
- Saddle point problems
- When negative curvature information needed
- Often combined with trust region

## Fixed-Point Methods

### Picard Iteration (Fixed-Point)

**Form:** Rewrite f(x) = 0 as x = g(x)

**Update:**
```
x_{k+1} = g(x_k)
```

**Convergence:**
- Linear: ||e_{k+1}|| ≤ L ||e_k|| where L = ||g'(x*)||
- Converges if L < 1

**When to use:**
- Natural fixed-point form available
- Stability more important than speed
- Initial guess may be far from solution

### Anderson Acceleration

**Idea:** Accelerate fixed-point iteration using history

**Algorithm:**
```
1. Compute g_k = g(x_k)
2. Define f_k = g_k - x_k (residual)
3. Minimize ||Σ α_i f_{k-i}||² subject to Σ α_i = 1
4. x_{k+1} = Σ α_i g_{k-i}
```

**Parameters:**
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| m | 3-10 | History depth |
| beta | 1.0 | Mixing parameter |
| regularization | 1e-10 | For least-squares solve |

**Properties:**
- Transforms linear convergence to superlinear (often)
- Robust when simple acceleration fails
- Works well for multiphysics coupling

**When to use:**
- Accelerating Picard iteration
- Coupled multiphysics problems
- When Jacobian unavailable or ill-conditioned

## Specialized Methods

### Levenberg-Marquardt

**For least-squares:** min ||r(x)||²

**Update:** Solve
```
(J^T J + λ D^T D) δ = -J^T r
```
where D is scaling (often diagonal of J^T J or identity)

**λ strategy:**
| Ratio ρ | Action |
|---------|--------|
| ρ < 0.25 | Increase λ (more gradient-like) |
| ρ > 0.75 | Decrease λ (more Newton-like) |
| 0.25 ≤ ρ ≤ 0.75 | Keep λ |

**Parameters:**
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| λ_init | 1e-3 | Initial damping |
| λ_min | 1e-10 | Minimum damping |
| λ_max | 1e10 | Maximum damping |
| factor_up | 10 | Increase factor |
| factor_down | 10 | Decrease factor |

**When to use:**
- Nonlinear least-squares problems
- Small to medium problems
- Data fitting, parameter estimation

### Gauss-Newton

**Same as Levenberg-Marquardt with λ = 0:**
```
J^T J δ = -J^T r
```

**Properties:**
- Fast when residual is small at solution
- May fail when residual is large
- Requires J^T J invertible

**When to use:**
- Zero-residual or small-residual problems
- Well-conditioned J
- Combined with line search for robustness

### Dogleg Method

**For trust region:** Combine Cauchy point and Newton step

```
1. Cauchy point: p_C = -α_C ∇F where α_C = ||∇F||² / (∇F^T B ∇F)
2. Newton step: p_N = -B^{-1} ∇F
3. Dogleg path: interpolate between origin, p_C, and p_N
4. Find intersection with trust region boundary
```

**When to use:**
- Trust region subproblem
- When exact subproblem solve too expensive
- Positive definite Hessian approximation

### Steihaug-CG

**For large trust region subproblem:** Solve using CG

```
1. Start CG iteration for B p = -∇F
2. If direction of negative curvature, go to boundary
3. If CG step exits trust region, go to boundary
4. Otherwise, continue CG until convergence
```

**When to use:**
- Large-scale trust region
- Truncated Newton methods
- When B may be indefinite

## Parameter Guidelines

### Convergence Tolerances

| Problem Type | Absolute Tol | Relative Tol |
|--------------|--------------|--------------|
| Engineering | 1e-6 to 1e-8 | 1e-4 to 1e-6 |
| Scientific | 1e-10 to 1e-12 | 1e-8 to 1e-10 |
| Financial | 1e-8 to 1e-10 | 1e-6 to 1e-8 |
| Machine precision | 1e-14 | 1e-12 |

### Maximum Iterations

| Method | Typical max_iter |
|--------|------------------|
| Newton | 20-50 |
| Quasi-Newton | 50-200 |
| Anderson | 100-500 |
| Picard | 100-1000 |

### Line Search Parameters

| Parameter | Conservative | Aggressive |
|-----------|--------------|------------|
| c1 (Armijo) | 1e-4 | 1e-2 |
| c2 (Wolfe) | 0.9 | 0.5 |
| max_backtracks | 20 | 10 |
| α_init | 0.5 | 1.0 |

### Trust Region Parameters

| Parameter | Conservative | Aggressive |
|-----------|--------------|------------|
| Δ_init | 0.1 | 1.0 |
| Δ_max | 10 | 100 |
| η_1 | 0.1 | 0.25 |
| η_2 | 0.9 | 0.75 |
| γ_1 | 0.25 | 0.5 |
| γ_2 | 2.5 | 2.0 |
