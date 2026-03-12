# Globalization Strategies

Guide to line search, trust region, and damping strategies for nonlinear solvers.

## Overview

### Why Globalization?

Newton's method converges quadratically **near the solution**, but may:
- Diverge if initial guess is far
- Overshoot if step is too large
- Fail on singular or near-singular Jacobians

Globalization ensures convergence from "any" starting point.

### Strategy Comparison

| Aspect | Line Search | Trust Region |
|--------|-------------|--------------|
| Step control | Scale full Newton step | Constrain step norm |
| Implementation | Simpler | More robust |
| Cost per iter | Multiple f evaluations | One subproblem solve |
| Best for | Smooth, well-conditioned | Ill-conditioned, near-singular |
| Typical methods | Armijo, Wolfe | Dogleg, Steihaug |

## Line Search Methods

### Backtracking Line Search

**Algorithm:**
```
Given: step p_k, initial α = 1, ρ ∈ (0,1), c ∈ (0,1)

while f(x_k + α p_k) > f(x_k) + c α ∇f^T p_k:
    α = ρ α

x_{k+1} = x_k + α p_k
```

**Parameters:**
| Parameter | Typical | Range | Notes |
|-----------|---------|-------|-------|
| c | 1e-4 | (0, 0.5) | Sufficient decrease |
| ρ | 0.5 | (0.1, 0.9) | Backtrack factor |
| α_min | 1e-10 | - | Safety bound |
| max_backtracks | 20 | 10-50 | Iteration limit |

**Advantages:**
- Simple to implement
- Low overhead per backtrack
- Works for most problems

**Disadvantages:**
- May take many backtracks
- No curvature information
- Can be inefficient for quasi-Newton

### Armijo Condition

**The Armijo (sufficient decrease) condition:**
```
f(x_k + α p_k) ≤ f(x_k) + c₁ α ∇f(x_k)^T p_k
```

**Geometric interpretation:**
```
f(x)
  |
  |  *   Armijo condition requires staying
  | / \  below the dashed line
  |/   \____
  |-------- α
```

### Wolfe Conditions

**Strong Wolfe conditions:**
```
Sufficient decrease: f(x_k + α p_k) ≤ f(x_k) + c₁ α ∇f^T p_k
Curvature:          |∇f(x_k + α p_k)^T p_k| ≤ c₂ |∇f(x_k)^T p_k|
```

**Weak Wolfe (curvature):**
```
∇f(x_k + α p_k)^T p_k ≥ c₂ ∇f(x_k)^T p_k
```

**Parameters:**
| Parameter | Typical | Range | Notes |
|-----------|---------|-------|-------|
| c₁ | 1e-4 | (0, 0.5) | Sufficient decrease |
| c₂ (Newton) | 0.9 | (c₁, 1) | Less restrictive |
| c₂ (quasi-Newton) | 0.5 | (c₁, 1) | More restrictive for BFGS |
| c₂ (CG) | 0.1 | (c₁, 0.5) | Very restrictive |

**When to use Wolfe:**
- BFGS and L-BFGS (curvature condition ensures valid update)
- Conjugate gradient methods
- When backtracking is expensive

### Interpolation Line Search

**Quadratic interpolation:**
Given f(0), f'(0), f(α):
```
α_new = -f'(0) α² / (2 (f(α) - f(0) - f'(0) α))
```

**Cubic interpolation:**
Given f(α₀), f(α₁), f'(0):
```
Better approximation using two function values and derivative
```

**Advantages:**
- Fewer function evaluations than backtracking
- Better estimate of optimal step

**Safeguards:**
```
if α_new < 0.1 α or α_new > 0.9 α:
    α_new = 0.5 α  # Bisection fallback
```

## Trust Region Methods

### Basic Trust Region Framework

**Subproblem:**
```
Minimize m_k(p) = f_k + ∇f_k^T p + ½ p^T B_k p
subject to ||p|| ≤ Δ_k
```

**Algorithm:**
```
1. Compute step p_k by solving subproblem
2. Compute ratio ρ_k = (f(x_k) - f(x_k + p_k)) / (m_k(0) - m_k(p_k))
3. Update trust radius based on ρ_k:
   - ρ_k < 0.25: shrink Δ
   - ρ_k > 0.75 and ||p_k|| = Δ_k: expand Δ
   - otherwise: keep Δ
4. If ρ_k > η: accept step x_{k+1} = x_k + p_k
```

**Parameters:**
| Parameter | Typical | Range | Notes |
|-----------|---------|-------|-------|
| Δ_0 | 1.0 | (0.1, 10) | Initial radius |
| Δ_max | 100 | > Δ_0 | Maximum radius |
| η | 0.1 | (0, 0.25) | Accept threshold |
| η₁ | 0.25 | (0, 0.5) | Shrink threshold |
| η₂ | 0.75 | (0.5, 1) | Expand threshold |
| γ₁ | 0.25 | (0, 0.5) | Shrink factor |
| γ₂ | 2.0 | (1, 4) | Expand factor |

### Cauchy Point

**The Cauchy point minimizes along steepest descent:**
```
p_C = -α_C ∇f  where α_C minimizes m_k(-α ∇f) for ||−α∇f|| ≤ Δ
```

**Computation:**
```
If ∇f^T B ∇f ≤ 0:  (negative curvature)
    α_C = Δ / ||∇f||
Else:
    α_C = min(||∇f||² / (∇f^T B ∇f), Δ / ||∇f||)

p_C = -α_C ∇f
```

**Properties:**
- Always well-defined
- Provides sufficient decrease guarantee
- Usually not optimal step

### Dogleg Method

**Combines Cauchy and Newton steps:**
```
1. Compute Cauchy point p_C
2. Compute Newton step p_N = -B^{-1} ∇f
3. If ||p_N|| ≤ Δ: return p_N (Newton step in region)
4. If ||p_C|| ≥ Δ: return scaled p_C (Cauchy on boundary)
5. Otherwise: interpolate between p_C and p_N to hit boundary
```

**The dogleg path:**
```
     p_N *
        /
       /
      /
     /  <- leg 2
    *----
   p_C  ^
        |
        leg 1
        |
   -----O (origin)
```

**Advantages:**
- Simple closed-form solution
- Works well when B is positive definite
- Good balance of cost and quality

**Disadvantages:**
- Requires B positive definite
- May not be optimal for indefinite B

### Steihaug-CG (Truncated Conjugate Gradient)

**For large-scale or indefinite problems:**
```
1. Start CG iteration for B p = -∇f
2. If direction of negative curvature detected:
   - Extend current direction to trust region boundary
3. If CG step exits trust region:
   - Interpolate to boundary
4. If converged within region:
   - Return CG solution
```

**Advantages:**
- Handles indefinite B
- Efficient for large problems
- Exploits Hessian structure

**Disadvantages:**
- More complex implementation
- May terminate early

### Levenberg-Marquardt (Damped Newton)

**Modified Newton step:**
```
(J^T J + λ I) p = -J^T r   (least-squares)
(B + λ I) p = -∇f          (general optimization)
```

**λ interpretation:**
- λ = 0: Pure Newton/Gauss-Newton
- λ → ∞: Steepest descent
- λ interpolates between them

**λ update strategy:**
```
ρ = actual_reduction / predicted_reduction

if ρ < 0.25:
    λ = λ × 4
elif ρ > 0.75:
    λ = λ / 2
```

**Advantages:**
- Regularizes singular/ill-conditioned Jacobian
- Smooth transition from steepest descent to Newton
- Standard for nonlinear least-squares

## Damping Strategies

### Constant Damping

**Simple damped Newton:**
```
x_{k+1} = x_k - ω J^{-1} f(x_k)
```

**Typical ω values:**
| Situation | ω | Notes |
|-----------|---|-------|
| Initial iterations | 0.5-0.7 | Avoid overshoot |
| Near solution | 1.0 | Full Newton |
| Very nonlinear | 0.1-0.3 | Conservative |

### Adaptive Damping

**Increase when successful, decrease on failure:**
```
if ||f(x_{k+1})|| < ||f(x_k)||:
    ω = min(1.0, ω × 1.1)  # Gradually increase
else:
    ω = ω × 0.5            # Reduce on failure
```

### Under-relaxation for Coupled Systems

**For multiphysics coupling:**
```
x_1^{n+1} = x_1^n + ω_1 (x̃_1 - x_1^n)
x_2^{n+1} = x_2^n + ω_2 (x̃_2 - x_2^n)
```

**Aitken acceleration:**
```
ω_{k+1} = ω_k × (r_k^T (r_k - r_{k-1})) / ||r_k - r_{k-1}||²
```

## Parameter Selection Guidelines

### Line Search Selection

| Problem Type | Recommended | Parameters |
|--------------|-------------|------------|
| Newton, root-finding | Armijo backtracking | c=1e-4, ρ=0.5 |
| BFGS optimization | Wolfe | c₁=1e-4, c₂=0.9 |
| L-BFGS large-scale | Wolfe with interpolation | c₁=1e-4, c₂=0.9 |
| CG optimization | Strong Wolfe | c₁=1e-4, c₂=0.1 |

### Trust Region Selection

| Problem Type | Recommended | Initial Δ |
|--------------|-------------|-----------|
| Small, well-conditioned | Dogleg | 1.0 |
| Large scale | Steihaug-CG | 1.0 |
| Near-singular Jacobian | LM | 1.0 with λ=0.01 |
| Least-squares | LM | Based on problem scale |

### Switching Strategy

When to switch from line search to trust region:
- Multiple failed line searches
- Oscillating residuals
- Very ill-conditioned Jacobian
- Near-singular system detected

## Practical Recommendations

### Starting Points

1. **Conservative start:**
   - Trust region: Δ₀ = 0.1 × ||x₀||
   - Line search: α₀ = 0.5 (not full Newton)

2. **Aggressive start:**
   - Trust region: Δ₀ = 1.0
   - Line search: α₀ = 1.0 (full Newton)

### Monitoring

Track these quantities:
```
- ρ = actual/predicted reduction (trust region)
- α = step size taken (line search)
- ||p|| / Δ (fraction of trust region used)
- ||∇f|| (gradient norm)
- ||f|| (residual/objective)
```

### Common Adjustments

| Observation | Adjustment |
|-------------|------------|
| Many failed steps | Increase c₁, decrease Δ₀ |
| Very small steps | Check Jacobian accuracy |
| Always at boundary | Increase Δ_max |
| Slow decrease | Improve preconditioner |

### Failure Recovery

```
1. Failed line search:
   - Try trust region instead
   - Check Jacobian sign
   - Reduce initial α

2. Failed trust region:
   - Decrease Δ more aggressively
   - Add LM regularization
   - Check for local minimum

3. Repeated failures:
   - Problem may be infeasible
   - Try different formulation
   - Use continuation methods
```
