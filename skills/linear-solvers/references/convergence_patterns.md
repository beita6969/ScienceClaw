# Convergence Patterns

Comprehensive guide for diagnosing and improving iterative solver convergence.

## Convergence Monitoring

### Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Residual norm | \|\|b - Ax_k\|\| | Current error magnitude |
| Relative residual | \|\|r_k\|\| / \|\|b\|\| | Normalized by RHS |
| Reduction factor | \|\|r_k\|\| / \|\|r_{k-1}\|\| | Per-iteration improvement |
| True error | \|\|x - x_k\|\| | If exact solution known |

### Convergence Rate

```
Asymptotic convergence rate ρ = lim_{k→∞} ||r_k|| / ||r_{k-1}||

Linear convergence: ||r_k|| ≤ ρ^k ||r_0||, ρ < 1
Superlinear: ρ_k → 0 as k → ∞
```

| Rate | Value | Iterations to 10⁻⁶ |
|------|-------|-------------------|
| Excellent | ρ < 0.1 | < 10 |
| Good | ρ = 0.1-0.5 | 10-30 |
| Slow | ρ = 0.5-0.9 | 30-100 |
| Stagnation | ρ ≈ 1 | Does not converge |

## Typical Convergence Patterns

### Pattern 1: Fast Monotonic Decay

```
Residual:
1e0   ****
1e-2       ****
1e-4            ****
1e-6                 ****
      |----|----|----|----|
      0    10   20   30   40  iterations
```

**Characteristics:**
- Smooth, consistent reduction
- Constant or decreasing rate
- Reaches tolerance quickly

**Indicates:**
- Good preconditioner match
- Well-conditioned problem
- Appropriate solver choice

**Action:** None needed, this is ideal.

### Pattern 2: Initial Stall Then Decay

```
Residual:
1e0   ****----
1e-2          \____
1e-4               \____
1e-6                    \****
      |----|----|----|----|
      0    10   20   30   40  iterations
```

**Characteristics:**
- Slow start, then acceleration
- Superlinear convergence later
- Common with GMRES

**Indicates:**
- Krylov subspace building useful information
- Eventually finds good direction

**Action:**
- Be patient
- Increase max iterations
- Don't restart GMRES too early

### Pattern 3: Stagnation

```
Residual:
1e0   ****
1e-2       ****--------------------
1e-4
1e-6
      |----|----|----|----|----|----|
      0    20   40   60   80   100  iterations
```

**Characteristics:**
- Residual stops decreasing
- Convergence rate ≈ 1
- May plateau at various levels

**Indicates:**
- Preconditioner inadequate
- Matrix too ill-conditioned
- Reached limits of floating-point precision

**Actions:**
| Plateau Level | Likely Cause | Fix |
|---------------|--------------|-----|
| 1e-2 to 1e-4 | Weak preconditioner | Strengthen preconditioner |
| 1e-6 to 1e-8 | Condition number limit | Scale matrix, regularize |
| 1e-12 to 1e-14 | Machine precision | Accept, use extended precision |

### Pattern 4: Oscillation

```
Residual:
1e0   *   *   *
1e-2   * * * * *
1e-4    *   *   *
1e-6
      |----|----|----|----|
      0    10   20   30   40  iterations
```

**Characteristics:**
- Residual bounces up and down
- Net progress may be slow
- Common with BiCGSTAB

**Indicates:**
- Near-singular or indefinite matrix
- Eigenvalue close to origin
- Loss of orthogonality (GMRES)

**Actions:**
- Try different solver (GMRES if using BiCGSTAB)
- Increase GMRES restart parameter
- Better preconditioner
- Check matrix for issues

### Pattern 5: Divergence

```
Residual:
1e0   ****
1e2        ****
1e4             ****
NaN                  ****
      |----|----|----|----|
      0    10   20   30   40  iterations
```

**Characteristics:**
- Residual increases each iteration
- Eventually overflow/NaN

**Indicates:**
- Matrix singular or nearly so
- Instability in method
- Severe ill-conditioning

**Actions:**
- Check matrix (is it singular?)
- Apply scaling
- Try direct solver
- Regularize if appropriate

### Pattern 6: Plateau with Breakthrough

```
Residual:
1e0   ****
1e-2       ****--------****
1e-4                        \____
1e-6                             ****
      |----|----|----|----|----|----|
      0    20   40   60   80   100  iterations
```

**Characteristics:**
- Long plateau, then sudden progress
- Multiple plateaus possible

**Indicates:**
- Multiple scales in problem
- Different eigenvalue clusters

**Action:**
- Increase max iterations
- Consider multigrid or multilevel preconditioner

## Solver-Specific Patterns

### CG (Conjugate Gradient)

**Expected behavior:**
- Monotonic residual decrease
- At most n iterations for exact arithmetic
- Affected by eigenvalue distribution

**Warning signs:**
| Observation | Likely Problem |
|-------------|----------------|
| Non-monotonic | Matrix not SPD |
| Very slow | High condition number |
| Breakdown (0 division) | Indefinite matrix |

### GMRES

**Expected behavior:**
- Monotonically decreasing residual (in exact arithmetic)
- May stall then accelerate
- Memory grows with iterations

**Restart effects:**
| Restart m | Effect |
|-----------|--------|
| m small (10-20) | May stagnate |
| m medium (30-50) | Good balance |
| m large (100+) | Memory intensive |
| m = n | Optimal but expensive |

### BiCGSTAB

**Expected behavior:**
- May oscillate
- Two matrix-vector products per iteration
- Residual can temporarily increase

**Warning signs:**
| Observation | Likely Problem |
|-------------|----------------|
| Wild oscillation | Matrix strongly nonsymmetric |
| Breakdown | Lucky/unlucky breakdowns |
| Slow overall | Try GMRES instead |

## Diagnostic Procedures

### Step 1: Basic Health Check

```
1. Verify matrix properties:
   - Is it supposed to be SPD? Check.
   - Approximate condition number?
   - Check for zero rows/columns.

2. Check RHS:
   - Is ||b|| reasonable?
   - No NaN or Inf?

3. Verify preconditioner:
   - Did it build successfully?
   - Is it appropriate for this matrix type?
```

### Step 2: Convergence Analysis

```
1. Plot log(||r_k||) vs k
2. Compute average convergence rate
3. Identify pattern (decay, stagnation, oscillation)
4. Compare to expected behavior for solver type
```

### Step 3: Condition Number Estimation

```
1. Run CG/GMRES without preconditioner
2. Estimate κ from convergence rate:
   ρ ≈ (√κ - 1)/(√κ + 1) for CG
3. If κ > 10⁶, scaling/preconditioning essential
```

### Step 4: Preconditioner Quality Check

```
1. Compare iterations with/without preconditioner
2. If minimal improvement:
   - Preconditioner too weak
   - Wrong type for this matrix
   - Implementation bug

3. Target: 10-50× fewer iterations with preconditioner
```

## Convergence Criteria

### Relative Residual

```
Stop when: ||r_k|| / ||b|| < tol
Typical tol: 1e-6 to 1e-10
```

**Caution:** Can be fooled by ill-conditioned systems.

### Absolute Residual

```
Stop when: ||r_k|| < tol
Use when: ||b|| is very small or known scale
```

### True Error (if available)

```
Stop when: ||x - x_k|| / ||x|| < tol
Best criterion but rarely computable
```

### Practical Multi-Criteria

```python
def converged(r_k, r_0, b, x_k, tol_rel=1e-6, tol_abs=1e-10):
    rel_resid = np.linalg.norm(r_k) / np.linalg.norm(b)
    abs_resid = np.linalg.norm(r_k)
    reduction = np.linalg.norm(r_k) / np.linalg.norm(r_0)

    # Any criterion met
    return (rel_resid < tol_rel or
            abs_resid < tol_abs or
            reduction < tol_rel)
```

## Improving Convergence

### General Strategies

| Strategy | When to Apply | Expected Improvement |
|----------|---------------|----------------------|
| Better preconditioner | Slow convergence | 2-10× fewer iterations |
| Matrix scaling | High condition number | 2-100× fewer iterations |
| Different solver | Wrong solver type | May converge vs diverge |
| Increase restart (GMRES) | Stagnation | Variable |
| Add regularization | Near-singular | Enables convergence |

### Preconditioner Strengthening Ladder

```
Level 0: No preconditioner
  ↓ slow →
Level 1: Jacobi (diagonal)
  ↓ slow →
Level 2: ILU(0) or IC(0)
  ↓ slow →
Level 3: ILU(1) or IC(1)
  ↓ slow →
Level 4: ILUT with moderate fill
  ↓ slow →
Level 5: AMG or high-fill ILU
  ↓ slow →
Level 6: Direct solver
```

### When to Switch Solvers

| From | To | When |
|------|-----|------|
| CG | GMRES | Matrix not SPD |
| BiCGSTAB | GMRES | Oscillation, breakdown |
| GMRES | BiCGSTAB | Memory limited |
| Iterative | Direct | n < 10000, multiple RHS |

## Implementation: Convergence Logger

```python
class ConvergenceMonitor:
    """Track and analyze iterative solver convergence."""

    def __init__(self, b_norm):
        self.residuals = []
        self.b_norm = b_norm

    def log(self, r_norm):
        self.residuals.append(r_norm)

    def relative_residual(self):
        return [r / self.b_norm for r in self.residuals]

    def convergence_rate(self, window=5):
        """Average convergence rate over last window iterations."""
        if len(self.residuals) < window + 1:
            return None
        recent = self.residuals[-window-1:]
        rates = [recent[i+1]/recent[i] for i in range(window)]
        return np.mean(rates)

    def diagnose(self):
        if len(self.residuals) < 2:
            return "Insufficient data"

        rate = self.convergence_rate()
        if rate is None:
            return "Need more iterations"

        rel_final = self.residuals[-1] / self.b_norm

        if rel_final < 1e-10:
            return "Converged well"
        elif rate > 0.99:
            return "Stagnated - strengthen preconditioner"
        elif rate > 0.9:
            return "Slow - consider better preconditioner"
        elif rate < 0:
            return "Oscillating - try different solver"
        elif rate > 1:
            return "Diverging - check matrix properties"
        else:
            return "Good progress - continue"

    def plot(self):
        """Plot convergence history."""
        import matplotlib.pyplot as plt
        plt.semilogy(self.residuals, 'b-o')
        plt.xlabel('Iteration')
        plt.ylabel('Residual norm')
        plt.title(f'Convergence: final rate = {self.convergence_rate():.3f}')
        plt.grid(True)
        plt.show()
```

## Quick Troubleshooting Guide

| Symptom | First Check | Quick Fix |
|---------|-------------|-----------|
| No convergence after 1000 iter | Matrix singular? | Add regularization |
| Stagnation at 1e-4 | Preconditioner strength | Increase ILU fill |
| Oscillating residual | Solver appropriate? | Switch to GMRES |
| Very slow decay | Condition number | Apply scaling |
| Breakdown/NaN | Matrix properties | Check for issues, scale |
| Works but slow | Preconditioner type | Try AMG for elliptic |
