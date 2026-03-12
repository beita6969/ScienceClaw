# Scaling and Equilibration

Comprehensive guide for matrix scaling to improve conditioning and solver performance.

## When Scaling is Needed

### Indicators of Poor Scaling

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Residual stagnates | Large condition number | Row/column scaling |
| Small parameter changes → big solution changes | Ill-conditioning | Equilibration |
| Preconditioner fails | Extreme value ranges | Scale before factorization |
| Overflow/underflow | Very large/small entries | Scaling to O(1) |
| Slow convergence | Unbalanced row/column norms | Equilibration |

### Matrix Value Ranges

| Range (max/min) | Status | Action |
|-----------------|--------|--------|
| < 10³ | Good | Usually no scaling needed |
| 10³ - 10⁶ | Moderate | Consider scaling |
| 10⁶ - 10¹² | Poor | Scaling recommended |
| > 10¹² | Severe | Scaling essential |

## Scaling Methods

### Row Scaling

Multiply each row by a scalar: D_r × A

```
For each row i:
    scale_i = 1 / max_j |A_ij|  (max scaling)
    or
    scale_i = 1 / ||row_i||₂    (norm scaling)
    or
    scale_i = 1 / ||row_i||₁    (sum scaling)
```

**Effect:** All row maxima (or norms) become 1.

**Preserves:** Column space structure.

### Column Scaling

Multiply each column by a scalar: A × D_c

```
For each column j:
    scale_j = 1 / max_i |A_ij|
```

**Effect:** All column maxima become 1.

**Preserves:** Row space structure.

### Row and Column Scaling

Apply both: D_r × A × D_c

**Balanced approach:**
```
Scaled system: (D_r × A × D_c) × (D_c⁻¹ × x) = D_r × b
Solve for: y = D_c⁻¹ × x
Recover: x = D_c × y
```

### Symmetric Scaling

For symmetric matrices, use same scaling for rows and columns:
D × A × D

```
Preserves symmetry: (DAD)ᵀ = DAD
Essential for CG, MINRES
```

**Computing symmetric scaling:**
```
d_i = 1 / sqrt(|A_ii|)  (diagonal scaling)
or
d_i = 1 / sqrt(max_j |A_ij|)  (max scaling)
```

## Equilibration Algorithms

### Ruiz Equilibration

Iterative algorithm to make row and column norms approximately equal:

```
Repeat until converged:
    For each row i: r_i = ||row_i||_∞
    For each col j: c_j = ||col_j||_∞
    D_r = diag(1/sqrt(r_i))
    D_c = diag(1/sqrt(c_j))
    A = D_r × A × D_c
```

**Properties:**
- Converges to doubly stochastic-like scaling
- Usually 5-10 iterations sufficient
- Works for nonsymmetric matrices

### Sinkhorn-Knopp (Doubly Stochastic)

For nonnegative matrices, scale to doubly stochastic:

```
All row sums = 1
All column sums = 1
```

**Algorithm:**
```
Repeat:
    Normalize rows to sum to 1
    Normalize columns to sum to 1
Until converged
```

### Geometric Mean Scaling

Scale by geometric mean of max and min absolute values:

```
For row i:
    r_max = max_j |A_ij|
    r_min = min_{j: A_ij≠0} |A_ij|
    scale_i = 1 / sqrt(r_max × r_min)
```

**Good for:** Matrices with entries spanning many orders of magnitude.

## Special Considerations

### Preserving Symmetry

**CRITICAL:** For symmetric matrices, use symmetric scaling only!

```
Wrong: D_r × A × D_c with D_r ≠ D_c (destroys symmetry)
Right: D × A × D with same D for rows and columns
```

Check after scaling:
```
||A_scaled - A_scaled^T||_F / ||A_scaled||_F < ε
```

### Preserving Positive Definiteness

Symmetric scaling preserves definiteness:
```
If A is SPD and D is nonsingular diagonal:
Then D × A × D is also SPD
```

Check after scaling (if needed):
```
Try Cholesky factorization
Or check smallest eigenvalue > 0
```

### Scaling the RHS

When scaling Ax = b → (D_r A D_c) y = D_r b:

```
Solve: (D_r A D_c) y = D_r b
Recover: x = D_c y
Residual: r = b - Ax = b - A(D_c y)
```

**Important:** Scale b with the same D_r used for rows!

## Practical Scaling Strategies

### Safe Default Strategy

```python
def scale_matrix(A, symmetric=False):
    # Row scaling
    row_norms = np.max(np.abs(A), axis=1)
    D_r = 1.0 / np.maximum(row_norms, 1e-15)

    if symmetric:
        # Use same scaling for rows and columns
        D_c = D_r
    else:
        # Scale by rows first, then columns
        A_scaled = np.diag(D_r) @ A
        col_norms = np.max(np.abs(A_scaled), axis=0)
        D_c = 1.0 / np.maximum(col_norms, 1e-15)

    return D_r, D_c
```

### When to Apply Scaling

| Stage | Apply Scaling? |
|-------|---------------|
| Before analysis | Yes (check properties) |
| Before preconditioner | Often yes |
| Before solve | Yes |
| After solve | Unscale solution |

### Scaling Order

1. **Analyze** original matrix (symmetry, definiteness)
2. **Choose** appropriate scaling method
3. **Scale** matrix and RHS
4. **Build** preconditioner on scaled system
5. **Solve** scaled system
6. **Unscale** solution

## Diagnosing Scaling Issues

### Before Scaling

```
Check: ratio of max to min nonzero absolute values
       ||A||_∞ / min_{ij: A_ij≠0} |A_ij|

If ratio > 10⁶: scaling strongly recommended
```

### After Scaling

```
Verify:
- Row norms approximately equal
- Column norms approximately equal
- No overflow/underflow
- Symmetry preserved (if was symmetric)
```

### Monitoring Convergence

| Metric | Before Scaling | After Scaling | Status |
|--------|----------------|---------------|--------|
| Iterations | 500+ | 50 | Good |
| Residual decrease | 0.999/iter | 0.9/iter | Good |
| Final residual | Stagnated | Converged | Good |

## Common Mistakes

### Mistake 1: Destroying Symmetry

```
Problem: Used different row and column scaling on symmetric matrix
Result: CG fails (requires symmetric)
Fix: Use symmetric scaling D × A × D
```

### Mistake 2: Forgetting to Unscale

```
Problem: Returned y instead of x = D_c × y
Result: Wrong solution
Fix: Always unscale: x = D_c × y
```

### Mistake 3: Not Scaling RHS

```
Problem: Scaled A but not b
Result: Solving wrong system
Fix: b_scaled = D_r × b
```

### Mistake 4: Scaling Zero Rows

```
Problem: Row of zeros gets 1/0 = inf scaling
Result: NaN in solution
Fix: Handle zero rows specially (or they indicate singular matrix)
```

## Advanced Topics

### Iterative Refinement with Scaling

For very ill-conditioned systems:

```
1. Scale: Ã = D_r A D_c, b̃ = D_r b
2. Solve: Ã ỹ = b̃ (in lower precision if available)
3. Compute residual: r = b - A(D_c ỹ) (in high precision)
4. Solve: Ã δỹ = D_r r
5. Update: ỹ = ỹ + δỹ
6. Repeat until converged
7. Unscale: x = D_c ỹ
```

### Scaling for Eigenvalue Problems

When solving Ax = λx after scaling D_r A D_c:

```
Eigenvalues: Same (scaling is similarity transform)
Eigenvectors: v_original = D_c × v_scaled
```

### Block Scaling

For block-structured matrices:

```
Scale each block independently, or
Scale entire rows/columns of blocks
```

**Useful for:** Multiphysics with different variable scales.

## Implementation Example

```python
def equilibrate_matrix(A, tol=1e-6, max_iter=20):
    """Ruiz equilibration for general matrices."""
    n, m = A.shape
    D_r = np.ones(n)
    D_c = np.ones(m)

    for _ in range(max_iter):
        # Row scaling
        row_norms = np.max(np.abs(A), axis=1)
        row_norms = np.maximum(row_norms, 1e-15)
        d_r = 1.0 / np.sqrt(row_norms)
        A = np.diag(d_r) @ A
        D_r *= d_r

        # Column scaling
        col_norms = np.max(np.abs(A), axis=0)
        col_norms = np.maximum(col_norms, 1e-15)
        d_c = 1.0 / np.sqrt(col_norms)
        A = A @ np.diag(d_c)
        D_c *= d_c

        # Check convergence
        if np.max(np.abs(row_norms - 1)) < tol:
            if np.max(np.abs(col_norms - 1)) < tol:
                break

    return A, D_r, D_c

def solve_scaled(A, b, solver, symmetric=False):
    """Solve with scaling and unscaling."""
    A_scaled, D_r, D_c = equilibrate_matrix(A)
    b_scaled = D_r * b

    if symmetric:
        # Ensure symmetric scaling
        D = np.sqrt(D_r * D_c)
        A_scaled = np.diag(D) @ A @ np.diag(D)
        b_scaled = D * b
        y = solver(A_scaled, b_scaled)
        x = D * y
    else:
        y = solver(A_scaled, b_scaled)
        x = D_c * y

    return x
```

## Quick Reference

| Matrix Type | Scaling Method | Notes |
|-------------|----------------|-------|
| General | Row + column equilibration | Ruiz algorithm |
| Symmetric | Symmetric diagonal scaling | D × A × D |
| SPD | Symmetric diagonal scaling | Preserve definiteness |
| Block-structured | Block-aware scaling | Match physics scales |
| Very ill-conditioned | Equilibration + iterative refinement | May need extended precision |
