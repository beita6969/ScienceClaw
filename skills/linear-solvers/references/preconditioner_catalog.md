# Preconditioner Catalog

Comprehensive reference for preconditioners in iterative linear solvers.

## Preconditioner Fundamentals

### Purpose

Transform Ax = b into M⁻¹Ax = M⁻¹b where M⁻¹A has better conditioning.

**Goals:**
- Cluster eigenvalues away from zero
- Reduce condition number κ(M⁻¹A)
- Make M⁻¹ cheap to apply

### Application Modes

| Mode | System Solved | When to Use |
|------|---------------|-------------|
| Left | M⁻¹Ax = M⁻¹b | Most common |
| Right | AM⁻¹y = b, x = M⁻¹y | Preserves residual meaning |
| Split | L⁻¹AR⁻¹y = L⁻¹b | Symmetric preconditioning |

### Key Trade-offs

| Factor | Cheap Precond. | Expensive Precond. |
|--------|----------------|-------------------|
| Setup cost | Low | High |
| Apply cost | Low | High |
| Iterations | Many | Few |
| Total time | May be optimal | May be optimal |

## Incomplete Factorization Family

### Incomplete Cholesky (IC)

For SPD matrices: A ≈ LLᵀ where L is sparse.

**IC(0)** - Zero fill-in:
- Same sparsity pattern as lower triangle of A
- Cheap, often effective
- May fail for indefinite or poorly scaled

**IC(k)** - Level-k fill:
- Allow fill-in up to k levels from original pattern
- More robust, higher cost
- k = 1 or 2 often sufficient

**Modified IC (MIC)**:
- Add dropped entries to diagonal
- Better for M-matrices (e.g., Laplacian)
- Preserves row sums

### Incomplete LU (ILU)

For general matrices: A ≈ LU where L, U are sparse.

**ILU(0)** - Zero fill-in:
```
Pattern(L + U) = Pattern(A)
Cheap, first try for nonsymmetric
```

**ILU(k)** - Level-k fill:
```
Allow fill paths up to k edges
k = 1: moderate fill
k = 2: substantial fill
```

**ILUT** - Threshold-based:
```
Parameters: τ (drop tolerance), p (max fill per row)
Drop if |entry| < τ × ||row||
Keep at most p entries per row
```

| Parameter | Effect |
|-----------|--------|
| τ small | More fill, better approximation |
| τ large | Less fill, weaker preconditioner |
| p small | Limit memory, may reduce quality |
| p large | Better quality, more memory |

**Typical values:**
```
τ = 1e-4 to 1e-2
p = 10 to 50 (or 2× to 5× original nnz/row)
```

### Choosing IC vs ILU Parameters

| Symptom | Adjustment |
|---------|------------|
| Convergence too slow | Increase k or decrease τ |
| Too much memory | Increase τ or decrease p |
| Factorization fails | Add diagonal shift, try different ordering |
| Negative pivot (IC) | Matrix not SPD, use ILU |

## Algebraic Multigrid (AMG)

### When to Use

| Good for | Poor for |
|----------|----------|
| Elliptic PDEs | Highly nonsymmetric |
| Diffusion-dominated | Pure advection |
| Smooth error | Oscillatory error |
| Large systems | Small systems (overhead) |

### AMG Components

**Coarsening:**
- Classical (Ruge-Stüben): Strength-based C/F splitting
- Aggregation: Group nodes into aggregates
- Smoothed aggregation: SA-AMG, good for elasticity

**Interpolation:**
- Direct: Use strong connections
- Standard: Include weak connections
- Extended+i: For harder problems

**Smoothing:**
- Jacobi: Simple, parallelizable
- Gauss-Seidel: Better smoothing, less parallel
- Polynomial: Good for GPU

### AMG Tuning

| Parameter | Effect |
|-----------|--------|
| Strong threshold | Lower = more connections = slower coarsening |
| Coarsening ratio | 2:1 to 4:1 typical |
| Max levels | 10-20 typical |
| Smoother | Jacobi (parallel) vs GS (sequential) |
| Cycles | V-cycle (cheap) vs W-cycle (robust) |

### AMG for Nonsymmetric

Some AMG can handle mildly nonsymmetric:
- Use symmetric part for coarsening
- May need more smoothing
- Verify convergence experimentally

## Specialized Preconditioners

### Jacobi and Block Jacobi

**Point Jacobi:** M = diag(A)
```
Simple, parallel, weak
Use as smoother in multigrid
```

**Block Jacobi:** M = block_diag(A)
```
Blocks from natural structure (elements, nodes)
Stronger than point Jacobi
Embarrassingly parallel
```

### Gauss-Seidel

**Forward/Backward GS:**
```
M = L + D (forward) or D + U (backward)
Stronger than Jacobi
Sequential, hard to parallelize
```

**Symmetric GS (SSOR):**
```
Apply forward then backward
Good smoother for symmetric problems
Parameter ω (relaxation): typically 1.0
```

### SSOR (Symmetric SOR)

For SPD systems:
```
M = (D/ω + L) × (D/ω)⁻¹ × (D/ω + U)
ω: overrelaxation parameter (0 < ω < 2)
ω = 1: Symmetric Gauss-Seidel
ω optimal ≈ 2/(1 + sin(πh)) for model problem
```

### Polynomial Preconditioners

Approximate M⁻¹ ≈ p(A) for some polynomial p.

**Neumann series:**
```
M⁻¹ ≈ I + (I - A) + (I - A)² + ...
Requires ρ(I - A) < 1
```

**Chebyshev:**
```
Optimal polynomial for given eigenvalue bounds
Requires [λ_min, λ_max] estimates
Good for GPU (only matrix-vector products)
```

## Block Preconditioners

### For Saddle-Point Systems

```
K = [A   B ]    x = [u]    b = [f]
    [Bᵀ  -C]        [p]        [g]
```

**Block Diagonal:**
```
P = [Â   0 ]
    [0   Ŝ ]

Â ≈ A (e.g., AMG for A)
Ŝ ≈ S = C + BᵀA⁻¹B (Schur complement)
```

**Block Triangular:**
```
P = [Â   0 ]    or    P = [Â   B]
    [Bᵀ  -Ŝ]              [0  -Ŝ]
```

### Schur Complement Approximations

| Approximation | Formula | Use Case |
|---------------|---------|----------|
| Mass matrix | Ŝ = M_p (pressure mass) | Stokes |
| BFBt | Ŝ = B diag(A)⁻¹ Bᵀ | General |
| LSC | Ŝ = (B Bᵀ)(B A Bᵀ)⁻¹(B Bᵀ) | Navier-Stokes |
| PCD | Convection-diffusion-reaction | Navier-Stokes |

### Field-Split Preconditioners

For multi-physics with fields u₁, u₂, ...:
```
Multiplicative: Solve u₁, then u₂ using updated u₁
Additive: Solve each independently, sum corrections
```

## Preconditioner Selection Guide

### By Matrix Type

| Matrix | First Choice | Alternative |
|--------|--------------|-------------|
| SPD, diffusion | AMG | IC(k) |
| SPD, elasticity | SA-AMG | IC(k) |
| SPD, general | IC(0) | AMG |
| Nonsymmetric, mild | ILU(0) | ILUT |
| Nonsymmetric, advection | ILUT (strong) | Stream-wise ILU |
| Saddle-point | Block diagonal/triangular | Uzawa |
| Dense | - (direct solver) | - |

### By Problem Size

| Size | Recommendation |
|------|----------------|
| n < 1000 | Direct solver |
| n = 1000-10000 | ILU/IC or AMG |
| n = 10000-1M | AMG (if applicable) |
| n > 1M | AMG, domain decomposition |

### By Available Resources

| Resource | Recommendation |
|----------|----------------|
| Single core | Sequential GS, ILU |
| Multi-core | Block Jacobi, AMG |
| GPU | Polynomial, Block Jacobi |
| Distributed | Domain decomposition + local precond. |

## Troubleshooting

### Preconditioner Fails to Build

| Error | Cause | Fix |
|-------|-------|-----|
| Zero pivot | Singular or structurally singular | Reorder, add diagonal shift |
| Negative pivot (IC) | Not SPD | Use ILU, check matrix |
| Out of memory | Too much fill | Increase τ, reduce p or k |

### Poor Convergence Despite Preconditioner

| Symptom | Cause | Fix |
|---------|-------|-----|
| Slow decay | Weak preconditioner | Strengthen (lower τ, higher k) |
| Stagnation | Unfavorable eigenvalue distribution | Try different preconditioner |
| Oscillation | Near-singular modes | Check matrix, regularize |

### Parameter Tuning Strategy

1. **Start simple:** ILU(0) or IC(0)
2. **Monitor:** iteration count, residual curve
3. **Adjust:** If slow, strengthen preconditioner
4. **Balance:** Setup time vs iteration time
5. **Validate:** Check solution accuracy

## Implementation Notes

### Setup vs Apply Cost

| Preconditioner | Setup | Apply | When Setup Dominates |
|----------------|-------|-------|----------------------|
| Jacobi | O(n) | O(n) | Never |
| ILU(0) | O(nnz) | O(nnz) | Many solves |
| AMG | O(n log n) | O(n) | Few solves |

### Reusing Preconditioners

When matrix changes slightly:
```
Same pattern: May reuse structure, update values
Small changes: Lag preconditioner (update every k solves)
Large changes: Rebuild preconditioner
```

### Quality Metrics

| Metric | Good Value |
|--------|------------|
| Fill ratio (nnz(LU)/nnz(A)) | 2-10 for ILU |
| Operator complexity (AMG) | 1.2-2.0 |
| Convergence factor | < 0.3 |
| Iterations | < 50 typically |
