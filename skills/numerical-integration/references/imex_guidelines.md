# IMEX Guidelines

Comprehensive guide for Implicit-Explicit time integration methods.

## When to Use IMEX

### Problem Structure

IMEX methods are designed for problems of the form:
```
du/dt = f_stiff(u) + f_nonstiff(u)
```

Where:
- f_stiff: Requires implicit treatment for stability
- f_nonstiff: Can be handled explicitly (cheaper)

### Canonical Examples

| Problem | Stiff Term | Non-stiff Term |
|---------|------------|----------------|
| Advection-diffusion | Diffusion (∇²u) | Advection (v·∇u) |
| Reaction-diffusion | Diffusion | Mild reactions |
| Phase-field | Laplacian | Bulk free energy |
| Combustion | Fast chemistry | Convection |
| Incompressible flow | Pressure (implicit) | Advection |

### Decision Criteria

Use IMEX when:
1. **Clear separation** exists between stiff and non-stiff terms
2. **Stiff term is linear** or easy to solve implicitly
3. **Non-stiff term dominates cost** if treated implicitly
4. **Explicit dt limit** would be too restrictive for stiff part

Avoid IMEX when:
1. All terms have similar stiffness
2. Terms are strongly coupled (nonlinear interaction)
3. Stiff term is highly nonlinear
4. Splitting error is unacceptable

## IMEX Scheme Classes

### IMEX-Runge-Kutta

Combines explicit and implicit RK methods:

```
Stage i:
  U_i = u_n + dt × Σⱼ aᵢⱼᴱ f_explicit(Uⱼ) + dt × Σⱼ aᵢⱼᴵ f_implicit(Uⱼ)

Update:
  u_{n+1} = u_n + dt × Σᵢ bᵢᴱ f_explicit(Uᵢ) + dt × Σᵢ bᵢᴵ f_implicit(Uᵢ)
```

**Common Schemes:**

| Name | Order | ERK Stages | DIRK Stages | Properties |
|------|-------|------------|-------------|------------|
| IMEX-Euler | 1 | 1 | 1 | Simple, first-order |
| ARS(2,2,2) | 2 | 2 | 2 | L-stable implicit part |
| ARK3(2)4L | 3 | 4 | 4 | L-stable, stiff accurate |
| ARK4(3)6L | 4 | 6 | 6 | High order |

### IMEX-Multistep (SBDF)

Semi-implicit BDF combines BDF for stiff with Adams-Bashforth for non-stiff:

**SBDF1:**
```
(u_{n+1} - u_n) / dt = f_implicit(u_{n+1}) + f_explicit(u_n)
```

**SBDF2:**
```
(3u_{n+1} - 4u_n + u_{n-1}) / (2dt) = f_implicit(u_{n+1}) + 2f_explicit(u_n) - f_explicit(u_{n-1})
```

| Order | Stability | Startup | Memory |
|-------|-----------|---------|--------|
| SBDF1 | A-stable | None | 1 level |
| SBDF2 | A-stable | RK2 | 2 levels |
| SBDF3 | A(86°)-stable | RK3 | 3 levels |
| SBDF4 | A(73°)-stable | RK4 | 4 levels |

### IMEX-Peer Methods

Multi-stage multi-step methods with good parallel properties:
- Each stage can be computed in parallel
- Combine strengths of RK and multistep
- Emerging class with ongoing research

## Splitting Strategies

### Term Assignment

| Physics | Treatment | Reasoning |
|---------|-----------|-----------|
| Diffusion (∇²u) | Implicit | Stiff, linear |
| Fast linear reactions | Implicit | Stiff |
| Advection (v·∇u) | Explicit | CFL-limited anyway |
| Slow reactions | Explicit | Not stiff |
| Nonlinear bulk terms | Explicit | Difficult implicit solve |
| External forcing | Explicit | Usually smooth |

### Semi-implicit Linearization

When nonlinear terms must be implicit, linearize:

```
Original: du/dt = -∇·(D(u)∇u)

Linearized: du/dt = -∇·(D(u_n)∇u_{n+1})
             ↑            ↑         ↑
          implicit    lagged    new value
```

**Lagging strategies:**
- Lag coefficient by one step (first-order in time for coefficient)
- Extrapolate coefficient (higher order)
- Iterate until converged (fully implicit)

## Stability Analysis

### Combined Stability Region

The effective stability region is the intersection of:
1. Implicit method's region for stiff term
2. Explicit method's region for non-stiff term

```
For stability: both λ_stiff × dt and λ_nonstiff × dt must be in respective regions
```

### Stability Plots

For IMEX methods, plot stability in (λ_E × dt, λ_I × dt) plane:
- Horizontal axis: explicit eigenvalue × dt
- Vertical axis: implicit eigenvalue × dt
- Shaded region: stable combinations

### CFL for Explicit Part

Even with IMEX, the explicit part imposes a CFL constraint:

```
dt ≤ C × dx / |v_max|  (for advection)
```

The implicit treatment of diffusion removes:
```
dt ≤ dx² / (2D)  (this constraint is gone!)
```

## Accuracy Considerations

### Order Conditions

For IMEX-RK, both tableaux must satisfy order conditions:
- Individual ERK and DIRK must be consistent
- Coupling conditions between tableaux
- Stiff accuracy (optional but helpful)

### Splitting Error

IMEX introduces splitting error beyond method truncation error:

```
Total error = O(dt^p) + splitting error
```

Splitting error depends on:
- Commutator [f_stiff, f_nonstiff]
- Method coupling (how stages interact)

For additive IMEX-RK: splitting error = O(dt^p) when properly designed.

### Order Reduction

Near stability boundaries:
- Classical order reduction for stiff ODEs applies
- Some IMEX schemes are "stiff accurate" to mitigate this

## Implementation Guidelines

### Linear Solver Requirements

The implicit part requires solving:
```
(I - dt × γ × J_implicit) × δu = RHS
```

Where J_implicit is the Jacobian of f_implicit.

**Requirements:**
1. Jacobian or Jacobian-vector product
2. Linear solver (direct or iterative)
3. Preconditioner for large systems

### Jacobian Reuse

For IMEX-RK with SDIRK (same diagonal):
- Same Jacobian matrix at all implicit stages
- Factor once per step, reuse for all stages
- Significant cost savings

For varying Jacobian:
- Recompute each stage OR
- Lag Jacobian across stages (lose some order)

### Iteration Strategy

**Fixed number of iterations:**
- Predictable cost
- May not converge fully
- OK if splitting dominates error

**Converge to tolerance:**
- Fully implicit behavior
- Variable cost per step
- Use when high accuracy needed

### Memory Management

| Scheme | Storage Required |
|--------|-----------------|
| IMEX-RK (s stages) | 2s arrays |
| SBDF2 | 2 time levels |
| SBDF3 | 3 time levels |

## Practical Examples

### Advection-Diffusion

```
∂u/∂t + v · ∇u = D ∇²u

f_explicit = -v · ∇u  (advection)
f_implicit = D ∇²u    (diffusion)

dt constraint: CFL for advection only
dt ≤ dx / |v_max|  (instead of min with dx²/D)
```

### Allen-Cahn Equation

```
∂φ/∂t = M(ε²∇²φ - φ(φ²-1))

Option 1 (linear implicit):
  f_implicit = M ε² ∇²φ
  f_explicit = -M φ(φ²-1)

Option 2 (stabilized explicit):
  f_implicit = M ε² ∇²φ - M s φ  (add stabilizing term)
  f_explicit = -M φ(φ²-1) + M s φ  (subtract same term)

  Choose s to make explicit part bounded.
```

### Cahn-Hilliard Equation

```
∂φ/∂t = M ∇²(f'(φ) - ε² ∇²φ)

Convex-concave splitting:
  f_implicit = M ∇²(c₁ φ - ε² ∇²φ)  (convex part)
  f_explicit = M ∇²(f'(φ) - c₁ φ)   (concave part)

  c₁ chosen so that f''(φ) - c₁ ≤ 0
```

## Coupling Strength

### Weak Coupling

Terms interact slowly:
```
Example: Heat conduction + slow reaction
Solution: Simple splitting works well
```

### Moderate Coupling

Terms have some interaction:
```
Example: Fast diffusion + moderate reaction
Solution: IMEX-RK with good coupling coefficients
```

### Strong Coupling

Terms are tightly coupled:
```
Example: Phase-field with strong anisotropy
Solution: Fully coupled IMEX-RK or iteration
```

## Verification

### Order Verification

1. Run with dt, dt/2, dt/4
2. Compute error vs exact (if known) or Richardson
3. Confirm error ∝ dt^p

### Splitting Error Check

1. Run pure implicit (full problem)
2. Run IMEX with same dt
3. Difference reveals splitting error

### Stability Boundary

1. Gradually increase dt
2. Find maximum stable dt
3. Compare with theoretical prediction
