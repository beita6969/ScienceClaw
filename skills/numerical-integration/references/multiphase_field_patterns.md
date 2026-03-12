# Multiphase-Field Splitting Patterns

Specialized integration patterns for phase-field and multi-order-parameter models.

## Phase-Field Model Classes

### Allen-Cahn (Non-Conserved)

Model A kinetics for non-conserved order parameters:
```
∂φ/∂t = -M × δF/δφ = -M × (f'(φ) - ε²∇²φ)
```

Where:
- φ: order parameter (e.g., phase indicator)
- M: mobility
- f(φ): bulk free energy density
- ε: interface width parameter

### Cahn-Hilliard (Conserved)

Model B kinetics for conserved order parameters:
```
∂c/∂t = ∇·(M∇μ)
μ = δF/δc = f'(c) - ε²∇²c
```

Where:
- c: conserved field (e.g., concentration)
- μ: chemical potential
- M: mobility (may be M(c))

### Multi-Order-Parameter

Multiple coupled order parameters:
```
∂φᵢ/∂t = -Mᵢ × δF/δφᵢ + Σⱼ coupling_ij(φⱼ)
```

Examples:
- Grain growth: φᵢ for each grain
- Eutectic: φ_α, φ_β, c
- Polycrystal: φᵢ + θᵢ (orientation)

## Allen-Cahn Splitting

### Standard Double-Well

```
f(φ) = (1/4)(φ² - 1)² = (1/4)φ⁴ - (1/2)φ² + 1/4
```

**IMEX Split (Linear Implicit):**
```
Implicit: ε²∇²φ           (stiffest term)
Explicit: -φ(φ² - 1)      (bulk driving force)
```

**Stability:** Explicit part bounded for |φ| ≤ 1 + O(ε)

### Stabilized Splitting

Add/subtract linear stabilizing term:
```
∂φ/∂t = M[ε²∇²φ - sφ] + M[-φ³ + φ + sφ]
         └─implicit─┘   └───explicit───┘
```

Choose s > max|f''(φ)| = 3 for double-well.

**Effect:** Explicit term becomes contractive, unconditionally stable.

### Convex-Concave Splitting (Eyre)

Split f(φ) into convex and concave parts:
```
f(φ) = f_convex(φ) + f_concave(φ)

f_convex = (c/2)φ²  where c ≥ max f''
f_concave = f(φ) - (c/2)φ²
```

**Scheme:**
```
(φ_{n+1} - φ_n)/dt = M[ε²∇²φ_{n+1} - f'_convex(φ_{n+1}) - f'_concave(φ_n)]
```

**Properties:**
- Unconditionally energy stable
- Unique solvability
- First-order in time

### Second-Order Convex Splitting

Using Crank-Nicolson for spatial:
```
(φ_{n+1} - φ_n)/dt = M[ε²∇²(φ_{n+1}+φ_n)/2
                      - f'_convex(φ_{n+1})
                      - f'_concave(φ_n)]
```

Or BDF2:
```
(3φ_{n+1} - 4φ_n + φ_{n-1})/(2dt) = M[ε²∇²φ_{n+1}
                                     - f'_convex(φ_{n+1})
                                     - f'_concave(2φ_n - φ_{n-1})]
```

## Cahn-Hilliard Splitting

### Fourth-Order Challenge

Cahn-Hilliard involves ∇⁴:
```
∂c/∂t = M∇²μ = M∇²(f'(c) - ε²∇²c)
```

**Direct discretization:**
- Explicit: dt ~ dx⁴ (very restrictive!)
- Implicit: fourth-order operator

### Mixed Formulation

Introduce chemical potential as separate variable:
```
∂c/∂t = M∇²μ
μ = f'(c) - ε²∇²c
```

**Two second-order equations** instead of one fourth-order.

### IMEX for Mixed Form

```
Implicit: -ε²∇²c (in μ equation)
          ∇²μ (in c equation)
Explicit: f'(c)
```

**Linear system per step:**
```
[I      ε²L  ] [c_{n+1}]   [c_n + dt×M×L×μ_{n+1}]
[-1   I     ] [μ_{n+1}] = [f'(c_n)              ]
```

Where L = discrete Laplacian.

### Convex Splitting for Cahn-Hilliard

```
f(c) = f_convex(c) + f_concave(c)

(c_{n+1} - c_n)/dt = M∇²[f'_convex(c_{n+1}) + f'_concave(c_n) - ε²∇²c_{n+1}]
```

**Properties:**
- Energy decreasing: F[c_{n+1}] ≤ F[c_n]
- Mass conserving: ∫c dx = constant
- Unconditionally stable

### Scalar Auxiliary Variable (SAV)

Introduce r(t) = √(F[c] + C) where C ensures positivity:

```
dc/dt = M∇²μ
μ = r(t)/√(F+C) × δF/δc - ε²∇²c
dr/dt = (1/2√(F+C)) ∫ (δF/δc) × (dc/dt) dx
```

**Advantages:**
- Linear implicit solve
- Energy stable
- Second-order possible

## Coupled Multi-Phase Systems

### Two-Phase with Concentration

φ: phase field, c: concentration
```
∂φ/∂t = -M_φ × δF/δφ
∂c/∂t = ∇·(M_c ∇(δF/δc))
```

**Splitting strategy:**
```
Step 1: Evolve φ with c fixed (Allen-Cahn)
Step 2: Evolve c with φ fixed (diffusion with phase-dependent mobility)
```

**Coupling terms:**
- Free energy: f(φ,c) couples fields
- Mobility: M_c(φ) may depend on phase

### Multi-Grain Systems

N order parameters φ₁, ..., φ_N with constraint:
```
Σᵢ φᵢ² = 1  (at interfaces)
```

**Sequential update:**
```
For i = 1 to N:
    Update φᵢ with φⱼ (j≠i) fixed
    Enforce constraint
```

**Projection method:**
```
Update all φᵢ without constraint
Project: φᵢ → φᵢ / √(Σⱼ φⱼ²)
```

### Phase-Field Crystal

```
∂ψ/∂t = ∇²[(r + (1+∇²)²)ψ + ψ³]
```

**Splitting:**
```
Implicit: (1+∇²)²ψ (sixth-order!)
Explicit: rψ + ψ³
```

Requires careful treatment due to high-order derivatives.

## Anisotropy Handling

### Weakly Anisotropic

Interface energy: γ(n) = γ₀(1 + ε₄cos(4θ))

**Gradient energy:**
```
W(∇φ) = (1/2)|∇φ|² × a(n)²
```

**IMEX approach:**
```
Implicit: isotropic part (1/2)|∇φ|²
Explicit: anisotropic correction
```

### Strongly Anisotropic

When a(n)² can become negative (missing orientations):

**Regularization:**
```
W_reg = (1/2)|∇φ|² × a(n)² + (β/2)|∇∇φ|²
```

Corner regularization term requires implicit treatment.

### Faceted Interfaces

For crystalline anisotropy (Wulff shapes with corners):
```
Use Willmore regularization or
Level-set with crystalline curvature
```

Specialized schemes needed (not standard IMEX).

## Adaptive Mesh Refinement (AMR)

### Where to Refine

Refine near interfaces where gradients are large:
```
Criterion: |∇φ| > threshold
        or |φ - 0.5| < δ
```

### Time Stepping with AMR

**Subcycling:**
```
Fine levels: dt_fine = dt_coarse / ratio
Synchronize at coarse step
```

**Splitting with AMR:**
```
1. Coarse step for bulk
2. Fine steps for interface region
3. Synchronize solutions at boundaries
```

## Practical Guidelines

### Time Step Selection

| Model | Limiting Factor | Typical dt |
|-------|-----------------|------------|
| Allen-Cahn | Interface motion | dx²/(M×ε²) |
| Cahn-Hilliard | Fourth derivative | dx⁴/(M×ε²) |
| Coupled | Slowest physics | min of above |

With IMEX: can exceed explicit limits by 10-100×.

### Convergence Checking

1. **Energy decay:** dF/dt ≤ 0 (thermodynamically consistent)
2. **Mass conservation:** ∫c dx = const (for Cahn-Hilliard)
3. **Interface width:** Check ε² matches expected profile
4. **Steady state:** φ converges to equilibrium

### Common Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| Energy increase | F[n+1] > F[n] | Use convex splitting |
| Interface sharpening | φ goes outside [0,1] | Reduce dt, add stabilization |
| Mass loss | ∫c drifts | Use conservative discretization |
| Wrong interface width | Profile too sharp/wide | Check ε, dx relationship |
| Slow convergence | Many iterations | Better preconditioner |

### Interface Resolution

Rule of thumb:
```
dx ≤ ε / 3  (at least 3 points across interface)
Better: dx ≤ ε / 5  (5-6 points)
```

For IMEX stability:
```
dt ≤ C × dx² / (M × ε²)  (explicit part)
```

With stabilization, C can be O(1) instead of O(0.1).

## Verification Benchmarks

### Allen-Cahn

1. **Shrinking circle:** R(t) = √(R₀² - 2Mt)
2. **Traveling wave:** Compare to analytical profile
3. **Equilibrium:** Flat interface, f'(φ) = 0

### Cahn-Hilliard

1. **Spinodal decomposition:** Compare coarsening rate
2. **Ostwald ripening:** R ~ t^{1/3} for droplets
3. **Equilibrium:** μ = constant, phase fractions correct

### Multi-Order-Parameter

1. **Grain growth:** Normal grain growth law
2. **Triple junction:** 120° angles for isotropic
3. **Wetting:** Contact angle vs surface energies
