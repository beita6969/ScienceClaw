# Stability Criteria Reference

## Overview

This document provides comprehensive stability criteria for explicit time integration of PDEs. Use these formulas to determine maximum stable time steps.

---

## Core Nondimensional Numbers

### Courant-Friedrichs-Lewy (CFL) Number

For advection-dominated problems:

```
C = v × dt / dx
```

Where:
- `v` = advection velocity (m/s)
- `dt` = time step (s)
- `dx` = grid spacing (m)

**Physical meaning**: Ratio of distance traveled per time step to grid spacing.

### Fourier Number

For diffusion-dominated problems:

```
Fo = D × dt / dx²
```

Where:
- `D` = diffusivity (m²/s)
- `dt` = time step (s)
- `dx` = grid spacing (m)

**Physical meaning**: Ratio of diffusion distance per time step to grid spacing squared.

### Reaction Number

For reaction-dominated problems:

```
R = k × dt
```

Where:
- `k` = reaction rate constant (1/s)
- `dt` = time step (s)

**Physical meaning**: Number of reaction time constants per time step.

---

## Explicit Stability Limits

### Advection (1D)

| Scheme | CFL Limit | Notes |
|--------|-----------|-------|
| Upwind (first-order) | C ≤ 1 | Dissipative, stable |
| Lax-Friedrichs | C ≤ 1 | More dissipative |
| Lax-Wendroff | C ≤ 1 | Second-order, dispersive |
| FTCS (central) | **Unstable** | Never use for advection |
| Leapfrog | C ≤ 1 | Neutral stability, needs filter |

### Advection (Multi-dimensional)

For d dimensions with velocity components `v_i`:

```
dt ≤ dx / (sum of |v_i|)    [L1 norm]
dt ≤ dx / sqrt(sum of v_i²) [L2 norm, less restrictive]
```

### Diffusion (1D)

| Scheme | Fourier Limit | Notes |
|--------|---------------|-------|
| FTCS | Fo ≤ 0.5 | Standard explicit |
| DuFort-Frankel | Unconditional | But conditionally consistent |

### Diffusion (Multi-dimensional)

```
Fo ≤ 1 / (2 × d)
```

| Dimensions | Fourier Limit |
|------------|---------------|
| 1D | Fo ≤ 0.50 |
| 2D | Fo ≤ 0.25 |
| 3D | Fo ≤ 0.167 |

### Reaction

```
R ≤ 1   (or R ≤ 0.2 for safety margin)
```

For stiff reactions (k >> 1), explicit methods are inefficient.

---

## Combined Criteria

### Advection + Diffusion

When both are present, apply **both** limits:

```
dt ≤ min(dt_advection, dt_diffusion)

dt_advection = C_limit × dx / v
dt_diffusion = Fo_limit × dx² / D
```

### Grid Péclet Number

Ratio of advection to diffusion:

```
Pe = v × dx / D
```

| Pe | Dominant Physics |
|----|------------------|
| Pe << 1 | Diffusion-dominated |
| Pe ≈ 1 | Both important |
| Pe >> 1 | Advection-dominated |

**Note**: High Péclet numbers (Pe > 2) cause spurious oscillations with central schemes.

### Advection + Diffusion + Reaction

```
dt ≤ min(dt_adv, dt_diff, dt_react)
```

Use the most restrictive limit.

---

## Safety Factors

### Recommended Safety Margins

| Scenario | Safety Factor | Resulting dt |
|----------|---------------|--------------|
| Standard | 0.9 | 90% of limit |
| Strong coupling | 0.5-0.7 | 50-70% of limit |
| Stiff systems | 0.3-0.5 | 30-50% of limit |
| Near stability boundary | 0.8 | 80% of limit |

### When to Use Smaller Safety Factors

- Multiple coupled physics
- Nonlinear problems (limits may vary in time)
- Near phase transitions or sharp gradients
- Adaptive mesh refinement (dx changes)

---

## Special Cases

### Anisotropic Meshes

When `dx ≠ dy ≠ dz`:

```
dt_diff ≤ 1/(2D) × 1/(1/dx² + 1/dy² + 1/dz²)
```

Use the smallest spacing for the most conservative estimate.

### Variable Coefficients

When `v(x)` or `D(x)` vary in space:

```
Use maximum values: v_max, D_max
```

### Time-Dependent Coefficients

Re-evaluate stability at each time step or periodically.

### Nonlinear Problems

Estimate local velocity/diffusivity from solution and re-check limits.

---

## Decision Algorithm

```python
def compute_stable_dt(dx, v, D, k, dimensions, safety=0.9):
    dt_candidates = []

    # Advection limit
    if v > 0:
        dt_adv = dx / v  # CFL = 1
        dt_candidates.append(dt_adv)

    # Diffusion limit
    if D > 0:
        fo_limit = 1.0 / (2.0 * dimensions)
        dt_diff = fo_limit * dx**2 / D
        dt_candidates.append(dt_diff)

    # Reaction limit
    if k > 0:
        dt_react = 1.0 / k
        dt_candidates.append(dt_react)

    if dt_candidates:
        return min(dt_candidates) * safety
    else:
        return None  # No limits apply
```

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Ignoring multi-D factor | Instability in 2D/3D | Use `Fo ≤ 1/(2d)` |
| Using central diff for advection | Oscillations | Use upwind or limiters |
| No safety factor | Borderline stability | Apply 0.8-0.9 factor |
| Forgetting anisotropy | Instability | Use smallest dx |
| Ignoring variable coefficients | Instability in some regions | Use max values |

---

## Quick Reference Card

```
ADVECTION:  dt ≤ dx / v           (CFL ≤ 1)
DIFFUSION:  dt ≤ dx² / (2·d·D)    (Fo ≤ 1/(2d))
REACTION:   dt ≤ 1 / k            (R ≤ 1)
COMBINED:   dt = min(all limits) × safety
```
