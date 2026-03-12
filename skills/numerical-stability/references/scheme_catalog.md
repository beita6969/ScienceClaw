# Numerical Scheme Stability Catalog

## Overview

This catalog summarizes stability properties of common time integration and spatial discretization schemes used in PDE simulations.

---

## Time Integration Schemes

### Explicit Methods

| Scheme | Order | Stability | CFL Limit | Memory | Notes |
|--------|-------|-----------|-----------|--------|-------|
| Forward Euler | 1 | Conditional | Region-dependent | Low | Simple but diffusive |
| RK2 (Heun) | 2 | Conditional | C ≤ ~1 | Low | Better accuracy |
| RK4 (Classic) | 4 | Conditional | C ≤ ~1.4 | Low | Most common explicit |
| RK45 (Dormand-Prince) | 4/5 | Conditional | Adaptive | Medium | Error estimation |
| Adams-Bashforth 2 | 2 | Conditional | C ≤ ~0.5 | Low | Multi-step, cheaper |
| Adams-Bashforth 4 | 4 | Conditional | C ≤ ~0.3 | Low | More restrictive |
| Leapfrog | 2 | Neutral | C ≤ 1 | Low | No damping, needs filter |

### Implicit Methods

| Scheme | Order | Stability | Properties | Memory | Notes |
|--------|-------|-----------|------------|--------|-------|
| Backward Euler | 1 | A-stable | Unconditional | Medium | Very diffusive |
| Crank-Nicolson | 2 | A-stable | Unconditional | Medium | May oscillate |
| BDF2 | 2 | A-stable | Unconditional | Medium | Good for stiff |
| BDF3-6 | 3-6 | A(α)-stable | Nearly unconditional | Medium | Higher order |
| Radau IIA | 3,5 | L-stable | Unconditional | High | Very stiff problems |
| SDIRK | 2-4 | L-stable | Unconditional | Medium | Good balance |

### IMEX Methods

| Scheme | Order | Implicit Part | Explicit Part | Use Case |
|--------|-------|---------------|---------------|----------|
| IMEX-Euler | 1 | Backward Euler | Forward Euler | Simple stiff/nonstiff |
| IMEX-RK2 | 2 | SDIRK | RK2 | Moderate stiffness |
| ARK4(3)6L | 4 | L-stable | RK4 | High-order IMEX |

---

## Spatial Discretization Schemes

### First Derivatives (Advection)

| Scheme | Order | Stability | Dispersion | Diffusion |
|--------|-------|-----------|------------|-----------|
| Upwind (1st order) | 1 | Stable for C ≤ 1 | High | Adds artificial |
| Central (2nd order) | 2 | **Unstable** | Moderate | None |
| Upwind (3rd order) | 3 | Stable for C ≤ 1 | Low | Small artificial |
| QUICK | 3 | Conditional | Low | Small |
| WENO5 | 5 | Conditional | Very low | Adaptive |

### Second Derivatives (Diffusion)

| Scheme | Order | Stability (1D) | Notes |
|--------|-------|----------------|-------|
| Central (3-point) | 2 | Fo ≤ 0.5 | Standard |
| Central (5-point) | 4 | Fo ≤ 0.5 | Higher accuracy |
| Compact (Padé) | 4-6 | Similar | Implicit system |

---

## Stability Regions

### A-Stability

A method is **A-stable** if stable for all `z = λ·dt` with `Re(λ) < 0`.

```
A-stable methods:
✓ Backward Euler
✓ Crank-Nicolson
✓ BDF1-2
✓ Radau IIA
✓ All fully implicit methods

Not A-stable:
✗ All explicit methods
✗ BDF3-6 (A(α)-stable only)
✗ Adams methods
```

### L-Stability

A method is **L-stable** if A-stable and `|R(z)| → 0` as `Re(z) → -∞`.

```
L-stable methods:
✓ Backward Euler
✓ Radau IIA
✓ SDIRK methods

A-stable but not L-stable:
✗ Crank-Nicolson (oscillates for very stiff)
✗ Trapezoidal rule
```

---

## Scheme Selection Guide

### By Problem Type

| Problem Type | Recommended Schemes |
|--------------|---------------------|
| Smooth advection | RK4 + Upwind/Central |
| Advection with shocks | TVD, WENO, DG |
| Diffusion-dominated | Implicit or Crank-Nicolson |
| Stiff reactions | BDF, Radau, Rosenbrock |
| Wave propagation | Leapfrog + filter, DG |
| Navier-Stokes | IMEX, projection methods |

### By Accuracy Requirement

| Accuracy | Time Integration | Spatial |
|----------|------------------|---------|
| Low (1st order) | Forward/Backward Euler | Upwind |
| Medium (2nd order) | RK2, Crank-Nicolson | Central, MUSCL |
| High (4th order) | RK4, BDF4 | Compact, DG |
| Very high (6th+) | DOP853, Spectral | Spectral, high-order DG |

---

## Stability Tables

### Forward Euler + Central Diffusion

| Dimensions | Fourier Limit | Max dt Formula |
|------------|---------------|----------------|
| 1D | Fo ≤ 0.5 | dt ≤ dx²/(2D) |
| 2D | Fo ≤ 0.25 | dt ≤ dx²/(4D) |
| 3D | Fo ≤ 0.167 | dt ≤ dx²/(6D) |

### RK4 + Central Advection

| Property | Value |
|----------|-------|
| CFL Limit | C ≤ 2√2 ≈ 2.83 (linear) |
| Practical Limit | C ≤ 1.5 (nonlinear) |

### BDF Methods

| Order | Stability | Error Constant |
|-------|-----------|----------------|
| BDF1 | A-stable, L-stable | 1 |
| BDF2 | A-stable | 2/3 |
| BDF3 | A(86°)-stable | 6/11 |
| BDF4 | A(73°)-stable | 12/25 |
| BDF5 | A(51°)-stable | 60/137 |
| BDF6 | A(17°)-stable | 60/147 |

---

## Comparison Matrix

| Feature | FE | BE | CN | RK4 | BDF2 | Radau |
|---------|----|----|----|----|------|-------|
| Order | 1 | 1 | 2 | 4 | 2 | 3-5 |
| A-stable | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| L-stable | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| Linear solve | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| Memory | Low | Med | Med | Low | Med | High |
| Stiff OK | ✗ | ✓ | ⚠️ | ✗ | ✓ | ✓ |

Legend: FE=Forward Euler, BE=Backward Euler, CN=Crank-Nicolson

---

## Quick Reference

```
NON-STIFF:        RK4, RK45, Adams-Bashforth
STIFF:            BDF2, Radau, SDIRK, Rosenbrock
MIXED:            IMEX schemes
WAVE PROPAGATION: Leapfrog, symplectic
SHOCK CAPTURING:  TVD, WENO, DG
```
