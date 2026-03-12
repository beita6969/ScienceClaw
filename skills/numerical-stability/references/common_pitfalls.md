# Common Stability Pitfalls

## Overview

This document catalogs frequent stability failures and their fixes. Use this as a diagnostic guide when simulations blow up.

---

## Pitfall Categories

| Category | Frequency | Severity |
|----------|-----------|----------|
| CFL Violations | Very Common | High |
| Stiffness Issues | Common | High |
| Boundary Conditions | Common | Medium |
| Numerical Precision | Occasional | Medium |
| Initialization | Occasional | High |

---

## CFL/Fourier Violations

### Symptom: Exponential Growth

**Pattern**: Solution grows exponentially, values reach 10^30+ within a few steps.

**Cause**: Time step exceeds stability limit.

**Diagnosis**:
```bash
python3 scripts/cfl_checker.py --dx YOUR_DX --dt YOUR_DT --diffusivity YOUR_D --velocity YOUR_V --dimensions YOUR_D
```

**Fix**:
1. Reduce `dt` to meet all limits
2. Or switch to implicit scheme
3. Or increase `dx` (reduces resolution)

### Symptom: Oscillations Then Blow-up

**Pattern**: Solution develops 2dx oscillations that grow unbounded.

**Cause**: Central differences for advection-dominated problems.

**Fix**:
1. Use upwind scheme for advection terms
2. Or add artificial diffusion
3. Or use flux limiters (TVD schemes)

---

## Stiffness Issues

### Symptom: Extremely Small Time Steps

**Pattern**: Adaptive stepping reduces dt to 10^-15 or smaller.

**Cause**: Stiff system with widely separated time scales.

**Diagnosis**:
```bash
python3 scripts/stiffness_detector.py --eigs=FAST_EIGENVALUE,SLOW_EIGENVALUE
```

**Fix**:
1. Use implicit or IMEX scheme
2. Identify and treat stiff terms implicitly
3. Use BDF, Radau, or Rosenbrock methods

### Symptom: Correct Results But Very Slow

**Pattern**: Simulation runs but takes millions of small steps.

**Cause**: Explicit scheme on stiff problem.

**Fix**: Same as above - switch to stiff solver.

---

## Boundary Condition Issues

### Symptom: Instability at Boundaries

**Pattern**: Blow-up starts at domain edges.

**Causes**:
1. Incompatible BC with interior scheme
2. Reflection of outgoing waves
3. Wrong-sided stencil at boundaries

**Fixes**:
| Problem | Solution |
|---------|----------|
| Wave reflection | Use absorbing/sponge layers |
| Stencil mismatch | Use one-sided differences at boundaries |
| Dirichlet oscillations | Check that BC is applied correctly |

### Symptom: Spurious Waves from Boundaries

**Pattern**: Waves emanate from boundaries into domain.

**Cause**: Non-physical boundary treatment.

**Fix**:
1. Use characteristic BCs for hyperbolic problems
2. Implement proper Neumann/Robin conditions
3. Add buffer zones

---

## Initialization Issues

### Symptom: Immediate Blow-up

**Pattern**: Solution diverges within first 1-10 steps.

**Causes**:
1. Initial condition violates conservation
2. Discontinuous IC with central schemes
3. IC incompatible with BCs

**Fixes**:
| Problem | Solution |
|---------|----------|
| Sharp discontinuities | Smooth IC or use shock-capturing |
| IC/BC mismatch | Ensure IC satisfies BCs |
| Non-physical IC | Check conservation laws |

### Symptom: Transient Spike Then Settling

**Pattern**: Large values early, then stabilizes.

**Cause**: IC not in equilibrium with dynamics.

**Fix**:
1. Ramp parameters gradually (see time-stepping skill)
2. Use smaller dt during startup
3. Pre-condition IC to equilibrium

---

## Numerical Precision Issues

### Symptom: Late-Time Instability

**Pattern**: Stable for many steps, then suddenly diverges.

**Causes**:
1. Accumulating round-off error
2. Loss of conservation
3. Condition number degradation

**Fixes**:
| Problem | Solution |
|---------|----------|
| Round-off accumulation | Use double precision, Kahan summation |
| Conservation loss | Use conservative discretization |
| Conditioning | Re-scale/equilibrate matrices |

### Symptom: Checkerboard Pattern

**Pattern**: Alternating high/low values in 2dx wavelength.

**Cause**: Odd-even decoupling (central schemes without stabilization).

**Fix**:
1. Add small diffusion
2. Use staggered grids
3. Apply filtering

---

## Multi-Physics Coupling Issues

### Symptom: Instability When Coupling

**Pattern**: Each physics stable alone, unstable when coupled.

**Causes**:
1. Operator splitting error
2. Incompatible time scales
3. Conservation violation at interface

**Fixes**:
| Problem | Solution |
|---------|----------|
| Splitting error | Use smaller dt or higher-order splitting |
| Time scale mismatch | Use IMEX or subcycling |
| Interface issues | Ensure conservation across coupling |

---

## Setup Mistakes

### Unit Mixing

**Pattern**: Inconsistent results or immediate blow-up.

**Cause**: Mixing units (e.g., `dx` in microns, `v` in m/s).

**Fix**: Convert all quantities to consistent units before computing.

### Mesh Refinement

**Pattern**: Instability after mesh refinement.

**Cause**: Reusing old `dt` after refining (dx changed).

**Fix**: Recompute dt after any mesh change.

### Anisotropic Grids

**Pattern**: Instability in one direction.

**Cause**: Using dx but ignoring smaller dy.

**Fix**: Use smallest grid spacing for stability calculations.

---

## Diagnostic Workflow

When simulation blows up:

```
1. WHERE does it start?
   ├── Boundaries → Check BC implementation
   ├── Interior → Check CFL/Fourier
   └── Everywhere → Check IC or stiffness

2. WHEN does it start?
   ├── Immediately → IC or gross CFL violation
   ├── After some time → Gradual instability or round-off
   └── At specific event → Check event handling

3. HOW does it manifest?
   ├── Exponential growth → Linear instability
   ├── Oscillations → Dispersion or central diff issue
   └── Checkerboard → Odd-even decoupling
```

---

## Prevention Checklist

Before running:
- [ ] Verify CFL/Fourier with `cfl_checker.py`
- [ ] Check stiffness ratio if multiple scales
- [ ] Validate IC satisfies BCs
- [ ] Test with smaller dt first
- [ ] Enable conservation monitoring

During run:
- [ ] Monitor max/min values
- [ ] Track residuals or energy
- [ ] Watch for early warning signs
- [ ] Log dt changes (if adaptive)

---

## Quick Fixes by Symptom

| Symptom | Quick Fix |
|---------|-----------|
| Exponential blow-up | Reduce dt by 2x |
| 2dx oscillations | Add upwinding or diffusion |
| Boundary instability | Check BC implementation |
| Very slow convergence | Switch to implicit |
| Checkerboard | Add stabilization or filter |
| Conservation drift | Use conservative form |
