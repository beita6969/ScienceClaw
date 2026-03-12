# Tolerance Guidelines

Comprehensive guide for setting integration tolerances in time-dependent simulations.

## Fundamental Concepts

### Relative vs Absolute Tolerance

```
Local error estimate ≤ atol + rtol × |y|
```

| Tolerance | Controls | When Dominates |
|-----------|----------|----------------|
| rtol | Relative accuracy (significant digits) | Large values |
| atol | Absolute accuracy (noise floor) | Small values |

### Error Interpretation

- **rtol = 1e-3**: ~3 significant digits accuracy
- **rtol = 1e-6**: ~6 significant digits accuracy
- **atol = 1e-10**: Values below 1e-10 treated as "zero"

## Default Starting Points

### By Application Type

| Application | rtol | atol | Notes |
|-------------|------|------|-------|
| Exploratory/debugging | 1e-2 | 1e-4 | Fast but coarse |
| Engineering design | 1e-3 | 1e-6 | Good balance |
| Validation studies | 1e-4 | 1e-8 | Publication quality |
| Benchmark/reference | 1e-6 | 1e-10 | Near machine precision |
| Coupled multiphysics | 1e-4 | 1e-7 | Conservative choice |

### By Problem Type

| Problem | rtol | atol | Reasoning |
|---------|------|------|-----------|
| Diffusion (smooth) | 1e-3 | 1e-6 | Errors smooth out |
| Wave propagation | 1e-4 | 1e-8 | Phase errors accumulate |
| Stiff chemistry | 1e-5 | 1e-10 | Fast modes need accuracy |
| Phase-field | 1e-4 | 1e-8 | Interface needs precision |
| Turbulence (DNS) | 1e-4 | 1e-8 | Energy cascade sensitive |

## Multicomponent Systems

### The Scaling Problem

When variables have vastly different magnitudes:

```
Temperature: O(1000 K)
Concentration: O(1e-6 mol/L)
Velocity: O(10 m/s)
```

A single atol cannot work for all!

### Solution: Component-wise atol

```python
# Python/SciPy example
atol = [1e-3, 1e-12, 1e-5]  # T, C, V
rtol = 1e-4  # Same for all
```

### Determining Component atol

| Variable Type | atol Rule |
|---------------|-----------|
| Temperature | 0.001 × typical_T |
| Concentration | 1e-3 × smallest_meaningful_C |
| Phase fraction | 1e-6 (order parameter) |
| Velocity | 1e-3 × typical_U |
| Pressure | 1e-3 × typical_P |

### Nondimensionalization Alternative

```
Scale all variables to O(1):
T* = T / T_ref
C* = C / C_ref
Then use atol = 1e-6 for all
```

**Advantages:**
- Simpler tolerance specification
- Better numerical conditioning
- Easier debugging

## Dimensional Analysis Approach

### Physical Scale Identification

1. **Identify characteristic scales**
   - Length: L (domain size or feature size)
   - Time: τ (diffusion time, reaction time)
   - Velocity: U (imposed or derived)
   - Temperature: ΔT (driving temperature difference)

2. **Compute derived scales**
   - Concentration change: ΔC = C_initial - C_eq
   - Energy: ρ × c_p × ΔT × L³

3. **Set atol**
   - atol_T = 1e-3 × ΔT
   - atol_C = 1e-3 × ΔC
   - atol_U = 1e-3 × U

## Diagnosing Tolerance Issues

### Symptoms and Solutions

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Many rejected steps | rtol too tight | Loosen rtol by 10× |
| Noisy solution | rtol too loose | Tighten rtol by 10× |
| Negative concentrations | atol too large | Reduce atol for that component |
| Conservation violation | tolerances too loose | Tighten both |
| Slow convergence | atol >> solution values | Reduce atol |
| dt shrinks to minimum | Stiffness, not tolerance | Switch to implicit method |

### Diagnostic Workflow

```
1. Run with default (rtol=1e-3, atol=1e-6)
2. Check:
   - Step rejection rate < 10%? If not, loosen
   - Conservation errors acceptable? If not, tighten
   - Physical variables positive? If not, reduce atol
3. Tighten by 10× and re-run
4. If results change significantly, tolerances were too loose
```

## Conservation and Tolerance

### Mass Conservation

For conserved quantities (mass, energy):
```
Relative conservation error ≈ rtol × (integration time / characteristic time)
```

**Example:**
- rtol = 1e-4
- Integration: 1000 time units
- Characteristic time: 1 unit
- Expected drift: ~1e-4 × 1000 = 0.1 (10% error!)

**Solution:** Use tighter rtol for long integrations or specialized conservative schemes.

### Energy Drift in Hamiltonian Systems

| Method | Energy Error |
|--------|--------------|
| RK4 | Drifts linearly with t |
| Symplectic | Bounded, oscillates |
| Energy-preserving | Exact to tolerance |

For long-time simulations, prefer structure-preserving methods.

## Stiff Problems

### Special Considerations

1. **Stiff components need tighter atol**
   - Fast modes decay to small values quickly
   - Large atol masks stiff dynamics

2. **Watch for order reduction**
   - Near stability boundary, effective order drops
   - May need tighter rtol to compensate

3. **Jacobian accuracy matters**
   - Numerical Jacobian errors ∝ sqrt(eps)
   - May need tighter tolerances with numerical Jacobian

### Recommended Settings for Stiff

| Stiffness | rtol | atol |
|-----------|------|------|
| Mildly stiff (ratio ~100) | 1e-4 | 1e-8 |
| Moderately stiff (~10⁴) | 1e-5 | 1e-10 |
| Very stiff (~10⁶+) | 1e-6 | 1e-12 |

## Practical Examples

### Phase-Field Simulation

```python
# Two-phase Allen-Cahn
# phi: order parameter [-1, 1]
# T: temperature [300-2000 K]

rtol = 1e-4
atol_phi = 1e-6  # Order parameter precision
atol_T = 1e-1    # 0.1 K absolute accuracy

atol = [atol_phi, atol_T]
```

### Reactive Flow

```python
# Combustion with trace species
# Major species: O(1)
# Minor species: O(1e-6)
# Temperature: O(1000)

rtol = 1e-5  # Tighter for stiff chemistry
atol_major = 1e-8
atol_minor = 1e-12  # Detect small species
atol_T = 1e-2

atol = [..., atol_minor, ..., atol_T]
```

### Solidification

```python
# Phase-field solidification
# phi: solid fraction [0, 1]
# C: concentration [0.01 - 0.05]
# T: temperature [1700-1800 K]

rtol = 1e-4
atol = [1e-6,   # phi (interface precision)
        1e-6,   # C (microsegregation)
        0.01]   # T (0.01 K accuracy)
```

## Tolerance Tightening Protocol

### Convergence Study

1. Run with baseline tolerances (tol₀)
2. Run with tol₀/10
3. Compare key metrics (peak values, integrals, times)
4. If difference < acceptable threshold, tol₀ is adequate
5. If not, repeat with tighter tolerances

### What to Compare

| Metric | Acceptable Difference |
|--------|----------------------|
| Peak temperature | < 1% |
| Total energy | < 0.1% |
| Interface position | < dx/2 |
| Reaction completion time | < 1% |
| Species concentrations | < 1% |

## Common Mistakes

### 1. Same atol for all components
**Problem:** Variables with small physical values are ignored.
**Solution:** Use component-wise atol scaled to each variable.

### 2. atol = 0
**Problem:** Division by zero when solution passes through zero.
**Solution:** Always use small positive atol.

### 3. rtol too tight for noisy data
**Problem:** Excessive step rejection, slow progress.
**Solution:** Match rtol to data precision.

### 4. Ignoring units
**Problem:** atol = 1e-6 means nothing without units.
**Solution:** Express atol in simulation units: "1e-6 mol/L" or "0.1 K".
