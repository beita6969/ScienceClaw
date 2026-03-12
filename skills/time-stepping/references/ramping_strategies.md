# Ramping Strategies

Comprehensive guide for time step ramping during simulation startup and transients.

## Why Ramp Time Steps

### Initial Condition Problems

| Issue | Without Ramping | With Ramping |
|-------|-----------------|--------------|
| Sharp gradients | Instability | Smooth relaxation |
| Unphysical IC | Blow-up | Gradual correction |
| Stiff transients | Solver failure | Stable startup |
| Initialization artifacts | Propagate/amplify | Dissipate quickly |

### Physics Motivations

**Phase-field simulations:**
- Random initial conditions have high-frequency modes
- These modes are stiff and need small dt
- After relaxation, larger dt is stable

**Reactive flow:**
- Initial ignition creates steep gradients
- Flame establishes, then becomes smoother
- Full dt works after flame stabilizes

**Incompressible flow:**
- Pressure initialization may be inconsistent
- Ramping allows pressure to adjust

## Ramping Methods

### Linear Ramp

```
dt(n) = dt_init + (dt_target - dt_init) × (n / N_ramp)

for n = 0, 1, ..., N_ramp
then dt = dt_target for n > N_ramp
```

**Properties:**
- Simple to implement
- Predictable total ramp time
- Smooth increase

**Parameters:**
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| N_ramp | 10-50 steps | Short for simple problems |
| dt_init | 0.01 × dt_target | Conservative start |
| dt_target | CFL-limited dt | Full speed |

### Geometric (Exponential) Ramp

```
dt(n) = dt_init × growth_factor^n

until dt(n) ≥ dt_target, then use dt_target
```

**Properties:**
- Spans many orders of magnitude efficiently
- Good for very stiff startup
- Reaches target faster for extreme cases

**Parameters:**
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| growth_factor | 1.1 - 1.5 | Higher = faster ramp |
| dt_init | 1e-6 × dt_target | For very stiff |

### Adaptive Ramp

```
dt(n+1) = min(
    dt(n) × growth_factor,
    dt_limit,  # CFL/Fourier limit
    dt_target
)

where growth_factor = f(error_norm) or fixed (e.g., 1.2)
```

**Properties:**
- Respects stability limits
- Accelerates when safe
- Most robust approach

### Sigmoid (S-curve) Ramp

```
dt(t) = dt_init + (dt_target - dt_init) × S(t/t_ramp)

where S(x) = x³(10 - 15x + 6x²)  for x ∈ [0,1]
or    S(x) = 1/(1 + exp(-k(x - 0.5)))  (logistic)
```

**Properties:**
- Smooth start and finish
- No sudden jumps
- Good for sensitive physics

## Problem-Specific Strategies

### Phase-Field Initialization

**Random initial condition:**
```
Start: dt = 0.001 × (dx² / (M × ε²))
Ramp: geometric with factor 1.2
End: dt = 0.8 × (dx² / (M × ε²))
Duration: typically 10-20 steps
```

**Sharp interface initial condition:**
```
Start: dt = 0.01 × dt_target
Ramp: linear over 10 steps
End: dt_target
```

### Solidification Simulations

**Undercooled melt:**
```
Phase 1: Nucleation (very small dt)
  dt ~ 0.01 × dt_limit, 5-10 steps
Phase 2: Early growth (moderate dt)
  dt ~ 0.1-0.5 × dt_limit, 10-20 steps
Phase 3: Steady growth (full dt)
  dt = dt_limit
```

### Combustion/Ignition

**Pre-ignition:**
```
dt = dt_flow (CFL-limited)
```

**Ignition phase:**
```
dt = min(dt_flow, 1/k_max) where k_max is reaction rate
Ramp back up as flame stabilizes
```

### Cold-Start Fluid Dynamics

**Impulsive start:**
```
Step 0-5: dt = 0.1 × dt_CFL
Step 5-20: geometric ramp
Step 20+: dt = dt_CFL
```

**Smooth startup:**
```
Ramp velocity/BC instead of dt
Keeps dt constant, more predictable
```

## Implementation Patterns

### Simple Linear Ramp

```python
def get_dt_ramped(step, dt_init, dt_target, n_ramp):
    """Linear ramp from dt_init to dt_target."""
    if step >= n_ramp:
        return dt_target
    factor = step / n_ramp
    return dt_init + (dt_target - dt_init) * factor
```

### Geometric Ramp with Cap

```python
def get_dt_geometric(step, dt_init, growth, dt_limit, dt_target):
    """Geometric ramp, capped at limits."""
    dt = dt_init * (growth ** step)
    dt = min(dt, dt_limit)  # Stability limit
    dt = min(dt, dt_target)  # User target
    return dt
```

### Adaptive Ramp with Error Control

```python
def adapt_dt(dt_current, error_norm, dt_limit, growth_max=1.5):
    """Adapt dt based on error, respecting limits."""
    if error_norm < 0.5:
        # Can increase
        factor = min(growth_max, 0.9 / max(error_norm, 0.1))
    elif error_norm < 1.0:
        # Keep or slight increase
        factor = 1.0 + 0.1 * (1.0 - error_norm)
    else:
        # Must decrease
        factor = 0.7

    dt_new = dt_current * factor
    return min(dt_new, dt_limit)
```

## Ramping Duration Guidelines

### By Problem Type

| Problem | N_ramp (steps) | dt_init / dt_target |
|---------|----------------|---------------------|
| Smooth IC | 5-10 | 0.1-0.5 |
| Random IC | 10-30 | 0.01-0.1 |
| Sharp gradients | 20-50 | 0.01-0.05 |
| Stiff chemistry | 50-100 | 0.001-0.01 |
| Cold start | 10-20 | 0.1-0.2 |
| Phase change | 20-50 | 0.01-0.1 |

### By Initial Condition Quality

| IC Quality | Recommendation |
|------------|----------------|
| Exact equilibrium | Minimal ramp (5 steps) |
| Approximate equilibrium | Short ramp (10 steps) |
| Perturbed equilibrium | Moderate ramp (20 steps) |
| Random/noisy | Long ramp (50+ steps) |
| Discontinuous | Very conservative (100+ steps) |

## Safety Considerations

### Never Exceed Stability Limit

```
dt_ramped = min(
    ramp_value(step),
    stability_limit,
    user_maximum
)
```

Even during ramping, stability limits must be respected!

### Monitor for Problems

| Symptom | Action |
|---------|--------|
| Residuals growing | Slow ramp, reduce growth factor |
| Solution unbounded | Start with smaller dt_init |
| Oscillations | Use geometric instead of linear |
| Still unstable | Check if dt_target is stable |

### Ramp Restart After Events

**When to restart ramping:**
- After load balancing / mesh changes
- After remeshing / refinement
- After physical events (nucleation, fracture)
- After restoring from checkpoint with different parameters

## Integration with Checkpointing

### Saving Ramp State

```python
checkpoint = {
    'solution': u,
    'time': t,
    'dt_current': dt,
    'ramp_step': n,
    'ramp_complete': (n >= n_ramp)
}
```

### Restoring Ramp State

```python
def restore_from_checkpoint(checkpoint):
    if checkpoint['ramp_complete']:
        # Full speed
        return checkpoint['dt_current']
    else:
        # Continue ramping
        return get_dt_ramped(
            checkpoint['ramp_step'],
            dt_init, dt_target, n_ramp
        )
```

## Debugging Ramping Issues

### Issue: Ramp Too Fast

**Symptoms:**
- Instability during ramp
- Large residual spikes

**Solutions:**
- Reduce growth factor (1.1 instead of 1.5)
- Increase N_ramp
- Start with smaller dt_init

### Issue: Ramp Too Slow

**Symptoms:**
- Excessive wall time in startup
- Physics already stabilized

**Solutions:**
- Increase growth factor
- Decrease N_ramp
- Use adaptive ramp

### Issue: Never Reaches Target

**Symptoms:**
- dt stays small forever
- Error stays high

**Solutions:**
- Check if dt_target is actually stable
- Verify initial conditions are sensible
- Look for underlying physics issue

## Quick Reference

### Recommended Default Settings

```python
# Conservative default
n_ramp = 20
dt_init = 0.1 * dt_target
method = 'linear'

# For stiff problems
n_ramp = 50
dt_init = 0.01 * dt_target
method = 'geometric'
growth = 1.2

# For well-behaved problems
n_ramp = 10
dt_init = 0.2 * dt_target
method = 'linear'
```

### Ramping Checklist

- [ ] Determined appropriate N_ramp for problem type
- [ ] Set dt_init conservatively for IC quality
- [ ] Ensured stability limit respected at all times
- [ ] Tested that dt_target is stable after ramp
- [ ] Added monitoring for ramp issues
- [ ] Documented ramping strategy for reproducibility
