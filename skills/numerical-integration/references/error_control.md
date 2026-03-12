# Error Control

Comprehensive guide for adaptive step size control in time integration.

## Error Norm Computation

### Scaled Error Norm

The standard approach scales the error by tolerance:

```
scale_i = atol_i + rtol × max(|y_n,i|, |y_{n+1},i|)
err_scaled_i = error_i / scale_i
```

### Norm Types

| Norm | Formula | Use Case |
|------|---------|----------|
| RMS (L2) | sqrt(Σ err_i² / n) | General purpose, smooth |
| Max (L∞) | max(\|err_i\|) | Strict component control |
| Weighted RMS | sqrt(Σ w_i × err_i² / Σ w_i) | Variable importance |

**RMS Norm (Recommended Default):**
```
error_norm = sqrt( (1/n) × Σᵢ (error_i / scale_i)² )
```

**Max Norm (Conservative):**
```
error_norm = max_i |error_i / scale_i|
```

### When to Use Each Norm

| Scenario | Recommended | Reasoning |
|----------|-------------|-----------|
| Most problems | RMS | Balanced, smooth control |
| Safety-critical | Max | No component exceeds tolerance |
| Many components | RMS | Max overly conservative |
| Localized dynamics | Max | Catch local errors |
| Smooth problems | RMS | Reduces rejected steps |

## Step Acceptance Criterion

```
IF error_norm ≤ 1.0:
    ACCEPT step
    Possibly increase dt
ELSE:
    REJECT step
    Decrease dt
    Retry with smaller step
```

### Near-Acceptance Zone

| error_norm | Action |
|------------|--------|
| < 0.5 | Accept, increase dt aggressively |
| 0.5 - 1.0 | Accept, mild dt increase |
| 1.0 - 2.0 | Reject, mild dt decrease |
| > 2.0 | Reject, aggressive dt decrease |

## Step Size Controllers

### I-Controller (Integral/Proportional)

Simplest controller based on current error only:

```
dt_new = dt × safety × error_norm^(-1/(p+1))
```

Where:
- safety = 0.8 - 0.9 (prevents oscillation)
- p = method order

**Characteristics:**
- Simple implementation
- May oscillate if error varies rapidly
- Suitable for smooth problems

### PI-Controller (Proportional-Integral)

Uses current and previous error for smoother adaptation:

```
dt_new = dt × safety × error_norm^(-α) × error_prev^(β)
```

Standard coefficients:
```
α = 0.7 / (p + 1)
β = 0.4 / (p + 1)
```

Alternative (Gustafsson):
```
α = 1 / (p + 1)
β = 1 / (p + 1) × (dt_prev / dt)
```

**Characteristics:**
- Smoother step size evolution
- Reduces oscillations
- Industry standard choice

### PID-Controller

Adds derivative term for aggressive adaptation:

```
dt_new = dt × safety × error_norm^(-α) × error_prev^(β) × error_prev2^(γ)
```

Typical coefficients:
```
α = 0.49 / (p + 1)
β = 0.34 / (p + 1)
γ = 0.10 / (p + 1)
```

**Characteristics:**
- Most aggressive adaptation
- Best for rapidly changing dynamics
- Risk of oscillation if poorly tuned

### Controller Comparison

| Controller | Smoothness | Responsiveness | Complexity |
|------------|------------|----------------|------------|
| I | Low | Immediate | Simple |
| PI | Medium | Balanced | Moderate |
| PID | High | Fast | Complex |

## Safety Factors and Limits

### Safety Factor

```
safety = 0.8 - 0.9 (typical)
```

**Purpose:**
- Prevents accepting steps at exact tolerance boundary
- Accounts for error estimate uncertainty
- Reduces rejected steps due to fluctuations

**Tuning:**
- Conservative (stiff/sensitive): safety = 0.8
- Aggressive (smooth): safety = 0.95

### Step Size Limits

```
dt_min = ε_machine × |t| × 100  (prevent stall)
dt_max = T_end / 10  (or physics-based limit)
```

**Max step increase per step:**
```
dt_new ≤ factor_max × dt  (typically factor_max = 2-5)
```

**Max step decrease per step:**
```
dt_new ≥ factor_min × dt  (typically factor_min = 0.1-0.2)
```

### Limiting Formulas

```python
# Apply all limits
dt_new = dt × safety × error_norm^(-1/(p+1))
dt_new = max(dt_new, dt × factor_min)
dt_new = min(dt_new, dt × factor_max)
dt_new = max(dt_new, dt_min)
dt_new = min(dt_new, dt_max)
```

## Error Estimation Methods

### Embedded Methods

Use two methods of different orders in one step:

```
y_high = O(h^{p+1}) method result
y_low  = O(h^p) embedded result
error  = y_high - y_low
```

| Method Pair | Orders | Evaluations |
|-------------|--------|-------------|
| RK45 (Dormand-Prince) | 5(4) | 6 |
| RK78 (Fehlberg) | 7(8) | 13 |
| RK23 (Bogacki-Shampine) | 3(2) | 4 |

**Advantage:** Error estimate is "free" from the integration.

### Richardson Extrapolation

Compare full step with two half-steps:

```
y_full = one step of size h
y_half = two steps of size h/2
error = (y_half - y_full) / (2^p - 1)
```

**Advantage:** Works with any method.
**Disadvantage:** Doubles function evaluations.

### Milne Device (Multi-step)

Compare predictor and corrector:

```
y_predict = Adams-Bashforth (explicit)
y_correct = Adams-Moulton (implicit)
error = (y_correct - y_predict) × C
```

Where C depends on method orders.

## Handling Rejected Steps

### Standard Approach

```
1. Compute y_new and error estimate
2. If error_norm > 1:
   a. Reduce dt using controller
   b. Discard y_new
   c. Retry from y_current with smaller dt
3. If many consecutive rejections:
   a. Check for stiffness
   b. Consider method change
```

### Rejection Statistics

| Rejection Rate | Interpretation | Action |
|----------------|----------------|--------|
| < 5% | Normal | None |
| 5-20% | Borderline | Consider loosening tol |
| 20-50% | Problem | Check for stiffness |
| > 50% | Severe | Wrong method or extreme stiffness |

### After Rejection Strategy

**Conservative:**
```
dt_new = dt × 0.5  (halve step)
```

**Controller-based:**
```
dt_new = dt × safety × error_norm^(-1/(p+1))
```

**Minimum reduction:**
```
dt_new = max(dt × 0.1, dt_min)  (never reduce by more than 10×)
```

## Special Situations

### Output Points

When output is required at specific times:

```
1. Integrate to t_out with adaptive stepping
2. Last step may need adjustment to hit t_out exactly
3. Options:
   a. Interpolate from nearby steps (dense output)
   b. Force exact step to t_out
   c. Accept small overstep and interpolate back
```

### Discontinuities

When a discontinuity is detected (event, switch):

```
1. Locate discontinuity time t_disc (bisection/Newton)
2. Integrate exactly to t_disc
3. Apply discontinuity (jump conditions)
4. Restart integrator with small initial step
5. Rebuild multi-step history if needed
```

### Stiffness Detection

Monitor for implicit stiffness transition:

```
IF step rejections increase rapidly:
   Estimate stiffness ratio
   IF ratio > threshold (e.g., 1000):
       Switch to implicit method
```

## Implementation Example

```python
def adaptive_step(y, t, dt, f, tol, order, controller='pi'):
    """One adaptive step with PI control."""
    safety = 0.9
    factor_min, factor_max = 0.2, 5.0

    # Error history for PI controller
    global error_prev

    while True:
        # Take step and estimate error
        y_new, error = step_with_error(y, t, dt, f)

        # Compute scaled error norm
        scale = tol['atol'] + tol['rtol'] * np.maximum(np.abs(y), np.abs(y_new))
        error_norm = np.sqrt(np.mean((error / scale)**2))

        if error_norm <= 1.0:
            # Accept step
            if controller == 'pi':
                factor = safety * error_norm**(-0.7/(order+1)) * error_prev**(0.4/(order+1))
            else:  # 'i' controller
                factor = safety * error_norm**(-1/(order+1))

            factor = np.clip(factor, factor_min, factor_max)
            dt_new = dt * factor
            error_prev = error_norm

            return y_new, t + dt, dt_new, True
        else:
            # Reject step
            factor = safety * error_norm**(-1/(order+1))
            factor = max(factor, factor_min)
            dt = dt * factor
            # Loop continues with smaller dt
```

## Tuning Guidelines

### Start Conservative

1. Begin with safety = 0.8, PI controller
2. Use default α, β coefficients
3. Set factor_max = 2.0

### Tune for Efficiency

1. If few rejections, increase safety to 0.9
2. If smooth evolution, increase factor_max to 5.0
3. If oscillating dt, reduce α or switch to PI

### Monitor Health

Track and report:
- Total step count
- Rejected step count
- Minimum dt reached
- Maximum error norm encountered
