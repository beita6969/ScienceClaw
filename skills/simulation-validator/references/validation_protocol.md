# Validation Protocol

## Overview

This document defines the complete validation protocol for materials simulations. Each stage has specific checks, criteria, and actions.

---

## Stage 1: Pre-flight Checks

Run before simulation starts. Any BLOCKER prevents simulation launch.

### 1.1 Configuration Validation

| Check | Criteria | Severity |
|-------|----------|----------|
| Required parameters present | All mandatory fields exist in config | BLOCKER |
| Parameter types correct | Numeric fields are numbers, not strings | BLOCKER |
| Parameter ranges valid | Values within physical bounds | BLOCKER |
| Units consistent | All parameters use compatible units | WARNING |

### 1.2 Stability Checks

| Check | Criteria | Severity |
|-------|----------|----------|
| CFL condition | `dt <= dx / v_max` for explicit schemes | BLOCKER |
| Diffusion stability | `dt <= dx^2 / (2 * D * dim)` | BLOCKER |
| Courant number | `C = v * dt / dx < 1` for advection | WARNING |
| Mesh resolution | `dx` resolves smallest features | WARNING |

### 1.3 Material Properties

| Check | Criteria | Severity |
|-------|----------|----------|
| Source documented | Material data has citation/source | WARNING |
| Physical bounds | Properties are physically reasonable | BLOCKER |
| Temperature dependence | If T-dependent, valid for T range | WARNING |
| Consistency | No contradictory property values | BLOCKER |

### 1.4 System Resources

| Check | Criteria | Severity |
|-------|----------|----------|
| Output directory exists | Path is valid and accessible | WARNING |
| Directory writable | Write permissions confirmed | BLOCKER |
| Sufficient disk space | Free space > estimated output size | BLOCKER |
| Memory estimate | RAM sufficient for mesh size | WARNING |

### 1.5 Pre-flight Decision Matrix

| Blockers | Warnings | Action |
|----------|----------|--------|
| 0 | 0 | PASS - Proceed immediately |
| 0 | 1+ | WARN - Review and document |
| 1+ | Any | BLOCK - Do not proceed |

---

## Stage 2: Runtime Checks

Monitor during simulation execution. Alert thresholds are configurable.

### 2.1 Convergence Monitoring

| Metric | Alert Condition | Likely Cause |
|--------|-----------------|--------------|
| Residual growth | `r[n] / r[n-1] > threshold` | Instability, bad IC |
| Residual stagnation | `r[n] ≈ r[n-k]` for many k | Solver stuck |
| Iteration count | Near max iterations | Ill-conditioned system |

**Default Thresholds:**
- Residual growth threshold: 10x (single step)
- Cumulative growth threshold: 1000x (over run)
- Stagnation window: 100 iterations

### 2.2 Time Step Monitoring

| Metric | Alert Condition | Likely Cause |
|--------|-----------------|--------------|
| dt reduction | `dt_max / dt_min > threshold` | Stability issues |
| dt collapse | `dt < dt_min_allowed` | Imminent failure |
| dt oscillation | Repeated up/down cycles | Instability boundary |

**Default Thresholds:**
- dt reduction threshold: 100x
- dt minimum: 1e-12 (problem-dependent)

### 2.3 Conservation Monitoring

| Quantity | Alert Condition | Tolerance |
|----------|-----------------|-----------|
| Mass | `|M(t) - M(0)| / M(0) > tol` | 1e-6 to 1e-3 |
| Energy | `E(t) > E(0)` (if dissipative) | 1e-6 |
| Momentum | `|P(t) - P(0)| / P(0) > tol` | 1e-6 |

### 2.4 Runtime Alert Severity

| Severity | Action |
|----------|--------|
| INFO | Log only, no action needed |
| WARNING | Continue but monitor closely |
| CRITICAL | Consider stopping simulation |
| FATAL | Stop immediately, diagnose |

---

## Stage 3: Post-flight Checks

Run after simulation completes. Failed checks invalidate results.

### 3.1 Conservation Validation

| Check | Criteria | Action if Failed |
|-------|----------|------------------|
| Mass conserved | Drift < tolerance | Results invalid |
| Energy behavior | Monotonic if variational | Review physics |
| Momentum conserved | Drift < tolerance | Results invalid |

### 3.2 Field Validation

| Check | Criteria | Action if Failed |
|-------|----------|------------------|
| No NaN values | `count(NaN) == 0` | Results invalid |
| No Inf values | `count(Inf) == 0` | Results invalid |
| Bounds respected | `field_min >= bound_min` | Review BC/IC |
| Physical range | Values physically meaningful | Review model |

**Common Field Bounds:**
- Phase field: `0 <= phi <= 1`
- Concentration: `0 <= c <= c_max`
- Temperature: `T > 0` (absolute)
- Density: `rho > 0`

### 3.3 Statistical Validation

| Check | Criteria | Notes |
|-------|----------|-------|
| Mean reasonable | Within expected range | Problem-dependent |
| Variance stable | Not exploding over time | Check late times |
| Extremes bounded | No outliers beyond physical | Check mesh refinement |

### 3.4 Confidence Score Calculation

```
confidence = (passed_checks) / (total_checks)

Interpretation:
- 1.0: All checks passed, high confidence
- 0.75-0.99: Minor issues, review before use
- 0.5-0.74: Significant issues, use with caution
- < 0.5: Major problems, do not use results
```

---

## Validation Documentation

### What to Record

For each simulation, document:
1. Pre-flight status and any accepted warnings
2. Runtime alerts and actions taken
3. Post-flight validation results
4. Confidence score
5. Any manual review notes

### Traceability

Link validation records to:
- Simulation configuration (version-controlled)
- Software version and commit hash
- Hardware/environment details
- Date and operator

---

## Quick Reference Checklist

### Before Running
- [ ] Config file validated
- [ ] All required parameters present
- [ ] Stability conditions satisfied
- [ ] Output directory ready
- [ ] Disk space sufficient

### During Run
- [ ] Residuals monitored
- [ ] Time step behavior normal
- [ ] No critical alerts

### After Run
- [ ] No NaN/Inf in results
- [ ] Conservation checks passed
- [ ] Field bounds respected
- [ ] Confidence score acceptable
- [ ] Results documented
