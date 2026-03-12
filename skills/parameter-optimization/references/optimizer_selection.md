# Optimizer Selection Guide

## Overview

Choosing the right optimization algorithm depends on problem characteristics: dimensionality, evaluation budget, noise level, and whether gradients are available.

## Algorithm Comparison

| Algorithm | Best Dim | Min Budget | Noise Tolerance | Gradient Needed |
|-----------|----------|------------|-----------------|-----------------|
| Bayesian Optimization | 1-10 | 20 | Medium | No |
| CMA-ES | 5-100 | 50 | Low | No |
| Gradient Descent | Any | 10 | Very Low | Yes |
| Random Search | Any | 100+ | High | No |
| Nelder-Mead | 1-10 | 20 | Low | No |

---

## Bayesian Optimization (BO)

### How It Works

BO builds a probabilistic surrogate model (typically Gaussian Process) of the objective function and uses an acquisition function to balance exploration vs exploitation.

```
Iteration loop:
1. Fit GP to observed (x, y) pairs
2. Compute acquisition function (EI, UCB, PI)
3. Find x_next = argmax(acquisition)
4. Evaluate y_next = f(x_next)
5. Add (x_next, y_next) to dataset
6. Repeat until budget exhausted
```

### When to Use

- Expensive simulations (minutes to hours per evaluation)
- Low to moderate dimensions (d <= 10)
- Smooth or moderately noisy objectives
- Small evaluation budgets (20-100)

### When to Avoid

- High dimensions (d > 15) - GP scales as O(n^3)
- Very noisy objectives without noise model
- Fast evaluations where random search suffices

### Key Hyperparameters

| Parameter | Typical Values | Notes |
|-----------|----------------|-------|
| Kernel | Matern 5/2, RBF | Matern more robust for non-smooth |
| Acquisition | EI, UCB | EI for exploitation, UCB for exploration |
| Initial points | 5-10 | Use LHS or Sobol |

### Libraries

- `botorch` (PyTorch-based, production-ready)
- `scikit-optimize` (sklearn-compatible)
- `GPyOpt` (flexible but less maintained)

---

## CMA-ES (Covariance Matrix Adaptation)

### How It Works

CMA-ES is an evolutionary strategy that adapts the search distribution covariance matrix based on successful mutations.

```
Iteration loop:
1. Sample lambda offspring from N(m, sigma^2 * C)
2. Evaluate and rank offspring
3. Update mean m toward better solutions
4. Update covariance C and step-size sigma
5. Repeat until convergence
```

### When to Use

- Moderate to high dimensions (5-100)
- Non-convex, multimodal landscapes
- No gradients available
- Budget of 100+ evaluations

### When to Avoid

- Very small budgets (< 50)
- Low dimensions where BO is more efficient
- Highly noisy objectives

### Key Hyperparameters

| Parameter | Typical Values | Notes |
|-----------|----------------|-------|
| Population size | 4 + 3*ln(d) | Default is usually good |
| Initial sigma | 0.3 | Fraction of search range |
| Restarts | 1-5 | IPOP or BIPOP strategies |

### Libraries

- `cma` (official Python package)
- `pycma` (pure Python)
- `nevergrad` (includes CMA-ES)

---

## Gradient-Based Methods

### When to Use

- Gradients available (adjoint methods, autodiff)
- Smooth objectives with single optimum
- Very large dimensions

### Algorithms

| Method | Order | Best For |
|--------|-------|----------|
| L-BFGS-B | 2nd (approx) | Bounded problems |
| Adam | 1st | Stochastic objectives |
| Gradient Descent | 1st | Simple problems |

### Convergence Criteria

```
Stop when ANY condition is met:
- ||grad|| < tol (gradient norm)
- |f_k - f_{k-1}| < ftol (function change)
- ||x_k - x_{k-1}|| < xtol (step size)
- k > max_iter (iteration limit)
```

---

## Random Search

### How It Works

Sample uniformly at random from the parameter space and keep the best result.

### When to Use

- Very high dimensions (d > 50)
- Highly noisy objectives
- Large budgets (1000+ evaluations)
- Baseline comparison

### Efficiency Note

Random search with `n` samples finds a point in the top `1/n` fraction with probability `1 - (1 - 1/n)^n ≈ 0.63`.

---

## Decision Flowchart

```
START
  |
  v
Are gradients available AND noise is low?
  |
  +-- YES --> GRADIENT-BASED (L-BFGS-B)
  |
  +-- NO --> Is dimension <= 10 AND budget <= 100?
               |
               +-- YES --> BAYESIAN OPTIMIZATION
               |
               +-- NO --> Is dimension <= 100?
                            |
                            +-- YES --> CMA-ES
                            |
                            +-- NO --> RANDOM SEARCH
```

## Handling Noise

| Noise Level | Recommendation |
|-------------|----------------|
| None | Any method works; prefer gradient if available |
| Low | BO with homoscedastic noise model |
| Medium | BO with heteroscedastic noise or CMA-ES with resampling |
| High | Robust BO, evolutionary strategies, or increase replicates |

## Handling Constraints

| Constraint Type | Approach |
|-----------------|----------|
| Box bounds | Most optimizers support natively |
| Linear | Transform to box or use barrier methods |
| Nonlinear | Penalty methods or constrained BO |
| Black-box | Feasibility-aware acquisition (cEI) |

## Expected Evaluation Counts

| Problem Type | Algorithm | Expected Evals to Converge |
|--------------|-----------|---------------------------|
| 3D smooth | BO | 20-40 |
| 5D smooth | BO | 40-80 |
| 10D multimodal | CMA-ES | 200-500 |
| 20D multimodal | CMA-ES | 500-2000 |
| 50D+ | Random/screening | 1000+ |

## Implementation Notes

The `optimizer_selector.py` script in this skill provides recommendations based on:
- Dimension
- Budget
- Noise level
- Presence of constraints

It returns the most suitable algorithm(s) with expected evaluation counts and notes about configuration.
