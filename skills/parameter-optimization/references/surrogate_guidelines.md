# Surrogate Model Guidelines

## Overview

Surrogate models (metamodels, emulators) approximate expensive simulations with fast-to-evaluate functions. They enable rapid optimization, sensitivity analysis, and uncertainty quantification.

## Model Comparison

| Model | Complexity | Interpretable | Best Dimension | Handles Noise |
|-------|------------|---------------|----------------|---------------|
| Polynomial | Low | Yes | 1-5 | Poor |
| RBF | Medium | No | 1-20 | Poor |
| Gaussian Process | High | Partially | 1-15 | Yes |
| Neural Network | High | No | Any | Yes |

---

## Polynomial Response Surface

### How It Works

Fit polynomial of specified order to simulation data:

```
y = b0 + sum(b_i * x_i) + sum(b_ij * x_i * x_j) + ...
```

Orders:
- Linear: `1 + d` terms
- Quadratic: `1 + d + d*(d+1)/2` terms

### When to Use

- Very limited data (< 20 points)
- Smooth, nearly quadratic response
- Need interpretable coefficients
- Quick approximation

### When to Avoid

- Highly nonlinear responses
- Extrapolation required
- High dimensions (coefficient explosion)

### Sample Requirements

| Order | Minimum Samples | Recommended |
|-------|-----------------|-------------|
| Linear | d + 1 | 2*(d+1) |
| Quadratic | (d+1)*(d+2)/2 | 2x minimum |

---

## Radial Basis Functions (RBF)

### How It Works

Interpolate using weighted sum of radial basis functions centered at data points:

```
y(x) = sum(w_i * phi(||x - x_i||))
```

Common kernels:
- Gaussian: `exp(-r^2 / epsilon^2)`
- Multiquadric: `sqrt(1 + (epsilon*r)^2)`
- Thin-plate spline: `r^2 * log(r)`

### When to Use

- Exact interpolation required
- Moderate dimensions (< 20)
- No noise in data
- Complex response surfaces

### When to Avoid

- Noisy data (will fit noise)
- Very large datasets (O(n^3) fitting)
- Need uncertainty estimates

### Key Hyperparameters

| Parameter | Description | Tuning |
|-----------|-------------|--------|
| epsilon | Width of basis function | Cross-validation or rule of thumb |
| Kernel type | Shape of basis | Try multiple, pick lowest CV error |

---

## Gaussian Process (Kriging)

### How It Works

Model output as realization of Gaussian Process with specified mean and covariance (kernel):

```
y(x) ~ GP(m(x), k(x, x'))
```

Provides:
- Mean prediction: E[y(x)]
- Uncertainty: Var[y(x)]

### When to Use

- Need uncertainty quantification
- Expensive simulations (BO framework)
- Smooth responses
- Sequential/adaptive sampling

### When to Avoid

- Very large datasets (> 1000 points)
- High dimensions (> 15-20)
- Very fast simulations

### Kernel Selection

| Kernel | Properties | Best For |
|--------|------------|----------|
| RBF (SE) | Infinitely smooth | Very smooth responses |
| Matern 3/2 | Once differentiable | Typical engineering |
| Matern 5/2 | Twice differentiable | Default choice |
| Periodic | Captures periodicity | Cyclic phenomena |

### Libraries

- `GPyTorch` (PyTorch, scalable)
- `GPy` (numpy, flexible)
- `scikit-learn` (simple API)

---

## Neural Networks

### When to Use

- Very large datasets (> 10000 points)
- High dimensions
- Complex nonlinear patterns
- Have GPU resources

### When to Avoid

- Small datasets (< 100 points)
- Need uncertainty (without extra work)
- Interpretability required
- Training time is limited

---

## Validation Strategies

### Cross-Validation

| Type | Procedure | When to Use |
|------|-----------|-------------|
| Leave-one-out | Predict each point from rest | Small datasets (< 50) |
| k-fold (k=5,10) | Split into k groups | Medium datasets |
| Hold-out | Reserve 20% for testing | Large datasets |

### Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| RMSE | sqrt(mean((y - y_hat)^2)) | Lower is better |
| R^2 | 1 - SS_res/SS_tot | > 0.9 good, > 0.95 excellent |
| Max Error | max(|y - y_hat|) | Depends on application |
| NRMSE | RMSE / range(y) | < 5% good |

### Red Flags

| Observation | Possible Cause |
|-------------|----------------|
| R^2 < 0.5 | Model too simple or data too noisy |
| Max error >> RMSE | Outliers or localized bad fit |
| Training error << CV error | Overfitting |

---

## Adaptive Sampling

When initial surrogate is poor, add samples strategically:

| Strategy | How It Works |
|----------|--------------|
| Max uncertainty | Sample where GP variance is highest |
| Max error | Sample where CV error is highest |
| Space-filling | Add LHS points to sparse regions |
| Exploitation | Sample near current optimum |

---

## Decision Flowchart

```
START
  |
  v
Is data noisy?
  |
  +-- YES --> Need uncertainty?
  |             |
  |             +-- YES --> GAUSSIAN PROCESS
  |             |
  |             +-- NO --> NEURAL NETWORK (if large data)
  |
  +-- NO --> Is response smooth and low-dim?
               |
               +-- YES --> Is data < 20 points?
               |             |
               |             +-- YES --> POLYNOMIAL
               |             |
               |             +-- NO --> RBF or GP
               |
               +-- NO --> RBF or GP with Matern kernel
```

---

## Implementation Notes

The `surrogate_builder.py` script in this skill:
- Is a **placeholder** for demonstration
- Computes basic MSE metric only
- Does not fit actual models

For real surrogate modeling, use:
- `scikit-learn` (polynomial, GP, neural networks)
- `scipy.interpolate` (RBF)
- `GPyTorch` (scalable GPs)
- `SMT` (Surrogate Modeling Toolbox)
