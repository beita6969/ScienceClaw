# Nonlinear Solver Decision Tree

Comprehensive decision guide for selecting nonlinear solvers for f(x)=0 or min F(x).

## Problem Classification

### Key Properties to Determine

| Property | How to Check | Impact |
|----------|--------------|--------|
| Problem type | Root-finding, optimization, least-squares | Determines solver class |
| Jacobian availability | Analytic vs finite-difference | Newton vs quasi-Newton |
| Problem size | Number of unknowns | Memory and algorithm choice |
| Smoothness | Continuous derivatives | Enables fast convergence |
| Constraints | Bounds, equalities, inequalities | Specialized methods needed |
| Hessian SPD | Optimization: F''(x) > 0 | BFGS maintains this property |

### Quick Classification

```
Problem type:
├── f(x) = 0 (root-finding/nonlinear equations)
├── min F(x) (unconstrained optimization)
├── min F(x) s.t. g(x) = 0 (equality constrained)
├── min F(x) s.t. l ≤ x ≤ u (bound constrained)
└── min ||r(x)||² (nonlinear least-squares)
```

## Primary Decision Tree

```
START: Need to solve nonlinear problem
│
├─ What type of problem?
│   │
│   ├─ ROOT-FINDING (f(x) = 0)
│   │   │
│   │   ├─ Is analytic Jacobian available?
│   │   │   │
│   │   │   ├── YES, cheap to compute
│   │   │   │   ├── Small problem (n < 1000) → Newton (full)
│   │   │   │   ├── Large problem → Newton-Krylov (GMRES/BiCGSTAB)
│   │   │   │   └── Sparse Jacobian → Newton-Krylov with ILU
│   │   │   │
│   │   │   ├── YES, expensive to compute
│   │   │   │   ├── Modified Newton (reuse Jacobian)
│   │   │   │   └── Broyden update
│   │   │   │
│   │   │   └── NO (finite-diff or unavailable)
│   │   │       ├── Smooth problem → Broyden (good/bad)
│   │   │       ├── Fixed-point form → Anderson acceleration
│   │   │       └── Very large → Newton-Krylov (matrix-free)
│   │   │
│   │   └─ Convergence issues?
│   │       ├── Diverging → Add line search or trust region
│   │       ├── Stagnating → Better preconditioner
│   │       └── Oscillating → Reduce step, add damping
│   │
│   ├─ UNCONSTRAINED OPTIMIZATION (min F(x))
│   │   │
│   │   ├─ Is Hessian available?
│   │   │   │
│   │   │   ├── YES → Newton with trust region
│   │   │   │   └── Large problem → Truncated Newton (CG)
│   │   │   │
│   │   │   └── NO → Use quasi-Newton
│   │   │       ├── Moderate size → BFGS
│   │   │       └── Large problem → L-BFGS
│   │   │
│   │   └─ Is objective smooth?
│   │       ├── YES → Standard quasi-Newton
│   │       └── NO → Subgradient methods, bundle methods
│   │
│   ├─ CONSTRAINED OPTIMIZATION
│   │   │
│   │   ├─ Bound constraints only
│   │   │   ├── Smooth → L-BFGS-B
│   │   │   └── General → Trust-region reflective
│   │   │
│   │   ├─ Equality constraints
│   │   │   ├── Few constraints → SQP
│   │   │   └── Many constraints → Augmented Lagrangian
│   │   │
│   │   └── Inequality constraints
│   │       ├── Smooth → SQP or Interior Point
│   │       └── Nonsmooth → Penalty methods
│   │
│   └─ NONLINEAR LEAST-SQUARES (min ||r(x)||²)
│       │
│       ├─ Is Jacobian of r(x) available?
│       │   ├── YES → Gauss-Newton or Levenberg-Marquardt
│       │   └── NO → Variable projection or L-BFGS
│       │
│       └─ Zero residual problem?
│           ├── YES → May converge faster (quadratic near solution)
│           └── NO → LM more robust
```

## Method Selection by Problem Type

### Root-Finding (f(x) = 0)

| Condition | Method | Notes |
|-----------|--------|-------|
| Small, Jacobian available | Newton | Quadratic convergence |
| Large, Jacobian available | Newton-Krylov | Matrix-free inner solve |
| Jacobian expensive | Modified Newton | Reuse J for k steps |
| No Jacobian, smooth | Broyden | Superlinear convergence |
| Fixed-point form | Anderson acceleration | Accelerates Picard |
| Very large, sparse | Newton-Krylov + ILU | Preconditioned |

### Unconstrained Optimization (min F(x))

| Condition | Method | Notes |
|-----------|--------|-------|
| Hessian available | Newton-TR | Quadratic convergence |
| Gradient only | BFGS or L-BFGS | Superlinear |
| Large scale | L-BFGS | O(n) memory |
| Nonsmooth | Subgradient, Bundle | Slower convergence |

### Least-Squares (min ||r(x)||²)

| Condition | Method | Notes |
|-----------|--------|-------|
| Small residual | Gauss-Newton | Fast near solution |
| Large residual | Levenberg-Marquardt | More robust |
| Very large | Variable projection | Separable structure |
| No Jacobian | L-BFGS on ||r||² | Suboptimal but works |

## Application-Specific Recommendations

### Phase-Field Simulations

| Equation Type | Recommended | Notes |
|---------------|-------------|-------|
| Allen-Cahn | Newton-Krylov | Smooth, can be stiff |
| Cahn-Hilliard | Newton + preconditioner | 4th order, ill-conditioned |
| Crystal plasticity | Modified Newton | Expensive Jacobian |
| Multiphase | Anderson/Picard + acceleration | Fixed-point nature |

### Navier-Stokes

| Formulation | Recommended | Notes |
|-------------|-------------|-------|
| Steady | Newton-Krylov + block precond | Saddle point structure |
| Unsteady (implicit) | Modified Newton | Reuse Jacobian over time |
| Turbulent (RANS) | Under-relaxed Picard | Stability first |
| Large Reynolds | Continuation in Re | Globalization |

### Solid Mechanics

| Problem | Recommended | Notes |
|---------|-------------|-------|
| Linear elasticity | Direct (if linear) | Not truly nonlinear |
| Hyperelasticity | Newton + line search | May need regularization |
| Plasticity | Modified Newton | Tangent expensive |
| Contact | SQP or Augmented Lagrangian | Inequality constraints |

## Failure Modes and Remedies

### Common Problems

| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| No convergence | Poor initial guess | Continuation, better starting point |
| Divergence | Step too large | Line search, trust region |
| Slow convergence | Poor Jacobian | Better preconditioner, exact Jacobian |
| Oscillation | Step size issues | Damping, trust region |
| Stagnation | Singular Jacobian | Regularization, different formulation |

### When Newton Fails

1. **Check Jacobian accuracy**: Compare with finite-difference
2. **Add globalization**: Line search or trust region
3. **Try continuation**: Gradually increase difficult parameters
4. **Check conditioning**: May need scaling or preconditioning
5. **Reformulate**: Sometimes a different formulation is better

### When Quasi-Newton Fails

1. **Reset Hessian approximation**: Start fresh with identity
2. **Switch to BFGS**: More stable than SR1/Broyden for optimization
3. **Try L-BFGS**: Less aggressive updates
4. **Use Newton**: If Jacobian/Hessian is available

## Globalization Summary

| Method | Use When |
|--------|----------|
| Line search (Armijo) | Standard, root-finding |
| Line search (Wolfe) | Optimization, quasi-Newton |
| Backtracking | Simple, when Armijo sufficient |
| Trust region | Ill-conditioned, near-singular |
| Levenberg-Marquardt | Least-squares |
| Damped Newton | Simple alternative to line search |

## Quick Reference Table

| Problem | First Choice | Alternative | Globalization |
|---------|--------------|-------------|---------------|
| Small root-finding | Newton | Broyden | Line search |
| Large root-finding | Newton-Krylov | Anderson | Trust region |
| Small optimization | BFGS | Newton | Wolfe line search |
| Large optimization | L-BFGS | Truncated Newton | Trust region |
| Least-squares | Levenberg-Marquardt | Gauss-Newton | Trust region |
| Bound constrained | L-BFGS-B | Trust-region reflective | Projected |
| General constrained | SQP | Interior Point | Merit function |
