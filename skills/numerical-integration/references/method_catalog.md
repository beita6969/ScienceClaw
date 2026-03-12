# Method Catalog

Comprehensive reference for time integration methods in ODE/PDE simulations.

## Explicit Methods (Non-Stiff)

### Runge-Kutta Family

| Method | Order | Stages | Error Est. | Best For |
|--------|-------|--------|------------|----------|
| Euler | 1 | 1 | No | Prototyping only |
| RK2 (Heun) | 2 | 2 | No | Simple problems |
| RK4 (Classical) | 4 | 4 | No | Fixed-step, smooth |
| RK45 (Dormand-Prince) | 5(4) | 6 | Yes | General adaptive |
| DOP853 | 8(5,3) | 12 | Yes | High accuracy |

#### RK45 (Dormand-Prince)
- **Default choice** for non-stiff problems
- Embedded 4th-order method for error estimation
- FSAL (First Same As Last) optimization
- Recommended tolerances: rtol=1e-3, atol=1e-6

#### DOP853
- 8th-order with 5th and 3rd-order error estimates
- Excellent for high-precision requirements
- More expensive per step but fewer steps needed
- Use when rtol < 1e-6 is required

### Adams-Bashforth (Multi-step)

| Order | Points | Formula |
|-------|--------|---------|
| 1 | 1 | y_{n+1} = y_n + h*f_n |
| 2 | 2 | y_{n+1} = y_n + h*(3f_n - f_{n-1})/2 |
| 3 | 3 | y_{n+1} = y_n + h*(23f_n - 16f_{n-1} + 5f_{n-2})/12 |
| 4 | 4 | y_{n+1} = y_n + h*(55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3})/24 |

**Advantages:**
- Only one function evaluation per step
- Efficient for smooth, non-stiff problems

**Disadvantages:**
- Requires startup procedure
- Less robust for discontinuous forcing
- Needs variable-step modification for adaptivity

## Implicit Methods (Stiff)

### BDF (Backward Differentiation Formulas)

| Order | Stability | Formula |
|-------|-----------|---------|
| BDF1 | A-stable | y_{n+1} = y_n + h*f_{n+1} |
| BDF2 | A-stable | (3y_{n+1} - 4y_n + y_{n-1})/2 = h*f_{n+1} |
| BDF3 | A(α)-stable | (11y_{n+1} - 18y_n + 9y_{n-1} - 2y_{n-2})/6 = h*f_{n+1} |
| BDF4 | A(α)-stable | Higher order, smaller stability region |
| BDF5 | A(α)-stable | Use only for mildly stiff |
| BDF6 | Not A-stable | Avoid - stability issues |

**When to use:**
- Large eigenvalue spread (> 100)
- Chemistry with fast/slow reactions
- Diffusion-dominated problems

**Considerations:**
- Requires nonlinear solver (Newton)
- Jacobian needed (analytical or numerical)
- Order reduction near stability boundary

### Radau IIA

| Order | Stages | Properties |
|-------|--------|------------|
| Radau IIA-3 | 2 | L-stable, 3rd order |
| Radau IIA-5 | 3 | L-stable, 5th order |

**Properties:**
- L-stable (strong damping of stiff modes)
- Excellent for very stiff problems
- Superconvergent at endpoints

**Use when:**
- BDF order reduction is problematic
- DAE (differential-algebraic) systems
- Very stiff chemistry

### Rosenbrock Methods

**Characteristics:**
- Linearly implicit (one Jacobian factorization per step)
- No nonlinear iteration needed
- Excellent for moderate stiffness

| Method | Order | Stages |
|--------|-------|--------|
| ROS2 | 2 | 2 |
| ROS3P | 3 | 3 |
| ROS4 | 4 | 4 |
| RODAS | 4 | 6 |

**Advantages over BDF:**
- No iteration convergence issues
- Fixed number of Jacobian evaluations
- Better for time-varying Jacobians

## Structure-Preserving Methods

### Symplectic Integrators

For Hamiltonian systems: dp/dt = -∂H/∂q, dq/dt = ∂H/∂p

| Method | Order | Type |
|--------|-------|------|
| Symplectic Euler | 1 | Explicit |
| Störmer-Verlet | 2 | Explicit |
| Ruth's 3rd order | 3 | Explicit |
| Forest-Ruth | 4 | Explicit |

**Use for:**
- Long-time molecular dynamics
- Orbital mechanics
- Oscillatory systems with energy conservation

**Properties:**
- Exactly conserve symplectic structure
- Near-conservation of Hamiltonian for exponentially long times
- Time-reversible (symmetric methods)

### Geometric Integrators

| Property | Methods |
|----------|---------|
| Volume-preserving | Implicit midpoint, Gauss-Legendre |
| Energy-preserving | Discrete gradient methods |
| Momentum-preserving | Variational integrators |

## IMEX Methods

For problems with mixed stiff/non-stiff terms: du/dt = f_stiff(u) + f_nonstiff(u)

### Common IMEX-RK Schemes

| Scheme | Implicit | Explicit | Order |
|--------|----------|----------|-------|
| IMEX-Euler | BE | FE | 1 |
| IMEX-SSP2 | Trapezoid | SSP-RK2 | 2 |
| IMEX-ARK2 | SDIRK | ERK | 2 |
| IMEX-ARK4 | SDIRK | ERK | 4 |

### IMEX-BDF

| Order | Properties |
|-------|------------|
| SBDF1 | BE + AB1 extrapolation |
| SBDF2 | BDF2 + AB2 extrapolation |
| SBDF3 | BDF3 + AB3 extrapolation |

## Selection Guide

### Quick Decision Table

| Stiffness | Accuracy | Smoothness | Recommended |
|-----------|----------|------------|-------------|
| Non-stiff | Moderate | Smooth | RK45 |
| Non-stiff | High | Smooth | DOP853 |
| Non-stiff | Low | Smooth | Adams-Bashforth |
| Stiff | Moderate | Any | BDF |
| Very stiff | Any | Any | Radau IIA |
| Moderate stiff | Any | Jacobian available | Rosenbrock |
| Mixed | Any | Split possible | IMEX |
| Hamiltonian | Long-time | Oscillatory | Symplectic |

### Cost Comparison

| Method | f evals/step | Jacobian evals | Linear solves |
|--------|--------------|----------------|---------------|
| RK45 | 6 | 0 | 0 |
| BDF2 | 1 + Newton | 1 per few steps | 1 per Newton |
| Radau5 | 3 + Newton | 1 per step | 3 per Newton |
| Rosenbrock4 | 4 | 1 | 4 |

## Stability Regions

### Explicit Methods
- RK4: Extends to Re(λh) ≈ -2.8 on real axis
- RK45: Similar to RK4
- Adams-Bashforth: Smaller regions, decreasing with order

### Implicit Methods
- BDF1-2: A-stable (entire left half-plane)
- BDF3-5: A(α)-stable (wedge-shaped regions)
- Radau: L-stable (A-stable + stiff decay)

## Implementation Notes

### Jacobian Handling
1. **Analytical**: Most accurate, requires code derivation
2. **Automatic differentiation**: Accurate, some overhead
3. **Numerical (finite difference)**: Simple, may be inaccurate for stiff
4. **Jacobian-free (GMRES)**: For very large systems

### Step Size Limits
- Minimum dt: Floating-point precision, typically 1e-15 * t_current
- Maximum dt: Physical time scale or output frequency
- Safety factor: Typically 0.8-0.9 for adaptive methods
