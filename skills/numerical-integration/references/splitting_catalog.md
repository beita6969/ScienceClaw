# Operator Splitting Catalog

Comprehensive reference for operator splitting methods in PDEs and multiphysics.

## Fundamental Concepts

### Problem Setup

Given:
```
du/dt = A(u) + B(u)
```

Where A and B are operators (may be differential, algebraic, or mixed).

**Goal:** Solve A and B separately, combine solutions.

### Why Split?

| Reason | Example |
|--------|---------|
| Different physics | Advection + reaction |
| Different solvers | Spectral + finite difference |
| Different time scales | Fast chemistry + slow transport |
| Code modularity | Reuse existing solvers |
| Parallelization | Different operators on different processors |

## First-Order Methods

### Lie-Trotter Splitting

```
Step 1: Solve du/dt = A(u) for time dt → u*
Step 2: Solve du/dt = B(u) for time dt starting from u* → u_{n+1}
```

**Properties:**
- Order: 1 (error = O(dt))
- Simplest implementation
- Non-symmetric (order matters)

**Error:**
```
Error ≈ (dt/2) × [A, B] × u
```
Where [A, B] = AB - BA is the commutator.

### Choosing Order (A then B vs B then A)

| Order | Preferred When |
|-------|---------------|
| A → B | B's result needed for output |
| B → A | A is diagnostic, B is physics |
| Symmetric | Use Strang instead |

## Second-Order Methods

### Strang Splitting

```
Step 1: Solve du/dt = A(u) for time dt/2 → u*
Step 2: Solve du/dt = B(u) for time dt → u**
Step 3: Solve du/dt = A(u) for time dt/2 → u_{n+1}
```

**Properties:**
- Order: 2 (error = O(dt²))
- Symmetric (time-reversible)
- Standard choice for moderate accuracy

**Error:**
```
Error ≈ (dt²/24) × [[A, B], A + B] × u
```
Involves nested commutators.

### Marchuk-Strang Alternating

```
Step n (even): A/2 → B → A/2
Step n (odd):  B/2 → A → B/2
```

Alternating between Strang orderings:
- Averages directional bias
- Can improve symmetry for anisotropic problems

## Higher-Order Methods

### Triple-Jump (Order 4)

```
γ = 1 / (2 - 2^{1/3}) ≈ 1.3512

Sub-step 1: Strang with dt × γ
Sub-step 2: Strang with dt × (1 - 2γ)  [negative!]
Sub-step 3: Strang with dt × γ
```

**Note:** Requires a negative time step!
- Problematic for irreversible operators (diffusion)
- Works for Hamiltonian systems

### Yoshida (Order 4, Symmetric)

For symmetric operators only:
```
w₀ = -2^{1/3} / (2 - 2^{1/3})
w₁ = 1 / (2 - 2^{1/3})

Sequence: S(w₁ dt) → S(w₀ dt) → S(w₁ dt)
```

Where S is Strang splitting.

### Forest-Ruth (Order 4)

For Hamiltonian splitting H = T(p) + V(q):
```
θ = 1 / (2 - 2^{1/3})
Sequence: V(θ/2) → T(θ) → V((1-θ)/2) → T(1-2θ) → V((1-θ)/2) → T(θ) → V(θ/2)
```

## Directional Splitting (ADI)

### 2D Diffusion

```
∂u/∂t = D(∂²u/∂x² + ∂²u/∂y²)
```

Split by direction:
```
A: ∂u/∂t = D ∂²u/∂x²  (x-diffusion)
B: ∂u/∂t = D ∂²u/∂y²  (y-diffusion)
```

### Douglas-Gunn (Unconditionally Stable)

```
Step 1: (I - dt/2 × D_xx) u* = (I + dt/2 × D_xx + dt × D_yy) u_n
Step 2: (I - dt/2 × D_yy) u_{n+1} = u* - dt/2 × D_yy u_n
```

**Properties:**
- Second-order accurate
- Unconditionally stable for diffusion
- Tridiagonal solves only

### Peaceman-Rachford

```
Step 1: (I - dt/2 × D_xx) u* = (I + dt/2 × D_yy) u_n
Step 2: (I - dt/2 × D_yy) u_{n+1} = (I + dt/2 × D_xx) u*
```

**Properties:**
- Second-order
- Symmetric form
- Classic ADI method

### 3D Extension

For 3D: ∂u/∂t = D(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)

**Douglas-Rachford-Gunn:**
```
(I - dt/2 × D_xx) u* = (I + dt/2 × D_xx + dt × D_yy + dt × D_zz) u_n
(I - dt/2 × D_yy) u** = u* - dt/2 × D_yy u_n
(I - dt/2 × D_zz) u_{n+1} = u** - dt/2 × D_zz u_n
```

## Physics-Based Splitting

### Advection-Reaction

```
∂u/∂t + v·∇u = R(u)

A: ∂u/∂t + v·∇u = 0  (pure advection)
B: du/dt = R(u)       (pure reaction, ODE at each point)
```

**Advantages:**
- Use specialized advection scheme (upwind, WENO)
- Use stiff ODE solver for reaction
- Each solver optimized for its physics

### Diffusion-Reaction

```
∂u/∂t = D∇²u + R(u)

A: ∂u/∂t = D∇²u    (diffusion)
B: du/dt = R(u)    (reaction)
```

**Common Approach:**
- Implicit for diffusion (larger dt)
- CVODE/LSODA for stiff reaction

### Full Advection-Diffusion-Reaction

```
∂u/∂t + v·∇u = D∇²u + R(u)

Three-way split:
A: Advection
B: Diffusion
C: Reaction

Strang-like: A/2 → B/2 → C → B/2 → A/2
```

## Splitting Error Analysis

### Commutator Size

Splitting error depends on how much operators "fail to commute":
```
[A, B] = AB - BA
```

| Commutator | Splitting Error | Example |
|------------|-----------------|---------|
| [A,B] = 0 | Zero | Linear independent operators |
| [A,B] small | O(dt²) for Strang | Diffusion + mild reaction |
| [A,B] large | Significant | Coupled nonlinear terms |

### Error Estimation

**A posteriori estimate:**
```
Compare: Strang (dt) vs 2 × Strang (dt/2)
Error ≈ (result_dt - result_dt/2) / 3
```

**Adaptive splitting:**
- If error small: increase dt
- If error large: decrease dt or refine splitting

### Error Accumulation

Over long times:
```
Total error ≈ (T_final / dt) × local_splitting_error
           = O(dt^{p-1}) for order-p method
```

## Special Techniques

### Balanced Splitting

For problems with conservation laws:
```
Ensure: ∫ u dx conserved by each sub-step
```

Modify operators to preserve conservation:
- Conservative discretization for each piece
- Flux-form splitting

### Stabilized Splitting

Add and subtract stabilizing terms:
```
Original: du/dt = A(u)
Modified: du/dt = [A(u) + Su] - Su
          Split:   implicit      explicit
```

Choose S to improve stability of explicit part.

### Iterative Splitting

Iterate between operators until converged:
```
While not converged:
    Solve A with current B(u)
    Solve B with updated A(u)
```

Converges to coupled solution (removes splitting error).

## Implementation Patterns

### Modular Code Structure

```python
def splitting_step(u, dt, A_solver, B_solver, method='strang'):
    if method == 'lie':
        u = A_solver(u, dt)
        u = B_solver(u, dt)
    elif method == 'strang':
        u = A_solver(u, dt/2)
        u = B_solver(u, dt)
        u = A_solver(u, dt/2)
    return u
```

### State Management

For multi-step splittings:
```python
class SplitSolver:
    def __init__(self):
        self.u_history = []  # For multi-step methods

    def step(self, u, dt):
        # Store for multi-step
        self.u_history.append(u.copy())
        if len(self.u_history) > 3:
            self.u_history.pop(0)
        # Splitting step...
```

### Parallel Splitting

When operators act on independent domains:
```
A: operates on spatial points 0:N/2
B: operates on spatial points N/2:N
→ Solve A and B in parallel!
```

## Choosing a Method

### Decision Table

| Accuracy Need | Coupling | Recommended |
|---------------|----------|-------------|
| Low (1st order) | Any | Lie |
| Moderate (2nd) | Weak | Strang |
| Moderate (2nd) | Directional | ADI |
| High (4th) | Weak, reversible | Yoshida |
| High | Strong | Iterative or monolithic |

### Red Flags

| Warning Sign | Problem | Solution |
|--------------|---------|----------|
| Solution blows up | Splitting unstable | Reduce dt, change order |
| Conservation violated | Unbalanced split | Use conservative form |
| Oscillations at interface | Commutator large | Reduce dt or iterate |
| Wrong steady state | Splitting error | Tighter tolerance |

## Verification Tests

### Manufactured Solutions

1. Choose exact u(x,t)
2. Compute source S = ∂u/∂t - A(u) - B(u)
3. Solve with splitting + source S
4. Compare to exact

### Order Verification

1. Run with dt, dt/2, dt/4
2. Measure error vs reference
3. Compute order = log(e₁/e₂) / log(2)
4. Should match theoretical order

### Conservation Test

1. Compute ∫ u dx at t=0
2. Run splitting simulation
3. Check ∫ u dx drift over time
4. Should be O(dt^p) or machine precision
