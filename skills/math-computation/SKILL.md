---
name: math-computation
description: Mathematical computation including symbolic math, numerical methods, linear algebra, calculus, differential equations, optimization, and mathematical modeling. Uses Python with SymPy, NumPy, SciPy. Use when user asks to solve equations, compute integrals/derivatives, do matrix operations, solve ODEs/PDEs, optimize functions, or build mathematical models. Triggers on "solve equation", "integral", "derivative", "matrix", "eigenvalue", "differential equation", "optimization", "linear algebra", "symbolic math", "proof".
---

# Mathematical Computation

Symbolic and numerical mathematics. Venv: `source /Users/zhangmingda/clawd/.venv/bin/activate`

## Symbolic Math (SymPy)

```python
from sympy import *
x, y, z, t = symbols('x y z t')
a, b, c = symbols('a b c', real=True)
n, k = symbols('n k', integer=True, positive=True)

# Solve equations
solve(x**2 - 5*x + 6, x)  # [2, 3]
solve([x + y - 5, x - y - 1], [x, y])  # {x: 3, y: 2}

# Calculus
diff(sin(x)*exp(x), x)           # derivative
integrate(x**2 * exp(-x), (x, 0, oo))  # definite integral
limit(sin(x)/x, x, 0)            # limit
series(exp(x), x, 0, 5)          # Taylor series

# Linear algebra
M = Matrix([[1, 2], [3, 4]])
M.eigenvals()    # eigenvalues
M.eigenvects()   # eigenvectors
M.det()          # determinant
M.inv()          # inverse

# Differential equations
f = Function('f')
dsolve(f(x).diff(x, 2) + f(x), f(x))  # y'' + y = 0

# Simplification
simplify(sin(x)**2 + cos(x)**2)  # 1
trigsimp(expr)
factor(expr)
expand(expr)

# LaTeX output
latex(expr)  # for paper-ready equations
```

## Numerical Methods (SciPy)

```python
from scipy import optimize, integrate, linalg, interpolate
import numpy as np

# Root finding
root = optimize.brentq(lambda x: x**3 - 2*x - 5, 2, 3)

# Optimization
result = optimize.minimize(lambda x: (x[0]-1)**2 + (x[1]-2.5)**2,
                          x0=[0, 0], method='Nelder-Mead')

# Constrained optimization
from scipy.optimize import linprog, minimize
result = minimize(objective, x0, constraints=constraints, bounds=bounds)

# Numerical integration
val, err = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)  # √π

# ODE solving
from scipy.integrate import solve_ivp
def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]
sol = solve_ivp(lorenz, [0, 50], [1, 1, 1], dense_output=True, max_step=0.01)

# Interpolation
f_interp = interpolate.interp1d(x_data, y_data, kind='cubic')

# FFT
from scipy.fft import fft, fftfreq
yf = fft(signal)
xf = fftfreq(N, 1/sample_rate)
```

## Linear Algebra

```python
# NumPy
A = np.array([[1, 2], [3, 4]])
np.linalg.eig(A)        # eigendecomposition
np.linalg.svd(A)        # SVD
np.linalg.solve(A, b)   # solve Ax = b
np.linalg.norm(A)       # matrix norm
np.linalg.matrix_rank(A)

# Sparse matrices (SciPy)
from scipy.sparse import csr_matrix, linalg as sparse_linalg
```

## Mathematical Modeling Workflow

1. **Define** the system and variables
2. **Formulate** equations (conservation laws, constitutive relations)
3. **Non-dimensionalize** if appropriate
4. **Solve** analytically (SymPy) or numerically (SciPy)
5. **Validate** against known solutions or data
6. **Sensitivity analysis** on parameters
7. **Visualize** results

## Common Models

- **Population dynamics**: Lotka-Volterra, SIR/SEIR epidemiological
- **Diffusion**: Heat equation, Fick's law
- **Mechanics**: Newton's laws, Lagrangian/Hamiltonian
- **Economics**: Supply-demand, game theory, optimal control
- **Networks**: Graph theory, flow optimization

## Tips
- Use SymPy for exact solutions, SciPy for numerical
- Always verify numerical solutions against analytical when possible
- Check units and dimensional consistency
- Use `latex()` to generate paper-ready equations
- For large systems, consider sparse matrix methods
