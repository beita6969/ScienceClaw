---
name: sympy-math
description: "Symbolic mathematics via SymPy. Use when: user asks for algebraic manipulation, calculus, equation solving, or mathematical proofs. NOT for: numerical computation (use scipy) or statistical analysis."
metadata: { "openclaw": { "emoji": "∑", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-sympy", "kind": "uv", "packages": ["sympy"], "label": "Install SymPy via uv" }] } }
---

# SymPy Mathematics Skill

Symbolic mathematics: algebra, calculus, equation solving, and LaTeX output.

## When to Use

- "Solve this equation symbolically"
- "Differentiate/integrate this expression"
- "Simplify this algebraic expression"
- "Convert to LaTeX"
- Symbolic matrix operations
- Solving ODEs/PDEs symbolically

## When NOT to Use

- Numerical computation (use scipy-analysis)
- Statistical analysis (use statsmodels-stats)
- Data visualization (use matplotlib-viz)

## Setup

```bash
uv pip install sympy
```

## Common Commands

### Algebra

```python
python3 -c "
from sympy import *
x, y = symbols('x y')
print('Expanded:', expand((x + y)**3))
print('Factored:', factor(x**3 - 1))
print('Simplified:', simplify((x**2 - 1)/(x - 1)))
"
```

### Calculus

```python
python3 -c "
from sympy import *
x = symbols('x')
f = sin(x) * exp(-x)
print('Derivative:', diff(f, x))
print('Integral:', integrate(f, x))
print('Definite:', integrate(f, (x, 0, oo)))
print('Limit:', limit(sin(x)/x, x, 0))
print('Series:', series(exp(x), x, 0, 5))
"
```

### Equation Solving

```python
python3 -c "
from sympy import *
x, y = symbols('x y')
print(solve(x**2 - 5*x + 6, x))
print(solve([x + y - 5, x - y - 1], [x, y]))
y = Function('y')
t = symbols('t')
print(dsolve(y(t).diff(t, 2) + y(t), y(t)))
"
```

### Linear Algebra

```python
python3 -c "
from sympy import *
M = Matrix([[1, 2], [3, 4]])
print('Det:', M.det())
print('Inverse:', M.inv())
print('Eigenvals:', M.eigenvals())
print('RREF:', M.rref())
"
```

### LaTeX Output

```python
python3 -c "
from sympy import *
x = symbols('x')
expr = Integral(exp(-x**2), (x, -oo, oo))
print(latex(expr))
print(latex(expr.doit()))
"
```

## Notes

- Operates symbolically, not numerically
- Use `N(expr)` or `expr.evalf()` for numerical evaluation
- Supports assumptions: `x = symbols('x', positive=True)`
- For numerical work, use scipy/numpy instead
