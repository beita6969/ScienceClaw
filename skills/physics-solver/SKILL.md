---
name: physics-solver
description: Physics problem solving including classical mechanics, electromagnetism, thermodynamics, quantum mechanics, optics, and computational physics. Use when user asks to solve physics problems, simulate physical systems, derive equations, or do unit conversions. Triggers on "physics problem", "Newton's law", "electromagnetic", "quantum", "thermodynamics", "optics", "wave equation", "Schrödinger", "relativity", "unit conversion", "circuit analysis".
---

# Physics Solver

Physics computation and problem solving. Venv: `source /Users/zhangmingda/clawd/.venv/bin/activate`

## Physical Constants

```python
from scipy import constants as const
import numpy as np

# Key constants
c = const.c           # speed of light (m/s)
h = const.h           # Planck's constant (J·s)
hbar = const.hbar     # reduced Planck's constant
k_B = const.k         # Boltzmann constant (J/K)
e = const.e           # elementary charge (C)
m_e = const.m_e       # electron mass (kg)
m_p = const.m_p       # proton mass (kg)
G = const.G           # gravitational constant
N_A = const.N_A       # Avogadro's number
epsilon_0 = const.epsilon_0  # vacuum permittivity
mu_0 = const.mu_0     # vacuum permeability
sigma = const.sigma   # Stefan-Boltzmann constant
```

## Classical Mechanics

```python
from sympy import *

t = symbols('t')
m, g, k, L = symbols('m g k L', positive=True)

# Lagrangian mechanics
# Example: Simple pendulum
theta = Function('theta')(t)
T = Rational(1,2) * m * (L * diff(theta, t))**2  # kinetic energy
V = -m * g * L * cos(theta)                        # potential energy
Lag = T - V

# Euler-Lagrange equation
EL = diff(diff(Lag, diff(theta, t)), t) - diff(Lag, theta)
eq = simplify(EL)
print(f"Equation of motion: {eq} = 0")

# Numerical simulation (projectile, pendulum, etc.)
from scipy.integrate import solve_ivp

def pendulum(t, state, g=9.81, L=1.0):
    theta, omega = state
    return [omega, -g/L * np.sin(theta)]

sol = solve_ivp(pendulum, [0, 10], [np.pi/4, 0], max_step=0.01)
```

## Electromagnetism

```python
# Coulomb's law
def coulomb_force(q1, q2, r):
    """Force between two charges (N)"""
    return const.k * q1 * q2 / r**2  # k = 1/(4πε₀)

# Capacitor energy
def capacitor_energy(C, V):
    return 0.5 * C * V**2

# RC circuit
def rc_discharge(V0, R, C, t):
    tau = R * C
    return V0 * np.exp(-t / tau)

# Electromagnetic wave
def em_wavelength(frequency):
    return const.c / frequency

def photon_energy(wavelength):
    return const.h * const.c / wavelength
```

## Quantum Mechanics

```python
# Particle in a box energy levels
def particle_in_box(n, L, m=const.m_e):
    """Energy of nth level, box length L"""
    return (n**2 * const.h**2) / (8 * m * L**2)

# Hydrogen atom energy levels
def hydrogen_energy(n):
    """Energy in eV"""
    return -13.6 / n**2

# de Broglie wavelength
def de_broglie(p):
    return const.h / p

# Heisenberg uncertainty
# Δx · Δp ≥ ℏ/2
```

## Thermodynamics & Statistical Mechanics

```python
# Ideal gas
def ideal_gas_pressure(n, T, V):
    return n * const.R * T / V

# Carnot efficiency
def carnot_efficiency(T_hot, T_cold):
    return 1 - T_cold / T_hot

# Blackbody radiation (Planck's law)
def planck_spectral_radiance(wavelength, T):
    """W/(m²·sr·m)"""
    return (2 * const.h * const.c**2 / wavelength**5) / \
           (np.exp(const.h * const.c / (wavelength * const.k * T)) - 1)

# Maxwell-Boltzmann speed distribution
def mb_speed_dist(v, T, m):
    return 4 * np.pi * (m / (2 * np.pi * const.k * T))**1.5 * \
           v**2 * np.exp(-m * v**2 / (2 * const.k * T))
```

## Unit Conversion

```python
# scipy.constants has conversion factors
from scipy.constants import eV, atm, calorie, mile, inch

# Common conversions
def eV_to_J(energy_eV): return energy_eV * eV
def J_to_eV(energy_J): return energy_J / eV
def celsius_to_kelvin(T_C): return T_C + 273.15
def atm_to_Pa(P_atm): return P_atm * atm
```

## Problem-Solving Framework

1. **Identify** the physical system and relevant principles
2. **Draw** a diagram (describe it textually)
3. **List** knowns and unknowns
4. **Choose** appropriate equations/laws
5. **Solve** symbolically first (SymPy), then substitute numbers
6. **Check** units, limiting cases, and order of magnitude
7. **Interpret** the result physically

## Tips
- Always carry units through calculations
- Check dimensional consistency
- Verify with limiting cases (e.g., v << c for classical limit)
- Use SymPy for symbolic derivations, SciPy for numerical
- For complex simulations, consider specialized tools (COMSOL, OpenFOAM)
