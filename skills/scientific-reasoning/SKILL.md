---
name: scientific-reasoning
description: Mathematical and physical reasoning with formal proof construction and problem solving
---

# Scientific Reasoning & Problem Solving

## Purpose
Solve mathematical and scientific reasoning problems with rigorous step-by-step proofs and physical intuition.

## Key Datasets
- **MATH Dataset** (hendrycks/competition_math): 12,500 competition math problems across 7 categories (Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus), difficulty levels 1-5
- **PhysReason** (zhibei1204/PhysReason): Physics reasoning benchmark requiring multi-step physical reasoning

## Protocol
1. **Problem classification** — Identify domain (algebra, calculus, geometry, mechanics, thermodynamics, etc.)
2. **Known-unknowns analysis** — List given quantities, target quantity, constraints
3. **Strategy selection** — Choose approach (direct computation, proof by contradiction, dimensional analysis, symmetry arguments)
4. **Step-by-step solution** — Show each derivation step with justification
5. **Verification** — Check units, boundary conditions, limiting cases

## Problem Categories
- **Pure mathematics**: Number theory, combinatorics, algebra, analysis, geometry
- **Applied mathematics**: Differential equations, optimization, numerical methods
- **Physics reasoning**: Mechanics, E&M, thermodynamics, quantum, relativity
- **Cross-domain**: Mathematical modeling of physical systems

## Rules
- Show all intermediate steps; never skip derivations
- Verify dimensional consistency in physics problems
- Check answer against limiting cases and physical intuition
- For competition problems, aim for elegant solutions, not just correct ones
- Distinguish between exact and approximate answers
