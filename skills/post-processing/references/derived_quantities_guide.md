# Derived Quantities Guide

## Overview

Derived quantities are computed from raw simulation output to extract physical meaning. This guide explains each quantity and its physical interpretation.

## Volume Fraction

**Definition:** Fraction of domain with field value above/below threshold.

**Formula:** V_f = N_above / N_total

**Physical Meaning:**
- For order parameter φ ∈ [0,1]: fraction of phase with φ > 0.5
- For concentration: fraction above solubility limit
- For temperature: fraction above/below critical temperature

**Usage:**
```bash
python derived_quantities.py --input field.json --quantity volume_fraction \
    --field phi --threshold 0.5 --json
```

**Interpretation:**
- V_f = 0.5: Equal phase fractions (equilibrium for symmetric systems)
- V_f → 0 or 1: Predominantly single phase
- Time evolution of V_f: Phase transformation kinetics

## Interface Area/Length

**Definition:** Surface area (3D) or perimeter (2D) of iso-surface at threshold.

**Method:** Marching squares/cubes approximation
- Count cells where threshold is crossed
- Multiply by average cell size

**Physical Meaning:**
- Interfacial area controls reaction/diffusion rates
- Decreases during coarsening (curvature-driven)
- Increases during nucleation/growth

**Usage:**
```bash
python derived_quantities.py --input field.json --quantity interface_area \
    --field phi --threshold 0.5 --json
```

**Interpretation:**
- A(t) ∝ t^(-1/3): Normal coarsening (2D)
- A(t) ∝ t^(-1/2): Interface-controlled growth
- Sudden increase: New phase nucleation

## Gradient Magnitude

**Definition:** |∇φ| = √[(∂φ/∂x)² + (∂φ/∂y)² + ...]

**Method:** Central finite differences
- Interior: (φᵢ₊₁ - φᵢ₋₁) / (2Δx)
- Boundaries: Forward/backward differences

**Physical Meaning:**
- High gradients: Sharp interfaces, steep concentration profiles
- Low gradients: Smooth fields, equilibrium regions
- Maximum gradient: Interface width indicator

**Usage:**
```bash
python derived_quantities.py --input field.json --quantity gradient_magnitude \
    --field phi --json
```

**Interpretation:**
- Max|∇φ| ≈ 1/η for phase-field (η = interface width)
- Mean|∇φ| decreases during coarsening
- High gradients in wrong regions: Numerical instability

## Integral / Total Mass

**Definition:** ∫φ dV ≈ Σφᵢ × ΔV

**Physical Meaning:**
- Total amount of conserved quantity
- Should be constant for Cahn-Hilliard dynamics
- May change for Allen-Cahn (non-conserved)

**Usage:**
```bash
python derived_quantities.py --input field.json --quantity integral \
    --field concentration --json
```

**Conservation Check:**
- |M(t) - M(0)| / M(0) < 10⁻¹⁰: Well-conserved
- Drift > 10⁻⁶: Possible discretization issue

## Total Variation

**Definition:** TV(φ) = ∫|∇φ| dV ≈ Σ|φᵢ₊₁ - φᵢ|

**Physical Meaning:**
- Measures total "roughness" of field
- Related to interface area for binary fields
- Regularization term in image processing

**Usage:**
```bash
python derived_quantities.py --input field.json --quantity total_variation \
    --field phi --json
```

**Interpretation:**
- TV decreases: Smoothing/coarsening
- TV increases: Sharpening/nucleation
- Constant TV: Steady state

## Centroid

**Definition:** Center of mass weighted by field values

**Formula:**
- x_c = ∫x φ dV / ∫φ dV
- y_c = ∫y φ dV / ∫φ dV

**Physical Meaning:**
- Center of precipitate or inclusion
- Drift indicates asymmetric growth
- Multiple centroids for multi-particle systems

**Usage:**
```bash
python derived_quantities.py --input field.json --quantity centroid \
    --field phi --json
```

## Quantity Selection Guide

| Question | Quantity to Use |
|----------|----------------|
| How much of each phase? | volume_fraction |
| How large is the interface? | interface_area |
| How sharp is the interface? | gradient_magnitude (max) |
| Is mass conserved? | integral (over time) |
| Is the system coarsening? | total_variation or interface_area |
| Where is the particle? | centroid |

## Physical Units

All quantities assume consistent units based on grid spacing:

| Quantity | Units (if dx in meters) |
|----------|------------------------|
| Volume fraction | Dimensionless |
| Interface area | m² (3D) or m (2D) |
| Gradient | 1/m |
| Integral | m³ × [field units] |
| Total variation | m² × [field units] |
| Centroid | m |
