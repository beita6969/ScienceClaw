# Quality Metrics

Comprehensive guide for evaluating and ensuring mesh quality.

## Why Quality Matters

Poor mesh quality leads to:
- Reduced accuracy (truncation error increases)
- Solver convergence problems
- Non-physical solutions
- Instability

## Primary Quality Metrics

### Aspect Ratio

Ratio of longest to shortest edge (or dimension).

```
AR = L_max / L_min

For rectangle: AR = max(dx, dy) / min(dx, dy)
For triangle: AR = L_max / h_min (height to opposite edge)
```

| Aspect Ratio | Quality | Impact |
|--------------|---------|--------|
| 1:1 | Excellent | Optimal accuracy |
| 1:1 - 3:1 | Good | Acceptable |
| 3:1 - 5:1 | Fair | May affect accuracy |
| 5:1 - 10:1 | Poor | Accuracy degradation |
| > 10:1 | Bad | Solver issues likely |

**Effect by physics:**
| Physics | AR Tolerance |
|---------|--------------|
| Isotropic diffusion | AR < 5 |
| Anisotropic diffusion | Can align with anisotropy |
| Advection | Align with flow: AR ~ 10 OK |
| Boundary layer | AR ~ 100 along wall OK |

### Skewness

Measures deviation from ideal shape.

**For quadrilaterals/hexahedra:**
```
Skewness = max(|90° - θ_i|) / 90°

where θ_i are the angles at vertices
Ideal: all angles = 90°, skewness = 0
```

**For triangles:**
```
Equilateral skewness:
Skewness = (θ_max - 60°) / (180° - 60°)
         = (θ_max - 60°) / 120°

Or: Skewness = 1 - (θ_min / 60°)
```

| Skewness | Quality | Notes |
|----------|---------|-------|
| 0 - 0.25 | Excellent | Ideal for any simulation |
| 0.25 - 0.50 | Good | Acceptable |
| 0.50 - 0.75 | Fair | May affect accuracy |
| 0.75 - 0.90 | Poor | Significant errors |
| > 0.90 | Bad | Likely problems |

### Orthogonality

Alignment of cell faces with face normals.

```
Non-orthogonality angle θ:
θ = angle between face normal and line connecting cell centers

Perfect: θ = 0°
```

| Non-orthogonality | Quality | Notes |
|-------------------|---------|-------|
| < 20° | Excellent | Standard schemes work |
| 20° - 40° | Good | May need correction |
| 40° - 60° | Fair | Correction required |
| 60° - 70° | Poor | Strong correction needed |
| > 70° | Bad | Likely convergence issues |

**Correction in FV:**
```
Standard gradient: uses face normal
Corrected gradient: adds non-orthogonal correction term
```

### Cell Volume/Area Ratio

Ratio of neighboring cell sizes.

```
Volume ratio = max(V_i, V_j) / min(V_i, V_j)

for adjacent cells i, j
```

| Volume Ratio | Quality | Impact |
|--------------|---------|--------|
| < 1.5 | Excellent | Smooth variation |
| 1.5 - 2.0 | Good | Acceptable |
| 2.0 - 3.0 | Fair | May affect accuracy |
| > 3.0 | Poor | Large interpolation errors |

## Element-Specific Metrics

### Triangle Quality Measures

**Radius ratio:**
```
q = 2 × r_inscribed / r_circumscribed

Ideal: q = 1 (equilateral)
Acceptable: q > 0.5
Poor: q < 0.25
```

**Area-based:**
```
q = 4√3 × Area / (L₁² + L₂² + L₃²)

Ideal: q = 1 (equilateral)
```

**Minimum angle:**
```
θ_min ≥ 20° (minimum recommended)
θ_min ≥ 30° (good quality)
```

### Quadrilateral Quality Measures

**Jacobian ratio:**
```
J_ratio = J_min / J_max

where J is the Jacobian at each corner

Ideal: J_ratio = 1 (parallelogram)
Acceptable: J_ratio > 0.3
Poor: J_ratio < 0.1
```

**Warpage (3D faces):**
```
Maximum angle between sub-triangle normals
Flat: warpage = 0°
Acceptable: warpage < 15°
```

### Tetrahedral Quality Measures

**Radius ratio:**
```
q = 3 × r_inscribed / r_circumscribed

Ideal: q = 1 (regular tetrahedron)
Acceptable: q > 0.2
```

**Dihedral angles:**
```
θ_min: smallest angle between faces
θ_max: largest angle between faces

Good: 40° < θ < 120°
```

**Volume ratio:**
```
q = 6√2 × Volume / sum(face_areas × opposite_edge_length)

Ideal: q = 1
Acceptable: q > 0.2
```

### Hexahedral Quality Measures

**Jacobian:**
```
J_min / J_max at all integration points

Good: > 0.3
Poor: < 0.1
Invalid: ≤ 0 (inverted element)
```

**Edge ratio:**
```
max(L_edge) / min(L_edge)

Good: < 5
Acceptable: < 10
```

## Quality Thresholds by Application

### Phase-Field Simulations

| Metric | Threshold | Reason |
|--------|-----------|--------|
| Aspect ratio | < 3:1 | Interface resolution |
| Skewness | < 0.5 | Gradient accuracy |
| Size ratio | < 2.0 | Smooth interpolation |

### Fluid Dynamics

| Region | AR Limit | Skewness Limit |
|--------|----------|----------------|
| Interior | < 5 | < 0.85 |
| Boundary layer | < 100 (aligned) | < 0.75 |
| Wake region | < 10 | < 0.80 |

### Structural Analysis

| Metric | Threshold |
|--------|-----------|
| Jacobian ratio | > 0.2 |
| Aspect ratio | < 10 |
| Warpage | < 15° |

### Heat Transfer

| Metric | Threshold |
|--------|-----------|
| Aspect ratio | < 5 |
| Non-orthogonality | < 40° |
| Size ratio | < 2.0 |

## Improving Mesh Quality

### Smoothing Techniques

**Laplacian smoothing:**
```
Move node to average of neighbors
x_new = (1/n) × Σ x_neighbors

Caution: can invert elements
Use with quality check
```

**Optimization-based smoothing:**
```
Minimize: Σ (quality penalty)
Subject to: no inverted elements
```

### Local Reconnection

**Edge/face swapping:**
```
If swap improves minimum quality:
  Perform swap
```

**Node insertion/deletion:**
```
Insert node in poor element
Delete nodes causing poor quality
```

### Refinement/Coarsening

```
Poor quality from size variation:
  Refine large cells
  Coarsen too-fine cells
  Maintain 2:1 balance
```

## Quality Checking Workflow

### Pre-Solve Check

1. **Statistics:**
   - Minimum/maximum of each metric
   - Distribution histograms
   - Location of worst elements

2. **Thresholds:**
   - Flag elements below threshold
   - Count problem elements
   - Identify clusters

3. **Visualization:**
   - Color by quality metric
   - Highlight problem regions
   - Check boundary layers

### During Solve (Adaptive)

```python
def check_quality_runtime(mesh):
    """Check quality at runtime for adaptive methods."""
    min_quality = compute_min_quality(mesh)

    if min_quality < threshold_critical:
        raise MeshQualityError("Mesh degenerated")

    if min_quality < threshold_warning:
        log_warning(f"Poor mesh quality: {min_quality}")
        trigger_remesh()
```

### Post-Solve Verification

1. Compare solution smoothness to mesh quality
2. Check conservation errors vs quality
3. Identify if quality limited accuracy

## Quality Metrics Implementation

```python
def triangle_quality(p1, p2, p3):
    """Compute triangle quality measures."""
    # Edge lengths
    L1 = np.linalg.norm(p2 - p1)
    L2 = np.linalg.norm(p3 - p2)
    L3 = np.linalg.norm(p1 - p3)

    # Semi-perimeter and area
    s = (L1 + L2 + L3) / 2
    area = np.sqrt(s * (s - L1) * (s - L2) * (s - L3))

    # Inscribed and circumscribed radii
    r_in = area / s
    r_circ = L1 * L2 * L3 / (4 * area)

    # Quality measures
    radius_ratio = 2 * r_in / r_circ
    area_quality = 4 * np.sqrt(3) * area / (L1**2 + L2**2 + L3**2)

    # Angles
    angles = []
    for i, (a, b, c) in enumerate([(L1, L2, L3), (L2, L3, L1), (L3, L1, L2)]):
        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
        angles.append(np.arccos(np.clip(cos_angle, -1, 1)))

    min_angle = np.min(angles) * 180 / np.pi
    max_angle = np.max(angles) * 180 / np.pi

    return {
        'radius_ratio': radius_ratio,
        'area_quality': area_quality,
        'min_angle': min_angle,
        'max_angle': max_angle,
        'aspect_ratio': max(L1, L2, L3) / min(L1, L2, L3),
        'skewness': (max_angle - 60) / 120
    }
```

## Quick Reference

### Acceptable Ranges Summary

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Aspect ratio | < 3 | < 5 | > 10 |
| Skewness | < 0.25 | < 0.50 | > 0.75 |
| Non-orthogonality | < 20° | < 40° | > 60° |
| Min angle (tri) | > 30° | > 20° | < 10° |
| Jacobian ratio | > 0.5 | > 0.2 | < 0.1 |
| Volume ratio | < 1.5 | < 2.0 | > 3.0 |

### Priority by Problem Type

| Problem | Priority Metrics |
|---------|------------------|
| FD diffusion | Aspect ratio, uniformity |
| FV flow | Skewness, non-orthogonality |
| FE structural | Jacobian, aspect ratio |
| Phase-field | Aspect ratio, size ratio |
| Boundary layer | Near-wall AR, growth rate |
