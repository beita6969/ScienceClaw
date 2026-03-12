# Mesh Types

Comprehensive guide for selecting mesh types in numerical simulations.

## Structured Meshes

### Cartesian (Regular)

Uniform spacing in each direction.

```
Grid points at: (i×dx, j×dy, k×dz)
where i, j, k are integers
```

**Properties:**
| Property | Value |
|----------|-------|
| Indexing | Simple (i,j,k) |
| Storage | Minimal (just dx, dy, dz) |
| Stencils | Efficient, regular |
| Parallelization | Easy domain decomposition |
| Complex geometry | Poor fit |

**Best for:**
- Phase-field simulations
- Spectral methods
- Regular domains (boxes)
- Prototyping and testing

### Rectilinear (Non-uniform Cartesian)

Variable spacing, still axis-aligned.

```
Grid points at: (x_i, y_j, z_k)
where x_i, y_j, z_k are 1D arrays
```

**Properties:**
| Property | Value |
|----------|-------|
| Indexing | Still (i,j,k) |
| Storage | 1D arrays for coordinates |
| Stencils | Variable coefficients |
| Refinement | Local stretching |

**Use for:**
- Boundary layer refinement
- Interface refinement
- Graded meshes

**Stretching functions:**
```
Geometric: dx_i = dx_0 × r^i
Hyperbolic tangent: concentrated at boundaries
Polynomial: smooth variation
```

### Curvilinear (Body-Fitted)

General structured mesh, not axis-aligned.

```
Physical: (x, y, z)
Computational: (ξ, η, ζ) on [0,1]³

Mapping: x = x(ξ,η,ζ), etc.
```

**Properties:**
| Property | Value |
|----------|-------|
| Indexing | Still (i,j,k) |
| Storage | Full coordinate arrays |
| Stencils | Metric terms required |
| Geometry | Good for smooth boundaries |

**Metric terms:**
```
∂/∂x = (1/J)[∂ξ/∂x × ∂/∂ξ + ∂η/∂x × ∂/∂η + ∂ζ/∂x × ∂/∂ζ]
where J = Jacobian of mapping
```

**Common transformations:**
- Polar/cylindrical
- Elliptic smoothing
- Algebraic stretching

### Block-Structured

Multiple structured blocks patched together.

```
Block 1: (i,j,k) ∈ [0,N1] × [0,M1] × [0,P1]
Block 2: (i,j,k) ∈ [0,N2] × [0,M2] × [0,P2]
...
Interface: matching or non-matching
```

**Properties:**
| Property | Value |
|----------|-------|
| Flexibility | Better than single block |
| Parallelism | Block = parallel unit |
| Complexity | Interface handling |
| Geometry | Moderate complexity |

## Unstructured Meshes

### Triangular (2D) / Tetrahedral (3D)

Elements: triangles (2D) or tetrahedra (3D).

**Properties:**
| Property | Value |
|----------|-------|
| Geometry | Arbitrary boundaries |
| Adaptivity | Easy local refinement |
| Stencils | Variable, need connectivity |
| Storage | Element-node connectivity |
| Generation | Delaunay, advancing front |

**Data structures:**
```
Nodes: [(x_0, y_0), (x_1, y_1), ...]
Elements: [(n_0, n_1, n_2), ...]  # node indices
Edges: derived from elements
```

**Quality metrics:**
- Aspect ratio
- Minimum angle
- Circumradius/inradius ratio

### Quadrilateral (2D) / Hexahedral (3D)

Elements: quads (2D) or hexahedra (3D).

**Properties:**
| Property | Value |
|----------|-------|
| Efficiency | Better per-element accuracy |
| Stiffness | Can be over-constrained |
| Generation | Harder than tri/tet |
| Quality control | More challenging |

**Advantages over tri/tet:**
- Fewer elements for same accuracy
- Better alignment with flow/field directions
- Lower numerical diffusion for advection

### Mixed/Hybrid

Combine different element types.

```
Near walls: structured quad/hex (boundary layer)
Interior: unstructured tri/tet (flexibility)
```

**Common patterns:**
- Prism layers near walls + tets in bulk
- Quad faces on boundaries + tet interior
- Hanging nodes with transitions

## Special Mesh Types

### Octree/Quadtree

Hierarchical refinement by recursive subdivision.

```
Root cell covers domain
Subdivide cells based on criterion
Continue until resolution satisfied
```

**Properties:**
| Property | Value |
|----------|-------|
| Adaptivity | Automatic, hierarchical |
| Load balancing | Natural with Z-ordering |
| Hanging nodes | At refinement interfaces |
| Conservation | Needs special treatment |

**Refinement criteria:**
- Gradient magnitude
- Error estimator
- Geometric features
- Distance to interface

### Voronoi/Polyhedral

Cells are general polygons/polyhedra.

**Properties:**
| Property | Value |
|----------|-------|
| Flexibility | Maximum |
| Quality | Depends on generation |
| FV methods | Natural fit |
| Stencils | Per-cell connectivity |

### Overset (Chimera)

Multiple overlapping meshes.

```
Background mesh: covers domain
Body-fitted mesh: around objects
Interpolation: at overlap boundaries
```

**Use for:**
- Moving bodies
- Multiple components
- Complex geometry with relative motion

## Mesh Selection Guide

### By Geometry Complexity

| Geometry | Recommended |
|----------|-------------|
| Box/rectangle | Cartesian |
| Cylinder/sphere | Curvilinear |
| Single body, smooth | Body-fitted structured |
| Complex single body | Unstructured |
| Multiple bodies | Block-structured or overset |
| Arbitrary | Unstructured |

### By Physics

| Physics | Recommended |
|---------|-------------|
| Diffusion only | Any (Cartesian often sufficient) |
| Advection-dominated | Aligned with flow if possible |
| Boundary layers | Structured near wall |
| Shocks | Adaptive (octree/AMR) |
| Interface tracking | Refined at interface |
| Phase-field | Uniform or locally refined |

### By Method

| Method | Compatible Meshes |
|--------|-------------------|
| Finite difference | Structured (Cartesian, curvilinear) |
| Finite volume | Any |
| Finite element | Any (often unstructured) |
| Spectral | Structured |
| Lattice Boltzmann | Cartesian |

## Mesh Refinement Strategies

### H-Refinement

Subdivide cells/elements.

```
Original: element of size h
Refined: 4 elements (2D) or 8 elements (3D) of size h/2
```

### P-Refinement

Increase polynomial order within elements.

```
Original: linear elements (p=1)
Refined: quadratic elements (p=2)
```

### R-Refinement (Mesh Movement)

Move existing nodes without changing connectivity.

```
Nodes move toward regions needing resolution
Total node count unchanged
```

### HP-Refinement

Combine h and p adaptively.

```
Smooth regions: increase p
Non-smooth regions: decrease h
```

## Mesh Generation Considerations

### Input Requirements

| Input | Purpose |
|-------|---------|
| Geometry (CAD, STL) | Domain boundary |
| Target element size | Resolution control |
| Refinement regions | Local sizing |
| Boundary conditions | Layer requirements |

### Output Quality Checks

- [ ] No inverted elements
- [ ] Aspect ratio within bounds
- [ ] Skewness within bounds
- [ ] Minimum angle acceptable
- [ ] Smooth size transitions
- [ ] Boundary conformity

### Common Tools

| Tool | Type | Mesh Types |
|------|------|------------|
| Gmsh | Open source | Tri, tet, structured |
| Triangle | Open source | 2D Delaunay |
| TetGen | Open source | 3D Delaunay |
| CGAL | Library | Various |
| ANSYS Meshing | Commercial | All types |
| Pointwise | Commercial | High quality |

## Quick Reference

### Mesh Type Trade-offs

| Property | Structured | Unstructured |
|----------|------------|--------------|
| Setup time | Low (simple) | Higher |
| Memory | Low | Higher |
| Solver efficiency | High | Lower |
| Geometry flexibility | Low | High |
| Adaptivity | Harder | Easier |
| Parallelism | Easy | More complex |
