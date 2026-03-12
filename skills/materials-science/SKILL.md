---
name: materials-science
description: Analyzes material properties including crystal structures, phase diagrams, mechanical/thermal/electronic properties, and supports materials discovery through computational approaches; trigger when users discuss alloys, ceramics, polymers, nanomaterials, or materials characterization.
---

## When to Trigger

Activate this skill when the user mentions:
- Crystal structures, lattice parameters, space groups, unit cells
- Phase diagrams, phase transitions, thermodynamic stability
- Mechanical properties (tensile strength, hardness, elastic modulus)
- Electronic properties (band gap, conductivity, dielectric constant)
- Materials characterization (XRD, SEM, TEM, AFM)
- Nanomaterials, thin films, composites, polymers
- Materials databases, high-throughput screening, materials informatics

## Step-by-Step Methodology

1. **Define the materials question** - Specify the material system (elements, compounds), property of interest, and application context (structural, electronic, optical, catalytic).
2. **Database search** - Query Materials Project, AFLOW, ICSD, or OQMD for known structures and computed properties. Check experimental databases (Springer Materials, NIST) for measured values.
3. **Structure analysis** - Identify crystal system, space group, and Wyckoff positions. Compute lattice parameters and density. For disordered systems, characterize using pair distribution functions or radial distribution functions.
4. **Property evaluation** - Retrieve or compute relevant properties: formation energy (thermodynamic stability), band structure (electronic), phonon dispersion (thermal), elastic tensor (mechanical). Compare with target specifications.
5. **Phase diagram analysis** - Construct or retrieve phase diagrams (binary, ternary). Identify stable phases, invariant reactions (eutectic, peritectic), and solid solutions. Use CALPHAD method for complex systems.
6. **Characterization guidance** - Recommend appropriate techniques: XRD for crystal structure, SEM/TEM for microstructure, XPS for surface chemistry, DSC for thermal transitions. Specify expected peaks/features.
7. **Design recommendations** - Suggest composition or processing modifications to achieve target properties. Consider trade-offs between competing properties (strength vs. ductility, conductivity vs. transparency).

## Key Databases and Tools

- **Materials Project** - Computed materials properties (DFT)
- **AFLOW** - Automatic FLOW for materials discovery
- **ICSD** - Inorganic Crystal Structure Database
- **NIST Materials Data** - Experimental property data
- **Springer Materials** - Curated materials data
- **Thermo-Calc / FactSage** - CALPHAD thermodynamic modeling

## Output Format

- Crystal structures with space group, lattice parameters (in Angstroms), and atomic positions.
- Properties in SI units with comparison to reference values.
- Phase diagrams with labeled phases, invariant points, and temperature/composition axes.
- Characterization predictions (expected XRD peaks with 2-theta and hkl, expected spectral features).

## Quality Checklist

- [ ] Crystal structure validated against experimental data when available
- [ ] Property values compared between computational and experimental sources
- [ ] Temperature and pressure conditions specified for all properties
- [ ] Appropriate computational method noted (DFT functional, basis set)
- [ ] Phase diagram includes metastable phases if relevant
- [ ] Synthesis feasibility and processing conditions considered
- [ ] Units consistent and clearly stated throughout
- [ ] Uncertainty or accuracy of computational predictions discussed
