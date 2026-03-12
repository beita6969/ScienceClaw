---
name: materials-screening
description: "Orchestrates a materials screening workflow from database search through property filtering to stability assessment and ranking. Use when identifying candidate materials for batteries, catalysts, semiconductors, or other applications. NOT for molecular chemistry or biological compound analysis."
metadata: { "openclaw": { "emoji": "🔋" } }
---

# Materials Screening (Meta Skill)

This meta-skill orchestrates a computational materials screening pipeline by
combining database querying, property-based filtering, structural analysis,
and multi-criteria ranking. It coordinates three specialized skills to
systematically identify and evaluate candidate materials for target applications.

## Workflow

### Step 1: Database Search and Candidate Retrieval

Query the Materials Project API to build an initial candidate pool based on
application-specific criteria:
- Chemical system constraints (e.g., Li-containing oxides for battery cathodes)
- Space group or crystal system requirements
- Elemental composition filters (include/exclude specific elements)
- Property range pre-filters (band gap, formation energy, density)

Retrieve structural data (CIF files), computed properties, and literature
references for each candidate material.

### Step 2: Property-Based Filtering

Apply quantitative property thresholds to narrow the candidate pool:
- **Electronic**: Band gap range for semiconductors, metals, or insulators
- **Thermodynamic**: Formation energy cutoffs for synthesizability
- **Mechanical**: Bulk/shear modulus for structural applications
- **Physical**: Density, volume per atom, coordination preferences
- **Magnetic**: Magnetic ordering for spintronic applications

Define application-specific filter chains (e.g., for photovoltaics: band gap
1.0-1.8 eV, direct gap preferred, low effective mass).

### Step 3: Structure Analysis with Pymatgen

Perform detailed structural characterization on filtered candidates:
- Symmetry analysis: space group verification, site symmetries
- Bonding analysis: coordination environments, bond lengths and angles
- Defect tolerance: vacancy formation energies, anti-site energies
- Surface analysis: slab models, surface energy estimation
- Structural similarity: comparison across candidates using fingerprints

### Step 4: Stability Assessment

Evaluate thermodynamic and dynamic stability of remaining candidates:
- **Thermodynamic**: Energy above the convex hull (Ehull < 25 meV/atom typical)
- **Phase stability**: Competing phases, decomposition products
- **Phonon stability**: Check for imaginary frequencies indicating dynamic instability
- **Aqueous stability**: Pourbaix diagram analysis for electrochemical applications
- **Thermal stability**: Estimated decomposition temperatures

Flag materials with marginal stability for experimental verification.

### Step 5: Multi-Criteria Ranking and Selection

Use scipy optimization to rank candidates through weighted scoring:
- Define objective function combining normalized property values
- Apply Pareto front analysis for multi-objective screening
- Weight criteria by application importance (user-configurable)
- Calculate composite figure of merit for final ranking
- Sensitivity analysis on weight choices to assess ranking robustness

Output a ranked shortlist with property cards and selection rationale.

## Integration Points

- **materials-project** -- Database queries, computed properties, phase diagrams, crystal structures
- **pymatgen-materials** -- Structure manipulation, symmetry, bonding, surface analysis, fingerprints
- **scipy-analysis** -- Optimization, Pareto analysis, statistical ranking, sensitivity analysis

## Output Formats

- **Candidate table**: Formula, space group, key properties, stability metrics
- **Property cards**: Per-material summary with structure visualization description
- **Ranking report**: Ordered list with composite scores and contributing factors
- **Stability summary**: Ehull values, competing phases, phonon stability flags
- **Comparison matrix**: Side-by-side property comparison of top candidates

## Best Practices

1. Define the target application clearly before constructing filter chains
2. Start with broad chemical systems and narrow progressively to avoid missing candidates
3. Use energy above hull as a primary stability filter before detailed analysis
4. Validate database-computed properties against experimental values when available
5. Consider synthesizability alongside thermodynamic stability
6. Apply cost and toxicity filters for practical applications
7. Use multiple ranking weight schemes to test sensitivity of the final shortlist
8. Cross-reference with ICSD or experimental databases for synthesis precedent
9. Document all filter thresholds and ranking weights for reproducibility
10. Flag materials requiring experimental validation of computed properties
