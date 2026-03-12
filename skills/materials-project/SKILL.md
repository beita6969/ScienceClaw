---
name: materials-project
description: "Query the Materials Project API v3 for crystal structures, band gaps, formation energies, and thermodynamic stability of 150k+ inorganic materials. Use when: (1) searching materials by chemical formula, (2) looking up material properties by MP ID, (3) filtering materials by band gap, energy, or density, (4) finding stable phases for a composition. NOT for: organic molecules (use pubchem-compound), proteins (use uniprot-protein), drug compounds (use chembl-drug)."
metadata: { "openclaw": { "emoji": "\U0001F9F1", "requires": { "bins": ["curl"] } } }
---

# Materials Project API

Search and retrieve computed materials data from the Materials Project database
(v3 API). Covers 150k+ inorganic crystalline materials with DFT-computed properties.

## Authentication

All requests require the `X-API-KEY` header set to your Materials Project API key.
Store it in the `MP_API_KEY` environment variable.

```bash
export MP_API_KEY="your_api_key_here"
```

## API Base URL

```
https://api.materialsproject.org/v3
```

## Search by Chemical Formula

```bash
curl -s -H "X-API-KEY: $MP_API_KEY" \
  "https://api.materialsproject.org/v3/materials/summary/?formula=Fe2O3&_fields=material_id,formula_pretty,band_gap,formation_energy_per_atom,energy_above_hull,symmetry,density&_limit=10"
```

## Lookup by Material ID

```bash
curl -s -H "X-API-KEY: $MP_API_KEY" \
  "https://api.materialsproject.org/v3/materials/summary/?material_ids=mp-149&_fields=material_id,formula_pretty,band_gap,formation_energy_per_atom,energy_above_hull,symmetry,density,structure"
```

## Filter by Band Gap Range

```bash
curl -s -H "X-API-KEY: $MP_API_KEY" \
  "https://api.materialsproject.org/v3/materials/summary/?band_gap_min=1.0&band_gap_max=2.0&_fields=material_id,formula_pretty,band_gap,formation_energy_per_atom&_limit=20"
```

## Filter by Thermodynamic Stability

Find materials on or near the convex hull (energy_above_hull close to 0 = stable):

```bash
curl -s -H "X-API-KEY: $MP_API_KEY" \
  "https://api.materialsproject.org/v3/materials/summary/?energy_above_hull_max=0.025&elements=Li,Fe,O&_fields=material_id,formula_pretty,energy_above_hull,formation_energy_per_atom&_limit=20"
```

## Filter by Elements

Search for materials containing specific elements:

```bash
curl -s -H "X-API-KEY: $MP_API_KEY" \
  "https://api.materialsproject.org/v3/materials/summary/?elements=Si,Ge&_fields=material_id,formula_pretty,band_gap,density&_limit=15"
```

## Key Properties

| Property                    | Description                                   | Unit       |
|-----------------------------|-----------------------------------------------|------------|
| `band_gap`                  | Electronic band gap                           | eV         |
| `formation_energy_per_atom` | Formation energy per atom from elements        | eV/atom    |
| `energy_above_hull`         | Energy above convex hull (0 = stable)          | eV/atom    |
| `symmetry`                  | Space group and crystal system                 | -          |
| `density`                   | Computed density                               | g/cm^3     |
| `volume`                    | Unit cell volume                               | Angstrom^3 |
| `nsites`                    | Number of sites in the unit cell               | -          |

## Parse Results with Python

```bash
curl -s -H "X-API-KEY: $MP_API_KEY" \
  "https://api.materialsproject.org/v3/materials/summary/?formula=TiO2&_fields=material_id,formula_pretty,band_gap,energy_above_hull,density&_limit=10" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for mat in data.get('data', []):
    mid = mat.get('material_id', 'N/A')
    formula = mat.get('formula_pretty', 'N/A')
    bg = mat.get('band_gap', 'N/A')
    ehull = mat.get('energy_above_hull', 'N/A')
    rho = mat.get('density', 'N/A')
    print(f'{mid:12s} {formula:10s} Eg={bg} eV  Ehull={ehull} eV/at  rho={rho} g/cm3')
"
```

## Pagination

Use `_limit` and `_skip` for pagination:

```bash
# First page
curl -s -H "X-API-KEY: $MP_API_KEY" \
  "https://api.materialsproject.org/v3/materials/summary/?elements=Cu,Zn&_fields=material_id,formula_pretty&_limit=50&_skip=0"

# Second page
curl -s -H "X-API-KEY: $MP_API_KEY" \
  "https://api.materialsproject.org/v3/materials/summary/?elements=Cu,Zn&_fields=material_id,formula_pretty&_limit=50&_skip=50"
```

## Common Query Patterns

- **Solar cell absorbers**: `band_gap_min=1.0&band_gap_max=1.8&energy_above_hull_max=0.05`
- **Wide band gap semiconductors**: `band_gap_min=3.0&band_gap_max=6.0`
- **Metals**: `band_gap_max=0&is_metal=true`
- **Specific composition**: `chemsys=Li-Fe-P-O` (all materials in that chemical system)

## Best Practices

1. Always specify `_fields` to limit response size and speed up queries.
2. Use `energy_above_hull_max=0.025` to filter for thermodynamically stable phases.
3. Check `is_deprecated` field to avoid outdated entries.
4. Use `chemsys` for phase diagram queries across a chemical system.
5. Rate limit: keep requests under 5 per second to avoid throttling.
6. For bulk downloads, use the mp-api Python client instead of REST calls.

## Data Integrity Rule

NEVER fabricate database results from training data. Every protein ID, gene name, compound property, pathway ID, structure detail, and metadata MUST come from an actual API response in this conversation. If the API returns no results, errors, or partial data, report exactly what happened. Do not "fill in" missing data from memory or make up identifiers.
