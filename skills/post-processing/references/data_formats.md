# Data Formats Reference

## Supported Input Formats

### JSON Format

The primary format for structured simulation data.

**Field Data Structure:**
```json
{
    "timestep": 100,
    "time": 0.5,
    "phi": [[0.0, 0.1, ...], [0.2, 0.3, ...], ...],
    "concentration": [[0.5, 0.5, ...], [0.5, 0.5, ...], ...],
    "dx": 0.01,
    "dy": 0.01,
    "Lx": 1.0,
    "Ly": 1.0
}
```

**Time Series Structure:**
```json
{
    "history": {
        "time": [0.0, 0.01, 0.02, ...],
        "energy": [1.0, 0.99, 0.98, ...],
        "residual": [1e-2, 1e-3, 1e-4, ...],
        "mass": [1.0, 1.0, 1.0, ...]
    }
}
```

**Profile Data Structure:**
```json
{
    "coordinates": [0.0, 0.1, 0.2, ...],
    "values": [0.0, 0.5, 1.0, ...],
    "field": "phi",
    "axis": "x",
    "slice_position": 0.5
}
```

### CSV Format

Tabular data with header row.

**Time Series Example:**
```csv
time,energy,residual,mass
0.0,1.0,1e-2,1.0
0.01,0.99,1e-3,1.0
0.02,0.98,1e-4,1.0
```

**Profile Example:**
```csv
x,phi,concentration
0.0,0.0,0.5
0.1,0.25,0.5
0.2,0.5,0.5
```

## Output Formats

All scripts support `--json` flag for machine-readable output.

### Standard JSON Output Structure

```json
{
    "script": "script_name",
    "source_file": "input_file.json",
    "field": "phi",
    "result_key": "result_value",
    "metadata": {}
}
```

### CSV Export

For tabular results, use `--export-csv` where available:
- Comma-separated values
- Header row with field names
- Numeric precision: 6 significant figures

## Field Naming Conventions

| Convention | Example | Description |
|------------|---------|-------------|
| Snake case | `phi_field` | Preferred for field names |
| Timestep suffix | `field_0100.json` | For sequential output |
| History files | `history.json` | Time series data |
| Config files | `config.json` | Simulation parameters |

## Grid Information

Scripts attempt to extract grid information from:

1. **Explicit fields:** `dx`, `dy`, `dz`, `nx`, `ny`, `nz`
2. **Domain fields:** `Lx`, `Ly`, `Lz` (computed: `dx = Lx / (nx - 1)`)
3. **Bounds object:** `{"bounds": {"x": [0, 1], "y": [0, 1]}}`
4. **Default:** `dx = dy = dz = 1.0` if not found

## Data Validation

Scripts validate:
- File exists and is readable
- JSON is well-formed
- Requested fields exist
- Numeric data is present
- No unexpected NaN/Inf values (where applicable)
