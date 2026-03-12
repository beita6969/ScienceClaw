---
name: code-science
description: Scientific programming best practices including reproducible research, computational notebooks, version control for research, data management, HPC/parallel computing, and research software engineering. Use when user needs help with research code organization, reproducibility, scientific Python/R workflows, or computational infrastructure. Triggers on "reproducible research", "research code", "scientific computing", "HPC", "parallel computing", "Jupyter", "notebook", "data management plan", "research software", "code review for science".
---

# Scientific Programming

Best practices for research software and reproducible computation.

## Project Structure

```
project/
├── README.md              # Project overview, how to reproduce
├── LICENSE                 # MIT, Apache 2.0, or GPL
├── requirements.txt       # or environment.yml (conda)
├── setup.py / pyproject.toml
├── data/
│   ├── raw/               # Never modify raw data
│   ├── processed/         # Cleaned/transformed data
│   └── external/          # Third-party data
├── src/ or scripts/
│   ├── data_processing.py
│   ├── analysis.py
│   ├── models.py
│   └── visualization.py
├── notebooks/             # Exploratory analysis
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_figures.ipynb
├── results/
│   ├── figures/
│   └── tables/
├── tests/
└── docs/
```

## Reproducibility Checklist

1. **Environment**: Pin all dependencies with versions
   ```bash
   pip freeze > requirements.txt
   # or conda
   conda env export > environment.yml
   ```

2. **Random seeds**: Set and document all random seeds
   ```python
   import numpy as np
   import random
   SEED = 42
   np.random.seed(SEED)
   random.seed(SEED)
   # torch.manual_seed(SEED)
   # tf.random.set_seed(SEED)
   ```

3. **Data versioning**: Use DVC or git-lfs for large data
   ```bash
   dvc init
   dvc add data/raw/dataset.csv
   git add data/raw/dataset.csv.dvc
   ```

4. **Configuration**: Separate config from code
   ```python
   # config.yaml
   # experiment:
   #   learning_rate: 0.001
   #   batch_size: 32
   #   epochs: 100
   import yaml
   with open('config.yaml') as f:
       config = yaml.safe_load(f)
   ```

5. **Logging**: Record all experiments
   ```python
   import logging
   logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s %(levelname)s: %(message)s',
                       filename='experiment.log')
   ```

## Parallel Computing

```python
# Multiprocessing (CPU-bound)
from multiprocessing import Pool
import numpy as np

def process_chunk(data):
    return heavy_computation(data)

with Pool(processes=8) as pool:
    results = pool.map(process_chunk, data_chunks)

# Concurrent futures (simpler API)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_func, items))

# For I/O-bound tasks (API calls, file reading)
with ThreadPoolExecutor(max_workers=20) as executor:
    results = list(executor.map(fetch_data, urls))
```

## Performance Optimization

```python
# Profiling
import cProfile
cProfile.run('my_function()', sort='cumulative')

# Line profiling
# pip install line_profiler
# @profile decorator, then: kernprof -l -v script.py

# NumPy vectorization (avoid loops)
# Bad:
result = [x**2 + 2*x + 1 for x in data]
# Good:
result = data**2 + 2*data + 1

# Memory profiling
# pip install memory_profiler
# @profile decorator, then: python -m memory_profiler script.py
```

## Data Management

### FAIR Principles
- **Findable**: Persistent identifiers (DOI), rich metadata
- **Accessible**: Open protocols, authentication when needed
- **Interoperable**: Standard formats (CSV, JSON, HDF5, NetCDF)
- **Reusable**: Clear license, provenance, community standards

### File Formats for Science
| Format | Best For | Size | Speed |
|--------|----------|------|-------|
| CSV | Small tabular, universal | Large | Slow |
| Parquet | Large tabular, columnar | Small | Fast |
| HDF5 | Multidimensional arrays | Small | Fast |
| NetCDF | Climate/geospatial | Small | Fast |
| FITS | Astronomy | Medium | Fast |
| Feather | DataFrame interchange | Small | Very fast |

```python
# Parquet (recommended for large datasets)
df.to_parquet('data.parquet', compression='snappy')
df = pd.read_parquet('data.parquet')

# HDF5 (for arrays)
import h5py
with h5py.File('data.h5', 'w') as f:
    f.create_dataset('experiment1', data=array)
```

## Testing Scientific Code

```python
import numpy as np
import pytest

def test_conservation_law():
    """Physical quantities should be conserved"""
    initial_energy = compute_energy(initial_state)
    final_energy = compute_energy(simulate(initial_state))
    np.testing.assert_allclose(initial_energy, final_energy, rtol=1e-6)

def test_known_solution():
    """Compare against analytical solution"""
    numerical = solve_numerically(params)
    analytical = analytical_solution(params)
    np.testing.assert_allclose(numerical, analytical, atol=1e-4)

def test_symmetry():
    """Result should be symmetric under transformation"""
    result1 = compute(data)
    result2 = compute(transform(data))
    np.testing.assert_array_equal(result1, result2)
```

## Tips
- Raw data is sacred — never modify it, only create processed copies
- Use version control (git) from day one
- Write README before writing code
- Automate the full pipeline (Makefile or Snakemake)
- Document assumptions and decisions in code comments
- Use type hints for clarity in scientific code
- Publish code alongside papers (GitHub + Zenodo for DOI)
