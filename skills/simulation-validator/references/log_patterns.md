# Log Patterns

## Overview

This document catalogs common log patterns that indicate simulation issues. Use these patterns for automated monitoring and failure diagnosis.

---

## Pattern Categories

| Category | Severity | Typical Action |
|----------|----------|----------------|
| Numerical Issues | CRITICAL | Stop, reduce dt |
| Solver Failures | WARNING-CRITICAL | Tune solver, check conditioning |
| Resource Errors | CRITICAL | Fix environment, reduce problem size |
| Warnings | INFO-WARNING | Monitor, may be acceptable |

---

## Numerical Issues

### NaN/Inf Detection

| Pattern | Regex | Meaning |
|---------|-------|---------|
| NaN appeared | `nan\|NaN\|NAN` | Not a Number in computation |
| Infinity | `inf\|Inf\|INF` | Overflow to infinity |
| Overflow | `overflow\|Overflow` | Numeric overflow |
| Underflow | `underflow\|Underflow` | Numeric underflow (usually okay) |

**Example Log Lines:**
```
[ERROR] Field contains NaN at step 1523
[WARN] Detected Inf in temperature field
Solution overflow at t = 0.0234
```

**Regex for Detection:**
```python
import re
nan_inf_pattern = re.compile(r'\b(nan|inf|overflow)\b', re.IGNORECASE)
```

### Divergence Indicators

| Pattern | Regex | Meaning |
|---------|-------|---------|
| Divergence | `diverg` | Solution blowing up |
| Explosion | `explo` | Rapid growth |
| Unstable | `unstable` | Instability detected |
| Blow-up | `blow.?up` | Numerical explosion |

**Example Log Lines:**
```
[ERROR] Solution diverging at iteration 5000
Newton method: divergence detected
Residual exploding: 1e+38
```

**Regex for Detection:**
```python
diverge_pattern = re.compile(r'diverg|explo|unstable|blow.?up', re.IGNORECASE)
```

---

## Solver Failures

### Convergence Issues

| Pattern | Regex | Meaning |
|---------|-------|---------|
| Max iterations | `max.?iter\|maximum.?iter` | Hit iteration limit |
| Did not converge | `did not converge\|failed to converge` | Convergence failure |
| Stagnation | `stagnat` | No progress |
| Slow convergence | `slow.?converg` | Very slow progress |

**Example Log Lines:**
```
[WARN] Linear solver: max iterations (1000) reached
Newton: did not converge in 50 iterations
GMRES stagnation detected at iteration 234
```

**Regex for Detection:**
```python
converge_pattern = re.compile(
    r'max.?iter|did not converge|failed to converge|stagnat',
    re.IGNORECASE
)
```

### Linear Solver Issues

| Pattern | Regex | Meaning |
|---------|-------|---------|
| Singular matrix | `singular` | Matrix not invertible |
| Ill-conditioned | `ill.?condition` | Poor conditioning |
| Breakdown | `breakdown` | Krylov method failure |
| Negative pivot | `negative.?pivot` | LU decomposition issue |

**Example Log Lines:**
```
[ERROR] Matrix is singular or nearly singular
Condition number: 1e+15 (ill-conditioned)
GMRES breakdown at iteration 45
```

---

## Time Step Issues

### Adaptive Time Stepping

| Pattern | Regex | Meaning |
|---------|-------|---------|
| dt reduced | `dt.{0,10}reduc` | Time step decreased |
| dt too small | `dt.{0,10}(too small\|minimum)` | Hit minimum dt |
| Step rejected | `step.{0,10}reject` | Step failed, retrying |
| CFL violation | `cfl\|courant` | Stability limit exceeded |

**Example Log Lines:**
```
[INFO] dt reduced from 1e-3 to 5e-4
[WARN] dt approaching minimum: 1.2e-10
Step rejected: error estimate 1.5e-2 > tolerance
CFL condition violated, reducing time step
```

**Regex for Detection:**
```python
dt_pattern = re.compile(r'dt[^0-9]*([0-9][0-9eE+\.-]*)', re.IGNORECASE)
dt_issue_pattern = re.compile(r'dt.{0,10}(reduc|too small|minimum)', re.IGNORECASE)
```

---

## Resource Errors

### Memory Issues

| Pattern | Regex | Meaning |
|---------|-------|---------|
| Out of memory | `out of memory\|OOM` | RAM exhausted |
| Allocation failed | `alloc.{0,10}fail` | Cannot get memory |
| Memory limit | `memory.{0,10}limit` | Hit memory cap |
| Bad alloc | `bad_alloc\|bad alloc` | C++ allocation error |

**Example Log Lines:**
```
[FATAL] Out of memory allocating 16 GB array
std::bad_alloc: failed to allocate 8589934592 bytes
Memory limit exceeded, terminating
```

### Disk/IO Issues

| Pattern | Regex | Meaning |
|---------|-------|---------|
| Disk full | `disk.{0,10}full\|no space` | Storage exhausted |
| Permission denied | `permission.{0,10}denied` | Access error |
| File not found | `file.{0,10}not found\|no such file` | Missing file |
| IO error | `io.?error\|i/o.?error` | Input/output failure |

**Example Log Lines:**
```
[ERROR] Disk full: cannot write checkpoint
Permission denied: /output/results.h5
IOError: failed to write 1.2 GB to disk
```

---

## Residual Patterns

### Extracting Residual Values

Common log formats and regex patterns:

| Format | Example | Regex |
|--------|---------|-------|
| Labeled | `residual = 1.23e-5` | `residual[^0-9eE+\-]*([0-9][0-9eE+\.-]*)` |
| Tabular | `1000  1.23e-5  2.34e-6` | Depends on format |
| Bracketed | `[res: 1.23e-5]` | `\[res[^\]]*([0-9][0-9eE+\.-]*)\]` |

**Example Extraction:**
```python
residual_re = re.compile(r'residual[^0-9eE+\-]*([0-9][0-9eE+\.-]*)', re.IGNORECASE)
for line in log_lines:
    match = residual_re.search(line)
    if match:
        residual = float(match.group(1))
```

---

## Simulator-Specific Patterns

### FEniCS/DOLFIN

| Pattern | Meaning |
|---------|---------|
| `PETSc error` | PETSc solver issue |
| `UFL error` | Form definition problem |
| `DOLFIN warning` | Runtime warning |

### OpenFOAM

| Pattern | Meaning |
|---------|---------|
| `FOAM FATAL ERROR` | Critical failure |
| `Maximum number of iterations exceeded` | Solver limit |
| `Courant Number` | CFL monitoring |

### MOOSE Framework

| Pattern | Meaning |
|---------|---------|
| `*** ERROR ***` | MOOSE error |
| `Solve Did NOT Converge` | Nonlinear failure |
| `Time Step` | Time integration info |

---

## Pattern Priority

When multiple patterns match, prioritize by severity:

1. **FATAL**: NaN/Inf, out of memory, disk full
2. **CRITICAL**: Divergence, solver breakdown
3. **WARNING**: Max iterations, dt reduction
4. **INFO**: Normal progress messages

---

## Implementation Example

```python
import re
from typing import List, Tuple

PATTERNS = [
    (re.compile(r'nan|inf|overflow', re.I), 'Numerical blow-up', 'FATAL'),
    (re.compile(r'diverg|explo', re.I), 'Divergence', 'CRITICAL'),
    (re.compile(r'out of memory|oom|bad_alloc', re.I), 'Memory exhaustion', 'FATAL'),
    (re.compile(r'disk.{0,10}full|no space', re.I), 'Disk full', 'FATAL'),
    (re.compile(r'max.?iter|did not converge', re.I), 'Convergence failure', 'WARNING'),
    (re.compile(r'dt.{0,10}reduc', re.I), 'Time step reduced', 'INFO'),
]

def scan_log(text: str) -> List[Tuple[str, str]]:
    """Return list of (issue, severity) tuples found in log."""
    issues = []
    for pattern, description, severity in PATTERNS:
        if pattern.search(text):
            issues.append((description, severity))
    return issues
```
