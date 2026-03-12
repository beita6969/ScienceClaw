# Output and Checkpoint Guidelines

Comprehensive guide for scheduling solution output and checkpoints in long-running simulations.

## Output Cadence

### Physics-Based Output Frequency

Match output frequency to the dynamics you want to capture:

| Dynamics | Characteristic Time | Outputs per τ |
|----------|---------------------|---------------|
| Interface motion | τ = L/v | 20-50 |
| Diffusion across L | τ = L²/D | 20-50 |
| Oscillation period | τ = 1/f | 10-20 per period |
| Reaction half-life | τ = ln(2)/k | 5-10 |
| Growth/coarsening | τ varies | 10-20 per decade |

### Minimum Recommendation

```
N_outputs ≥ 20 per characteristic time scale

dt_output = τ_characteristic / 20
```

### For Visualization

| Purpose | Frames | Notes |
|---------|--------|-------|
| Static images | 10-50 | Key times |
| Smooth animation | 100-500 | 24-60 fps target |
| Detailed analysis | 500-2000 | High resolution |
| Machine learning | 1000+ | Training data |

### Adaptive Output

```python
def should_output(t, t_last, solution):
    """Adaptive output based on solution change."""
    # Time-based minimum
    if t - t_last > dt_output_max:
        return True

    # Change-based
    change = compute_change(solution)
    if change > threshold:
        return True

    return False
```

## Checkpoint Strategy

### Purpose of Checkpoints

| Purpose | Typical Frequency |
|---------|-------------------|
| Crash recovery | Every 30-60 min wall time |
| Analysis restart | Every τ_characteristic |
| Parameter studies | At key milestones |
| Long-term storage | End of simulation |

### Optimal Checkpoint Interval

**Daly's Formula** (minimize total time with failures):

```
τ_checkpoint = sqrt(2 × δ × MTBF)

where:
  δ = checkpoint write time
  MTBF = mean time between failures
```

**Example:**
```
δ = 2 minutes (checkpoint write time)
MTBF = 24 hours = 1440 minutes

τ_checkpoint = sqrt(2 × 2 × 1440) = 76 minutes
```

**Young's Refinement** (accounts for restart overhead):

```
τ_checkpoint = sqrt(2 × δ × (MTBF - δ)) + δ
```

### Practical Guidelines

| Compute Environment | Checkpoint Interval |
|---------------------|---------------------|
| Laptop/workstation | 30-60 minutes |
| Shared cluster | 15-30 minutes |
| Large HPC job | 30-60 minutes |
| Cloud with preemption | 5-15 minutes |
| Very long runs | Based on Daly formula |

### Maximum Lost Work

If MTBF is unknown, set checkpoint interval based on acceptable lost work:

```
τ_checkpoint ≤ max_lost_time

Typical: max_lost_time = 30-60 minutes
Conservative: max_lost_time = 15 minutes
```

## Checkpoint Content

### Minimum Required

```python
checkpoint = {
    'time': current_time,
    'step': step_number,
    'solution': solution_fields,  # Primary unknowns
    'dt': current_dt,
    'random_state': rng.state,  # For reproducibility
}
```

### Recommended Additional

```python
checkpoint.update({
    'parameters': simulation_parameters,
    'mesh': mesh_data,  # Or mesh filename
    'history': recent_history,  # For multi-step methods
    'integrator_state': integrator_internal_state,
    'version': code_version,
    'timestamp': datetime.now(),
})
```

### Optional for Debugging

```python
checkpoint.update({
    'residual': current_residual,
    'error_estimate': error_estimate,
    'performance': timing_data,
    'convergence_history': solver_history,
})
```

## I/O Scheduling

### Avoid I/O Storms

**Problem:** Simultaneous output + checkpoint causes I/O spike.

**Solution:** Stagger operations:

```python
def schedule_io(step, t):
    do_output = (step % output_interval == 0)
    do_checkpoint = (step % checkpoint_interval == 0)

    # Stagger if both due
    if do_output and do_checkpoint:
        if step % 2 == 0:
            do_checkpoint = False  # Delay to next step
        else:
            do_output = False
```

### Asynchronous I/O

For HPC simulations:

```python
# Start checkpoint write in background
async_checkpoint = AsyncWriter(checkpoint_data)
async_checkpoint.start()

# Continue computation
for step in range(n_substeps):
    advance_solution()

# Wait for write to complete before next checkpoint
async_checkpoint.wait()
```

### Compression Trade-offs

| Format | Write Speed | Size | Portability |
|--------|-------------|------|-------------|
| Raw binary | Fast | Large | Low |
| Compressed binary | Medium | Small | Low |
| HDF5 | Medium | Medium | High |
| NetCDF | Medium | Medium | High |
| Text/CSV | Slow | Large | Very high |

## Storage Management

### Estimating Storage Needs

```
Storage per output = n_fields × n_points × bytes_per_value

Example:
  3 fields × 1000³ points × 8 bytes = 24 GB per output

For 100 outputs: 2.4 TB
```

### Retention Policies

| Data Type | Retention | Notes |
|-----------|-----------|-------|
| Final solution | Permanent | Essential |
| Key checkpoints | Long-term | For analysis |
| Regular outputs | Medium-term | For visualization |
| Diagnostic outputs | Short-term | Delete after validation |
| Intermediate checkpoints | Until job completes | Delete after success |

### Thinning Strategy

Keep every Nth output as simulation progresses:

```python
def should_keep_output(step, current_step):
    """Logarithmic thinning: keep recent, thin old."""
    age = current_step - step

    if age < 100:
        return True  # Keep all recent
    elif age < 1000:
        return (step % 10 == 0)  # Every 10th
    elif age < 10000:
        return (step % 100 == 0)  # Every 100th
    else:
        return (step % 1000 == 0)  # Every 1000th
```

## Restart Workflow

### Clean Restart

```python
def restart_from_checkpoint(filename):
    checkpoint = load(filename)

    # Restore state
    solution = checkpoint['solution']
    time = checkpoint['time']
    step = checkpoint['step']
    dt = checkpoint['dt']

    # Restore RNG for reproducibility
    if 'random_state' in checkpoint:
        rng.set_state(checkpoint['random_state'])

    # Restore multi-step history if needed
    if 'history' in checkpoint:
        integrator.restore_history(checkpoint['history'])

    return solution, time, step, dt
```

### Verification After Restart

```python
def verify_restart(solution, checkpoint):
    """Check that restart is consistent."""
    # Recompute something that should match
    residual = compute_residual(solution)

    if 'residual' in checkpoint:
        diff = abs(residual - checkpoint['residual'])
        if diff > tolerance:
            warn(f"Residual mismatch: {diff}")

    # Check conservation
    mass = integrate(solution)
    if 'mass' in checkpoint:
        diff = abs(mass - checkpoint['mass'])
        if diff > tolerance:
            warn(f"Mass changed: {diff}")
```

### Handling Version Changes

```python
def load_checkpoint_versioned(filename):
    checkpoint = load(filename)

    if checkpoint.get('version') != current_version:
        warn(f"Checkpoint from version {checkpoint['version']}, "
             f"current is {current_version}")

        # Apply migrations if needed
        checkpoint = migrate_checkpoint(checkpoint)

    return checkpoint
```

## Documentation and Reproducibility

### Metadata to Record

```python
metadata = {
    'simulation_name': 'phase_field_solidification',
    'run_id': unique_run_id,
    'started': start_time,
    'code_version': git_commit_hash,
    'parameters': all_parameters,
    'machine': hostname,
    'n_processors': mpi_size,
}
```

### Output Naming Convention

```
{simulation_name}_{run_id}_{type}_{sequence}.{ext}

Examples:
solidification_run42_output_00100.h5
solidification_run42_checkpoint_00050.h5
solidification_run42_final.h5
```

### Log File

```
simulation.log should contain:
- Start time and parameters
- Each output with time, step, key metrics
- Each checkpoint with filename
- Warnings and errors
- End time and summary
```

## Practical Examples

### Phase-Field Solidification

```python
# Parameters
t_total = 100.0  # Total simulation time
tau_diffusion = 1.0  # Diffusion time scale
tau_growth = 10.0  # Characteristic growth time

# Output cadence
dt_output = min(tau_diffusion, tau_growth) / 30  # ~30 outputs per tau

# Checkpoint cadence
wall_time_limit = 48 * 3600  # 48 hours
checkpoint_interval = 30 * 60  # 30 minutes wall time
checkpoint_dt = estimate_sim_time_per_wall_time() * checkpoint_interval
```

### Turbulent Flow

```python
# Eddy turnover time
tau_eddy = L / u_rms

# Output for statistics
dt_output = tau_eddy / 100  # 100 samples per eddy time

# Flow-through time
tau_flow = L / U_mean

# Save fields every flow-through
dt_field_output = tau_flow / 10
```

## Quick Reference Checklist

### Output Setup
- [ ] Identified characteristic time scale(s)
- [ ] Set output interval to resolve key dynamics
- [ ] Chose appropriate file format
- [ ] Planned storage requirements
- [ ] Set up output directory structure

### Checkpoint Setup
- [ ] Estimated acceptable lost work time
- [ ] Calculated checkpoint interval (Daly or practical)
- [ ] Included all necessary restart data
- [ ] Tested restart procedure
- [ ] Set up checkpoint rotation/cleanup

### Best Practices
- [ ] Staggered output and checkpoint schedules
- [ ] Documented all cadence choices
- [ ] Set up monitoring/logging
- [ ] Planned storage cleanup
- [ ] Verified restart produces identical results
