# Campaign Patterns

Common patterns for organizing multi-simulation campaigns.

## Pattern 1: Parameter Sweep

**Use case**: Explore parameter sensitivity systematically

```
campaign/
├── manifest.json      # Sweep configuration
├── campaign.json      # Job tracking state
├── config_0000.json   # Base + dt=0.001
├── config_0001.json   # Base + dt=0.002
├── config_0002.json   # Base + dt=0.003
├── result_0000.json   # Output from job 0
├── result_0001.json   # Output from job 1
└── ...
```

**Workflow**:
1. Generate sweep: `sweep_generator.py --params "dt:0.001:0.01:10"`
2. Initialize campaign: `campaign_manager.py --action init`
3. Run simulations externally (parallel or sequential)
4. Track progress: `job_tracker.py --update`
5. Aggregate: `result_aggregator.py --metric objective`

## Pattern 2: Convergence Study

**Use case**: Verify mesh/time convergence

```
convergence_study/
├── mesh_10x10/
├── mesh_20x20/
├── mesh_40x40/
├── mesh_80x80/
└── convergence_analysis.json
```

**Approach**:
1. Create separate campaigns for each resolution
2. Or use single campaign with resolution as parameter
3. Post-process to compute convergence rates

## Pattern 3: Calibration Campaign

**Use case**: Fit model to experimental data

```
calibration/
├── iteration_01/      # Initial DOE
│   ├── manifest.json
│   ├── config_*.json
│   └── result_*.json
├── iteration_02/      # Refined search
│   └── ...
└── best_parameters.json
```

**Workflow**:
1. Start with DOE sampling (LHS)
2. Run initial batch
3. Identify promising region
4. Refine with focused sweep
5. Repeat until converged

## Pattern 4: Ensemble Runs

**Use case**: Uncertainty quantification with stochastic inputs

```
ensemble/
├── sample_0000.json   # Random seed 0
├── sample_0001.json   # Random seed 1
├── ...
└── statistics.json    # Mean, std, confidence intervals
```

**Key considerations**:
- Use different random seeds for each run
- Aggregate statistics: mean, variance, percentiles
- Check convergence of ensemble statistics

## Job Naming Conventions

| Convention | Example | Use Case |
|------------|---------|----------|
| Sequential | `job_0000`, `job_0001` | Simple sweeps |
| Parameter-based | `dt_0.001_kappa_0.5` | Easy identification |
| Hierarchical | `sweep_dt/job_001` | Nested studies |
| Hash-based | `job_a3f8b2c1` | Unique, collision-free |

## State Machine for Jobs

```
                    ┌─────────┐
                    │ pending │
                    └────┬────┘
                         │ start
                         ▼
                    ┌─────────┐
          timeout   │ running │   error
         ┌─────────►│         │◄─────────┐
         │          └────┬────┘          │
         │               │ finish        │
         ▼               ▼               ▼
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ timeout │     │completed│     │ failed  │
    └─────────┘     └─────────┘     └─────────┘
```

## File Locking for Parallel Execution

When running jobs in parallel, use lock files to prevent conflicts:

```bash
# Job script wrapper
flock -n job_${JOB_ID}.lock python sim.py --config ${CONFIG} || exit 1
```

## Campaign Metadata Best Practices

Always record in `manifest.json`:
- Creation timestamp
- User/machine identifier
- Git commit hash of simulation code
- Base configuration checksum
- Parameter ranges and method

This enables reproducibility and debugging.
