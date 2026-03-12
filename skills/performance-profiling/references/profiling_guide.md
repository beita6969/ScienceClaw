# Performance Profiling Guide

## Introduction

Performance profiling is the process of measuring and analyzing where computational resources (time, memory, communication) are spent during simulation execution. This guide explains key profiling concepts and how to interpret profiling results.

## Profiling Concepts

### Timing Analysis

**Definition**: Measuring how long different phases of a simulation take.

**Key Metrics**:
- **Total time**: Overall wall-clock time for the simulation
- **Phase time**: Time spent in each computational phase (assembly, solve, I/O, etc.)
- **Percentage**: Fraction of total time spent in each phase
- **Count**: Number of times each phase is executed

**Interpretation**:
- Phases consuming >50% of total time are **dominant** and should be optimized first
- Phases consuming 30-50% are **significant** and may benefit from optimization
- Phases consuming <30% are **minor** and typically not worth optimizing

**Common Phases**:
- **Mesh generation**: Creating the computational grid
- **Assembly**: Building matrices and vectors
- **Linear solver**: Solving Ax=b systems
- **Nonlinear solver**: Newton or fixed-point iterations
- **Update fields**: Computing derived quantities
- **I/O**: Writing output files
- **Communication**: MPI message passing (parallel simulations)

### Scaling Analysis

**Definition**: Measuring how performance changes with problem size or processor count.

#### Strong Scaling

**Definition**: Fixed problem size, varying processor count.

**Ideal behavior**: Time decreases proportionally with processor count.

**Metrics**:
- **Speedup**: S(N) = T(1) / T(N) where T(N) is time on N processors
- **Efficiency**: E(N) = S(N) / N = T(1) / (N * T(N))
- **Ideal efficiency**: 1.0 (100%)

**Interpretation**:
- E > 0.80: Excellent scaling
- 0.70 < E < 0.80: Good scaling
- 0.50 < E < 0.70: Poor scaling (communication/load imbalance issues)
- E < 0.50: Very poor scaling (not worth using more processors)

**Typical behavior**: Efficiency decreases as processor count increases due to:
- Communication overhead
- Load imbalance
- Serial bottlenecks (Amdahl's law)

#### Weak Scaling

**Definition**: Constant work per processor, varying processor count.

**Ideal behavior**: Time remains constant as processors increase.

**Metrics**:
- **Efficiency**: E(N) = T(1) / T(N)
- **Ideal efficiency**: 1.0 (constant time)

**Interpretation**:
- E > 0.90: Excellent weak scaling
- 0.80 < E < 0.90: Good weak scaling
- E < 0.80: Poor weak scaling

**Typical behavior**: Time increases slightly due to:
- Communication overhead (grows with processor count)
- Synchronization costs

### Memory Profiling

**Definition**: Estimating or measuring memory usage.

**Key Components**:
- **Field memory**: Storage for solution variables (concentration, temperature, etc.)
- **Solver workspace**: Temporary vectors for iterative solvers
- **Matrix storage**: Sparse matrix entries (if stored explicitly)
- **Communication buffers**: MPI send/receive buffers

**Estimation Formula**:
```
Total Memory = Field Memory + Solver Workspace + Matrix Storage

Field Memory = mesh_points × fields × components × bytes_per_value
Solver Workspace = mesh_points × workspace_multiplier × bytes_per_value
```

**Typical Values**:
- **bytes_per_value**: 8 (double precision), 4 (single precision)
- **workspace_multiplier**: 5-10 for iterative solvers, 0 for matrix-free

**Interpretation**:
- Memory usage < 60% available: Safe
- Memory usage 60-80% available: Moderate (monitor)
- Memory usage > 80% available: High (risk of swapping)
- Memory usage > 100% available: Exceeds capacity (will fail or swap)

### Bottleneck Detection

**Definition**: Identifying the computational phase or resource that limits overall performance.

**Types of Bottlenecks**:
1. **Timing bottlenecks**: Phases consuming disproportionate time
2. **Scaling bottlenecks**: Poor parallel efficiency
3. **Memory bottlenecks**: Insufficient memory or excessive memory usage
4. **I/O bottlenecks**: Slow disk writes

**Detection Criteria**:
- **Dominant phase**: Any phase >50% of total time
- **Poor scaling**: Efficiency <0.70
- **High memory**: Usage >80% of available

## Profiling Workflow

### 1. Baseline Profiling

**Goal**: Establish current performance characteristics.

**Steps**:
1. Run simulation with timing enabled
2. Extract timing data from logs
3. Identify dominant phases
4. Measure total runtime

**Tools**: `timing_analyzer.py`

### 2. Scaling Study

**Goal**: Understand parallel performance.

**Steps**:
1. Run simulation at multiple processor counts (strong scaling) or problem sizes (weak scaling)
2. Measure runtime for each configuration
3. Compute speedup and efficiency
4. Identify efficiency threshold

**Tools**: `scaling_analyzer.py`

**Best Practices**:
- Use at least 5 data points
- Double processor count between runs (1, 2, 4, 8, 16, ...)
- Keep problem size fixed (strong) or work per processor fixed (weak)
- Run multiple trials and average results

### 3. Memory Analysis

**Goal**: Ensure sufficient memory and identify memory-intensive components.

**Steps**:
1. Estimate memory from problem parameters
2. Compare to available system memory
3. Identify memory-intensive components
4. Plan resource allocation

**Tools**: `memory_profiler.py`

### 4. Bottleneck Identification

**Goal**: Pinpoint performance-limiting factors.

**Steps**:
1. Combine timing, scaling, and memory analyses
2. Identify bottlenecks using thresholds
3. Prioritize by severity (high, medium, low)
4. Generate optimization recommendations

**Tools**: `bottleneck_detector.py`

### 5. Optimization

**Goal**: Improve performance based on profiling insights.

**Steps**:
1. Implement recommended optimizations
2. Re-profile to measure improvement
3. Iterate until performance goals are met

**See**: `optimization_strategies.md` for detailed strategies

## Common Profiling Patterns

### Pattern 1: Solver-Dominated

**Symptoms**:
- Linear solver >70% of total time
- Assembly <20% of total time

**Causes**:
- Poor preconditioner
- Over-solving (tolerance too tight)
- Ill-conditioned matrix

**Solutions**:
- Use AMG or ILU preconditioner
- Relax solver tolerance
- Improve matrix conditioning (scaling, regularization)

### Pattern 2: Assembly-Dominated

**Symptoms**:
- Assembly >50% of total time
- Solver <30% of total time

**Causes**:
- Inefficient assembly routines
- Repeated assembly of static matrices
- Lack of vectorization

**Solutions**:
- Cache element matrices
- Vectorize assembly loops
- Parallelize assembly with coloring
- Consider matrix-free methods

### Pattern 3: I/O-Dominated

**Symptoms**:
- I/O >30% of total time
- Frequent output writes

**Causes**:
- Excessive output frequency
- Serial I/O in parallel simulations
- Slow storage

**Solutions**:
- Reduce output frequency
- Use parallel I/O (HDF5, MPI-IO)
- Write to fast scratch storage
- Compress output data

### Pattern 4: Poor Scaling

**Symptoms**:
- Efficiency <0.70 at moderate processor counts
- Speedup plateaus

**Causes**:
- Communication overhead
- Load imbalance
- Serial bottlenecks

**Solutions**:
- Investigate communication patterns
- Improve load balancing
- Reduce synchronization points
- Use asynchronous communication

## Profiling Best Practices

### Do's
- ✅ Profile on representative problems (not toy examples)
- ✅ Run multiple trials and average results
- ✅ Profile at multiple scales (small, medium, large)
- ✅ Document profiling methodology and results
- ✅ Re-profile after optimizations to verify improvements

### Don'ts
- ❌ Don't profile debug builds (use optimized builds)
- ❌ Don't profile on oversubscribed systems (more processes than cores)
- ❌ Don't optimize minor phases (<10% of runtime)
- ❌ Don't assume profiling results generalize to all problems
- ❌ Don't ignore statistical variation (run multiple trials)

## Interpreting Profiling Results

### Red Flags
- Any phase >70% of total time
- Efficiency <0.50 at low processor counts (<16)
- Memory usage >90% of available
- Speedup <2x when doubling processors

### Green Flags
- No phase >40% of total time
- Efficiency >0.80 at moderate processor counts
- Memory usage <60% of available
- Speedup >1.8x when doubling processors

## Advanced Topics

### Amdahl's Law

**Formula**: S(N) = 1 / (f + (1-f)/N)

Where:
- S(N) = speedup on N processors
- f = fraction of serial code

**Implication**: Even small serial fractions limit scalability.

**Example**: If 5% of code is serial (f=0.05), maximum speedup is 20x regardless of processor count.

### Gustafson's Law

**Formula**: S(N) = N - f(N-1)

**Implication**: Larger problems scale better (weak scaling perspective).

### Communication Overhead

**Sources**:
- Latency: Time to initiate communication
- Bandwidth: Time to transfer data
- Synchronization: Waiting for all processes

**Mitigation**:
- Overlap communication with computation
- Use non-blocking MPI calls
- Reduce message frequency (batch communications)
- Increase message size (amortize latency)

## Tools and Techniques

### Built-in Profiling
- Use simulation's built-in timing (if available)
- Instrument code with timers
- Log timing information to files

### External Profilers
- **gprof**: GNU profiler (function-level timing)
- **Valgrind**: Memory profiling and leak detection
- **Intel VTune**: Comprehensive performance analysis
- **TAU**: Parallel performance profiling
- **Scalasca**: Scalability analysis

### Profiling Overhead
- Timing analysis: <1% overhead
- Memory profiling: 5-10% overhead
- Detailed profiling (VTune, TAU): 10-50% overhead

## References

- Amdahl, G. M. (1967). "Validity of the single processor approach to achieving large scale computing capabilities"
- Gustafson, J. L. (1988). "Reevaluating Amdahl's law"
- Gropp, W., Lusk, E., & Skjellum, A. (1999). "Using MPI: Portable Parallel Programming with the Message-Passing Interface"

