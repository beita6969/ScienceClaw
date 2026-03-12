# Optimization Strategies

## Introduction

This document provides detailed optimization strategies for common performance bottlenecks in computational materials science simulations. Strategies are organized by bottleneck type and include implementation guidance.

## Solver Optimization

### Strategy 1: Preconditioner Selection

**When to use**: Linear solver >50% of runtime, many iterations

**Approach**: Choose appropriate preconditioner for problem type

**Preconditioner Options**:

| Preconditioner | Best For | Pros | Cons |
|----------------|----------|------|------|
| **None** | Well-conditioned problems | Zero setup cost | Many iterations |
| **Jacobi** | Diagonal-dominant matrices | Fast setup, parallel | Weak convergence |
| **ILU(k)** | General sparse matrices | Good convergence | Serial, expensive setup |
| **AMG** | Elliptic PDEs, diffusion | Excellent convergence | Complex setup |
| **Multigrid** | Structured grids | Optimal complexity | Problem-specific |

**Implementation**:
```python
# Example: Switch from no preconditioner to AMG
solver.set_preconditioner('amg')
solver.set_amg_levels(3)
solver.set_amg_smoother('gauss-seidel')
```

**Expected Improvement**: 2-10x reduction in solver time

### Strategy 2: Solver Tolerance Tuning

**When to use**: Solver converges to very tight tolerance, over-solving

**Approach**: Relax tolerance to minimum needed for accuracy

**Guidelines**:
- Start with relative tolerance 1e-6
- If solution quality is acceptable, try 1e-5 or 1e-4
- Monitor residual vs solution error
- Tighten tolerance only if solution degrades

**Implementation**:
```python
# Relax tolerance
solver.set_relative_tolerance(1e-5)  # was 1e-8
solver.set_absolute_tolerance(1e-10)  # was 1e-12
```

**Expected Improvement**: 1.5-3x reduction in solver time

### Strategy 3: Direct vs Iterative Solver

**When to use**: Small problems (<100k DOFs) or very ill-conditioned

**Approach**: Use direct solver for guaranteed convergence

**Trade-offs**:
- **Direct**: O(N^1.5) time, O(N^1.3) memory, guaranteed convergence
- **Iterative**: O(N) time per iteration, O(N) memory, may not converge

**Decision Rule**:
- N < 10k: Direct solver
- 10k < N < 100k: Try iterative first, fall back to direct
- N > 100k: Iterative solver (direct too expensive)

**Implementation**:
```python
if num_dofs < 100000:
    solver = DirectSolver('mumps')
else:
    solver = IterativeSolver('gmres', preconditioner='amg')
```

**Expected Improvement**: Varies (direct faster for small problems)

## Assembly Optimization

### Strategy 4: Matrix Caching

**When to use**: Geometry is static, matrix assembled repeatedly

**Approach**: Assemble matrix once, reuse for all time steps

**Implementation**:
```python
# Assemble once
if not matrix_cached:
    assemble_matrix(A)
    matrix_cached = True

# Reuse in time loop
for t in time_steps:
    # Only update RHS
    assemble_rhs(b, t)
    solve(A, x, b)
```

**Expected Improvement**: 2-5x reduction in assembly time

**Limitations**: Only works for linear problems with static geometry

### Strategy 5: Vectorized Assembly

**When to use**: Assembly is element-by-element, not vectorized

**Approach**: Batch element operations using NumPy/BLAS

**Implementation**:
```python
# Before: Loop over elements
for elem in elements:
    Ke = compute_element_matrix(elem)
    assemble_into_global(A, Ke, elem.dofs)

# After: Vectorized
Ke_all = compute_all_element_matrices(elements)  # Vectorized
assemble_batch(A, Ke_all, dof_map)
```

**Expected Improvement**: 2-4x reduction in assembly time

### Strategy 6: Parallel Assembly with Coloring

**When to use**: Assembly is serial, parallel simulation

**Approach**: Color elements to avoid race conditions, assemble in parallel

**Implementation**:
```python
# Color elements (one-time cost)
colors = graph_coloring(element_connectivity)

# Parallel assembly
for color in colors:
    # Elements in same color don't share DOFs
    parallel_for elem in elements_with_color(color):
        Ke = compute_element_matrix(elem)
        assemble_into_global(A, Ke, elem.dofs)
```

**Expected Improvement**: Near-linear speedup with processor count

## I/O Optimization

### Strategy 7: Reduce Output Frequency

**When to use**: I/O >20% of runtime, frequent output writes

**Approach**: Write output less frequently

**Guidelines**:
- Transient problems: Output every N time steps (N=10-100)
- Steady-state: Output only final solution
- Adaptive: Output when solution changes significantly

**Implementation**:
```python
# Before: Output every step
for t in time_steps:
    solve_step(t)
    write_output(t)

# After: Output every 10 steps
for i, t in enumerate(time_steps):
    solve_step(t)
    if i % 10 == 0:
        write_output(t)
```

**Expected Improvement**: 5-10x reduction in I/O time

### Strategy 8: Parallel I/O

**When to use**: Serial I/O in parallel simulation, large output files

**Approach**: Use parallel I/O library (HDF5, MPI-IO)

**Implementation**:
```python
# Before: Serial I/O (rank 0 writes all data)
if rank == 0:
    gather_data_from_all_ranks()
    write_file(data)

# After: Parallel I/O (each rank writes its data)
h5file = h5py.File('output.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
h5file.create_dataset('field', data=local_data)
h5file.close()
```

**Expected Improvement**: Near-linear speedup with processor count

### Strategy 9: Output Compression

**When to use**: Large output files, slow storage

**Approach**: Compress output data

**Options**:
- **gzip**: Good compression, moderate speed
- **lz4**: Fast compression, moderate ratio
- **zstd**: Balanced compression and speed

**Implementation**:
```python
# HDF5 with compression
h5file.create_dataset('field', data=data, compression='gzip', compression_opts=4)
```

**Expected Improvement**: 2-5x reduction in file size, 1.5-2x reduction in write time (if I/O bound)

## Parallel Optimization

### Strategy 10: Load Balancing

**When to use**: Poor scaling, uneven processor utilization

**Approach**: Redistribute work to balance load

**Techniques**:
- **Static partitioning**: Partition mesh evenly by element count
- **Dynamic partitioning**: Repartition based on measured work
- **Work stealing**: Idle processors steal work from busy ones

**Implementation**:
```python
# Use graph partitioning library (METIS, ParMETIS)
partition = metis.partition_graph(mesh, nparts=num_procs, objtype='vol')
redistribute_mesh(mesh, partition)
```

**Expected Improvement**: 1.5-3x improvement in scaling efficiency

### Strategy 11: Asynchronous Communication

**When to use**: Communication overhead >20%, blocking communication

**Approach**: Overlap communication with computation

**Implementation**:
```python
# Before: Blocking communication
send_data(neighbor)
receive_data(neighbor)
compute()

# After: Non-blocking communication
req_send = isend_data(neighbor)
req_recv = irecv_data(neighbor)
compute_interior()  # Compute while communicating
wait(req_send, req_recv)
compute_boundary()  # Compute boundary after communication
```

**Expected Improvement**: 1.5-2x reduction in communication overhead

### Strategy 12: Hybrid MPI+OpenMP

**When to use**: Poor MPI scaling, many cores per node

**Approach**: Use MPI between nodes, OpenMP within nodes

**Benefits**:
- Reduced MPI communication (fewer processes)
- Better memory locality
- Reduced memory footprint

**Implementation**:
```python
# Launch with fewer MPI ranks, more threads per rank
# mpirun -n 4 --bind-to socket --map-by socket:PE=8 ./simulation
# (4 MPI ranks, 8 OpenMP threads each = 32 cores)

# In code
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    compute_element(i);
}
```

**Expected Improvement**: 1.5-2x improvement in scaling efficiency

## Memory Optimization

### Strategy 13: Reduce Mesh Resolution

**When to use**: Memory usage >80%, out-of-memory errors

**Approach**: Use coarser mesh

**Guidelines**:
- Reduce resolution by factor of 2 in each dimension
- Memory reduction: 8x (3D), 4x (2D)
- Verify solution accuracy is acceptable

**Implementation**:
```python
# Before
mesh = create_mesh(nx=256, ny=256, nz=256)  # 16M elements

# After
mesh = create_mesh(nx=128, ny=128, nz=128)  # 2M elements (8x less memory)
```

**Expected Improvement**: 2-8x reduction in memory usage

### Strategy 14: Iterative vs Direct Solver

**When to use**: Memory-limited, using direct solver

**Approach**: Switch to iterative solver

**Memory Comparison**:
- **Direct**: O(N^1.3) memory (fill-in during factorization)
- **Iterative**: O(N) memory (matrix + workspace vectors)

**Implementation**:
```python
# Before: Direct solver
solver = DirectSolver('mumps')

# After: Iterative solver
solver = IterativeSolver('gmres', preconditioner='ilu')
```

**Expected Improvement**: 5-20x reduction in memory usage

### Strategy 15: Single Precision

**When to use**: Memory-limited, accuracy requirements allow

**Approach**: Use single precision (float32) instead of double (float64)

**Trade-offs**:
- Memory: 2x reduction
- Accuracy: ~7 digits (single) vs ~15 digits (double)
- Speed: Often faster (better cache utilization, SIMD)

**Implementation**:
```python
# Use float32 for field variables
field = np.zeros(n, dtype=np.float32)  # was np.float64
```

**Expected Improvement**: 2x reduction in memory usage, 1.2-1.5x speedup

## Algorithm Optimization

### Strategy 16: Adaptive Time Stepping

**When to use**: Fixed time step, solution varies slowly/rapidly

**Approach**: Adjust time step based on solution behavior

**Benefits**:
- Larger steps when solution is smooth (faster)
- Smaller steps when solution changes rapidly (accurate)

**Implementation**:
```python
dt = dt_initial
for t in time_range:
    solve_step(dt)
    error = estimate_error()
    
    if error < tol_low:
        dt *= 1.5  # Increase step
    elif error > tol_high:
        dt *= 0.5  # Decrease step
        redo_step()
```

**Expected Improvement**: 2-10x reduction in number of time steps

### Strategy 17: Matrix-Free Methods

**When to use**: Matrix assembly is expensive, matrix is not reused

**Approach**: Compute matrix-vector products on-the-fly

**Benefits**:
- No matrix storage (memory savings)
- No matrix assembly (time savings)
- Suitable for nonlinear problems

**Implementation**:
```python
# Instead of assembling A, define matrix-vector product
def matvec(x):
    y = np.zeros_like(x)
    for elem in elements:
        y_elem = compute_element_matvec(elem, x)
        add_to_global(y, y_elem, elem.dofs)
    return y

# Use with iterative solver
solver = IterativeSolver(matvec=matvec)
```

**Expected Improvement**: 2-5x reduction in memory, 1.5-3x reduction in time

## Profiling-Driven Optimization

### General Workflow

1. **Profile**: Identify bottleneck (timing, scaling, memory)
2. **Hypothesize**: Determine likely cause
3. **Optimize**: Implement targeted optimization
4. **Verify**: Re-profile to measure improvement
5. **Iterate**: Repeat until performance goals met

### Optimization Priority

1. **High priority**: Bottlenecks >50% of runtime or critical path
2. **Medium priority**: Bottlenecks 30-50% of runtime
3. **Low priority**: Bottlenecks <30% of runtime

### Diminishing Returns

- First optimization: Often 2-5x improvement
- Second optimization: Often 1.5-2x improvement
- Third+ optimization: Often <1.5x improvement

**Rule of thumb**: Stop optimizing when improvement <20% or effort exceeds benefit

## Case Studies

### Case Study 1: Materials Simulation (Phase-Field Example)

**Problem**: Simulation taking 10 hours, solver dominates (80% of time)

**Profiling**:
- Linear solver: 8 hours (80%)
- Assembly: 1.5 hours (15%)
- Other: 0.5 hours (5%)

**Optimizations**:
1. Switch from no preconditioner to AMG: 8h → 2h (4x improvement)
2. Relax tolerance from 1e-8 to 1e-6: 2h → 1.5h (1.3x improvement)
3. Cache matrix (geometry is static): 1.5h assembly → 0.1h (15x improvement)

**Result**: 10 hours → 2.1 hours (4.8x overall improvement)

### Case Study 2: Parallel Scaling

**Problem**: Poor scaling beyond 8 processors (efficiency 0.55 at 16 procs)

**Profiling**:
- 1 proc: 1000s
- 2 procs: 520s (efficiency 0.96)
- 4 procs: 270s (efficiency 0.93)
- 8 procs: 150s (efficiency 0.83)
- 16 procs: 90s (efficiency 0.69)

**Optimizations**:
1. Improve load balancing with METIS: efficiency 0.69 → 0.75
2. Use non-blocking communication: efficiency 0.75 → 0.82
3. Reduce synchronization points: efficiency 0.82 → 0.87

**Result**: Efficiency at 16 procs: 0.69 → 0.87 (26% improvement)

### Case Study 3: Memory-Limited

**Problem**: Out-of-memory error with 256³ mesh (16M elements)

**Profiling**:
- Field memory: 2.0 GB
- Direct solver: 18.0 GB
- Total: 20.0 GB (exceeds 16 GB available)

**Optimizations**:
1. Switch to iterative solver: 18 GB → 2 GB (9x reduction)
2. Use single precision for fields: 2 GB → 1 GB (2x reduction)

**Result**: 20 GB → 3 GB (6.7x reduction, fits in memory)

## References

- Saad, Y. (2003). "Iterative Methods for Sparse Linear Systems"
- Gropp, W., Lusk, E., & Skjellum, A. (1999). "Using MPI"
- Trottenberg, U., Oosterlee, C., & Schüller, A. (2001). "Multigrid"
- Karypis, G., & Kumar, V. (1998). "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs"

