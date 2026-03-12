---
name: quantum-computing
description: Designs and analyzes quantum computing solutions including quantum circuit construction, algorithm implementation, error correction, and quantum advantage assessment; trigger when users discuss qubits, quantum gates, quantum algorithms, or quantum hardware.
---

## When to Trigger

Activate this skill when the user mentions:
- Quantum circuits, quantum gates (Hadamard, CNOT, Toffoli)
- Qubits, superposition, entanglement, measurement
- Quantum algorithms (Shor's, Grover's, VQE, QAOA)
- Quantum error correction, decoherence, noise models
- Quantum advantage, quantum supremacy, complexity classes (BQP)
- Quantum hardware (superconducting, trapped ion, photonic)
- Quantum simulation, quantum chemistry applications

## Step-by-Step Methodology

1. **Problem formulation** - Determine if the problem has a known quantum advantage. Map the problem to a quantum computing framework: gate-based, adiabatic, or measurement-based. Identify required qubit count and circuit depth.
2. **Algorithm selection** - For search: Grover's (quadratic speedup). For factoring: Shor's. For optimization: QAOA or quantum annealing. For chemistry: VQE or QPE. For machine learning: quantum kernels or variational classifiers.
3. **Circuit design** - Construct the quantum circuit using elementary gates (H, CNOT, Rz, Ry). Decompose multi-qubit gates into native gate sets. Minimize circuit depth and CNOT count for near-term hardware compatibility.
4. **Simulation** - Simulate circuit on classical hardware using Qiskit Aer, Cirq, or PennyLane. For small systems (<30 qubits), use statevector simulation. For larger systems, use tensor network or MPS methods.
5. **Noise analysis** - Model realistic noise: single-qubit and two-qubit gate errors, measurement errors, T1/T2 decoherence times. Use noise models from real hardware backends (IBM Quantum, IonQ).
6. **Error mitigation / correction** - For near-term (NISQ): zero-noise extrapolation, probabilistic error cancellation, dynamical decoupling. For fault-tolerant: surface codes, repetition codes, logical qubit encoding.
7. **Results analysis** - Compare quantum vs. classical performance. Report circuit metrics (depth, gate count, qubit count). Assess scalability and resource requirements for practical problem sizes.

## Key Databases and Tools

- **Qiskit (IBM)** - Quantum SDK with hardware access
- **Cirq (Google)** - Quantum circuit framework
- **PennyLane (Xanadu)** - Quantum ML framework
- **Amazon Braket** - Cloud quantum computing
- **Quantum Algorithm Zoo** - Catalog of quantum algorithms
- **IBM Quantum / IonQ** - Real hardware backends

## Output Format

- Quantum circuits in standard notation (Qiskit/OpenQASM or circuit diagrams).
- State vectors or density matrices for small systems.
- Measurement histograms with shot counts and error bars.
- Resource estimates: qubit count, circuit depth, T-gate count, total gate count.

## Quality Checklist

- [ ] Problem-quantum advantage mapping justified
- [ ] Circuit decomposed into hardware-native gate set
- [ ] Qubit count and circuit depth reported
- [ ] Noise model specified if simulating realistic conditions
- [ ] Classical baseline comparison provided
- [ ] Error mitigation strategy appropriate for NISQ era
- [ ] Measurement shot count sufficient for statistical significance
- [ ] Scalability analysis for practical problem sizes included
