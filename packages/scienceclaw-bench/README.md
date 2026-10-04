# ScienceClaw benchmark package

This directory contains the Python benchmark rebuild embedded as an independent
package inside the TypeScript ScienceClaw gateway.  The package keeps its own
`scienceclaw/`, `scilib/`, `configs/`, `scripts/`, `tests/`, and `reports/`
boundaries so the gateway can evolve without changing the benchmark contracts.

The native gateway bridge invokes `benchctl.py` through a small JSON protocol.
Only task inventory, the offline TOY smoke run, and report generation are
exposed to the agent tool.  Formal ID/OOD evaluation remains an explicit
server-side operation and is not exposed through the gateway bridge.

Datasets, model weights, caches, logs, run outputs, and virtual environments
stay outside Git and are selected through deployment configuration.

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for the complete gateway,
benchmark, operator, skill, and self-evolution map.
The external model and checkpoint policy is in [docs/ASSETS.md](docs/ASSETS.md).
