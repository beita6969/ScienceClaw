# ScienceClaw benchmark package

This directory contains the Python benchmark rebuild embedded as an independent
package inside the TypeScript ScienceClaw gateway.  The package keeps its own
`scienceclaw/`, `scilib/`, `configs/`, `scripts/`, `tests/`, and `reports/`
boundaries so the gateway can evolve without changing the benchmark contracts.

The native gateway bridge invokes `benchctl.py` through a small JSON protocol.
Only the catalog, task availability, and report generation for existing runs are
exposed to the agent tool.  Formal ID/OOD evaluation remains an explicit
server-side operation and is not exposed through the gateway bridge.

Datasets, model weights, caches, logs, run outputs, and virtual environments
stay outside Git. Set `SCIENCECLAW_DATA_ROOT` to the mounted or Hugging Face
dataset snapshot before running; set `SCIENCECLAW_MODELS` for staged model
weights. The checked-in configs leave these roots empty so the same package
works on a laptop, Leonardo, or a container.

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for the complete gateway,
benchmark, operator, skill, and self-evolution map.
