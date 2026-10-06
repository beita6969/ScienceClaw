# ScienceClaw engine

The Python engine of the ScienceClaw agent system, embedded as an independent package inside the
TypeScript ScienceClaw gateway. It provides typed workflow (canvas) orchestration, the scientific tool
library `scilib`, the versioned Skill/Operator program and the replay-verified evolution gate. The
package keeps its own `scienceclaw/`, `scilib/`, `configs/` and `docs/` boundaries so the gateway can
evolve without changing the engine contracts.

The gateway plugin talks to the engine through `python -m scienceclaw.rpc`, a line-delimited JSON-RPC
service (canvas sessions, tool library, program store, evolution from live sessions).

Model weights, caches, logs, run outputs and virtual environments stay outside Git. Set
`SCIENCECLAW_MODELS` for staged model weights; `python -m scienceclaw.cli setup` stages them. The
checked-in `configs/default.yaml` contains no machine-specific paths, so the same package works on a
laptop, a cluster or a container.

## Companion benchmark

The ScienceClaw-Eval benchmark (23 disciplines across the natural and social sciences, FoR30-FoR52)
that accompanies the paper "ScienceClaw: Benchmarking Continual Self-Evolution of AI-for-Science Agents
Across the Natural and Social Sciences" is released separately. Its evaluation data is hosted on
Hugging Face: <https://huggingface.co/datasets/beita6969/scienceclaw-eval>. This package is the
agent system only; it contains no evaluation harness, datasets or tests.

See [docs/DESIGN.md](docs/DESIGN.md) for the design contract and
[docs/INTEGRATION.md](docs/INTEGRATION.md) for the gateway, plugin, skill and self-evolution map.
