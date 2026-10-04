# ScienceClaw benchmark bridge

This optional plugin exposes the isolated Python benchmark package through one
native agent tool, `scienceclaw_bench`. It supports `catalog`, `list_tasks`, the
offline `smoke` run, and `report`. Formal hidden ID/OOD evaluation stays
server-side and is deliberately not reachable through the agent tool.

Configure the plugin with an explicit `repoRoot` pointing at
`packages/scienceclaw-bench`. `dataRoot`, `modelRoot`, `runRoot`, and
`pythonBin` are deployment settings; they are never accepted as agent
parameters. Keep datasets, weights, caches, and credentials outside Git.
