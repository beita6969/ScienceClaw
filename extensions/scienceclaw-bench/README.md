# ScienceClaw benchmark bridge

This optional plugin exposes the isolated Python benchmark package through one
native agent tool, `scienceclaw_bench`. It supports `catalog`, `list_tasks`, and
`report` (for a run directory under `runRoot`). Formal hidden ID/OOD evaluation
stays server-side and is deliberately not reachable through the agent tool.

Configure the plugin with an explicit `repoRoot` pointing at
`packages/scienceclaw-bench`. `dataRoot`, `modelRoot`, `runRoot`, and
`pythonBin` are deployment settings; they are never accepted as agent
parameters. Keep datasets, weights, caches, and credentials outside Git.

Because the tool is optional, enable both the plugin entry and the tool in the
agent allowlist. The following JSON5 fragment is a template; replace the
checkout and Python paths for the host that runs the gateway:

```json5
{
  plugins: {
    entries: {
      "scienceclaw-bench": {
        enabled: true,
        config: {
          repoRoot: "<checkout>/packages/scienceclaw-bench",
          pythonBin: "<venv>/bin/python",
        },
      },
    },
  },
  agents: {
    list: [{ id: "main", tools: { allow: ["scienceclaw_bench"] } }],
  },
}
```
