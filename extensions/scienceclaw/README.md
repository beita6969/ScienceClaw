# ScienceClaw gateway plugin

Registers the ScienceClaw engine (`packages/scienceclaw`) as optional agent tools. The
gateway agent acts as the policy of the engine: it edits a typed workflow graph step by step
(`scienceclaw_canvas`), finds scientific tools and pretrained-model wrappers
(`scienceclaw_tools`), inspects or rolls back the versioned Skill/Operator program
(`scienceclaw_program`), and turns verified sessions into gated program updates
(`scienceclaw_evolve`).

| Tool | Operations |
| --- | --- |
| `scienceclaw_canvas` | `open`, `act`, `render`, `replay`, `finish`, `status`, `list` |
| `scienceclaw_tools` | `search`, `show`, `status`, `weights`, `setup` |
| `scienceclaw_program` | `summary`, `skills`, `operators`, `show`, `history`, `rollback` |
| `scienceclaw_evolve` | `val_add`, `val_list`, `val_remove`, `propose`, `gate`, `run`, `status`, `candidates`, `show` |

`scienceclaw_canvas(operation=open)` takes a live `task` declaration (objective, input files, the
schema of the deliverable and its hard constraints); inputs must lie under the configured input roots.

The plugin keeps one long-lived Python process (`python -m scienceclaw.rpc`, line-delimited
JSON) so canvas sessions survive between tool calls. Deployment settings (`packageRoot`,
`pythonBin`, `home`, `inputRoots`, `modelRoot`, `configPath`, `llm`) are
plugin configuration and are never accepted as tool parameters. Gateway and provider credentials
are not forwarded to the engine; the model used by `llm` nodes is the OpenAI-compatible endpoint
given under `llm`, or any backend registered through `scienceclaw.llm.interface`.

```json5
{
  plugins: {
    entries: {
      "scienceclaw": {
        enabled: true,
        config: {
          packageRoot: "<checkout>/packages/scienceclaw",
          pythonBin: "<venv>/bin/python",
          home: "<state-dir>",
          inputRoots: ["<workspace>"],
          llm: { baseUrl: "<endpoint>", model: "<model>" },
        },
      },
    },
  },
  agents: {
    list: [{
      id: "main",
      tools: {
        allow: ["scienceclaw_canvas", "scienceclaw_tools", "scienceclaw_program", "scienceclaw_evolve"],
      },
    }],
  },
}
```

Keep `home` (session receipts, program store) outside the directories the agent can read freely.
An admitted evolution candidate waits as `ready` until the user promotes it
(`python -m scienceclaw.cli live promote <id>`); `autoPromote: true` lets the gate promote by itself.

On first start the plugin installs the whole tool library (Python packages, pretrained weights, upstream sources; about 19 GB for the
full profile) in the background and the engine refuses tasks until that is done. `autoSetup: false` turns the automatic start off
(run `python -m scienceclaw.cli setup` yourself), `setupProfile: "light"` skips the assets larger than 1.5 GB.

This plugin is the agent system only. The ScienceClaw-Eval benchmark is released separately
(evaluation data: <https://huggingface.co/datasets/beita6969/scienceclaw-64-samples>) and is not
reachable through these tools.
