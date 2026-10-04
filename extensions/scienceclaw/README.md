# ScienceClaw gateway plugin

Registers the ScienceClaw engine (`packages/scienceclaw`) as optional agent tools. The
gateway agent acts as the policy of the engine: it edits a typed workflow graph step by step
(`scienceclaw_canvas`), finds scientific tools and pretrained-model wrappers
(`scienceclaw_tools`), and inspects or rolls back the versioned Skill/Operator program
(`scienceclaw_program`). `scienceclaw_eval` inspects the ScienceClaw-Eval catalog, the installed
task data and finished-run reports.

| Tool | Operations |
| --- | --- |
| `scienceclaw_canvas` | `open`, `act`, `render`, `replay`, `finish`, `status`, `list` |
| `scienceclaw_tools` | `search`, `show`, `status`, `weights` |
| `scienceclaw_program` | `summary`, `skills`, `operators`, `show`, `history`, `rollback` |
| `scienceclaw_eval` | `catalog`, `list_tasks`, `report` |

The plugin keeps one long-lived Python process (`python -m scienceclaw.rpc`, line-delimited
JSON) so canvas sessions survive between tool calls. Deployment settings (`packageRoot`,
`pythonBin`, `home`, `inputRoots`, `runRoot`, `dataRoot`, `modelRoot`, `configPath`, `llm`) are
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
      tools: { allow: ["scienceclaw_canvas", "scienceclaw_tools", "scienceclaw_program", "scienceclaw_eval"] },
    }],
  },
}
```

Formal hidden-split evaluation is not reachable through these tools.
