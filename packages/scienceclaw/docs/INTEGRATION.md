# ScienceClaw integration map

This package is the ScienceClaw engine - typed workflow orchestration, the scientific tool library and
the verified self-evolution of the Skill/Operator program - embedded in the original ScienceClaw
gateway. The integration keeps the two execution layers connected through a small, typed boundary:

| Layer | Location | Responsibility |
| --- | --- | --- |
| Gateway and long-lived agent | repository root (`src/`, `extensions/`, `skills/`) | OpenClaw routing, provider/session handling, memory, and the original skill/plugin system |
| Engine | `packages/scienceclaw/scienceclaw/` | typed graphs, execution, replay, receipts, the program store and evolution candidates |
| Scientific operators | `packages/scienceclaw/scilib/` | the scientific tool library and optional pretrained wrappers |
| Canvas skill | `skills/scienceclaw-canvas/` | how the gateway agent orchestrates a task on the canvas |
| Evolution skill | `skills/scienceclaw-evolution/` | candidate replay, validation gate, provenance, and promotion policy |
| Canvas, tools, program | `packages/scienceclaw/scienceclaw/{canvas,tools,program}/`, `evolution/live.py` | stepwise typed-workflow sessions, the scilib/weights registry, the versioned Skill/Operator store, gated evolution from finished sessions |
| Native plugin | `extensions/scienceclaw/` | `scienceclaw_canvas`, `scienceclaw_tools`, `scienceclaw_program`, `scienceclaw_evolve` over `python -m scienceclaw.rpc` |

The engine process does not receive provider credentials, and live task inputs are read only from the
configured input roots. The plugin exposes interactive canvas sessions, tool-library inspection, program
inspection and rollback, and gated evolution. A program change is learned from a finished session: a
candidate requires replay, validation on tasks the user registered, hard-constraint checks,
reproducibility, provenance and the configured budget gate, and an admitted candidate waits as `ready`
until the user promotes it. Model weights, caches, logs, virtual environments and credentials are
deployment-local and are excluded from the code checkout. Before adding a new operator or skill, keep its
source and routing metadata in the package or root skill directory, then record its provenance in the
candidate's bundle.

## Engine interface

The plugin starts one long-lived `python -m scienceclaw.rpc` process (line-delimited JSON-RPC), so canvas
sessions keep their graphs, checkpoints and evidence between tool calls. The methods are:

* `ping`, `setup.status`, `setup.start`: liveness and the tool-library installation;
* `canvas.open|act|render|replay|finish|status|list`: typed workflow sessions for a live `task` declaration;
* `tools.search|show|status`, `weights.status|plan`: the tool library and its pretrained weights;
* `program.summary|skills|operators|show|history|rollback`: the versioned Skill/Operator store;
* `evolve.val_add|val_list|val_remove|propose|gate|run|status|candidates|candidate`: evolution from finished
  sessions (`propose`, `gate` and `run` are background jobs polled with `evolve.status`).

The command line (`python -m scienceclaw.cli`, or `scienceclaw` once installed) offers `tools`
(search, show, status), `weights` (status, plan, verify), `setup`, `doctor` and `live`
(candidates, show, promote, rollback, history).

## Deployment roots and optional stacks

Install the lightweight engine with `pip install -e packages/scienceclaw`.
On a host that will run domain tools, add one or more optional stacks, for
example `pip install -e 'packages/scienceclaw[vision,audio,nlp]'` or
`[all]` on a prepared GPU image. The extras are dependency groups only; model
weights remain deployment-local and are never pulled into this repository.

Point the runtime at the externally managed weight root:

```bash
export SCIENCECLAW_MODELS=/srv/scienceclaw/models
```

`SCIENCECLAW_MODELS` is consumed by the pretrained wrappers and by `scienceclaw.cli setup`. The
state directory (`SCIENCECLAW_HOME`, default `~/.scienceclaw`) holds canvas sessions, the program store and
the evolution candidates; `SCIENCECLAW_INPUT_ROOTS` lists the directories live tasks may read.
`SCIENCECLAW_CONFIG` optionally points to a run configuration (see `configs/default.yaml`).

## Quick orientation

1. Load `skills/scienceclaw-canvas/SKILL.md` to orchestrate a task on the canvas
   (`scienceclaw_canvas`, `scienceclaw_tools`).
2. Use `scienceclaw_program` to inspect or roll back the active Skill/Operator version and
   `scienceclaw_evolve` to learn from a verified session through the validation gate; load
   `skills/scienceclaw-evolution/SKILL.md` for the procedure.
3. The user promotes an admitted candidate with `scienceclaw live promote <id>`.

## First install: every tool is set up before work starts

`python -m scienceclaw.cli setup` installs the `[all]` extra, stages every pretrained asset of `tools/weights.json` (plus the
pinned upstream sources some wrappers need) under `SCIENCECLAW_MODELS`, verifies them and records the result in
`$SCIENCECLAW_HOME/setup.json`. The full profile downloads about 19 GB and checks the free space first; `--profile light` leaves out
assets larger than 1.5 GB, `--only/--skip` select assets, `--check` only reports. `python -m scienceclaw.cli doctor` shows the setup
state and why any module cannot run. The root `setup.sh` runs it as part of the installation.

Until the setup is complete the engine refuses `canvas.open` and the evolution methods. The plugin installs
automatically when it first starts (`autoSetup`, on by default; `setupProfile` selects the profile);
`scienceclaw_tools(operation=setup)` shows the progress. `SCIENCECLAW_SKIP_SETUP_CHECK=1` disables the gate for development.

## Models, tools and safety switches

* **Language model.** `llm.backend` is `openai` (an OpenAI-compatible endpoint), `command` (a local program that reads the
  prompt on stdin and prints the completion; `llm.command` or `SCIENCECLAW_LLM_COMMAND`, with `{system}` and `{model}`
  placeholders) or `package.module:factory`. Model names and credentials come from the environment.
* **Tool library.** `python -m scienceclaw.cli tools status` shows which of the 47 scilib modules run here and why not;
  `python -m scienceclaw.cli weights status|plan|verify` stages the pretrained checkpoints and the upstream sources some
  wrappers need. The typed operators in `scienceclaw/program/specs/` wrap the main entry points; `program.check.check_operator`
  runs one through the real canvas on concrete inputs.
* **Code-node sandbox.** Workers run with an allow-listed environment, a runtime audit guard (protected data and state
  locations, sockets, programs other than the interpreter, native libraries, the engine's sources) and, where `unshare`
  works, in new user, network and pid namespaces (`SCIENCECLAW_SANDBOX_ISOLATION=auto|off|require`). Run untrusted
  workloads in a container as well.
* **Program changes learned from live sessions** wait as `ready` until the user promotes them:
  `python -m scienceclaw.cli live candidates|show|promote|rollback|history` (`autoPromote` in the plugin configuration
  makes the gate promote by itself).

## Companion benchmark

The ScienceClaw-Eval benchmark (23 disciplines, FoR30-FoR52; sequential task streams and independent reset
evaluation) from the paper "ScienceClaw: Benchmarking Continual Self-Evolution of AI-for-Science Agents Across
the Natural and Social Sciences" is released separately, and its evaluation data is hosted on Hugging Face:
<https://huggingface.co/datasets/beita6969/scienceclaw-64-samples>. This package contains the agent system
only: no evaluation harness, datasets or tests.
