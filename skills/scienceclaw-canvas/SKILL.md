---
name: scienceclaw-canvas
description: "Solve a data-analysis or scientific-computing request as a typed, executable workflow graph on the ScienceClaw canvas, using the scientific tool library and the evolved Skill/Operator program. Use for any task that has input files and a checkable deliverable; answer plain questions directly."
metadata: { "openclaw": { "emoji": "🧭" } }
---

# ScienceClaw canvas orchestration

The canvas is the working surface for scientific tasks. Instead of running ad-hoc
scripts, build one typed workflow graph: nodes are tools, library operators,
code or `llm` steps; edges carry typed values (type, shape, unit). The engine
executes every edit immediately and reruns only the affected nodes, so each step
returns real feedback.

## Procedure

1. **Declare the task.** Call `scienceclaw_canvas(operation=open, task={...})`:
   - `objective`: what must be computed and delivered.
   - `inputs`: `[{name, path, format?, description?}]` — each becomes a read-only
     `load_<name>` tool. Paths must lie under the configured input roots.
   - `required_output`: `{type, shape?, unit?}` of the deliverable.
   - `constraints`: hard, checkable conditions, each `{check, value?}` with
     `check` one of `finite`, `shape`, `type`, `range`, `nonempty`, `len_eq_input`, or `metric`
   (`{target, column?, metric, direction, value}`: an evaluator-only quality bar against a
   held-out file that the workflow never loads; the metric is `mae|mse|rmse|smape|r2|accuracy|f1_macro|auc`).
   Declare only constraints that the user actually requires; they are the
   acceptance test.
2. **Read the returned context.** It lists the protocol, the actions, the task's
   tools, the retrieved skills (routing recipes) and typed operators from the
   active program. Follow a retrieved skill when it fits; it is evidence, not a
   rule.
3. **Find tools.** `scienceclaw_tools(operation=search, query=...)` ranks the
   scientific library (classical toolkits and wrappers of pretrained models).
   `operation=show` prints the documentation and whether the module can run on
   this host; do not plan around a tool reported as unavailable.
4. **Edit one step at a time.** `scienceclaw_canvas(operation=act, sessionId,
   action={...})` applies a single atomic edit (add/modify/remove node or edge).
   Read the execution feedback before the next edit; fix the failing node
   instead of rewriting the graph.
5. **Verify.** When the submit node is wired, call `operation=replay`. The
   workflow is re-executed from a reset state and must reproduce its output and
   satisfy every constraint. Then `operation=finish` returns the verified
   deliverable and the workflow.
6. Report the deliverable together with what the workflow did, the tools used and
   any constraint that failed. Never present an unverified result as final.

## Rules

- Inputs are read through `load_<name>` only; never read files outside the input
  roots or invent data that is not in the task.
- Prefer a library tool or typed operator over reimplementing the method; pretrained
  wrappers need staged weights (`scienceclaw_tools(operation=weights)`).
- Never ask for, print or store credentials. The engine process does not receive
  gateway or provider keys.
- Use `scienceclaw_program` to see which Skills and Operators are active. When a verified
  session contained a real repair (a failed replay fixed by an edit), offer to learn from it
  with `scienceclaw_evolve`; see `scienceclaw-evolution` for the gate and rollback.
