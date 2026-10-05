# ScienceClaw engine — design contract

This document is the design contract of the ScienceClaw engine (`packages/scienceclaw`): typed workflow
(canvas) orchestration of scientific tasks, the scientific tool library `scilib`, the versioned
Skill/Operator program and its replay-verified self-evolution, the code-node sandbox, and the line-delimited
JSON-RPC service used by the OpenClaw gateway plugin (`extensions/scienceclaw`). Every module implements the
interfaces below. Paper references are to "ScienceClaw: Benchmarking Continual Self-Evolution of AI-for-Science
Agents Across the Natural and Social Sciences" (KDD'27 submission #358), which defines the method (Eq. 1–13) this
engine implements.

This repository is the agent system only. The companion benchmark, ScienceClaw-Eval (23 disciplines,
FoR30–FoR52), is released separately; its evaluation data is hosted on Hugging Face
(<https://huggingface.co/datasets/beita6969/scienceclaw-64-samples>). The engine package contains no evaluation
harness, datasets or tests.

---------------------------------------------------------------------------
## 0. Map from paper to code

| Paper object | Code |
|---|---|
| Task D_t = (D_T, D_V, D_E) | `task.Episode` (objective/required output = D_T; constraints + evaluator + acceptance = D_V; tools + data + budget = D_E); a live task declaration becomes an `Episode` through `canvas.live.build_live_episode` |
| Editable program A_r = (S_r, O_r) | `core.program.AgentProgram` (`skills: dict[str, Skill]`, `operators: dict[str, OperatorSpec]`), stored versioned by `program.store.ProgramStore` |
| Θ0 fixed foundation model | `llm.client.LLMClient` role `policy` (same model also used for `patch` and, by default, `executor`) — never trained |
| Eq.1 Solve_Θ0(D_t \| A_r) → Z_t = (G*_t, y_t, τ_t) | `agent.solver.Solver.solve(episode, program, mode)` → `SolveResult`; in the gateway the same loop is driven from outside by `canvas.session.CanvasSession` |
| Eq.5 typed workflow graph, Γ = (type, shape, unit, provenance), Compat | `core.schema.PortSchema`, `core.schema.compat`, `core.graph.WorkflowGraph` |
| Eq.6 policy π_Θ0 samples (a_{t,k}, ν_{t,k}) | `agent.policy.Policy.propose(...)` → `core.actions.Action` (has `uses` = ν); in the gateway the acting agent is the policy |
| Retrieve(D_t; S_r), Retrieve(D_t; O_r) | `core.retrieval.Retriever` (BM25 + tag match, top-k) |
| Eq.7 Execute_{D_E}(G, χ, a) → (G', χ', f) | `runtime.executor.Executor.apply(graph, checkpoint, action)` |
| Eq.8 Replay + Eval → e_{t,k} = (D_t, G, y, τ, q, h, c), Pass | `runtime.replay.replay(graph, episode, ...)`, `episode.evaluate(y, trace)` → `EvalResult`; `task.passes(...)` |
| Eq.9 evolution instance, e⁻, e⁺, δ_i | `evolution.attribution.extract_instances(trajectory)` → `EvolutionInstance` |
| Eq.10 Π_ctrl, Π_exec, CC_Γ | `evolution.split.split_edits(instance, program)` → `(control_edits, exec_components)` |
| Eq.11 Patch_Θ0 Skill candidate | `evolution.skill_patch.make_skill_candidates(...)` |
| Eq.12 Operator candidate ô = (G⁺[U], Γ_∂U, κ, ρ) + BReplay | `evolution.operator_abstraction.make_operator_candidate(...)`, `boundary_replay(...)` |
| Bundle B_i, Apply, version ids ω | `core.program.Bundle`, `evolution.bundle.build_bundle`, `AgentProgram.apply(bundle)` |
| Eq.13 R_src = Pass ∧ Use | `evolution.validation.source_replay_check(...)` |
| Eq.2 feasibility (R_src, H_val, C_val ⪯ B) + argmax Q_val | `evolution.validation.ValidationGate` |
| Eq.3 strict-improvement update, Θ fixed | `evolution.evolver.Evolver` (batch stream) and `evolution.live.LiveEvolution` (gateway) |
| Eq.4 MacroSR, z_i | `EvalResult.z`; MacroSR is Q_val over the validation tasks (`evolution.validation`) |
| D_src, D_val | the source stream of an `Evolver` plan; in the gateway, finished live sessions (D_src) and the validation tasks the user registered (D_val) |

---------------------------------------------------------------------------
## 1. Decisions the paper leaves open (fixed here, all configurable)

1. **Round / update schedule.** The batch `Evolver` walks the source stream of its plan
   (`plan.source_stream()`: `(round, episode)` pairs, round-major, in a fixed order). Default
   `update_schedule = "per_candidate"`: every candidate that passes R_src is
   validated immediately against the *current incumbent* and accepted iff
   Eq.2–3 hold; the next source episode already uses the updated program.
   Snapshots A_0..A_R are taken at round ends. Alternative
   `"per_round_argmax"` pools candidates of a round and applies one argmax
   (literal Eq.2–3 reading). In the gateway the source stream is the user's own work: each finished,
   replay-verified live session is a source episode and its candidates are gated one at a time (§7).
2. **Q_val.** Paper: MacroSR on D_val. Default `qval = "macrosr_then_score"`:
   lexicographic (MacroSR, mean normalized score on D_val); strict improvement
   means MacroSR ↑, or MacroSR equal and normalized score ↑ by ≥ `qval_eps`.
   `qval = "macrosr"` gives the literal paper rule.
3. **H_val.** Vector of non-negotiable integrity checks over D_val episodes:
   (a) no sandbox/leakage violation, (b) submit output schema valid,
   (c) no *new* hard-constraint violation relative to the incumbent
   (`hval_mode = "no_regression"`, default) or all hard constraints hold on
   every val episode (`hval_mode = "absolute"`). "Every val episode" means
   every *completed* one: an episode that produced no output (z = 0) is
   already penalised by MacroSR and has no output to violate a constraint with,
   so it cannot veto a candidate through H_val (otherwise a program could not
   be admitted while any val episode stays unsolved). A solver / gateway error
   entry is different: it blocks `no_solver_error` (decision 13).
4. **Pass.** `Pass(e) = completed ∧ all(h) ∧ reproducible ∧ within_budget ∧
   (acceptance if pass_requires_acceptance)`. Default
   `pass_requires_acceptance = True` (a replay-verified "success" is a solved
   episode in the z sense). `reproducible` = clean replay output equals the
   executor output within the episode tolerance.
5. **Feedback visibility.** The policy NEVER sees hidden-evaluator scores, in any
   mode. It sees: action validation errors, node execution status/errors,
   output summaries (type/shape/unit/finite/range), constraint-check results,
   and the *visible dev score* (a task-provided score computed on visible data,
   shown unless `show_dev_score` is off). Hidden evaluator results (labels,
   evaluator-only constraints) are used only by the evolution machinery
   (Pass/z).
6. **Replay schedule.** A clean replay (Eq.8) is run whenever an executed graph
   produces a submit output whose fingerprint differs from the last replayed
   one. Replay results give the evidence sequence e_{t,k}.
7. **Final solution G*.** In `source` mode: the replay-verified passing graph
   with the best visible dev score (ties: the later step), otherwise the graph
   of the latest step whose submit output was computed by a valid, fully wired
   workflow (an edit that leaves y uncomputed does not replace it), otherwise
   failure (y = None, z = 0). In `val` / `eval` mode only the latter rule
   applies (no per-step hidden verification). The policy prompt states this
   rule verbatim (`prompts._FINISH`), so what the policy is told about
   `finish` / budget exhaustion is what the solver does.
8. **Validation cost control.** Val episodes are re-solved for a candidate only
   if the candidate changes what that episode retrieves (retrieval-visible
   program slice hash); otherwise the incumbent's cached result is reused. The
   LLM response cache makes identical contexts produce identical outputs.
   Retrieval (BM25 + tag/applicability bonus) applies a **relevance floor**: a
   component is retrievable only if it matches the episode by metadata
   (discipline, task type or an episode tag among its tags / applicability) OR
   by a strong lexical match (BM25 >= `min_lexical_frac` = 0.15 of the query's
   idf mass). This deliberately replaces "applicability is only a bonus, never a
   filter": with a bare positive-BM25 test one shared token put every
   component into every episode's slice, so every candidate changed the slice
   of every val episode and lazy re-validation never fired (the D_val cost grew
   with the library instead of with the candidate). Cross-discipline transfer
   remains possible through task-type / tag / lexical relevance. The evolver
   records `val_resolved` (episodes re-solved) and `val_reused` per candidate in
   `candidates.jsonl` (and `ValReport.n_resolved / n_reused`).
   `lazy_revalidation: false` re-solves every val episode.
9. **Initial library.** `AgentProgram()` starts empty; D_E tools are always available. The gateway's
   seed program A0 (`program.seed.seed_program`) holds the `scienceclaw-*` Skills parsed from the `skills/`
   directory plus the typed library operators of `program/library_ops.py` (provenance `source == "library"`).
   The system prompt contains only (A) what each action/tool/node kind does, (B) the deliverable format, (C)
   interface rules. It never contains how-to procedures or task strategies — those come from Skills and
   learned Operators.
10. **Gateway gate.** `evolution.live.LIVE_GATE` applies Eq.2–3 as written: `hval_mode = "absolute"` (every
    hard constraint holds on every validation task) and `qval = "macrosr"`. A candidate needs at least
    `max(min_val_tasks (2), min_improved_episodes)` registered validation tasks; D_src and D_val are
    disjoint (the task key hashes the objective and the resolved input files, and the gate refuses a
    validation task whose files changed since registration); a candidate derived from an older program is
    refused when a component it changes has moved on. An admitted candidate is `ready`; promotion is the
    user's decision, unless the plugin enables `autoPromote`.
11. **Budget B.** The paper's `C_val ⪯ B` is made concrete as two conditions
    that must BOTH hold on the summed D_val cost: (a) an absolute cap that
    scales with the validation set, `B_abs = budget_tokens_per_val_episode ×
    |D_val|` (default 250 000 tokens per val episode; an explicit
    `budget_tokens > 0` overrides it), and (b) a cap relative to the incumbent,
    `C_val(cand) ≤ (1 + budget_beta) · C_val(inc)` (`budget_beta` default 0.5;
    negative disables it; skipped when the incumbent cost is 0). Wall time is
    checked against `budget_wall_s` or `budget_wall_s_per_val_episode ×
    |D_val|` (default 1800 s). Tokens are LOGICAL (spent + cached, see
    `agent.policy.logical_tokens`), so a run served from the response cache
    is judged exactly like the original run. The feasibility reasons record
    `within_budget_abs / _rel / _wall` and the violated ones (`budget_violated`).
12. **Noise guard on Eq. 3 (deliberate deviation, switchable).** Eq. 3 admits a
    candidate on any strict Q_val improvement. With one stochastic draw per val
    episode and a small D_val a single lucky episode is enough, so the incumbent drifts on
    noise. `evolution.min_improved_episodes` (default 2) additionally requires
    the Q_val gain to be supported by at least that many D_val episodes that
    *individually* improved (z 0 → 1, or the same z with a normalized-score
    gain ≥ `qval_eps`), and at most `max_regressed_episodes` (default 1; -1 =
    no cap) that regressed. No extra LLM draws are used: it is a comparison
    of the same per-episode entries. **`min_improved_episodes: 1` restores the
    literal rule** (and disables the regression cap). A val set smaller than
    the threshold can then never admit a candidate: set it to 1 for
    single-episode validation sets.
13. **Gateway outages are not task failures.** A solve cut short by an LLM /
    gateway outage (`SolveResult.infra_error`: the policy call raised —
    `stop_reason = "policy_error"` — or the final trace holds an executor
    error "LLM call failed" / "all N LLM calls failed") must not be scored as
    z = 0. `ValidationGate` retries such a val solve in place (`infra_retries`
    = 2, exponential backoff from `infra_backoff_s` = 30 s); if it still fails
    the entry becomes an *error entry* (z = 0, `error` set) that fails
    `H_val["no_solver_error"]`, is never reused by lazy re-validation, and is
    re-solved by the evolver's incumbent refresh; a candidate is not gated
    against an incumbent whose entries are still errored. A source solve
    is retried the same way; if it still fails the source episode is
    re-queued to the end of its round (one deferral; the last chance commits
    regardless) and is recorded in `stream.jsonl` with `infra_error` /
    `requeue`. A source-replay outage rejects the candidate with
    `infra_error` in its reasons (recorded, not silently z = 0).

---------------------------------------------------------------------------
## 2. Package layout

```
scienceclaw/
  config.py                 RunConfig and sub-configs (LLMConfig, SolverConfig, EvolutionConfig; YAML load/dump)
  task.py                   Episode, ToolSpec, ConstraintSpec, Budget, EvalResult, passes (the neutral task abstraction)
  bootstrap.py              first-install setup (Python packages, pretrained weights, upstream sources) and its gate
  core/
    schema.py               PortSchema, compat, UnitRegistry, summarize_value
    graph.py                Node, Edge, WorkflowGraph
    actions.py              Action, parse_action, apply_action (the atomic canvas edits)
    skills.py               Skill
    operators.py            OperatorSpec, Contract
    program.py              AgentProgram (+ save/load, apply), Bundle
    retrieval.py            Retriever (BM25 + tags)
    trace.py                NodeRecord, Trace, Evidence
  llm/
    client.py               LLMClient (OpenAI-compatible, cache, retries, accounting)
    interface.py            ChatModel protocol, register_backend / build_chat_model
    command.py              backend that runs a local program on the prompt
  runtime/
    sandbox.py              run_code_node(...) in an isolated subprocess worker
    node_worker.py          subprocess entry point
    executor.py             Executor (Eq.7), Checkpoint
    replay.py               replay (Eq.8)
    integrity.py            static scan of generated code
    values.py               value (de)serialisation and output comparison
  agent/
    prompts.py              system/user prompt builders (A/B/C only)
    policy.py               Policy (Eq.6)
    solver.py               Solver (Eq.1 via Eq.6–8), Trajectory, StepRecord, SolveResult
  canvas/
    live.py                 live task declarations -> Episode (input tools, constraints, input-root checks)
    session.py              CanvasSession: one atomic edit per call, replay, finish
  evolution/
    attribution.py          Eq.9
    split.py                Eq.10
    skill_patch.py          Eq.11
    operator_abstraction.py Eq.12 + boundary replay
    bundle.py               build_bundle
    validation.py           Eq.13 + Eq.2 gate
    evolver.py              batch stream loop + Eq.3
    live.py                 LiveEvolution: propose / gate / promote from finished live sessions
    variants.py             variants (frozen, workflow_only, skill_only, operator_only, unlinked, full)
  program/
    store.py                ProgramStore: versioned snapshots, HEAD, receipts, rollback
    seed.py                 seed program A0 (Skills from skills/ + library operators)
    library_ops.py          typed operators wrapping the main scilib entry points (specs/)
    check.py                check_operator: run one operator through the real canvas
  tools/
    registry.py             static catalog of scilib tools, BM25 search, availability probes
    weights.py, weights.json  registry of pretrained weights and upstream sources
  rpc.py                    line-delimited JSON-RPC service used by the gateway plugin
  cli.py                    `python -m scienceclaw.cli ...` (tools, weights, setup, doctor, live)
scilib/                     the scientific tool library (classical toolkits and pretrained-model wrappers)
configs/default.yaml        RunConfig defaults made explicit
docs/
```

---------------------------------------------------------------------------
## 3. Core data model (implemented in core/*, task.py — do not change
signatures without updating this file)

### 3.1 PortSchema Γ (core/schema.py)
```python
@dataclass(frozen=True)
class PortSchema:
    type: str                      # "any","number","text","array","table","list","dict","series"
    shape: tuple | None = None     # e.g. ("n",) or ("n", 12); ints or symbolic str; None = unspecified
    unit: str | None = None        # e.g. "K", "mg/L", "1" (dimensionless); None = not applicable
    dtype: str | None = None       # optional element dtype, e.g. "float", "int", "str", "prob"
    description: str = ""
    provenance: tuple[str, ...] = ()   # filled at runtime: upstream data ids and transforms
def compat(out: PortSchema, inp: PortSchema, conversion: dict | None = None) -> tuple[bool, str]
```
Rules: types equal or either is "any"; shapes unify (int vs same int, symbol
binds consistently, None matches anything); units equal, or either None, or an
explicit `conversion={"from": u1, "to": u2, "factor": a, "offset": b}` is on
the edge (recorded into provenance). Mismatch → (False, reason).

### 3.2 Graph (core/graph.py)
```python
NODE_KINDS = ("tool", "operator", "code", "llm", "submit")
@dataclass
class Node:
    id: str
    kind: str
    ref: str | None = None          # tool name (kind=tool) or operator id (kind=operator)
    code: str | None = None         # kind=code: python source defining `def run(inputs: dict, config: dict) -> dict`
    prompt: str | None = None       # kind=llm: template with {field} placeholders over each item
    config: dict = field(default_factory=dict)       # η
    inputs: dict[str, PortSchema] = ...
    outputs: dict[str, PortSchema] = ...
    origin: dict = ...              # {"step": k, "uses": [...], "generated": bool, "from_operator": id|None}
@dataclass(frozen=True)
class Edge:
    src: str; src_port: str; dst: str; dst_port: str
    conversion: dict | None = None
class WorkflowGraph:
    nodes: dict[str, Node]; edges: list[Edge]
    def copy(self) -> "WorkflowGraph"
    def validate(self) -> list[str]                  # [] if valid: DAG, ports exist, compat on every edge, ≤1 submit
    def topo_order(self) -> list[str]
    def predecessors(self, nid) -> list[str]; successors(self, nid) -> list[str]
    def descendants(self, nids) -> set[str]; ancestors(self, nids) -> set[str]
    def in_edges(self, nid) -> list[Edge]; out_edges(self, nid) -> list[Edge]
    def fingerprint(self, nid) -> str                # sha256 of node spec + upstream fingerprints + edge wiring
    def submit_node(self) -> str | None
    def boundary(self, U: set[str]) -> tuple[list[Edge], list[Edge]]   # (incoming edges into U, outgoing edges from U)
    def subgraph(self, U: set[str]) -> "WorkflowGraph"
    def to_dict(self) -> dict; @classmethod from_dict(d) -> "WorkflowGraph"
    def render_compact(self) -> str                  # text shown to the policy
```
The submit node has one input port `y` whose schema is the episode's
`required_output`; its output is the scientific output y_t.

`config` (η) is used by `tool` nodes (the tool options in its signature), `code` nodes (the second argument of
`run`) and `llm` nodes (`parse`, `choices`, `max_tokens`). **`operator` nodes take no config**: an operator's
behaviour is fixed by its body (versioned, learned), `add_node` / `modify_node` with a non-empty config on an
operator node is rejected by `apply_action`, and the prompts do not advertise one. `render_compact()` shows code
nodes in full (all code bodies together capped at ~12k characters; a body over its share is cut at a line boundary
and ends with a `# [cut for display: N more lines (M chars) ... the stored code is complete]` marker) and lists the
description of every port under the signature line.

### 3.3 Skills, Operators, Program
```python
@dataclass
class Skill:
    id: str; version: int; title: str; body: str
    tags: list[str]; provenance: dict; created: str   # created = "r{round}:e{episode}"
@dataclass
class Contract:        # κ
    pre: list[dict]       # e.g. {"port": "x", "check": "finite"} / {"port":"x","check":"shape","value":[...]} / {"port":"x","check":"unit","value":"K"}
    post: list[dict]
    applicability: dict   # {"disciplines": [...], "task_types": [...], "input_types": [...]}
@dataclass
class OperatorSpec:
    id: str; version: int; name: str; description: str
    body: WorkflowGraph                      # G⁺[U] (internal nodes: code/llm/tool/operator)
    inputs: dict[str, PortSchema]            # Γ over ∂⁻U
    outputs: dict[str, PortSchema]           # Γ over ∂⁺U
    input_map: dict[str, list[tuple[str, str]]]   # boundary input -> internal (node, port) targets
    output_map: dict[str, tuple[str, str]]        # boundary output -> internal (node, port) source
    contract: Contract                       # κ
    provenance: dict                         # ρ
    tags: list[str]
class AgentProgram:
    skills: dict[str, Skill]; operators: dict[str, OperatorSpec]
    version: str; parent: str | None; lineage: list[dict]
    def apply(self, bundle) -> "AgentProgram"       # new object, assigns ω (version ids) to every new/changed component
    def save(self, path); @classmethod load(path)
    def fingerprint(self) -> str
```
Contract checks (`core/operators.py: check_contract_entries(entries, values, schemas, units=None)`) run on the
*delivered values* of the boundary ports: `type` tests the value with `core.schema.value_has_type` (number: int/float/
0-d array; text: str; table: DataFrame; dict; list: list/tuple; array: ndarray, or list/tuple/Series; series: Series,
or ndarray/list; `any`: everything), `unit` compares the unit that actually arrives at the port (derived from the
incoming edge; `None` = unspecified upstream, compatible with every unit as in `schema.compat`) with the required one.
Neither check compares the declared schema with itself. Violations are diagnostics, not crashes.

### 3.4 Task / episode (task.py)
```python
@dataclass
class Budget:
    max_steps: int = 12; max_policy_tokens: int = 200_000; max_wall_s: float = 1800
    max_node_s: float = 300; max_llm_items: int = 256
@dataclass
class ToolSpec:
    name: str; description: str
    inputs: dict[str, PortSchema]; outputs: dict[str, PortSchema]
    fn: Callable[[dict, dict], dict]          # (inputs, config) -> outputs; runs in the parent (trusted)
@dataclass
class ConstraintSpec:
    name: str; description: str
    check: Callable[[Any, "Trace"], tuple[bool, str]]   # on final output y (+ trace)
    visible: bool = True        # visible checks are also reported to the policy
    grade: Callable[[Any], float] | None = None          # optional graded share in [0, 1] of the criterion
@dataclass
class EvalResult:
    metrics: dict[str, float]; primary: float | None; direction: str   # "max"|"min"
    h: dict[str, bool]; h_msgs: dict[str, str]
    cost: dict[str, float]; accepted: bool; z: int
    completed: bool; reproducible: bool | None; within_budget: bool
    details: dict
@dataclass
class Episode:                                  # selected fields
    id: str; discipline: str; task_type: str; tags: list[str]
    objective: str                          # D_T: objective, inputs, initial conditions, target outputs
    required_output: PortSchema             # schema of y
    tools: list[ToolSpec]                   # D_E
    constraints: list[ConstraintSpec]       # D_V hard constraints
    budget: Budget
    lineage: dict                           # provenance of the task's data
    acceptance: str                         # human-readable acceptance rule
    tolerance: dict                         # {"rtol":..., "atol":...} for reproducibility / boundary replay
    _evaluate: Callable[[Any, "Trace"], EvalResult]     # evaluator (never exposed to policy/sandbox)
    _dev_evaluate: Callable[[Any], dict] | None          # visible dev scoring on visible data
    def evaluate(self, y, trace) -> EvalResult
    def public_view(self) -> dict                        # what the policy may see: no evaluator, no hidden data
def passes(ev: EvalResult, require_acceptance: bool = True) -> bool    # Pass: completed, hard constraints, within budget,
                                                                       # reproducible (and accepted, if required)
```
Hidden data live only in the evaluator closure (parent process). Visible data is materialized by tools
into the run dir; code nodes may only read their inputs (the integrity scan flags absolute paths, `..`,
protected roots, network). `evaluate(None | malformed y)` returns z = 0 with the failure recorded in
`details`; `EvalResult.hard_ok()` is true when every entry of `h` holds.

### 3.5 Trace τ and evidence e (core/trace.py)
```python
@dataclass
class NodeRecord:
    node_id: str; fingerprint: str; status: str       # "ok"|"error"|"skipped"|"pending"
    wall_s: float; error: str | None; stdout_tail: str
    outputs_summary: dict[str, dict]; contract_violations: list[str]
    input_refs: dict[str, str]; output_refs: dict[str, str]   # paths of pickled values (for boundary replay)
    llm_usage: dict; kind: str; cached: bool
@dataclass
class Trace:
    records: dict[str, NodeRecord]; order: list[str]; llm_usage: dict; wall_s: float; run_dir: str
@dataclass
class Evidence:          # e_{t,k}
    step: int; graph_dict: dict; y: Any; trace: Trace; eval: EvalResult; passed: bool; graph_fp: str
```

---------------------------------------------------------------------------
## 4. Runtime (Eq.7–8)

* Every node runs with a timeout. `code` nodes run in a subprocess
  (`runtime/node_worker.py`) with cwd = episode run dir, inputs/outputs
  exchanged as pickles, stdout/stderr captured. `tool` nodes run in the parent
  (trusted tool code). `llm` nodes run in the parent through `LLMClient`
  (role `executor`), mapping the prompt template over `inputs["items"]` (list of
  dicts or strings), returning `{"outputs": [parsed per item]}`; `config`
  keys: `parse` ("text"|"json"|"number"|"choice"), `choices`, `max_tokens`.
  `operator` nodes expand their `body` inline (recursively) with boundary maps
  and check κ pre/post conditions (violations → diagnostics, not crashes).
* `Checkpoint χ` maps node fingerprint → NodeRecord (+ pickled outputs). After an
  edit, only nodes whose fingerprint is not in χ are (re)executed, in topo order;
  nodes whose inputs are unavailable are `pending`.
* Diagnostics f: per-node status/error/stdout tail, output summaries
  (`summarize_value`: type, shape, dtype, unit, finite fraction, min/max/mean,
  head), contract violations, visible constraint checks on the submit output,
  visible dev score (if the episode provides `_dev_evaluate`).
* `replay(graph, episode)` executes the full graph in a fresh run dir with an
  empty checkpoint → (y, trace). Reproducibility = replay y ≈ executor y within
  `episode.tolerance`.

Hardening rules (runtime/executor.py):

* **Transient nodes taint their descendants.** Fingerprints hash the node *spec* and its upstream specs, not
  values. A node whose outcome is not reproducible (partial LLM failure, API error, first timeout) is `transient`:
  neither it nor any descendant enters the checkpoint, so the next edit recomputes them instead of serving a
  value derived from a failed run.
* **Timeouts.** A node gets `max_node_s` (a node's `config["timeout_s"]` may only lower it); tool and llm calls
  run in a daemon thread that is abandoned on timeout. The first timeout of a fingerprint is transient (machine
  load, retried on the next execution); a repeat of the same spec is cached as an ordinary error. An `operator`
  node as a whole has `2 × max_node_s`, and nodes inside it stop at that deadline. Tool calls run under one tool
  lock (tool code need not be thread-safe); an abandoned tool call keeps the lock until it really ends, and a
  new tool call is then refused at once with a transient error instead of running concurrently.
* **llm nodes.** Placeholders are checked before any request is sent: attribute / index access is an error;
  when more than one item is mapped and the template uses neither `{item}` nor a key of the items, the node
  fails (all requests would be identical); other unresolved names are reported in the output summary
  (`unresolved_placeholders`). The rendered prompt of one request is capped (32 000 characters unless the
  Budget defines `max_llm_prompt_chars`; the error text says to pass fewer fields) and `max_tokens` is clamped to
  the executor role's configured limit (`max_tokens_clamped_to` in the summary). `parse = "number"` returns the
  one finite number in the answer (thousands separators and list numbering are handled; the same number repeated
  counts once) and None for an answer with no number, with several distinct numbers, or with a percentage
  (80 % is ambiguous) — a parse failure, counted in `parse_failures`, never a guessed value.
* **Summaries never raise.** `summarize_value` failures (exotic objects, broken `__len__`) become
  `summary_error` in the summary; the feedback path cannot crash on a value.
* **Feedback rendering is scrub-first.** `Feedback.render(max_chars, scrub)` applies the solver's
  `scrub_volatile` to every free-text field *before* `_clip_tail` / the final cut and once more to the assembled
  text (decision 8): a cut can never split an absolute run path or a `work/<node>-<uid>` suffix and leave a
  half-scrubbed, run-specific fragment in the policy context.
* **Code-node sandbox.** `code` nodes run as a separate process (`runtime/sandbox.py`) in their own working
  directory with a time limit, an allow-listed environment (no keys, tokens or credentials), data and model hubs
  in offline mode and, where `unshare` works, in new user, network and pid namespaces
  (`SCIENCECLAW_SANDBOX_ISOLATION=auto|off|require`). A runtime audit guard in the worker
  (`runtime/node_worker.py`) denies protected locations (engine state, run receipts, credential stores,
  protected data roots), sockets, programs other than the interpreter, native libraries and the engine's own
  sources; `runtime/integrity.py` additionally scans generated source statically for these patterns
  (H_val (a)). This is defence in depth, not a guarantee against deliberately obfuscated code: leakage
  protection rests on hidden data never being in the run environment (evaluators keep hidden data in the
  parent process; the held-out file of a live `metric` constraint is never loaded by the workflow), and
  untrusted workloads should run in a container as well.

## 5. Agent (Eq.1, Eq.6)

Action JSON emitted by the policy (exactly one per turn):
```json
{"thought": "<≤3 sentences>",
 "action": {"type": "add_node", "node": {...Node fields...}}
          | {"type": "remove_node", "id": "n3"}
          | {"type": "modify_node", "id": "n3", "patch": {"code"|"code_edit"|"prompt"|"config"|"inputs"|"outputs"|"wire": ...}}
          | {"type": "add_edge", "edge": {"src","src_port","dst","dst_port","conversion"?}}
          | {"type": "remove_edge", "edge": {...}}
          | {"type": "finish"},
 "uses": ["skill:<id>", "op:<id>"]}
```
Control edits (Π_ctrl): add/remove edge, remove_node, modify_node with only
`config`, add_node of kind tool/operator/submit, finish. Executable edits
(Π_exec): add_node of kind code/llm, modify_node touching `code`/`code_edit`/`prompt`/ports.

`code_edit` = `{"find": "<text>", "replace": "<text>"}` patches the current source of a `code` node without
resending it: it replaces the one exact occurrence of `find` (`core.actions.apply_code_edit`). It is rejected —
canvas unchanged, with a message that says which case — when `find` is empty, occurs zero or several times,
when the node is not a `code` node, when it is combined with `code` in the same patch, or when the result is not
valid Python source with a top-level `run`. It is documented in the prompt as an interface rule (not as advice) and
is accepted by the `fixed_workflow` orchestration together with `code` and `config`. `add_node` /
`modify_node` reject any config value other than null on `operator` nodes (§3.2). A `config` patch is merged
into the existing config (a null value deletes the key).

Prompt content (agent/prompts.py) — allowed: canvas description, node kinds,
port schema syntax, action JSON syntax, the episode objective and required
output schema, tool signatures (with the description of every port), retrieved
Skills (full text) and Operators (signature + description + contract), budget,
the feedback of the last step, compact history. Forbidden: any strategy/how-to
text not coming from a Skill (`find_strategy_phrases` scans every built-in
prompt variant).

Interface facts the prompt states (all of them descriptions of what the runtime does, none of them advice):

* **Python values per port type** (`value_has_type`, §3.3) and the list of importable packages. The package list is
  *probed in the running interpreter* (`prompts.available_packages()`, `importlib.util.find_spec` over a fixed
  candidate list), never hard-coded, because code nodes run as `sys.executable -m scienceclaw.runtime.node_worker`.
* **Acceptance (uniform disclosure).** A short `# Acceptance` section generated from one template for every task
  and every orchestration: the deliverable counts as solved when the evaluator accepts it, all constraints
  hold (including ones that are not listed), and the final workflow replays reproducibly within the budget.
  It names neither metric values, a reference method nor any task-specific recipe; a task's objective and
  tool strings must not contain such text either.
* **What `finish` and budget exhaustion select** — exactly the rule of decision 7 (`prompts._FINISH`), so the
  policy is not told that the current canvas is the deliverable when it is not.
* **Orchestration-specific components.** `single_operator` and `fixed_workflow` list neither `operator` / `llm`
  node kinds nor the operator library, nor the llm item budget; `fixed_workflow` additionally lists only the tools
  that are wired into the pre-built canvas (input-free tools), and only the actions and patch keys it accepts
  (`code`, `code_edit`, `config`).
* **Feedback description** is built from what is really shown (`show_dev_score=False` removes the development
  score from the feedback text and from the history, `prompts._feedback_section`).

Compact history (`prompts.summarize_feedback`, one line per earlier step): outcome, first validation error, the
summary of the submitted `y` (without its head), the visible development score, failed visible constraints, the
first integrity violation, and the nodes that ran again or failed (nodes served from the checkpoint are only
counted, "+N cached ok"). Every free-text fragment goes through `scrub_volatile` *before* it is shortened (decision
8), for the history, the per-step feedback and `Feedback.render` alike.

## 6. Evolution (Eq.9–13, Eq.2–3)

* `extract_instances(traj)`: for each replay-verified pass e⁺ (first one by
  default, `instances_per_episode=1`), e⁻ = latest replay-verified failure
  before it (or None); δ = actions with k⁻ ≤ k < k⁺ (all actions up to k⁺ if e⁻
  is None, but then no Skill candidate is formed).
* `split_edits`: Π_ctrl → control edits; Π_exec → the code/llm nodes of G⁺
  touched in δ (created or modified there) plus the executable nodes weakly
  connected to them (unchanged helpers wired into the repaired block travel
  with it); nodes expanded from a registered operator are never candidates
  unless δ touched them; with e⁻ = ∅ (δ covers the whole graph) every
  executable node that is not a registered-operator expansion is kept; unchanged
  nodes of unrelated components are not turned into operators. CC_Γ → maximal
  weakly connected components of those nodes in G⁺, made **convex**: a
  component from which a path leaves and re-enters through outside nodes (the
  code → tool → code shape) could not be wired into any DAG as ONE operator
  node, so it is split along a topological cut (`split.convex_components`);
  a candidate whose boundary would still create a cycle is dropped and logged
  (`operator_abstraction` guard, `convex_splits` in the split summary).
* Skill candidate: `Patch_Θ0` prompt with attributed skills S_r[ν^S] (ν from
  the steps in δ), the control edits, the failure evidence summary (e⁻ errors /
  violations) and the success summary (e⁺). Output: new Skill (if no attributed
  skill) or revised body of the attributed skill(s). Generalize beyond the
  instance: no instance-specific values, ids or answers.
* Operator candidate per component U: body = G⁺[U]; boundary schemas from the
  crossing edges, with concrete integer dimensions replaced by symbols (instance
  sizes must not make the operator incompatible with other episodes); κ from observed boundary values (types/shapes/units/finite/
  ranges) + episode tags; ρ = {episode, steps, parent program version}. Name and
  description may be produced by Θ0 (documentation only, not a judge).
  Boundary replay: load recorded inputs at ∂⁻U from τ⁺, execute the operator in
  isolation (fresh dir), compare outputs at ∂⁺U with τ⁺ within tolerance,
  `breplay_repeats` times. Only passing operators enter the bundle.
* Bundle B = (ΔS, ΔO); `Ã = A.apply(B)`; `R_src(Ã) = Pass(e_src) ∧ Use(ω)`
  where the fresh reset solve of the source episode with Ã supplies e_src =
  its passing replay-verified evidence (the solver's final one, else the
  first). Both conjuncts are judged on THAT evidence (`validation.use_check`):
  an Operator version in ω is used iff an operator node with that ref is in
  e_src's graph and ran ok in e_src's trace; a Skill version is used iff it
  is in ν of an *effective* step (action applied OK, step ≤ e_src.step).
  Citations of rejected / failed actions, of failing evidence graphs and of
  later steps do not count (the solver adds to ν only after the action was
  applied). A policy that merely names an Operator in `uses` without placing
  its node therefore fails Use.
* Gate (Eq.2): H_val, C_val ⪯ B (decision 11), Q_val strict improvement over
  the incumbent (Eq.3) with the noise guard (decision 12); outages
  (decision 13).
* Variants (`evolution.variants`): `frozen` (no evolution), `workflow_only` (persist the whole G⁺ as a
  retrievable workflow exemplar Skill, no abstraction), `skill_only`,
  `operator_only`, `unlinked` (ΔS and ΔO gated independently), `full` (linked
  bundle). Solver orchestration modes: `single_turn` (policy must emit
  the complete graph in one action list, no repair), `single_operator` (graph
  limited to one code node + submit), `fixed_workflow` (a fixed tool→code→submit
  template; only code/config edits allowed).

---------------------------------------------------------------------------
## 7. Live tasks, canvas sessions, program store, tool library

### 7.1 Live tasks (`canvas/live.py`)
A *live task* comes from a user request: the gateway agent declares a plain JSON document
`{objective, inputs:[{name, path, format?, description?}], required_output:{type, shape?, unit?},
constraints:[{check, value?}]}` and `build_live_episode` turns it into the same `task.Episode` the engine
orchestrates, executes, replays and verifies:

* every input file becomes a read-only `load_<name>` tool of D_E; paths must lie under the configured input
  roots (symlinks are resolved; the engine's state directory and credential stores are always denied; a root
  that is the filesystem root or the home directory is refused; files over 512 MB are refused);
* the declared constraints become the hard constraints of D_V. `check` is one of `finite`, `shape`, `type`,
  `range`, `nonempty`, `len_eq_input` or `metric` (`{target, column?, metric, direction, value}`: an
  evaluator-only quality bar against a held-out file the workflow never loads; the metric is one of
  `mae`, `mse`, `rmse`, `smape`, `r2`, `accuracy`, `f1_macro`, `auc`);
* verification is Pass for a task without hidden labels: the replayed workflow reproduces its output from a
  reset state, completes within budget and satisfies every hard constraint.

### 7.2 Canvas sessions (`canvas/session.py`)
In the batch solver the engine owns the loop; in a gateway the policy is the agent that is already talking
to the user, so the loop is inverted: the agent *acts* on a `CanvasSession`, one atomic edit per call
(`act`), and receives the visible execution feedback. Everything below that boundary is the engine
unchanged: typed graph and atomic edits (`core.actions`), incremental execution against a checkpoint
(`runtime.executor`, Eq.7), reset replay with the reproducibility check and the constraint verdict
(`replay`, Eq.8) and the solver's bookkeeping (steps, uses ν, evidence stream, budgets), so a finished
session yields a `SolveResult` the evolution engine can learn from. `finish` selects the final solution G*
by decision 7, replays it if needed and writes the receipts (`trajectory.jsonl`, `graph.json`,
`session.json`) under the session's run directory.

### 7.3 Program store (`program/store.py`)
`ProgramStore` keeps immutable snapshots `<root>/snapshots/<version>/` (written to a scratch directory,
read back and compared by fingerprint, then renamed into place), the active version in `<root>/HEAD`
(replaced atomically) and one JSON receipt per promotion, rollback or seed in `<root>/receipts/`. Writers
take an exclusive file lock; `commit(..., expected_parent=...)` raises `StaleHead` unless that version is
still the head, so two promotions derived from the same program cannot overwrite each other. `rollback`
makes an earlier snapshot the head again and records the change. On first use `open()` creates the seed
program (decision 9) as the first snapshot.

### 7.4 Evolution from live sessions (`evolution/live.py`)
`LiveEvolution` drives the method of §6 with the user's own work: `val_add|val_list|val_remove` maintain
D_val; `propose(session, variant)` builds the linked Skill/Operator bundle of a finished, replay-verified
live session and stores it as a `pending` candidate (nothing about the active program changes); `gate(id)`
runs R_src, evaluates the incumbent and the candidate on D_val with `ValidationGate` and settles the
candidate as `ready` (admitted), `rejected` or `pending` (blocked: too few validation tasks, the source
task is itself a validation task, a validation task changed, the program moved on, or the model was
unavailable); `promote(id)` commits an admitted candidate to the store with a receipt, after checking that
the active program is still the one it was validated on and that the resulting program has the validated
fingerprint. `propose`, `gate` and `run` are background jobs (one at a time) polled with `status`.
Candidate records, bundles, gate reports and cached validation reports live under
`<home>/evolution/`.

### 7.5 Tool library and weights (`scilib/`, `tools/`, `bootstrap.py`)
`scilib` holds classical scientific toolkits and wrappers of pretrained models, imported inside code nodes
(`from scilib import ...`). `tools.registry` parses every module statically (nothing heavy is imported) into
`ToolEntry` records, ranks them with the same BM25 index the Skill/Operator retriever uses (`search`) and
asks a module whether it can run here (`probe`, `status_table`). `tools/weights.json` is the single source of
truth for pretrained checkpoints and upstream sources: `weights status|plan|verify` inspect the local model
root (`SCIENCECLAW_MODELS`) and print the commands that stage an asset, so weights always arrive through an
explicit, reviewable step. `bootstrap.setup` installs the optional dependency stack, stages every
non-optional asset and verifies it (profiles `full` and `light`), recording the result in
`$SCIENCECLAW_HOME/setup.json`; the engine refuses `canvas.open` and the evolution methods until that setup
is complete (`bootstrap.require`).

### 7.6 Gateway plugin boundary (`rpc.py`, `extensions/scienceclaw`)
The plugin keeps one long-lived `python -m scienceclaw.rpc` process (one JSON request per line, one JSON
response per line; library output on stdout is redirected to stderr). The methods are `ping`, `setup.status`,
`setup.start`, `canvas.open|act|render|replay|finish|status|list`, `tools.search|show|status`,
`weights.status|plan`, `program.summary|skills|operators|show|history|rollback` and
`evolve.val_add|val_list|val_remove|propose|gate|run|status|candidates|candidate`. They back the optional
agent tools `scienceclaw_canvas`, `scienceclaw_tools`, `scienceclaw_program` and `scienceclaw_evolve`.
Deployment settings (state directory, input roots, model root, config file, LLM endpoint) are plugin
configuration, never tool parameters; gateway and provider credentials are not forwarded to the engine.

### 7.7 Companion benchmark
ScienceClaw-Eval, the benchmark that accompanies the paper (23 disciplines, FoR30–FoR52; sequential task
streams and independent reset evaluation), is released separately and is not part of this repository. Its
evaluation data is hosted on Hugging Face: <https://huggingface.co/datasets/beita6969/scienceclaw-64-samples>.

---------------------------------------------------------------------------
## 8. Cross-module API (exact names — code against these)

Layering: core ← llm ← runtime ← agent ← evolution; `canvas`, `program`, `tools` and `rpc` sit on top of the
engine (`evolution.live` uses `canvas.live` and `program.store`); `task.py` depends only on `core`.
`core/actions.py` holds the action model because the executor applies actions.

### 8.1 llm/client.py, llm/interface.py
```python
@dataclass
class LLMResponse:
    text: str; usage: dict   # {"prompt_tokens","completion_tokens","reasoning_tokens","calls":1}
    cached: bool; latency_s: float; model: str; finish_reason: str | None
class LLMClient:
    def __init__(self, cfg: LLMConfig, cache_dir: str | Path | None = None)
    def chat(self, role: str, messages: list[dict], *, json_mode: bool | None = None, max_tokens: int | None = None,
             temperature: float | None = None, cache_salt: str = "", tag: str = "") -> LLMResponse
    def chat_many(self, role: str, batch: list[list[dict]], **kw) -> list[LLMResponse]   # concurrent, order-preserving
    def usage(self) -> dict                    # {"total":{...}, "by_role":{...}, "by_tag":{...}} (thread-safe)
def extract_json(text: str) -> dict | None     # tolerant: code fences, leading prose, trailing commas
```
Rules: role ∈ {"policy","executor","patch"} maps to `cfg.<role>`; reasoning models (name contains "gpt-5", "o3", "o4")
get `max_completion_tokens` + `reasoning_effort`, others `max_tokens` + `temperature`; requests MUST send a
`User-Agent` header (the gateway returns 403 without it); the API key is read from `cfg.credentials_file`
and never logged, printed, pickled or written to disk; sqlite cache keyed by sha256 of (model, messages,
params, cache_salt); retries with exponential backoff + jitter on 408/409/429/5xx/timeouts/connection errors.

### 8.2 core/actions.py
```python
ACTION_TYPES = ("add_node", "remove_node", "modify_node", "add_edge", "remove_edge", "finish", "batch")
@dataclass
class Action:
    type: str; payload: dict; uses: list[str] = []; thought: str = ""; raw: str = ""
    def to_dict(self) -> dict; @classmethod from_dict(cls, d) -> "Action"
def parse_action(text: str) -> tuple[Action | None, str | None]           # (action, error)
def apply_action(graph, action, episode, program) -> tuple[WorkflowGraph, str | None]   # new graph (copy) or error
def is_control_edit(action, graph_before) -> bool                         # Π_ctrl membership
def is_exec_edit(action, graph_before) -> bool                            # Π_exec membership
def touched_nodes(action) -> set[str]
```
apply_action fills ports automatically: tool nodes from the episode ToolSpec, operator nodes from the program
OperatorSpec, submit node inputs = {"y": episode.required_output}. `batch` (payload {"actions":[...]}) is only
accepted when the solver runs `single_turn` orchestration.

### 8.3 runtime/*
```python
# runtime/values.py
def save_value(obj, path) -> None; def load_value(path) -> Any; def outputs_match(a, b, tol: dict) -> bool
# runtime/integrity.py
def scan_code(code: str) -> list[str]           # leakage / sandbox violations (absolute paths, '..', dataset roots,
                                                # network modules, subprocess/os.system/eval of strings, env access to keys)
# runtime/executor.py
@dataclass
class Checkpoint:                               # χ
    records: dict[str, NodeRecord]              # fingerprint -> record (with output_refs to pickles)
    values_dir: str
    def copy(self) -> "Checkpoint"
@dataclass
class Feedback:                                 # f_{t,k} plus action-level info
    step: int; action_ok: bool; action_error: str | None; validation_errors: list[str]
    records: dict[str, NodeRecord]              # node id -> record for the current graph
    submit_ready: bool; y_summary: dict | None
    visible_constraints: dict[str, list]        # name -> [ok, msg]
    dev: dict | None; integrity_violations: list[str]; llm_usage: dict; wall_s: float
    def render(self, max_chars: int = 6000, scrub: Callable[[str], str] | None = None) -> str   # scrub runs BEFORE any clip / cut
class Executor:
    def __init__(self, episode, program, llm, run_dir, max_parallel_subprocs: int = 4)
    def execute(self, graph, checkpoint) -> tuple[Checkpoint, dict[str, NodeRecord], Any]   # y or None
    def apply(self, graph, checkpoint, action, step: int) -> tuple[WorkflowGraph, Checkpoint, Feedback, Any]  # Eq.7; last = y
def replay(graph, episode, program, llm, run_dir) -> tuple[Any, Trace]                        # Eq.8 (fresh dir)
def run_operator_isolated(op, inputs: dict, program, llm, run_dir, episode=None) -> tuple[dict, NodeRecord]
```

### 8.4 core/retrieval.py, agent/*
```python
class Retriever:
    def __init__(self, program: AgentProgram)
    def skills(self, episode, k: int) -> list[Skill]; def operators(self, episode, k: int) -> list[OperatorSpec]
    def slice_hash(self, episode, k_skills: int, k_ops: int) -> str   # hash of what this episode would retrieve
class Policy:
    def __init__(self, llm, cfg: SolverConfig)
    def propose(self, system: str, messages: list[dict]) -> tuple[Action | None, str | None, str, dict]  # action, parse_error, raw, usage
@dataclass
class StepRecord:
    step: int; action: dict | None; parse_error: str | None; feedback: dict; uses: list[str]
    policy_usage: dict; graph_fp: str; evidence_idx: int | None; wall_s: float
@dataclass
class SolveResult:
    episode_id: str; mode: str; program_version: str
    final_graph: WorkflowGraph; y: Any; trace: Trace | None; eval: EvalResult
    evidence: list[Evidence]; steps: list[StepRecord]; actions: list[Action]
    retrieved: dict; uses: set[str]; usage: dict; run_dir: str
    infra_error: str | None    # gateway / LLM outage that cut the solve short (NOT a task failure), else None
    def save(self, path) -> None                                       # receipts (no hidden labels)
class Solver:
    def __init__(self, cfg: SolverConfig, llm, evo_cfg: EvolutionConfig | None = None)
    def solve(self, episode, program, mode: str, run_dir) -> SolveResult   # mode: "source" | "val" | "eval"
```
`usage` (cost vector c) keys: policy_prompt_tokens, policy_completion_tokens, executor_prompt_tokens,
executor_completion_tokens, total_tokens, llm_calls, wall_s, policy_wall_s, node_runs, replays; additive:
policy_logical_tokens, executor_logical_tokens, logical_tokens (= spent + cached, cache independent; what the
validation budget B is checked against).
In "source" mode every new submit output is replayed and hidden-evaluated into `evidence`; in "val"/"eval"
mode only the final graph is replayed and evaluated. The policy never sees hidden scores in any mode.
Prompt-side determinism (decision 8): the solver pre-scrubs feedback text (`_render_feedback`, `scrub_volatile`:
run-dir paths, `work/<node>-<8hex>` uids, wall times) *before* any truncation, and old history lines are scrubbed
before they are clipped; `_visible_feedback` drops the development score from feedback and history when
`show_dev_score` is off. `prompts.fixed_workflow_tools` lists the input-free tools a fixed-workflow prompt may name.

### 8.5 evolution/*
```python
@dataclass
class EvolutionInstance:
    episode_id: str; k_minus: int | None; k_plus: int; e_minus: Evidence | None; e_plus: Evidence
    delta: list[Action]; delta_steps: list[int]; uses_in_delta: set[str]
def extract_instances(result: SolveResult, max_instances: int = 1) -> list[EvolutionInstance]          # Eq.9
def split_edits(inst, program) -> tuple[list[Action], list[set[str]]]                                   # Eq.10
def make_skill_candidates(inst, control_edits, program, llm, episode) -> list[Skill]                    # Eq.11
def make_operator_candidate(inst, component, program, llm, episode) -> OperatorSpec | None              # Eq.12
def boundary_replay(op, inst, component, program, llm, episode, run_dir, repeats: int = 1) -> tuple[bool, dict]
def build_bundle(inst, program, llm, episode, run_dir, variant: str) -> tuple[Bundle, dict]
def source_replay_check(candidate, omega: list[str], episode, solver, run_dir, *, retries: int = 0,
                        backoff_s: float = 0.0) -> tuple[bool, SolveResult]                          # Eq.13
def use_check(omega: list[str], result) -> tuple[bool, list[str]]      # Use(ω) on the passing evidence e_src (§6)
@dataclass
class ValReport:
    program_version: str; per_episode: dict[str, dict]   # id -> {"z","norm_score","h_ok","hard_violations","cost","slice_hash",
                                                         #        "completed","error","reused","discipline",...}
    macro_sr: float; norm_score: float; h_val: dict[str, bool]; cost: dict
class ValidationGate:
    def __init__(self, evo_cfg, solver_cfg, solver, val_episodes, run_dir)
    def evaluate(self, program, reuse_from: ValReport | None = None) -> ValReport   # lazy re-validation; outage -> error entry
    def budget_for(self, n_val: int) -> dict                                          # absolute caps (decision 11)
    def feasible(self, cand: ValReport, inc: ValReport) -> tuple[bool, dict]          # Eq.2: H_val and C_val ⪯ B
    def improves(self, cand: ValReport, inc: ValReport) -> tuple[bool, dict]          # Eq.3 + noise guard (decision 12)
    def admit(self, cand: ValReport, inc: ValReport) -> tuple[bool, dict]             # feasible and improves
class Evolver:
    def __init__(self, cfg: RunConfig, llm, plan, run_dir)
    def run(self, program0: AgentProgram) -> list[AgentProgram]      # snapshots A_0..A_R; writes candidates.jsonl
```

### 8.6 canvas/*, program/*, evolution/live.py, rpc.py
```python
# canvas/live.py
class SpecError(ValueError)                    # the task declaration is invalid; the message says what to change
def build_live_episode(spec: dict, *, task_id: str, input_roots: list[Path]) -> Episode
# canvas/session.py
class CanvasSession:
    def __init__(self, episode, program, run_dir, *, llm=None, cfg=None, session_id=None, reveal_verdict=True,
                 kind="live", spec=None)
    def context(self) -> str                   # protocol, task, tools, retrieved Skills/Operators
    def act(self, action: str | dict) -> dict  # one atomic edit; execution feedback
    def render(self) -> str; def replay(self) -> dict; def finish(self) -> dict; def status(self) -> dict
    result: SolveResult | None                 # set by finish(); what evolution learns from
# program/store.py
class StaleHead(RuntimeError)
class ProgramStore:
    def __init__(self, root=None)
    def head(self) -> str | None; def versions(self) -> list[str]; def load(self, version=None) -> AgentProgram | None
    def history(self) -> list[dict]            # receipts
    def commit(self, program, receipt=None, *, activate=True, expected_parent=None) -> str
    def rollback(self, version: str) -> str
    def open(self, **seed_kw) -> AgentProgram  # the active program; creates the seed program on first use
# evolution/live.py
LIVE_GATE = {"hval_mode": "absolute", "qval": "macrosr"}
class LiveEvolution:
    def __init__(self, store, home, llm, cfg, input_roots, *, auto_promote=False, min_val_tasks=2)
    def val_add(self, spec, vid=None) -> dict; def val_list(self) -> list[dict]; def val_remove(self, vid) -> dict
    def propose(self, session, variant=None) -> dict      # pending candidates from a finished live session
    def gate(self, cid) -> dict                           # R_src, then D_val gate; ready | rejected | pending
    def promote(self, cid) -> dict                        # the user's decision
    def candidates(self) -> list[dict]; def candidate(self, cid) -> dict
    def run(self, session, variant=None) -> dict          # propose + gate every candidate
    def start_job(self, kind, fn) -> dict; def job(self, jid=None, wait_s=0.0) -> dict   # background jobs
```
`rpc.Service.handle(line)` maps one JSON request to `{"id", "ok": true, "result"}` or
`{"id", "ok": false, "error": {"type", "message"}}`; every failure becomes a structured answer.
