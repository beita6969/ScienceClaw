import { Type } from "@sinclair/typebox";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/scienceclaw";
import type { EngineWorker } from "./worker.js";

const MAX_TEXT = 120_000;
const Obj = Type.Record(Type.String(), Type.Unknown());

function enumOf<T extends readonly string[]>(values: T, description: string) {
  return Type.Unsafe<T[number]>({ type: "string", enum: [...values], description });
}

function clip(text: string): string {
  return text.length <= MAX_TEXT ? text : `${text.slice(0, MAX_TEXT)}\n...[truncated ${text.length - MAX_TEXT} characters]`;
}

function result(text: string, details: unknown) {
  return { content: [{ type: "text" as const, text: clip(text) }], details };
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function required(params: Record<string, unknown>, name: string): string {
  const v = params[name];
  if (typeof v !== "string" || !v.trim()) {
    throw new Error(`${name} is required for this operation`);
  }
  return v;
}

const CANVAS_OPS = ["open", "act", "render", "replay", "finish", "status", "list"] as const;

export function createCanvasTool(worker: EngineWorker) {
  return {
    name: "scienceclaw_canvas",
    label: "ScienceClaw Canvas",
    description:
      "Solve a scientific task as a typed, executable workflow graph (ScienceClaw canvas orchestration). " +
      "operation=open starts a session from a `task` ({objective, inputs:[{name,path}], required_output:{type,shape,unit}, constraints:[{check}]}) " +
      "and returns the protocol, the available tools and the retrieved skills/operators. operation=act applies ONE atomic edit " +
      "(add_node/modify_node/remove_node/add_edge/remove_edge, the JSON given in the protocol) and returns the execution feedback; " +
      "only the affected nodes rerun. operation=replay re-executes the submitted workflow from a reset state and verifies it " +
      "(reproducibility and hard constraints). operation=finish closes the session and returns the verified deliverable. " +
      "Search for tools with scienceclaw_tools first; library functions are imported inside code nodes (from scilib import ...).",
    parameters: Type.Object({
      operation: enumOf(CANVAS_OPS, "open | act | render | replay | finish | status | list"),
      task: Type.Optional(Obj),
      sessionId: Type.Optional(Type.String({ pattern: "^[A-Za-z0-9_-]{4,40}$", maxLength: 40 })),
      action: Type.Optional(Type.Union([Obj, Type.String()])),
    }),
    async execute(_id: string, params: Record<string, unknown>) {
      const op = String(params.operation ?? "");
      if (!(CANVAS_OPS as readonly string[]).includes(op)) {
        throw new Error(`operation must be one of: ${CANVAS_OPS.join(", ")}`);
      }
      if (op === "open") {
        const r = asRecord(await worker.call("canvas.open", { task: params.task }));
        const head =
          `Canvas session ${String(r.session_id)} opened (program ${String(r.program)}, ${String(r.steps)} steps).\n` +
          `Send ONE action per call: scienceclaw_canvas(operation="act", sessionId, action=<JSON object from the Actions section>). ` +
          `When the workflow's submit node is wired, call operation="replay", then operation="finish".\n\n`;
        return result(head + String(r.context), { ...r, context: undefined });
      }
      if (op === "list") {
        return result(pretty(await worker.call("canvas.list")), undefined);
      }
      const sessionId = required(params, "sessionId");
      if (op === "act") {
        if (params.action === undefined) {
          throw new Error("action is required for operation=act");
        }
        const r = asRecord(await worker.call("canvas.act", { session_id: sessionId, action: params.action }));
        if (r.done) {
          return result(String(r.text), r);
        }
        return result(`${String(r.text)}\n[step ${String(r.step)}, ${String(r.steps_left)} steps left]`, r);
      }
      if (op === "render") {
        const r = asRecord(await worker.call("canvas.render", { session_id: sessionId }));
        return result(String(r.graph), r.status);
      }
      if (op === "replay") {
        const r = asRecord(await worker.call("canvas.replay", { session_id: sessionId }));
        return result(String(r.text), r);
      }
      if (op === "finish") {
        const r = asRecord(await worker.call("canvas.finish", { session_id: sessionId }));
        return result(`${String(r.text)}\nDeliverable: ${pretty(r.deliverable)}`, r);
      }
      return result(pretty(await worker.call("canvas.status", { session_id: sessionId })), undefined);
    },
  };
}

const TOOL_OPS = ["search", "show", "status", "weights", "setup"] as const;

export function createToolsTool(worker: EngineWorker) {
  return {
    name: "scienceclaw_tools",
    label: "ScienceClaw Tool Library",
    description:
      "Find and inspect the scientific tool library (scilib): classical toolkits for forecasting, causal inference, chemistry, " +
      "genomics, geoscience, linguistics and more, plus wrappers of pretrained models (Chronos-2, ESM-2, Demucs, SAM 2.1, DINOv2, " +
      "Stanza, BGE, DeBERTa, SevenNet, ...). operation=search ranks tools for a natural-language need; show prints a tool or module " +
      "with its documentation and whether it can run here; status lists which modules are available and why not; weights shows which " +
      "pretrained weights are staged; setup shows the installation of the whole library (start=true begins it). Tools run inside canvas code nodes.",
    parameters: Type.Object({
      operation: enumOf(TOOL_OPS, "search | show | status | weights | setup"),
      query: Type.Optional(Type.String({ maxLength: 500 })),
      target: Type.Optional(Type.String({ maxLength: 120, description: "tool id (tsfm.forecast) or module (tsfm)" })),
      k: Type.Optional(Type.Integer({ minimum: 1, maximum: 30 })),
      kind: Type.Optional(enumOf(["pretrained", "library"] as const, "restrict to pretrained-model or classical tools")),
      availableOnly: Type.Optional(Type.Boolean()),
      modules: Type.Optional(Type.Array(Type.String({ maxLength: 60 }), { maxItems: 60 })),
      start: Type.Optional(Type.Boolean({ description: "with operation=setup: start the installation if it is not complete" })),
    }),
    async execute(_id: string, params: Record<string, unknown>) {
      const op = String(params.operation ?? "");
      if (op === "search") {
        const r = asRecord(
          await worker.call("tools.search", {
            query: required(params, "query"),
            k: params.k,
            kind: params.kind,
            available_only: params.availableOnly,
          }),
        );
        const tools = (r.tools as Array<Record<string, unknown>>) ?? [];
        return result(tools.map((t) => String(t.card)).join("\n\n") || "No matching tool.", { tools: tools.map((t) => t.id) });
      }
      if (op === "show") {
        const r = asRecord(await worker.call("tools.show", { target: required(params, "target") }));
        const probe = asRecord(r.probe);
        const state = probe.available ? "available here" : `not available here: ${String(probe.reason ?? "")}`;
        const body = r.card ? String(r.card) : `${String(r.doc)}\n\nfunctions: ${(r.functions as string[]).join(", ")}`;
        return result(`${body}\n\nStatus: ${state}`, probe);
      }
      if (op === "status") {
        return result(pretty(await worker.call("tools.status", { modules: params.modules })), undefined);
      }
      if (op === "weights") {
        return result(pretty(await worker.call("weights.status")), undefined);
      }
      if (op === "setup") {
        const st = (await worker.call("setup.status")) as { complete?: boolean; state?: string };
        const now = params.start === true && !st.complete && st.state !== "running" ? await worker.call("setup.start", {}) : st;
        return result(pretty(now), undefined);
      }
      throw new Error(`operation must be one of: ${TOOL_OPS.join(", ")}`);
    },
  };
}

const PROGRAM_OPS = ["summary", "skills", "operators", "show", "history", "rollback"] as const;

export function createProgramTool(worker: EngineWorker) {
  return {
    name: "scienceclaw_program",
    label: "ScienceClaw Program",
    description:
      "Inspect the agent's editable program A = (Skills, typed Operators): the versioned library the canvas retrieves from. " +
      "operation=summary shows the active version; skills / operators list the components; show prints one (skill:<id> or op:<id>); " +
      "history lists the promotion receipts; rollback makes an earlier version active again.",
    parameters: Type.Object({
      operation: enumOf(PROGRAM_OPS, "summary | skills | operators | show | history | rollback"),
      ref: Type.Optional(Type.String({ maxLength: 160 })),
      version: Type.Optional(Type.String({ maxLength: 120 })),
    }),
    async execute(_id: string, params: Record<string, unknown>) {
      const op = String(params.operation ?? "");
      if (!(PROGRAM_OPS as readonly string[]).includes(op)) {
        throw new Error(`operation must be one of: ${PROGRAM_OPS.join(", ")}`);
      }
      if (op === "show") {
        const r = asRecord(await worker.call("program.show", { ref: required(params, "ref") }));
        return result(String(r.text), { version: r.version });
      }
      if (op === "rollback") {
        return result(pretty(await worker.call("program.rollback", { version: required(params, "version") })), undefined);
      }
      return result(pretty(await worker.call(`program.${op}`)), undefined);
    },
  };
}

const EVOLVE_OPS = ["val_add", "val_list", "val_remove", "propose", "gate", "run", "status", "candidates", "show"] as const;

export function createEvolveTool(worker: EngineWorker) {
  return {
    name: "scienceclaw_evolve",
    label: "ScienceClaw Evolution",
    description:
      "Verifiable self-evolution of the agent program from finished canvas sessions. A session that was replay-verified " +
      "can yield a candidate Skill/Operator bundle (operation=propose). operation=gate re-solves the source task with the " +
      "candidate (it must pass and use the new components), then solves the registered validation tasks with the incumbent and " +
      "the candidate under the frozen model; the candidate is admitted only if every hard constraint holds on every validation " +
      "task, cost stays within budget and the validation success rate strictly improves. An admitted candidate is 'ready': the " +
      "USER promotes it (`scienceclaw live promote <id>`); you cannot, unless the user enabled automatic promotion. Register at " +
      "least two validation tasks first (val_add, with task=<declaration with constraints> or sessionId of a passed session), " +
      "only tasks the user confirms as representative; the source task cannot validate its own candidate, a validation task " +
      "whose files changed blocks the gate, and a candidate derived from an older program is refused when a component it " +
      "changes has moved on. operation=run does propose and gate in one job. propose/gate/run are background jobs: poll " +
      "operation=status (waitSeconds up to 300). candidates and show list the candidates and their decisions; " +
      "scienceclaw_program(operation=rollback) undoes a promotion.",
    parameters: Type.Object({
      operation: enumOf(EVOLVE_OPS, "val_add | val_list | val_remove | propose | gate | run | status | candidates | show"),
      sessionId: Type.Optional(Type.String({ pattern: "^[A-Za-z0-9_-]{4,40}$", maxLength: 40 })),
      task: Type.Optional(Obj),
      id: Type.Optional(Type.String({ pattern: "^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$" })),
      candidateId: Type.Optional(Type.String({ pattern: "^c[0-9]{4,}$" })),
      variant: Type.Optional(
        enumOf(["full", "workflow_only", "skill_only", "operator_only", "unlinked"] as const, "which parts of the repair to learn"),
      ),
      jobId: Type.Optional(Type.String({ pattern: "^[a-f0-9]{10}$" })),
      waitSeconds: Type.Optional(Type.Integer({ minimum: 0, maximum: 300 })),
    }),
    async execute(_id: string, params: Record<string, unknown>) {
      const op = String(params.operation ?? "");
      if (!(EVOLVE_OPS as readonly string[]).includes(op)) {
        throw new Error(`operation must be one of: ${EVOLVE_OPS.join(", ")}`);
      }
      switch (op) {
        case "val_add":
          return result(
            pretty(await worker.call("evolve.val_add", { task: params.task, session_id: params.sessionId, id: params.id })),
            undefined,
          );
        case "val_list":
          return result(pretty(await worker.call("evolve.val_list")), undefined);
        case "val_remove":
          return result(pretty(await worker.call("evolve.val_remove", { id: required(params, "id") })), undefined);
        case "propose":
        case "run": {
          const job = await worker.call(`evolve.${op}`, { session_id: required(params, "sessionId"), variant: params.variant });
          return result(`${pretty(job)}\nPoll with scienceclaw_evolve(operation=status, waitSeconds=120).`, job);
        }
        case "gate": {
          const job = await worker.call("evolve.gate", { candidate_id: required(params, "candidateId") });
          return result(`${pretty(job)}\nPoll with scienceclaw_evolve(operation=status, waitSeconds=120).`, job);
        }
        case "status": {
          const wait = typeof params.waitSeconds === "number" ? params.waitSeconds : 0;
          const job = await worker.call("evolve.status", { job_id: params.jobId, wait_s: wait }, (wait + 30) * 1000);
          return result(pretty(job), job);
        }
        case "candidates":
          return result(pretty(await worker.call("evolve.candidates")), undefined);
        default:
          return result(pretty(await worker.call("evolve.candidate", { candidate_id: required(params, "candidateId") })), undefined);
      }
    },
  };
}

export function createScienceClawTools(_api: OpenClawPluginApi, worker: EngineWorker) {
  return [
    createCanvasTool(worker),
    createToolsTool(worker),
    createProgramTool(worker),
    createEvolveTool(worker),
  ];
}
