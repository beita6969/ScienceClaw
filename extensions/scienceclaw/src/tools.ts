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
      episode: Type.Optional(Obj),
      sessionId: Type.Optional(Type.String({ pattern: "^[A-Za-z0-9_-]{4,40}$", maxLength: 40 })),
      action: Type.Optional(Type.Union([Obj, Type.String()])),
    }),
    async execute(_id: string, params: Record<string, unknown>) {
      const op = String(params.operation ?? "");
      if (!(CANVAS_OPS as readonly string[]).includes(op)) {
        throw new Error(`operation must be one of: ${CANVAS_OPS.join(", ")}`);
      }
      if (op === "open") {
        const r = asRecord(await worker.call("canvas.open", { task: params.task, episode: params.episode }));
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

const TOOL_OPS = ["search", "show", "status", "weights"] as const;

export function createToolsTool(worker: EngineWorker) {
  return {
    name: "scienceclaw_tools",
    label: "ScienceClaw Tool Library",
    description:
      "Find and inspect the scientific tool library (scilib): classical toolkits for forecasting, causal inference, chemistry, " +
      "genomics, geoscience, linguistics and more, plus wrappers of pretrained models (Chronos-2, ESM-2, Demucs, SAM 2.1, DINOv2, " +
      "Stanza, BGE, DeBERTa, SevenNet, ...). operation=search ranks tools for a natural-language need; show prints a tool or module " +
      "with its documentation and whether it can run here; status lists which modules are available and why not; weights shows which " +
      "pretrained weights are staged and how to stage the rest. Tools run inside canvas code nodes.",
    parameters: Type.Object({
      operation: enumOf(TOOL_OPS, "search | show | status | weights"),
      query: Type.Optional(Type.String({ maxLength: 500 })),
      target: Type.Optional(Type.String({ maxLength: 120, description: "tool id (tsfm.forecast) or module (tsfm)" })),
      k: Type.Optional(Type.Integer({ minimum: 1, maximum: 30 })),
      kind: Type.Optional(enumOf(["pretrained", "library"] as const, "restrict to pretrained-model or classical tools")),
      task: Type.Optional(Type.String({ maxLength: 12, description: "discipline code such as FoR37" })),
      availableOnly: Type.Optional(Type.Boolean()),
      modules: Type.Optional(Type.Array(Type.String({ maxLength: 60 }), { maxItems: 60 })),
    }),
    async execute(_id: string, params: Record<string, unknown>) {
      const op = String(params.operation ?? "");
      if (op === "search") {
        const r = asRecord(
          await worker.call("tools.search", {
            query: required(params, "query"),
            k: params.k,
            kind: params.kind,
            task: params.task,
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

const EVAL_OPS = ["catalog", "list_tasks", "report"] as const;

export function createEvalTool(worker: EngineWorker) {
  return {
    name: "scienceclaw_eval",
    label: "ScienceClaw-Eval",
    description:
      "Inspect the ScienceClaw-Eval benchmark of 23 disciplines: catalog lists adapters, tool refs, configs and launchers; " +
      "list_tasks shows which task datasets are installed here; report builds the report of a finished run (runId). " +
      "Formal hidden-split evaluation is intentionally unavailable through this tool. Benchmark episodes can be solved " +
      "interactively with scienceclaw_canvas(operation=open, episode={discipline, split, index}) when their data are installed.",
    parameters: Type.Object({
      operation: enumOf(EVAL_OPS, "catalog | list_tasks | report"),
      runId: Type.Optional(Type.String({ pattern: "^[A-Za-z0-9_.-]+$", maxLength: 128 })),
    }),
    async execute(_id: string, params: Record<string, unknown>) {
      const op = String(params.operation ?? "");
      if (!(EVAL_OPS as readonly string[]).includes(op)) {
        throw new Error(`operation must be one of: ${EVAL_OPS.join(", ")}`);
      }
      const payload = op === "report" ? { run_id: required(params, "runId") } : {};
      return result(pretty(await worker.call(`eval.${op}`, payload, 120_000)), undefined);
    },
  };
}

export function createScienceClawTools(_api: OpenClawPluginApi, worker: EngineWorker) {
  return [createCanvasTool(worker), createToolsTool(worker), createProgramTool(worker), createEvalTool(worker)];
}
