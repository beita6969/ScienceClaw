import { spawn } from "node:child_process";
import path from "node:path";
import { Type } from "@sinclair/typebox";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/scienceclaw-bench";

type PluginConfig = {
  pythonBin?: string;
  repoRoot?: string;
  runRoot?: string;
  dataRoot?: string;
  modelRoot?: string;
  timeoutMs?: number;
  maxStdoutBytes?: number;
};

const OPERATIONS = ["catalog", "list_tasks", "smoke", "report"] as const;
type Operation = (typeof OPERATIONS)[number];

function stringEnum<T extends readonly string[]>(values: T, description: string) {
  return Type.Unsafe<T[number]>({ type: "string", enum: [...values], description });
}

function asConfig(api: OpenClawPluginApi): PluginConfig {
  return (api.pluginConfig ?? {}) as PluginConfig;
}

function resolveConfigured(raw: unknown, fallback: string): string {
  if (typeof raw !== "string" || !raw.trim()) {
    return path.resolve(fallback);
  }
  return path.resolve(raw);
}

function makeEnv(repoRoot: string, cfg: PluginConfig): NodeJS.ProcessEnv {
  // Do not forward gateway/provider credentials into the Python benchmark.
  const env: NodeJS.ProcessEnv = {
    PATH: process.env.PATH,
    HOME: process.env.HOME,
    LANG: process.env.LANG,
    LC_ALL: process.env.LC_ALL,
    PYTHONPATH: [repoRoot, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
    SCIENCECLAW_RUN_ROOT: resolveConfigured(cfg.runRoot, path.join(repoRoot, "runs")),
  };
  if (cfg.dataRoot) {
    env.SCIENCECLAW_DATA_ROOT = path.resolve(cfg.dataRoot);
  }
  if (cfg.modelRoot) {
    env.SCIENCECLAW_MODELS = path.resolve(cfg.modelRoot);
  }
  return env;
}

async function runBench(params: {
  pythonBin: string;
  repoRoot: string;
  request: Record<string, unknown>;
  env: NodeJS.ProcessEnv;
  timeoutMs: number;
  maxStdoutBytes: number;
}): Promise<Record<string, unknown>> {
  const { pythonBin, repoRoot, request, env } = params;
  const timeoutMs = Math.max(1_000, params.timeoutMs);
  const maxStdoutBytes = Math.max(4_096, params.maxStdoutBytes);

  return await new Promise((resolve, reject) => {
    const child = spawn(pythonBin, ["benchctl.py"], {
      cwd: repoRoot,
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let stdoutBytes = 0;
    let settled = false;
    const finish = (result: { value: Record<string, unknown> } | { error: Error }) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if ("error" in result) reject(result.error);
      else resolve(result.value);
    };
    const fail = (message: string) => {
      child.kill("SIGKILL");
      finish({ error: new Error(message) });
    };

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdoutBytes += Buffer.byteLength(chunk, "utf8");
      if (stdoutBytes > maxStdoutBytes) {
        fail("scienceclaw benchmark output exceeded maxStdoutBytes");
        return;
      }
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.once("error", (error) => finish({ error }));
    child.once("exit", (code) => {
      if (code !== 0) {
        finish({ error: new Error(`scienceclaw benchmark failed (${code ?? "?"}): ${stderr.trim() || stdout.trim()}`) });
        return;
      }
      try {
        const parsed = JSON.parse(stdout.trim()) as unknown;
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("benchmark response must be a JSON object");
        }
        finish({ value: parsed as Record<string, unknown> });
      } catch (error) {
        finish({ error: new Error(`invalid benchmark JSON: ${String(error)}`) });
      }
    });

    const timer = setTimeout(() => fail("scienceclaw benchmark timed out"), timeoutMs);
    child.stdin.end(JSON.stringify(request));
  });
}

export function createScienceClawBenchTool(api: OpenClawPluginApi) {
  return {
    name: "scienceclaw_bench",
    label: "ScienceClaw Benchmark",
    description:
      "Run the isolated ScienceClaw benchmark inventory, offline TOY smoke, or report operation. Formal hidden-split evaluation is intentionally unavailable through this agent tool.",
    parameters: Type.Object({
      operation: stringEnum(OPERATIONS, "Safe operation to run; catalog lists all adapters, tool refs, configs, and scripts."),
      rounds: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
      workers: Type.Optional(Type.Integer({ minimum: 1, maximum: 16 })),
      seed: Type.Optional(Type.Integer()),
      runId: Type.Optional(Type.String({ pattern: "^[A-Za-z0-9_.-]+$", maxLength: 128 })),
    }),
    async execute(_id: string, params: Record<string, unknown>) {
      const operation = typeof params.operation === "string" ? params.operation : "";
      if (!OPERATIONS.includes(operation as Operation)) {
        throw new Error(`operation must be one of: ${OPERATIONS.join(", ")}`);
      }
      const cfg = asConfig(api);
      const repoRoot = resolveConfigured(cfg.repoRoot, path.resolve(process.cwd(), "packages/scienceclaw-bench"));
      const runRoot = resolveConfigured(cfg.runRoot, path.join(repoRoot, "runs"));
      const request: Record<string, unknown> = {
        schema: 1,
        op: operation,
        run_root: runRoot,
        data_root: cfg.dataRoot,
        model_root: cfg.modelRoot,
      };
      if (operation === "smoke") {
        request.rounds = typeof params.rounds === "number" ? params.rounds : 2;
        request.workers = typeof params.workers === "number" ? params.workers : 1;
        request.seed = typeof params.seed === "number" ? params.seed : 1;
      }
      if (operation === "report") {
        if (typeof params.runId !== "string" || !params.runId.trim()) {
          throw new Error("runId is required for report");
        }
        request.run_id = params.runId;
      }

      const timeoutMs = typeof cfg.timeoutMs === "number" ? cfg.timeoutMs : operation === "smoke" ? 300_000 : 60_000;
      const maxStdoutBytes = typeof cfg.maxStdoutBytes === "number" ? cfg.maxStdoutBytes : 2 * 1024 * 1024;
      const result = await runBench({
        pythonBin: typeof cfg.pythonBin === "string" && cfg.pythonBin.trim() ? cfg.pythonBin : "python3",
        repoRoot,
        request,
        env: makeEnv(repoRoot, cfg),
        timeoutMs,
        maxStdoutBytes,
      });
      if (result.status !== "ok") {
        throw new Error(typeof result.error === "string" ? result.error : "benchmark operation failed");
      }
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }], details: result };
    },
  };
}
