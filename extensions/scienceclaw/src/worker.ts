import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import path from "node:path";

export type EngineConfig = {
  pythonBin?: string;
  packageRoot?: string;
  repoRoot?: string;
  home?: string;
  inputRoots?: string[];
  runRoot?: string;
  dataRoot?: string;
  modelRoot?: string;
  configPath?: string;
  llm?: { baseUrl?: string; apiKey?: string; model?: string };
  timeoutMs?: number;
  maxResponseBytes?: number;
};

type Pending = {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  timer: NodeJS.Timeout;
};

const DEFAULT_TIMEOUT_MS = 15 * 60_000;
const DEFAULT_MAX_RESPONSE = 16 * 1024 * 1024;

export function resolvePackageRoot(cfg: EngineConfig): string {
  const raw = cfg.packageRoot ?? cfg.repoRoot;
  return path.resolve(
    typeof raw === "string" && raw.trim() ? raw : path.join(process.cwd(), "packages/scienceclaw"),
  );
}

function buildEnv(cfg: EngineConfig, packageRoot: string): NodeJS.ProcessEnv {
  // Gateway and provider credentials are never forwarded; the model used by `llm` nodes is opt-in through cfg.llm.
  const env: NodeJS.ProcessEnv = {
    PATH: process.env.PATH,
    HOME: process.env.HOME,
    LANG: process.env.LANG,
    LC_ALL: process.env.LC_ALL,
    PYTHONPATH: [packageRoot, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
    PYTHONUNBUFFERED: "1",
  };
  const set = (key: string, value: string | undefined) => {
    if (value && value.trim()) {
      env[key] = value;
    }
  };
  set("SCIENCECLAW_HOME", cfg.home ? path.resolve(cfg.home) : undefined);
  set("SCIENCECLAW_INPUT_ROOTS", cfg.inputRoots?.map((p) => path.resolve(p)).join(path.delimiter));
  set("SCIENCECLAW_RUN_ROOT", cfg.runRoot ? path.resolve(cfg.runRoot) : undefined);
  set("SCIENCECLAW_DATA_ROOT", cfg.dataRoot ? path.resolve(cfg.dataRoot) : undefined);
  set("SCIENCECLAW_MODELS", cfg.modelRoot ? path.resolve(cfg.modelRoot) : undefined);
  set("SCIENCECLAW_CONFIG", cfg.configPath ? path.resolve(cfg.configPath) : undefined);
  set("SCIENCECLAW_API_BASE_URL", cfg.llm?.baseUrl);
  set("SCIENCECLAW_API_KEY", cfg.llm?.apiKey);
  set("SCIENCECLAW_MODEL", cfg.llm?.model);
  return env;
}

/** A long-lived engine process spoken to over line-delimited JSON-RPC (see scienceclaw/rpc.py). */
export class EngineWorker {
  private child: ChildProcessWithoutNullStreams | null = null;
  private buffer = "";
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  private lastStderr = "";

  constructor(private readonly cfg: EngineConfig) {}

  private ensure(): ChildProcessWithoutNullStreams {
    if (this.child && !this.child.killed && this.child.exitCode === null) {
      return this.child;
    }
    const packageRoot = resolvePackageRoot(this.cfg);
    const cwd = this.cfg.inputRoots?.[0] ? path.resolve(this.cfg.inputRoots[0]) : process.cwd();
    const pythonBin = this.cfg.pythonBin?.trim() ? this.cfg.pythonBin : "python3";
    const child = spawn(pythonBin, ["-m", "scienceclaw.rpc"], {
      cwd,
      env: buildEnv(this.cfg, packageRoot),
      stdio: ["pipe", "pipe", "pipe"],
    });
    const limit = this.cfg.maxResponseBytes ?? DEFAULT_MAX_RESPONSE;
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      this.buffer += chunk;
      if (this.buffer.length > limit && !this.buffer.includes("\n")) {
        this.failAll(new Error("scienceclaw engine response exceeded maxResponseBytes"));
        child.kill("SIGKILL");
        return;
      }
      let nl = this.buffer.indexOf("\n");
      while (nl >= 0) {
        const line = this.buffer.slice(0, nl).trim();
        this.buffer = this.buffer.slice(nl + 1);
        if (line) {
          this.dispatch(line);
        }
        nl = this.buffer.indexOf("\n");
      }
    });
    child.stderr.on("data", (chunk: string) => {
      this.lastStderr = (this.lastStderr + chunk).slice(-4000);
    });
    child.once("error", (error) => this.failAll(error));
    child.once("exit", (code, signal) => {
      this.failAll(
        new Error(
          `scienceclaw engine exited (${code ?? signal ?? "?"}); open canvas sessions are lost. ${this.lastStderr.trim()}`.trim(),
        ),
      );
      if (this.child === child) {
        this.child = null;
      }
    });
    this.child = child;
    return child;
  }

  private dispatch(line: string): void {
    let msg: {
      id?: number;
      ok?: boolean;
      result?: unknown;
      error?: { type?: string; message?: string };
    };
    try {
      msg = JSON.parse(line);
    } catch {
      return; // not protocol output
    }
    const entry = typeof msg.id === "number" ? this.pending.get(msg.id) : undefined;
    if (!entry || typeof msg.id !== "number") {
      return;
    }
    this.pending.delete(msg.id);
    clearTimeout(entry.timer);
    if (msg.ok) {
      entry.resolve(msg.result);
    } else {
      entry.reject(new Error(msg.error?.message ?? "scienceclaw engine request failed"));
    }
  }

  private failAll(error: Error): void {
    for (const [id, entry] of this.pending) {
      clearTimeout(entry.timer);
      entry.reject(error);
      this.pending.delete(id);
    }
  }

  call(method: string, params: Record<string, unknown> = {}, timeoutMs?: number): Promise<unknown> {
    const child = this.ensure();
    const id = this.nextId++;
    const limit = timeoutMs ?? this.cfg.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`scienceclaw engine call ${method} timed out after ${limit} ms`));
      }, limit);
      this.pending.set(id, { resolve, reject, timer });
      child.stdin.write(`${JSON.stringify({ id, method, params })}\n`, (err) => {
        if (err) {
          clearTimeout(timer);
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }

  close(): void {
    if (this.child) {
      this.child.stdin.end();
      this.child.kill("SIGTERM");
      this.child = null;
    }
  }
}
