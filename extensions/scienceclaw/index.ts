import type { OpenClawPluginApi } from "openclaw/plugin-sdk/scienceclaw";
import { createScienceClawTools } from "./src/tools.js";
import { type EngineConfig, EngineWorker } from "./src/worker.js";

const plugin = {
  id: "scienceclaw",
  name: "ScienceClaw",
  description:
    "Verifiable program-level self-evolution and typed workflow orchestration for scientific agents: canvas sessions, the scientific tool library, the versioned Skill/Operator program and the ScienceClaw-Eval benchmark.",
  register(api: OpenClawPluginApi) {
    const worker = new EngineWorker((api.pluginConfig ?? {}) as EngineConfig);
    for (const tool of createScienceClawTools(api, worker)) {
      api.registerTool(tool, { optional: true });
    }
    process.once("exit", () => worker.close());
  },
};

export default plugin;
