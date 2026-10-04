import type { OpenClawPluginApi } from "openclaw/plugin-sdk/scienceclaw";
import { createScienceClawBenchTool } from "./src/tool.js";

const plugin = {
  id: "scienceclaw",
  name: "ScienceClaw Benchmark",
  description: "Optional bridge to the isolated Python ScienceClaw benchmark package.",
  register(api: OpenClawPluginApi) {
    api.registerTool(createScienceClawBenchTool(api), { optional: true });
  },
};

export default plugin;
