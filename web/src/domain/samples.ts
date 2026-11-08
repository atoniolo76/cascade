import type { GpuConfig, RunEntry, TraceSummary } from "../api/types";

export const SAMPLE_TOOLTIP =
  "A real profiling and Accel-Sim run included so you can explore Cascade without profiling (about 30 minutes). Results are real; it was not run in this workspace.";

export const SAMPLE_README_URL = "https://github.com/atoniolo76/cascade/tree/amani/samples";

export const SAMPLE_REPLAY_NOTE = "Sample trace: replays recorded Accel-Sim results for these configs.";

export const REPLAY_RUN_LINE = "Replays the sample's recorded Accel-Sim results; no new simulation ran.";

/** Why a trace can't start a run, or null when it can. */
export function runBlocker(trace: Pick<TraceSummary, "state">): string | null {
  return trace.state === "ready" ? null : `Trace is ${trace.state}; runs need a ready trace`;
}

/** A sample trace can only replay the configs it has recorded results for. */
export function launchableConfigs(trace: Pick<TraceSummary, "sample" | "sample_sim_configs">, configs: readonly GpuConfig[]): readonly GpuConfig[] {
  return trace.sample ? configs.filter((c) => trace.sample_sim_configs.includes(c.name)) : configs;
}

/** No trace of the workspace's own yet: none at all, or only samples. */
export function noOwnTraces(traces: readonly Pick<TraceSummary, "sample">[]): boolean {
  return traces.every((t) => t.sample);
}

export function sampleRunLine(run: { trace: Pick<TraceSummary, "id" | "model">; entries: readonly Pick<RunEntry, "gpu_config">[] }): string {
  const configs = run.entries.map((e) => e.gpu_config).join(", ");
  return `Sample run: real Accel-Sim results for ${run.trace.model ?? run.trace.id} on ${configs}, recorded earlier.`;
}
