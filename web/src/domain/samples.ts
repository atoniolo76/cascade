import type { RunEntry, TraceSummary } from "../api/types";

export const SAMPLE_TOOLTIP =
  "A real profiling and Accel-Sim run included so you can explore Cascade without profiling (about 30 minutes). Results are real; it was not run in this workspace.";

export const SAMPLE_README_URL = "https://github.com/atoniolo76/cascade/tree/amani/samples/opt125m-h100-e2e";

export const SAMPLE_RUN_BLOCKED = "Sample traces ship without their instruction traces, so Accel-Sim can't run them here. Open the sample run to explore its results.";

/** Why a trace can't start a run, or null when it can. */
export function runBlocker(trace: Pick<TraceSummary, "state" | "sample">): string | null {
  if (trace.state !== "ready") return `Trace is ${trace.state}; runs need a ready trace`;
  return trace.sample ? SAMPLE_RUN_BLOCKED : null;
}

/** No trace of the workspace's own yet: none at all, or only samples. */
export function noOwnTraces(traces: readonly Pick<TraceSummary, "sample">[]): boolean {
  return traces.every((t) => t.sample);
}

export function sampleRunLine(run: { trace: Pick<TraceSummary, "id" | "model">; entries: readonly Pick<RunEntry, "gpu_config">[] }): string {
  const configs = run.entries.map((e) => e.gpu_config).join(", ");
  return `Sample run: real Accel-Sim results for ${run.trace.model ?? run.trace.id} on ${configs}, recorded earlier.`;
}
