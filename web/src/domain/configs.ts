import type { GpuConfig, TraceSummary } from "../api/types";

/** The trace's suggested config when it matches the trace's arch, else the first config that does. */
export function suggestedConfig(trace: TraceSummary, configs: readonly GpuConfig[]): GpuConfig | undefined {
  const compatible = configs.filter((c) => c.arch === trace.trace_gpu_arch);
  return compatible.find((c) => c.name === trace.suggested_sim_config) ?? compatible[0];
}
