import type { GpuConfig, TraceSummary } from "../api/types";

/** The trace's suggested config when it matches the trace's arch, else the first config that does. */
export function suggestedConfig(trace: TraceSummary, configs: readonly GpuConfig[]): GpuConfig | undefined {
  const compatible = configs.filter((c) => c.arch === trace.trace_gpu_arch);
  return compatible.find((c) => c.name === trace.suggested_sim_config) ?? compatible[0];
}

/** All compatible sample configs, or the usual single-config default. */
export function initialConfigs(trace: TraceSummary, configs: readonly GpuConfig[]): GpuConfig[] {
  const samples = configs.filter((c) => c.arch === trace.trace_gpu_arch && trace.sample_sim_configs.includes(c.name));
  if (samples.length) return samples;
  const suggested = suggestedConfig(trace, configs);
  return suggested ? [suggested] : [];
}
