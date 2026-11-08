import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { GpuConfig, TraceSummary } from "../api/types";
import { initialConfigs } from "./configs.ts";

const configs: GpuConfig[] = [
  { name: "SM80_A100", arch: "sm_80", core_clock_mhz: 1410 },
  { name: "SM90_H100", arch: "sm_90", core_clock_mhz: 1980 },
  { name: "SM90_H200", arch: "sm_90", core_clock_mhz: 1980 },
];
const trace = (sample_sim_configs: string[] = [], suggested_sim_config: string | null = "SM90_H100"): TraceSummary => ({
  id: "sample", state: "ready", error: null, created_at: "", updated_at: "", model: "facebook/opt-125m",
  layers: [], source_gpu: "NVIDIA H100 80GB HBM3", trace_gpu_arch: "sm_90", suggested_sim_config,
  real_gpu_config: "SM90_H100", sample_sim_configs, sample_set: "cascade-samples",
  kernel_count: 712, bytes: 0, has_real_timing: true, has_step_timeline: true,
});

describe("initialConfigs", () => {
  it("selects every compatible sample config, without duplicates or unknown names", () => {
    const sample = trace(["SM90_H200", "SM80_A100", "unknown", "SM90_H100", "SM90_H100"]);
    assert.deepEqual(initialConfigs(sample, configs).map((c) => c.name), ["SM90_H100", "SM90_H200"]);
  });
  it("falls back to the suggested compatible config when sample configs are empty or incompatible", () => {
    for (const sample of [trace(), trace(["SM80_A100", "unknown"])]) {
      assert.deepEqual(initialConfigs(sample, configs).map((c) => c.name), ["SM90_H100"]);
    }
  });
  it("falls back to the first compatible config when the suggestion is incompatible", () => {
    assert.deepEqual(initialConfigs(trace([], "SM80_A100"), configs).map((c) => c.name), ["SM90_H100"]);
  });
  it("selects nothing when there is no compatible config", () => {
    assert.deepEqual(initialConfigs(trace(["SM90_H100"]), configs.filter((c) => c.arch === "sm_80")), []);
    assert.deepEqual(initialConfigs(trace(), []), []);
  });
});
