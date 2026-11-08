import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { GpuConfig } from "../api/types";
import { launchableConfigs, noOwnTraces, runBlocker, sampleRunLine } from "./samples.ts";

describe("runBlocker", () => {
  it("lets any ready trace run, sample or not", () => {
    assert.equal(runBlocker({ state: "ready" }), null);
  });

  it("names the state of a trace that is not ready", () => {
    assert.equal(runBlocker({ state: "processing" }), "Trace is processing; runs need a ready trace");
  });
});

describe("launchableConfigs", () => {
  const configs: GpuConfig[] = [
    { name: "SM80_A100", arch: "sm_80", core_clock_mhz: 1410 },
    { name: "SM90_H100", arch: "sm_90", core_clock_mhz: 1980 },
    { name: "SM90_H200", arch: "sm_90", core_clock_mhz: 1980 },
  ];
  const names = (cs: readonly GpuConfig[]) => cs.map((c) => c.name);

  it("limits a sample trace to its recorded configs, in config order", () => {
    assert.deepEqual(names(launchableConfigs({ sample: true, sample_sim_configs: ["SM90_H200", "SM90_H100"] }, configs)), ["SM90_H100", "SM90_H200"]);
  });

  it("leaves a trace of your own with every config", () => {
    assert.deepEqual(names(launchableConfigs({ sample: false, sample_sim_configs: ["SM90_H100"] }, configs)), ["SM80_A100", "SM90_H100", "SM90_H200"]);
  });

  it("does not invent a recorded config missing from the config list", () => {
    assert.deepEqual(names(launchableConfigs({ sample: true, sample_sim_configs: ["SM90_H100", "SM100_B200"] }, configs)), ["SM90_H100"]);
  });
});

describe("noOwnTraces", () => {
  it("is true for an empty workspace and one holding only samples", () => {
    assert.equal(noOwnTraces([]), true);
    assert.equal(noOwnTraces([{ sample: true }]), true);
  });

  it("is false once a trace of your own arrives", () => {
    assert.equal(noOwnTraces([{ sample: true }, { sample: false }]), false);
  });
});

describe("sampleRunLine", () => {
  it("names the model and configs", () => {
    const run = { trace: { id: "t", model: "facebook/opt-125m" }, entries: [{ gpu_config: "SM90_H100", count: 1 }] };
    assert.equal(sampleRunLine(run), "Sample run: real Accel-Sim results for facebook/opt-125m on SM90_H100, recorded earlier.");
  });
});
