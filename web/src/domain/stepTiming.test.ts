import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { SimStepTimeline } from "../api/types";
import { stepTimingByConfig } from "./stepTiming.ts";

const err = (real: number, simulated: number) => ({ real, simulated, error: simulated - real, error_pct: ((simulated - real) / real) * 100 });

/** Field names as cascade/sim/timeline.py simulate() and cascade/sim/compare.py compare() write them. */
function timeline(gpu_config: string, replica: number, scale = 1): SimStepTimeline {
  return {
    sim_id: `sim-${gpu_config}-${replica}`,
    gpu_config,
    replica,
    simulated: {
      meta: {},
      steps: [],
      requests: [],
      summary: {
        simulation: { coverage: { kernels: 683, simulated_kernels: 6, kernel_fraction: 6 / 683, gpu_time_fraction: 0.011, steps: [] } },
        per_kind: {
          prefill: { steps: 1, wall_ns_mean: 6_500_000 * scale, gpu_busy_ns_mean: 1 },
          decode: { steps: 2, wall_ns_mean: 4_800_000 * scale, gpu_busy_ns_mean: 1 },
        },
      },
    },
    compare: {
      steps: [
        { repeat: 0, step: 0, kind: "prefill", wall_ns: err(6_000_000, 6_500_000), gpu_busy_ns: err(1, 1) },
        { repeat: 0, step: 1, kind: "decode", wall_ns: err(4_000_000, 4_700_000), gpu_busy_ns: err(1, 1) },
        { repeat: 0, step: 2, kind: "decode", wall_ns: err(5_000_000, 4_900_000), gpu_busy_ns: err(1, 1) },
      ],
      requests: [
        { repeat: 0, ttft_ms: err(6, 6.5), itl_ms_mean: null, itl_ms_p50: null, e2e_ms: null },
        { repeat: 1, ttft_ms: err(8, 8.5), itl_ms_mean: null, itl_ms_p50: null, e2e_ms: null },
        { repeat: 2, ttft_ms: null, itl_ms_mean: null, itl_ms_p50: null, e2e_ms: null },
      ],
      summary: { wall_ns: { count: 3, mean_abs_error_pct: 7.5 }, gpu_busy_ns: { count: 3, mean_abs_error_pct: 0 } },
    },
  };
}

describe("stepTimingByConfig", () => {
  it("pairs predicted per-kind means from the summary with real means from compare", () => {
    const [h100] = stepTimingByConfig([timeline("SM90_H100", 0)]);
    assert.ok(h100);
    assert.deepEqual(h100.kinds, [
      { kind: "prefill", steps: 1, predicted: 6_500_000, real: 6_000_000 },
      { kind: "decode", steps: 2, predicted: 4_800_000, real: 4_500_000 },
    ]);
    assert.deepEqual(h100.ttft, { predicted: 7.5, real: 7 });
    assert.deepEqual(h100.stepError, { mape: 0.075, steps: 3 });
    assert.deepEqual(h100.coverage, { kernels: 683, simulatedKernels: 6, kernelFraction: 6 / 683, gpuTimeFraction: 0.011 });
  });

  it("gives one entry per config, read from its lowest replica", () => {
    const configs = stepTimingByConfig([timeline("SM90_H100", 1, 2), timeline("SM90_H100", 0), timeline("SM90_H200", 0, 3)]);
    assert.deepEqual(
      configs.map((c) => [c.config, c.replicas, c.kinds[0]?.predicted]),
      [
        ["SM90_H100", 2, 6_500_000],
        ["SM90_H200", 1, 19_500_000],
      ],
    );
  });

  it("keeps predictions without a compare report and drops what the summary lacks", () => {
    const t = timeline("SM90_H100", 0);
    const [only] = stepTimingByConfig([{ ...t, compare: null, simulated: { steps: [], summary: { per_kind: { decode: { steps: 3, wall_ns_mean: 5 } } } } }]);
    assert.ok(only);
    assert.deepEqual(only.kinds, [{ kind: "decode", steps: 3, predicted: 5, real: null }]);
    assert.equal(only.ttft, null);
    assert.equal(only.stepError, null);
    assert.equal(only.coverage, null);
  });
});
