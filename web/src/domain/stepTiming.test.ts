import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { RunSummary, SimStepTimeline, TraceSummary } from "../api/types";
import { architectureTimingRows, latestSampleRuns, stepTimingByConfig } from "./stepTiming.ts";

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

const trace = (id = "h100", real_gpu_config: string | null = "SM90_H100"): TraceSummary => ({
  id, real_gpu_config, sample_set: "cascade-samples", state: "ready", error: null,
  created_at: "", updated_at: "", model: "facebook/opt-125m", layers: [],
  source_gpu: real_gpu_config === "SM80_A100" ? "NVIDIA A100" : "NVIDIA H100 80GB HBM3",
  trace_gpu_arch: real_gpu_config === "SM80_A100" ? "sm_80" : "sm_90",
  suggested_sim_config: "SM90_H100", sample_sim_configs: [], kernel_count: 1, bytes: 1,
  has_real_timing: false, has_step_timeline: true,
});

function requests(config: string, simulated = 262.14): SimStepTimeline {
  const t = timeline(config, 0);
  return { ...t, compare: { ...t.compare, requests: [
    { e2e_ms: err(254.45, simulated), ttft_ms: err(8.63, 9), itl_ms_mean: err(7.93, 8.17) },
  ] } };
}

describe("architectureTimingRows", () => {
  it("shows signed E2E error only for the current trace's real config, never H200", () => {
    const current = trace();
    const rows = architectureTimingRows(current, [{ trace: current, timelines: [requests("SM90_H100"), requests("SM90_H200", 262.16)] }], [current], ["SM80_A100"]);
    const [real, h100, h200, pending] = rows;
    assert.ok(real && h100 && h200 && pending);
    assert.equal(real.label, "Real NVIDIA H100 80GB HBM3 (measured)");
    assert.equal(real.values.e2e, 254.45);
    assert.equal(h100.e2eError?.toFixed(4), "0.0302");
    assert.equal(h100.values.e2e, 262.14);
    assert.equal(h100.values.ttft, 9);
    assert.equal(h100.values.itl, 8.17);
    assert.equal(h100.values.prefillWall, 6.5);
    assert.equal(h100.values.decodeWall, 4.8);
    assert.equal(h100.values.prefillBusy, 0.000001);
    assert.equal(h200.values.e2e, 262.16);
    assert.equal(h200.e2eError, null);
    assert.match(h200.note, /No real SM90_H200 baseline/);
    assert.equal(pending.kind, "pending");
    assert.equal(pending.label, "SM80_A100");
    assert.equal(pending.note, "Pending: native A100 trace not captured yet");
  });

  it("includes other native traces with their own real values but no errors, even if configs match", () => {
    const current = trace(), a100 = trace("a100", "SM80_A100");
    const rows = architectureTimingRows(current, [
      { trace: current, timelines: [requests("SM90_H100")] },
      { trace: a100, timelines: [requests("SM80_A100", 200), requests("SM90_H100", 201)] },
    ], [current, a100], ["SM80_A100"]);
    assert.equal(rows.filter((r) => r.kind === "pending").length, 0);
    const other = rows.filter((r) => r.key.startsWith("a100:"));
    assert.equal(other.length, 3);
    assert.match(other[0]?.label ?? "", /Real NVIDIA A100/);
    assert.equal(other.find((r) => r.config === "SM80_A100" && r.kind === "simulated")?.values.e2e, 200);
    assert.ok(other.every((r) => r.e2eError === null));
  });

  it("does not infer a baseline from suggested_sim_config when real_gpu_config is missing", () => {
    const current = trace("legacy", null);
    const rows = architectureTimingRows(current, [{ trace: current, timelines: [requests("SM90_H100")] }], [current], []);
    assert.ok(rows.every((r) => r.e2eError === null));
    assert.match(rows.find((r) => r.kind === "simulated")?.note ?? "", /unidentified/);
  });

  it("keeps simulated metrics without compare and excludes warm-up requests", () => {
    const current = trace();
    const t = { ...timeline("SM90_H100", 0), compare: null };
    t.simulated.requests = [{ warmup: true, e2e_ms: 999 }, { warmup: false, e2e_ms: 10, ttft_ms: 2, itl_ms_mean: 1 }];
    const row = architectureTimingRows(current, [{ trace: current, timelines: [t] }], [current], []).find((r) => r.kind === "simulated");
    assert.equal(row?.values.e2e, 10);
    assert.equal(row.e2eError, null);
    assert.deepEqual(architectureTimingRows(current, [], [current], []), []);
  });
});

describe("latestSampleRuns", () => {
  const run = (id: string, trace_id: string, state: RunSummary["state"], created_at: string): RunSummary => ({ id, trace_id, state, created_at, entries: [], sim_counts: {} });
  it("chooses only the newest succeeded run per other trace in this set", () => {
    const current = trace(), a100 = trace("a100", "SM80_A100"), other = { ...trace("unrelated"), sample_set: "other" };
    const runs = [run("older", "a100", "succeeded", "2026-01-01"), run("active", "a100", "active", "2026-01-04"),
      run("failed", "a100", "failed", "2026-01-05"), run("latest", "a100", "succeeded", "2026-01-03"),
      run("self", "h100", "succeeded", "2026-01-06"), run("other", "unrelated", "succeeded", "2026-01-07")];
    assert.deepEqual(latestSampleRuns(current, [current, a100, other], runs).map((r) => r.id), ["latest"]);
    assert.deepEqual(latestSampleRuns({ ...current, sample_set: null }, [a100], runs), []);
    assert.deepEqual(latestSampleRuns(current, [current], runs), []);
  });
});
