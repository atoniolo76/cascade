import type { RunSummary, SimStepTimeline, TraceSummary } from "../api/types";

export const STEP_KINDS = ["prefill", "decode"] as const;
export type StepKind = (typeof STEP_KINDS)[number];

/** A predicted value and, when the compare report has one, the real value it predicts. */
export interface Predicted {
  predicted: number;
  real: number | null;
}

export interface KindTiming extends Predicted {
  kind: StepKind;
  steps: number;
}

/** One config's step timing, read from Alessio's simulated_timeline.json summary and compare.json. */
export interface ConfigStepTiming {
  config: string;
  replicas: number;
  /** Step wall time in ns, mean over that kind's non-warm-up steps. */
  kinds: KindTiming[];
  /** Mean over requests, in ms. */
  ttft: Predicted | null;
  e2e: Predicted | null;
  itl: Predicted | null;
  busy: KindTiming[];
  /** `compare.summary.wall_ns.mean_abs_error_pct`, as a fraction. */
  stepError: { mape: number; steps: number } | null;
  coverage: { kernels: number; simulatedKernels: number; kernelFraction: number; gpuTimeFraction: number } | null;
}

type Json = Record<string, unknown>;

const obj = (v: unknown): Json | null => (typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Json) : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const at = (v: unknown, ...path: string[]): unknown => path.reduce<unknown>((o, k) => obj(o)?.[k], v);

function mean(values: readonly number[]): number | null {
  return values.length === 0 ? null : values.reduce((a, b) => a + b, 0) / values.length;
}

/** Real and predicted means of one `error()` field across compare rows; Alessio's shape is `{real, simulated, error, error_pct}`. */
function pairedMean(rows: readonly unknown[], field: string): { real: number | null; predicted: number | null } {
  const pairs = rows.flatMap((r) => {
    const real = num(at(r, field, "real"));
    const predicted = num(at(r, field, "simulated"));
    return real === null || predicted === null ? [] : [{ real, predicted }];
  });
  return { real: mean(pairs.map((p) => p.real)), predicted: mean(pairs.map((p) => p.predicted)) };
}

function readOne(t: SimStepTimeline): Omit<ConfigStepTiming, "config" | "replicas"> {
  const steps = list(t.compare?.steps);
  const kinds = STEP_KINDS.flatMap((kind): KindTiming[] => {
    const perKind = at(t.simulated, "summary", "per_kind", kind);
    const predicted = num(at(perKind, "wall_ns_mean"));
    const count = num(at(perKind, "steps"));
    if (predicted === null || count === null) return [];
    const real = pairedMean(
      steps.filter((s) => at(s, "kind") === kind),
      "wall_ns",
    ).real;
    return [{ kind, steps: count, predicted, real }];
  });

  const busy = STEP_KINDS.flatMap((kind): KindTiming[] => {
    const perKind = at(t.simulated, "summary", "per_kind", kind);
    const predicted = num(at(perKind, "gpu_busy_ns_mean"));
    const count = num(at(perKind, "steps"));
    if (predicted === null || count === null) return [];
    return [{ kind, steps: count, predicted, real: pairedMean(steps.filter((s) => at(s, "kind") === kind), "gpu_busy_ns").real }];
  });
  const requestMetric = (field: string): Predicted | null => {
    const pair = pairedMean(list(t.compare?.requests), field);
    const requests = list(t.simulated.requests).filter((r) => at(r, "warmup") !== true);
    const predicted = mean(requests.flatMap((r) => { const v = num(at(r, field)); return v === null ? [] : [v]; })) ?? pair.predicted;
    return predicted === null ? null : { predicted, real: pair.real };
  };
  const ttft = requestMetric("ttft_ms");
  const e2e = requestMetric("e2e_ms");
  const itl = requestMetric("itl_ms_mean");

  const wall = at(t.compare, "summary", "wall_ns");
  const mapePct = num(at(wall, "mean_abs_error_pct"));
  const compared = num(at(wall, "count"));
  const stepError = mapePct === null || compared === null ? null : { mape: mapePct / 100, steps: compared };

  const cov = at(t.simulated, "summary", "simulation", "coverage");
  const kernels = num(at(cov, "kernels"));
  const simulatedKernels = num(at(cov, "simulated_kernels"));
  const kernelFraction = num(at(cov, "kernel_fraction"));
  const gpuTimeFraction = num(at(cov, "gpu_time_fraction"));
  const coverage =
    kernels === null || simulatedKernels === null || kernelFraction === null || gpuTimeFraction === null ? null : { kernels, simulatedKernels, kernelFraction, gpuTimeFraction };

  return { kinds, busy, ttft, e2e, itl, stepError, coverage };
}

/** One entry per config, in the order the API lists sims. Replicas of a config replay the same trace, so its first replica speaks for it. */
export function stepTimingByConfig(timelines: readonly SimStepTimeline[]): ConfigStepTiming[] {
  const byConfig = new Map<string, SimStepTimeline[]>();
  for (const t of timelines) byConfig.set(t.gpu_config, [...(byConfig.get(t.gpu_config) ?? []), t]);
  return [...byConfig].flatMap(([config, sims]) => {
    const first = sims.toSorted((a, b) => a.replica - b.replica)[0];
    return first === undefined ? [] : [{ config, replicas: sims.length, ...readOne(first) }];
  });
}

export const TIMING_METRICS = [
  { key: "e2e", label: "Request E2E" },
  { key: "ttft", label: "TTFT" },
  { key: "itl", label: "ITL" },
  { key: "prefillWall", label: "Prefill wall" },
  { key: "decodeWall", label: "Decode wall" },
  { key: "prefillBusy", label: "Prefill GPU" },
  { key: "decodeBusy", label: "Decode GPU" },
] as const;
export type TimingMetric = (typeof TIMING_METRICS)[number]["key"];

export interface ArchitectureTimingRow {
  key: string;
  kind: "measured" | "simulated" | "pending";
  config: string;
  label: string;
  note: string;
  values: Record<TimingMetric, number | null>;
  e2eError: number | null;
}

function metricValues(timing: ConfigStepTiming, side: "real" | "predicted"): ArchitectureTimingRow["values"] {
  const kind = (values: KindTiming[], name: StepKind) => {
    const value = values.find((k) => k.kind === name)?.[side];
    return value === undefined || value === null ? null : value / 1e6;
  };
  return {
    e2e: timing.e2e?.[side] ?? null, ttft: timing.ttft?.[side] ?? null, itl: timing.itl?.[side] ?? null,
    prefillWall: kind(timing.kinds, "prefill"), decodeWall: kind(timing.kinds, "decode"),
    prefillBusy: kind(timing.busy, "prefill"), decodeBusy: kind(timing.busy, "decode"),
  };
}

/** The latest succeeded run per other trace, never an active or failed replacement. */
export function latestSampleRuns(current: TraceSummary, traces: readonly TraceSummary[], runs: readonly RunSummary[]): RunSummary[] {
  if (!current.sample_set) return [];
  const ids = new Set(traces.filter((t) => t.id !== current.id && t.sample_set === current.sample_set).map((t) => t.id));
  const latest = new Map<string, RunSummary>();
  for (const run of runs.toSorted((a, b) => b.created_at.localeCompare(a.created_at))) {
    if (ids.has(run.trace_id) && run.state === "succeeded" && !latest.has(run.trace_id)) latest.set(run.trace_id, run);
  }
  return [...latest.values()];
}

export function architectureTimingRows(
  current: TraceSummary,
  sources: readonly { trace: TraceSummary; timelines: readonly SimStepTimeline[] }[],
  traces: readonly TraceSummary[],
  pendingConfigs: readonly string[],
): ArchitectureTimingRow[] {
  const rows: ArchitectureTimingRow[] = [];
  for (const { trace, timelines } of sources) {
    const configs = stepTimingByConfig(timelines);
    const reference = configs.find((c) => c.config === trace.real_gpu_config) ?? configs[0];
    if (reference && Object.values(metricValues(reference, "real")).some((v) => v !== null)) rows.push({
      key: `${trace.id}:real`, kind: "measured", config: trace.real_gpu_config ?? "",
      label: `Real ${trace.source_gpu ?? trace.real_gpu_config ?? "hardware"} (measured)`,
      note: trace.id === current.id ? "Current trace reference" : `Native trace · ${trace.id}`,
      values: metricValues(reference, "real"), e2eError: null,
    });
    for (const config of configs) {
      const eligible = trace.id === current.id && config.config === current.real_gpu_config;
      const values = metricValues(config, "predicted");
      const real = config.e2e?.real;
      rows.push({
        key: `${trace.id}:${config.config}`, kind: "simulated", config: config.config, label: config.config,
        note: eligible ? "Native hardware · error vs measured" : trace.id !== current.id
          ? `Other native trace · ${trace.id} · metrics only`
          : current.real_gpu_config ? `No real ${config.config} baseline · counterfactual` : "Real hardware config unidentified · metrics only",
        values, e2eError: eligible && values.e2e !== null && real !== undefined && real !== null && real > 0 ? (values.e2e - real) / real : null,
      });
    }
  }
  const seeded = new Set(traces.filter((t) => t.sample_set === current.sample_set).map((t) => t.real_gpu_config));
  for (const config of new Set(pendingConfigs)) {
    if (seeded.has(config)) continue;
    const gpu = config.split("_").slice(1).join(" ");
    rows.push({
      key: `pending:${config}`, kind: "pending", config, label: config,
      note: `Pending: native ${gpu} trace not captured yet`,
      values: { e2e: null, ttft: null, itl: null, prefillWall: null, decodeWall: null, prefillBusy: null, decodeBusy: null }, e2eError: null,
    });
  }
  return rows;
}
