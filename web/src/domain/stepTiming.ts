import type { SimStepTimeline } from "../api/types";

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

  const ttftPair = pairedMean(list(t.compare?.requests), "ttft_ms");
  const ttft = ttftPair.predicted === null ? null : { predicted: ttftPair.predicted, real: ttftPair.real };

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

  return { kinds, ttft, stepError, coverage };
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
