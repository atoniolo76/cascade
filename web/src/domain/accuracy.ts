import type { ComparisonRow } from "../api/types";

/** Signed relative error of a simulated time against the real one: +0.2 means the sim ran 20% slow. */
export function relativeError(sim: number | null | undefined, real: number | null): number | null {
  if (sim === null || sim === undefined || real === null || real <= 0) return null;
  return (sim - real) / real;
}

export interface ConfigAccuracy {
  config: string;
  /** Kernels with both a real and a simulated time. */
  matched: number;
  /** Mean absolute percentage error, as a fraction. */
  mape: number;
  /** Mean signed error, as a fraction; below 0 the sim runs fast. */
  bias: number;
  /** Summed time over the matched kernels, and its relative error. */
  total: { sim: number; real: number; error: number };
  worst: { index: number; name: string; error: number };
}

export function configAccuracy(rows: readonly ComparisonRow[], config: string): ConfigAccuracy | null {
  let matched = 0;
  let absSum = 0;
  let signedSum = 0;
  let simTotal = 0;
  let realTotal = 0;
  let worst: ConfigAccuracy["worst"] | null = null;
  for (const r of rows) {
    const sim = r.sim_ns[config] ?? null;
    const err = relativeError(sim, r.real_ns);
    if (err === null || sim === null || r.real_ns === null) continue;
    matched++;
    absSum += Math.abs(err);
    signedSum += err;
    simTotal += sim;
    realTotal += r.real_ns;
    if (worst === null || Math.abs(err) > Math.abs(worst.error)) worst = { index: r.kernel_index, name: r.kernel_name, error: err };
  }
  if (worst === null) return null;
  return {
    config,
    matched,
    mape: absSum / matched,
    bias: signedSum / matched,
    total: { sim: simTotal, real: realTotal, error: (simTotal - realTotal) / realTotal },
    worst,
  };
}

/** The lowest and highest error across configs, each clamped to include 0, so every config shares one axis. */
export function errorRange(rows: readonly ComparisonRow[], configs: readonly string[]): { min: number; max: number } {
  let min = 0;
  let max = 0;
  for (const r of rows) {
    for (const c of configs) {
      const e = relativeError(r.sim_ns[c], r.real_ns);
      if (e === null) continue;
      min = Math.min(min, e);
      max = Math.max(max, e);
    }
  }
  return { min, max };
}
