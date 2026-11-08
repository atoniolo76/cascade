import type { ComparisonRow, RunEntry, SimSummary } from "../api/types";
import type { KernelsBySim } from "./runLive";

export interface KernelRow {
  index: number;
  name: string;
  values: Record<string, number | null>;
}

export interface Series {
  key: string;
  label: string;
  color: string;
}

const SERIES_SLOTS = 8;

/** Configs keep their entry-order color everywhere in a run; past eight they share the reference gray. */
export function configSeries(entries: readonly RunEntry[]): Series[] {
  const seen: string[] = [];
  for (const e of entries) if (!seen.includes(e.gpu_config)) seen.push(e.gpu_config);
  return seen.map((name, i) => ({
    key: name,
    label: name,
    color: i < SERIES_SLOTS ? `var(--series-${i + 1})` : "var(--series-ref)",
  }));
}

export const REAL_SERIES: Series = { key: "__real__", label: "Real hardware", color: "var(--series-ref)" };

export function median(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  const hi = s[mid] ?? 0;
  return s.length % 2 === 1 ? hi : ((s[mid - 1] ?? hi) + hi) / 2;
}

export function simTimeRows({
  sims,
  kernels,
  configs,
}: {
  sims: readonly SimSummary[];
  kernels: KernelsBySim;
  configs: readonly string[];
}): KernelRow[] {
  const samples = new Map<number, { name: string; byConfig: Map<string, number[]> }>();
  for (const sim of sims) {
    for (const stat of Object.values(kernels[sim.id] ?? {})) {
      if (stat.sim_time_ns === null) continue;
      const entry = samples.get(stat.kernel_index) ?? { name: stat.kernel_name, byConfig: new Map<string, number[]>() };
      const list = entry.byConfig.get(sim.gpu_config) ?? [];
      list.push(stat.sim_time_ns);
      entry.byConfig.set(sim.gpu_config, list);
      samples.set(stat.kernel_index, entry);
    }
  }
  return [...samples.entries()]
    .sort(([a], [b]) => a - b)
    .map(([index, { name, byConfig }]) => ({
      index,
      name,
      values: Object.fromEntries(configs.map((c) => [c, median(byConfig.get(c) ?? [])])),
    }));
}

export function comparisonRows(rows: readonly ComparisonRow[]): KernelRow[] {
  return rows.map((r) => ({
    index: r.kernel_index,
    name: r.kernel_name,
    values: { ...r.sim_ns, [REAL_SERIES.key]: r.real_ns },
  }));
}

export function relativeError(sim: number | null | undefined, real: number | null): number | null {
  if (sim === null || sim === undefined || real === null || real === 0) return null;
  return (sim - real) / real;
}

export function aggregateProgress(sims: readonly SimSummary[]): { done: number; total: number } {
  let done = 0;
  let total = 0;
  for (const s of sims) {
    done += s.kernels_done;
    total += s.kernels_total;
  }
  return { done, total };
}
