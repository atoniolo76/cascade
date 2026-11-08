import type { ComparisonRow, RunEntry, SimSummary } from "../api/types";
import { relativeError } from "./accuracy.ts";
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

export interface KernelProgress {
  done: number;
  total: number;
  fraction: number;
  over: boolean;
}

/** A simulator can log more kernels than the trace lists; the fraction caps at 1 and `over` keeps the fact. */
export function kernelProgress(sim: Pick<SimSummary, "kernels_done" | "kernels_total">): KernelProgress {
  const { kernels_done: done, kernels_total: total } = sim;
  return { done, total, fraction: total > 0 ? Math.min(1, done / total) : 0, over: done > total };
}

export function aggregateProgress(sims: readonly SimSummary[]): KernelProgress {
  let done = 0;
  let capped = 0;
  let total = 0;
  for (const s of sims) {
    done += s.kernels_done;
    capped += Math.min(s.kernels_done, s.kernels_total);
    total += s.kernels_total;
  }
  return { done, total, fraction: total > 0 ? capped / total : 0, over: sims.some((s) => s.kernels_done > s.kernels_total) };
}

export const OVER_HINT = "more kernels than the trace lists";

export function orderedSims(sims: readonly SimSummary[], entries: readonly RunEntry[]): SimSummary[] {
  const rank = new Map(configSeries(entries).map((s, i) => [s.key, i]));
  return sims.toSorted((a, b) => (rank.get(a.gpu_config) ?? 99) - (rank.get(b.gpu_config) ?? 99) || a.replica - b.replica);
}

export type Eta = { kind: "done" } | { kind: "unknown" } | { kind: "remaining"; ms: number };

/** The run ends when its slowest sim does; each running sim projects its own pace over its remaining kernels. */
export function estimateRemaining(sims: readonly SimSummary[], now: number, isTerminal: (s: SimSummary) => boolean): Eta {
  let worst = 0;
  let open = 0;
  for (const s of sims) {
    if (isTerminal(s)) continue;
    open++;
    if (s.started_at === null || s.kernels_done === 0 || s.kernels_total === 0) return { kind: "unknown" };
    const elapsed = now - Date.parse(s.started_at);
    worst = Math.max(worst, (elapsed * Math.max(0, s.kernels_total - s.kernels_done)) / s.kernels_done);
  }
  return open === 0 ? { kind: "done" } : { kind: "remaining", ms: worst };
}

function csvCell(v: string | number | null | undefined): string {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function kernelRowsCsv(rows: readonly KernelRow[], series: readonly Series[]): string {
  const head = ["kernel_index", "kernel_name", ...series.map((s) => `${s.label}_ns`)];
  const body = rows.map((r) => [r.index, r.name, ...series.map((s) => r.values[s.key])].map(csvCell).join(","));
  return [head.map(csvCell).join(","), ...body].join("\n") + "\n";
}

export function comparisonCsv(rows: readonly ComparisonRow[], configs: readonly string[]): string {
  const head = ["kernel_index", "kernel_name", "real_ns", ...configs.flatMap((c) => [`${c}_ns`, `${c}_error_pct`])];
  const body = rows.map((r) =>
    [
      r.kernel_index,
      r.kernel_name,
      r.real_ns,
      ...configs.flatMap((c) => {
        const err = relativeError(r.sim_ns[c], r.real_ns);
        return [r.sim_ns[c], err === null ? null : Number((err * 100).toFixed(2))];
      }),
    ]
      .map(csvCell)
      .join(","),
  );
  return [head.map(csvCell).join(","), ...body].join("\n") + "\n";
}
