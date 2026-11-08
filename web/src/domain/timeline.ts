import type { ComparisonRow, KernelStat, RunEntry, SimState, SimSummary } from "../api/types";
import type { KernelsBySim } from "./runLive";
import { median } from "./stats.ts";

/** One kernel on a track. Times are simulated (or measured) nanoseconds from the track's first kernel. */
export interface Clip {
  kernelIndex: number;
  name: string;
  start: number;
  end: number;
  cycles: number | null;
  ipc: number | null;
  pending?: boolean;
}

export type TrackSource =
  | { kind: "real"; runId: string; traceId: string; model: string | null; gpu: string | null }
  | { kind: "config"; runId: string; gpuConfig: string }
  | { kind: "sim"; runId: string; simId: string; gpuConfig: string; replica: number; state: SimState };

export interface Track {
  id: string;
  label: string;
  source: TrackSource;
  color: string;
  clips: Clip[];
  /** End of the last finished kernel; a pending clip runs past it. */
  total: number;
}

export interface Timeline {
  tracks: Track[];
  duration: number;
}

export type Mode = "time" | "align";

export interface RunTimelineInput {
  run: { id: string; trace_id: string; entries: readonly RunEntry[]; sims: readonly SimSummary[] };
  trace: { id: string; model: string | null; source_gpu: string | null };
  kernels: KernelsBySim;
  /** Comparison rows; only `real_ns` is read. Null until the comparison loads. */
  real: readonly ComparisonRow[] | null;
  colorOf: (gpuConfig: string) => string;
}

const ACTIVE: ReadonlySet<SimState> = new Set(["running", "lost"]);
const FALLBACK_NS = 1000;

interface Sample {
  kernelIndex: number;
  name: string;
  ns: number;
  cycles: number | null;
  ipc: number | null;
}

function layOut(samples: readonly Sample[], pending: { kernelIndex: number; name: string; ns: number } | null): { clips: Clip[]; total: number } {
  const clips: Clip[] = [];
  let t = 0;
  for (const s of samples.toSorted((a, b) => a.kernelIndex - b.kernelIndex)) {
    clips.push({ kernelIndex: s.kernelIndex, name: s.name, start: t, end: t + s.ns, cycles: s.cycles, ipc: s.ipc });
    t += s.ns;
  }
  const total = t;
  if (pending) clips.push({ kernelIndex: pending.kernelIndex, name: pending.name, start: t, end: t + pending.ns, cycles: null, ipc: null, pending: true });
  return { clips, total };
}

function samplesOf(stats: Record<number, KernelStat> | undefined): Sample[] {
  const out: Sample[] = [];
  for (const s of Object.values(stats ?? {})) {
    if (s.sim_time_ns === null) continue;
    out.push({ kernelIndex: s.kernel_index, name: s.kernel_name, ns: s.sim_time_ns, cycles: s.gpu_sim_cycle, ipc: s.gpu_ipc });
  }
  return out;
}

function nextIndex(samples: readonly Sample[]): number {
  return samples.reduce((m, s) => Math.max(m, s.kernelIndex + 1), 0);
}

/** A best guess for an unfinished kernel's length: the same kernel elsewhere, else this track's mean. */
function estimate(k: number, known: ReadonlyMap<number, number>, own: readonly Sample[]): number {
  const same = known.get(k);
  if (same !== undefined) return same;
  return own.length > 0 ? own.reduce((a, s) => a + s.ns, 0) / own.length : FALLBACK_NS;
}

function pendingName(names: ReadonlyMap<number, string>, k: number): string {
  return names.get(k) ?? "Simulating…";
}

export function buildRunTracks(input: RunTimelineInput): Track[] {
  const { run, trace, kernels, real, colorOf } = input;
  const configs = [...new Set(run.entries.map((e) => e.gpu_config))];
  const known = new Map<number, number>();
  const names = new Map<number, string>();
  const byIndex = new Map<number, number[]>();
  for (const sim of run.sims) {
    for (const s of samplesOf(kernels[sim.id])) {
      names.set(s.kernelIndex, s.name);
      const list = byIndex.get(s.kernelIndex) ?? [];
      list.push(s.ns);
      byIndex.set(s.kernelIndex, list);
    }
  }
  for (const r of real ?? []) {
    if (r.kernel_name !== "") names.set(r.kernel_index, r.kernel_name);
    if (r.real_ns !== null && !byIndex.has(r.kernel_index)) byIndex.set(r.kernel_index, [r.real_ns]);
  }
  for (const [k, list] of byIndex) known.set(k, median(list) ?? FALLBACK_NS);

  const tracks: Track[] = [];
  const realRows = (real ?? []).filter((r): r is ComparisonRow & { real_ns: number } => r.real_ns !== null);
  if (realRows.length > 0) {
    const { clips, total } = layOut(
      realRows.map((r) => ({ kernelIndex: r.kernel_index, name: r.kernel_name, ns: r.real_ns, cycles: null, ipc: null })),
      null,
    );
    tracks.push({
      id: `real:${trace.id}`,
      label: trace.source_gpu ?? "Real hardware",
      source: { kind: "real", runId: run.id, traceId: trace.id, model: trace.model, gpu: trace.source_gpu },
      color: "var(--series-ref)",
      clips,
      total,
    });
  }

  for (const gpuConfig of configs) {
    const sims = run.sims.filter((s) => s.gpu_config === gpuConfig).toSorted((a, b) => a.replica - b.replica);
    const color = colorOf(gpuConfig);
    const perSim = sims.map((sim) => ({ sim, samples: samplesOf(kernels[sim.id]) }));

    const pooled = new Map<number, Sample[]>();
    for (const { samples } of perSim) {
      for (const s of samples) pooled.set(s.kernelIndex, [...(pooled.get(s.kernelIndex) ?? []), s]);
    }
    const medians: Sample[] = [...pooled].map(([kernelIndex, list]) => ({
      kernelIndex,
      name: list[0]?.name ?? "",
      ns: median(list.map((s) => s.ns)) ?? 0,
      cycles: median(list.flatMap((s) => (s.cycles === null ? [] : [s.cycles]))),
      ipc: median(list.flatMap((s) => (s.ipc === null ? [] : [s.ipc]))),
    }));
    const growing = sims.some((s) => ACTIVE.has(s.state));
    const k = nextIndex(medians);
    tracks.push({
      id: `cfg:${run.id}:${gpuConfig}`,
      label: gpuConfig,
      source: { kind: "config", runId: run.id, gpuConfig },
      color,
      ...layOut(medians, growing ? { kernelIndex: k, name: pendingName(names, k), ns: estimate(k, known, medians) } : null),
    });

    for (const { sim, samples } of perSim) {
      const growingSim = ACTIVE.has(sim.state);
      const ks = nextIndex(samples);
      tracks.push({
        id: `sim:${sim.id}`,
        label: `${gpuConfig} #${sim.replica}`,
        source: { kind: "sim", runId: run.id, simId: sim.id, gpuConfig, replica: sim.replica, state: sim.state },
        color,
        ...layOut(samples, growingSim ? { kernelIndex: ks, name: pendingName(names, ks), ns: estimate(ks, known, samples) } : null),
      });
    }
  }
  return tracks;
}

/** Joins several runs; real hardware appears once per trace. */
export function buildTimeline(runs: readonly RunTimelineInput[]): Timeline {
  const seen = new Set<string>();
  const tracks: Track[] = [];
  for (const input of runs) {
    for (const t of buildRunTracks(input)) {
      if (seen.has(t.id)) continue;
      seen.add(t.id);
      tracks.push(t);
    }
  }
  return { tracks, duration: timelineDuration(tracks) };
}

export function timelineDuration(tracks: readonly Track[]): number {
  return tracks.reduce((m, t) => Math.max(m, t.clips.at(-1)?.end ?? 0), 0);
}

export function groupKernels(stats: readonly KernelStat[]): KernelsBySim {
  const out: KernelsBySim = {};
  for (const s of stats) (out[s.sim_id] ??= {})[s.kernel_index] = s;
  return out;
}

export function configTrackId(runId: string, gpuConfig: string): string {
  return `cfg:${runId}:${gpuConfig}`;
}

export function parentOf(track: Track): string | null {
  return track.source.kind === "sim" ? configTrackId(track.source.runId, track.source.gpuConfig) : null;
}

/** Index of the clip under `t`, or the last clip when `t` is past the end; -1 before the start or on an empty track. */
export function clipIndexAt(clips: readonly Clip[], t: number): number {
  if (clips.length === 0 || t < 0) return -1;
  let lo = 0;
  let hi = clips.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if ((clips[mid]?.start ?? 0) <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

const kernelLookup = new WeakMap<readonly Clip[], Map<number, Clip>>();

export function clipForKernel(track: Track, kernelIndex: number): Clip | undefined {
  let map = kernelLookup.get(track.clips);
  if (!map) {
    map = new Map(track.clips.map((c) => [c.kernelIndex, c]));
    kernelLookup.set(track.clips, map);
  }
  return map.get(kernelIndex);
}

/** Align mode gives kernel i the slot [i, i+1) on every track; a clip fills its slot in proportion to the longest run of that kernel. */
export interface AlignScale {
  slots: number;
  longest: Float64Array;
}

export function alignScale(tracks: readonly Track[]): AlignScale {
  let slots = 0;
  for (const t of tracks) for (const c of t.clips) slots = Math.max(slots, c.kernelIndex + 1);
  const longest = new Float64Array(slots);
  for (const t of tracks) {
    for (const c of t.clips) {
      if (c.pending) continue;
      const d = c.end - c.start;
      if (d > (longest[c.kernelIndex] ?? 0)) longest[c.kernelIndex] = d;
    }
  }
  return { slots, longest };
}

/** A clip's span on the shared axis: nanoseconds in time mode, kernel slots in align mode. */
export function clipSpan(clip: Clip, mode: Mode, align: AlignScale): [number, number] {
  if (mode === "time") return [clip.start, clip.end];
  const k = clip.kernelIndex;
  if (clip.pending) return [k, k + 1];
  const longest = align.longest[k] ?? 0;
  const fill = longest > 0 ? (clip.end - clip.start) / longest : 1;
  return [k, k + Math.max(0.04, fill)];
}

export interface TrackReading {
  clip: Clip;
  /** 0..1 through the clip. */
  fraction: number;
  /** Track time at the playhead, in ns. */
  elapsed: number;
  done: boolean;
  before: boolean;
}

export function readTrack(track: Track, pos: number, mode: Mode, align: AlignScale): TrackReading | null {
  const clips = track.clips;
  if (clips.length === 0) return null;
  if (mode === "time") {
    const i = clipIndexAt(clips, pos);
    if (i === -1) {
      const first = clips[0];
      return first ? { clip: first, fraction: 0, elapsed: 0, done: false, before: true } : null;
    }
    const clip = clips[i];
    if (!clip) return null;
    const d = clip.end - clip.start;
    const done = pos >= clip.end;
    const fraction = done ? 1 : d > 0 ? (pos - clip.start) / d : 1;
    return { clip, fraction, elapsed: Math.min(pos, clip.end), done: done && i === clips.length - 1, before: false };
  }
  const k = Math.floor(pos);
  const clip = clipForKernel(track, k);
  if (!clip) {
    const last = clips.at(-1);
    if (last && k > last.kernelIndex) return { clip: last, fraction: 1, elapsed: last.end, done: true, before: false };
    return null;
  }
  const [x0, x1] = clipSpan(clip, mode, align);
  const fraction = Math.max(0, Math.min(1, (pos - x0) / (x1 - x0)));
  return { clip, fraction, elapsed: clip.start + fraction * (clip.end - clip.start), done: false, before: false };
}

/** Kernels whose start falls in [from, to] on the reference track's axis. */
export function kernelRange(track: Track, from: number, to: number, mode: Mode, align: AlignScale): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const c of track.clips) {
    if (c.pending) continue;
    const [x0] = clipSpan(c, mode, align);
    if (x0 >= from && x0 <= to) {
      lo = Math.min(lo, c.kernelIndex);
      hi = Math.max(hi, c.kernelIndex);
    }
  }
  return lo === Infinity ? null : [lo, hi];
}

export function rangeTime(track: Track, [lo, hi]: [number, number]): { ns: number; kernels: number } {
  let ns = 0;
  let kernels = 0;
  for (const c of track.clips) {
    if (c.pending || c.kernelIndex < lo || c.kernelIndex > hi) continue;
    ns += c.end - c.start;
    kernels++;
  }
  return { ns, kernels };
}

export interface View {
  start: number;
  end: number;
}

export function fitView(duration: number): View {
  const d = duration > 0 ? duration : 1;
  const pad = d * 0.015;
  return { start: -pad, end: d + pad };
}

export function clampView(view: View, duration: number, minSpan: number): View {
  const d = duration > 0 ? duration : 1;
  const maxSpan = d * 1.2;
  const span = Math.min(maxSpan, Math.max(minSpan, view.end - view.start));
  const lo = -d * 0.1;
  const hi = d * 1.1;
  const start = Math.min(hi - span, Math.max(lo, view.start));
  return { start, end: start + span };
}

export function zoomAround(view: View, factor: number, anchor: number): View {
  return { start: anchor - (anchor - view.start) * factor, end: anchor + (view.end - anchor) * factor };
}

export interface Tick {
  at: number;
  label: string;
}

export interface TickSet {
  major: Tick[];
  minor: number[];
}

const NICE = [1, 2, 5] as const;

export function niceStep(raw: number): number {
  if (!(raw > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(raw));
  for (const n of NICE) if (n * p >= raw) return n * p;
  return 10 * p;
}

function timeUnit(step: number): { unit: string; scale: number } {
  if (step < 1e2) return { unit: "ns", scale: 1 };
  if (step < 1e5) return { unit: "µs", scale: 1e3 };
  if (step < 1e8) return { unit: "ms", scale: 1e6 };
  return { unit: "s", scale: 1e9 };
}

function decimalsFor(step: number): number {
  return Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
}

export function ticks(view: View, widthPx: number, mode: Mode, minLabelPx = 84): TickSet {
  const span = view.end - view.start;
  if (widthPx <= 0 || span <= 0) return { major: [], minor: [] };
  const raw = (span / widthPx) * minLabelPx;
  const step = mode === "align" ? Math.max(1, niceStep(raw)) : niceStep(raw);
  const minorStep = step / (String(step / 10 ** Math.floor(Math.log10(step))).startsWith("2") ? 4 : 5);
  const { unit, scale } = timeUnit(step);
  const decimals = decimalsFor(step / scale);
  const major: Tick[] = [];
  const first = Math.ceil(view.start / step) * step;
  for (let at = first; at <= view.end; at += step) {
    const v = Math.abs(at) < step * 1e-6 ? 0 : at;
    major.push({ at: v, label: mode === "align" ? `#${Math.round(v)}` : `${(v / scale).toFixed(decimals)} ${unit}` });
  }
  const minor: number[] = [];
  if (mode === "time" || minorStep >= 1) {
    const firstMinor = Math.ceil(view.start / minorStep) * minorStep;
    for (let at = firstMinor; at <= view.end; at += minorStep) minor.push(at);
  }
  return { major, minor };
}

/** The playhead readout: microseconds, with as many decimals as one pixel resolves (down to whole ns). */
export function formatTimecode(ns: number, nsPerPx: number): string {
  const decimals = Math.max(0, Math.min(3, Math.ceil(-Math.log10(Math.max(nsPerPx, 1e-9) / 1000))));
  const us = Math.max(0, ns) / 1000;
  const [whole = "0", frac] = us.toFixed(decimals).split(".");
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return frac === undefined ? `${grouped} µs` : `${grouped}.${frac} µs`;
}

export interface Speed {
  label: string;
  /** Axis units per wall-clock second. */
  time: number;
  timeLabel: string;
  align: number;
  alignLabel: string;
}

export const SPEEDS: readonly Speed[] = [
  { label: "Slowest", time: 100, timeLabel: "100 ns = 1 s", align: 0.5, alignLabel: "1 kernel = 2 s" },
  { label: "Slow", time: 1e3, timeLabel: "1 µs = 1 s", align: 1, alignLabel: "1 kernel = 1 s" },
  { label: "Normal", time: 1e4, timeLabel: "10 µs = 1 s", align: 4, alignLabel: "4 kernels = 1 s" },
  { label: "Fast", time: 1e5, timeLabel: "100 µs = 1 s", align: 16, alignLabel: "16 kernels = 1 s" },
  { label: "Fastest", time: 1e6, timeLabel: "1 ms = 1 s", align: 64, alignLabel: "64 kernels = 1 s" },
];

/** The slowest preset that still plays the whole axis in under ~20 s. */
export function defaultSpeed(duration: number, mode: Mode): number {
  const i = SPEEDS.findIndex((s) => duration / (mode === "time" ? s.time : s.align) <= 20);
  return i === -1 ? SPEEDS.length - 1 : i;
}

/** NLE shuttle: L plays forward and each repeat doubles the rate up to 8×, J does the same in reverse, K stops. */
export function shuttle(rate: number, key: "j" | "k" | "l"): number {
  if (key === "k") return 0;
  const dir = key === "l" ? 1 : -1;
  if (rate === 0 || Math.sign(rate) !== dir) return dir;
  return Math.max(-8, Math.min(8, rate * 2));
}

/** Neighbouring kernel boundaries on a track, on the shared axis. */
export function boundaries(track: Track, mode: Mode): number[] {
  const out: number[] = [];
  for (const c of track.clips) {
    if (mode === "align") out.push(c.kernelIndex, c.kernelIndex + 1);
    else out.push(c.start, c.end);
  }
  return [...new Set(out)].sort((a, b) => a - b);
}

export function stepBoundary(points: readonly number[], pos: number, dir: 1 | -1): number | null {
  const eps = 1e-6 * Math.max(1, Math.abs(pos));
  if (dir === 1) return points.find((p) => p > pos + eps) ?? null;
  return points.findLast((p) => p < pos - eps) ?? null;
}
