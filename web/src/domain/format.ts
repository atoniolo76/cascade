import type { RunEntry } from "../api/types";

const compactFmt = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const intFmt = new Intl.NumberFormat("en-US");

export const formatInt = (n: number) => intFmt.format(n);
export const formatCompact = (n: number) => compactFmt.format(n);

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} ${units[i] ?? "TiB"}`;
}

export function formatAge(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

export function formatDuration(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

const sigFmt = [0, 1, 2].map((d) => new Intl.NumberFormat("en-US", { maximumFractionDigits: d }));

function scaled(v: number, unit: string): string {
  const a = Math.abs(v);
  const fmt = sigFmt[a < 10 ? 2 : a < 100 ? 1 : 0] ?? intFmt;
  return `${fmt.format(v)} ${unit}`;
}

export function formatNs(ns: number): string {
  const abs = Math.abs(ns);
  if (abs < 1e3) return scaled(ns, "ns");
  if (abs < 1e6) return scaled(ns / 1e3, "µs");
  if (abs < 1e9) return scaled(ns / 1e6, "ms");
  return scaled(ns / 1e9, "s");
}

export const formatDims = (d: readonly [number, number, number]) => d.join(" × ");

export function formatPercent(fraction: number, digits = 0): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function formatSigned(fraction: number): string {
  const pct = fraction * 100;
  const sign = pct > 0 ? "+" : pct < 0 ? "−" : "";
  return `${sign}${Math.abs(pct).toFixed(1)}%`;
}

export const ms = (iso: string) => Date.parse(iso);

export function entriesLabel(entries: readonly RunEntry[]) {
  return entries.map((e) => `${e.gpu_config} × ${e.count}`).join(" · ");
}

const exactFmt = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "medium" });
export const formatExact = (iso: string) => exactFmt.format(new Date(iso));

export const plural = (n: number, one: string, many = `${one}s`) => `${formatInt(n)} ${n === 1 ? one : many}`;
