/** Axis math for the kernel charts: 1-2-5 tick steps, and one time unit per axis so tick labels stay short. */

/** The 1-2-2.5-5 step giving about `targetTicks` intervals over `span` that overshoots it least. */
export function niceStep(span: number, targetTicks: number): number {
  if (!(span > 0) || targetTicks < 1) return 1;
  const mag = 10 ** Math.floor(Math.log10(span / targetTicks));
  let best = { step: 10 * mag, waste: Infinity };
  for (const m of [1, 2, 2.5, 5, 10, 20]) {
    const step = m * mag;
    const n = Math.ceil(span / step - 1e-9);
    if (n < Math.max(1, targetTicks - 1) || n > targetTicks + 1) continue;
    const waste = n * step - span;
    if (waste < best.waste - 1e-12) best = { step, waste };
  }
  return best.step;
}

/** Ticks from 0 to the first step at or above `max`; the last tick is the axis end. */
export function niceTicks(max: number, targetTicks = 4): number[] {
  if (!(max > 0)) return [0, 1];
  const step = niceStep(max, targetTicks);
  const n = Math.ceil(max / step - 1e-9);
  return Array.from({ length: n + 1 }, (_, i) => round(i * step));
}

/** Ticks covering [min, max] on one step that lands on 0, for an axis with values on both sides of it. */
export function spanTicks(min: number, max: number, targetTicks = 4): number[] {
  const lo = Math.min(0, min);
  const hi = Math.max(0, max);
  if (!(hi - lo > 0)) return [-0.1, 0, 0.1];
  const step = niceStep(hi - lo, targetTicks);
  const from = Math.floor(lo / step + 1e-9);
  const to = Math.ceil(hi / step - 1e-9);
  return Array.from({ length: to - from + 1 }, (_, i) => round((from + i) * step) + 0);
}

const round = (v: number) => Number(v.toPrecision(12));

export type TimeUnit = { readonly unit: "ns" | "µs" | "ms" | "s"; readonly ns: number };

const UNITS: readonly TimeUnit[] = [
  { unit: "s", ns: 1e9 },
  { unit: "ms", ns: 1e6 },
  { unit: "µs", ns: 1e3 },
  { unit: "ns", ns: 1 },
];

/** The largest unit in which `ns` is at least 1, so an axis ending at `ns` reads 0 … 250 µs, not 0 … 0.25 ms. */
export function timeUnitFor(ns: number): TimeUnit {
  const abs = Math.abs(ns);
  return UNITS.find((u) => abs >= u.ns) ?? { unit: "ns", ns: 1 };
}

const tickFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

export function formatTimeTick(ns: number, unit: TimeUnit): string {
  return ns === 0 ? "0" : `${tickFmt.format(ns / unit.ns)} ${unit.unit}`;
}

const pctFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

/** Signed percent for an error axis: −40%, 0, +40%. */
export function formatErrorTick(fraction: number): string {
  if (fraction === 0) return "0";
  const sign = fraction > 0 ? "+" : "−";
  return `${sign}${pctFmt.format(Math.abs(fraction) * 100)}%`;
}

/** Where `v` falls along [lo, hi], as a 0–100 percentage clamped to the axis. */
export function percentAlong(v: number, lo: number, hi: number): number {
  if (hi <= lo) return 0;
  return Math.min(100, Math.max(0, ((v - lo) / (hi - lo)) * 100));
}
