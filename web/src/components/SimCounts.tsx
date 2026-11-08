import type { SimState } from "../api/types";
import { SIM_STATE, SIM_STATE_ORDER, TONE_VARS } from "../domain/status";
import { cx } from "./cx";

type Counts = Partial<Record<SimState, number>>;

export function SimCountsBar({ counts, className }: { counts: Counts; className?: string }) {
  const total = SIM_STATE_ORDER.reduce((n, s) => n + (counts[s] ?? 0), 0);
  return (
    <div className={cx("flex h-1.5 gap-0.5 overflow-hidden rounded-full bg-surface-2", className)} aria-hidden>
      {SIM_STATE_ORDER.map((s) => {
        const n = counts[s] ?? 0;
        if (n === 0 || total === 0) return null;
        return (
          <div
            key={s}
            className="h-full transition-[flex-grow] duration-300 first:rounded-l-full last:rounded-r-full"
            style={{ flexGrow: n, background: TONE_VARS[SIM_STATE[s].tone].dot }}
          />
        );
      })}
    </div>
  );
}

export function SimCountsText({ counts }: { counts: Counts }) {
  const parts = SIM_STATE_ORDER.flatMap((s) => {
    const n = counts[s] ?? 0;
    return n > 0 ? [{ s, n }] : [];
  });
  if (parts.length === 0) return <span className="text-faint">No sims</span>;
  return (
    <span className="flex flex-wrap gap-x-2.5 gap-y-0.5">
      {parts.map(({ s, n }) => (
        <span key={s} className="inline-flex items-center gap-1 whitespace-nowrap tabular-nums">
          <span className="size-1.5 rounded-full" style={{ background: TONE_VARS[SIM_STATE[s].tone].dot }} />
          {n} {SIM_STATE[s].label.toLowerCase()}
        </span>
      ))}
    </span>
  );
}
