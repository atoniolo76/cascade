import type { ReactNode } from "react";
import { formatSigned } from "../../domain/format";
import { cx } from "../cx";

export type Glyph = "bar" | "tick";

export interface LegendItem {
  key: string;
  label: string;
  color: string;
  glyph: Glyph;
  mono: boolean;
}

function Key({ color, glyph, off = false }: { color: string; glyph: Glyph; off?: boolean }) {
  return glyph === "tick" ? (
    <span aria-hidden className="relative inline-block h-3 w-2.5 shrink-0">
      <span className="absolute inset-y-0 left-1/2 w-[2px] -translate-x-1/2 rounded-full" style={{ background: off ? "var(--faint)" : color }} />
    </span>
  ) : (
    <span
      aria-hidden
      className="h-[7px] w-3 shrink-0 rounded-[2px] transition-[background-color,box-shadow] duration-150"
      style={off ? { boxShadow: `inset 0 0 0 1.5px ${color}` } : { background: color }}
    />
  );
}

/** Each item toggles its series; the last visible bar series stays on so the chart never empties. */
export function ChartLegend({ items, hidden, onToggle }: { items: readonly LegendItem[]; hidden: ReadonlySet<string>; onToggle: (key: string) => void }) {
  const visibleBars = items.filter((i) => i.glyph === "bar" && !hidden.has(i.key)).length;
  return (
    <ul className="flex flex-wrap items-center gap-x-1 gap-y-1">
      {items.map((item) => {
        const off = hidden.has(item.key);
        const locked = !off && item.glyph === "bar" && visibleBars <= 1;
        return (
          <li key={item.key}>
            <button
              type="button"
              aria-pressed={!off}
              disabled={locked && items.length === 1}
              title={locked ? undefined : off ? `Show ${item.label}` : `Hide ${item.label}`}
              onClick={() => {
                if (!locked) onToggle(item.key);
              }}
              className={cx(
                "inline-flex h-7 items-center gap-2 rounded-[8px] px-2 text-[12px] transition-colors duration-150",
                items.length > 1 && !locked && "hover:bg-surface-2",
                locked && "cursor-default",
                off ? "text-faint" : "text-fg-2",
                items.length === 1 && "pl-0 disabled:cursor-default",
              )}
            >
              <Key color={item.color} glyph={item.glyph} off={off} />
              <span className={cx(item.mono && "font-mono", off && "line-through decoration-faint")}>{item.label}</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/** One line of a chart tooltip: value first, then whose value it is. */
export function TooltipLine({ color, glyph, value, label, mono = true, aside }: { color: string; glyph: Glyph; value: ReactNode; label: string; mono?: boolean; aside?: ReactNode }) {
  return (
    <li className="flex items-center gap-2.5 text-[12px] leading-5">
      <span aria-hidden className={cx("shrink-0 rounded-full", glyph === "tick" ? "h-3 w-[2px]" : "h-[3px] w-3")} style={{ background: color }} />
      <span className="w-[68px] shrink-0 font-medium text-fg tabular-nums">{value}</span>
      <span className={cx("min-w-0 flex-1 truncate text-muted", mono && "font-mono text-[11px]")}>{label}</span>
      {aside}
    </li>
  );
}

/** Signed error as a small pill; the tint deepens past 25% so the large misses stand out in a column. */
export function ErrorPill({ error, className }: { error: number | null; className?: string }) {
  if (error === null) return <span className={cx("inline-block w-[58px] text-right text-[11px] text-faint", className)}>—</span>;
  const big = Math.abs(error) >= 0.25;
  return (
    <span
      className={cx("inline-flex h-5 min-w-[58px] items-center justify-end rounded-[6px] px-1.5 text-[11px] font-medium tabular-nums", className)}
      style={big ? { background: "var(--st-lost-bg)", color: "var(--st-lost-ink)" } : { background: "var(--surface-2)", color: "var(--fg-2)" }}
    >
      {formatSigned(error)}
    </span>
  );
}
