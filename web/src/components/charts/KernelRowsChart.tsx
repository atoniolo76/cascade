import { useId, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from "react";
import { formatErrorTick, formatTimeTick, niceTicks, percentAlong, spanTicks, timeUnitFor } from "../../domain/scale";
import { cx } from "../cx";
import { setKernelFocus, useKernelFocus } from "./kernelFocus";
import { KernelLabel } from "./KernelLabel";

export interface ChartBar {
  key: string;
  color: string;
  value: number | null;
}

export interface ChartRow {
  index: number;
  name: string;
  bars: readonly ChartBar[];
  /** A reference value drawn as a tick across the bars, such as real hardware time. */
  marker: number | null;
  trailing: string | null;
}

/** `time` grows bars from 0 to the slowest value; `error` grows them either way from 0 across [min, max]. */
export type ChartScale = { kind: "time"; max: number } | { kind: "error"; min: number; max: number };

const MAX_VIEW = 600;
const NARROW = 520;
const OVERSCAN = 6;

function geometry(barCount: number, narrow: boolean) {
  const n = Math.max(1, barCount);
  const thick = n === 1 ? 10 : n === 2 ? 7 : 5;
  const bars = n * thick + (n - 1) * 2;
  const rowH = narrow ? 22 + bars + 14 : Math.max(30, bars + 16);
  return { thick, bars, rowH };
}

function axis(scale: ChartScale, plotWidth: number) {
  if (scale.kind === "time") {
    const ticks = niceTicks(scale.max, plotWidth < 320 ? 3 : 4);
    const lo = 0;
    const hi = ticks.at(-1) ?? 1;
    const unit = timeUnitFor(hi);
    return { ticks, lo, hi, label: (v: number) => formatTimeTick(v, unit) };
  }
  const ticks = spanTicks(scale.min, scale.max, plotWidth < 320 ? 3 : 5);
  return { ticks, lo: ticks[0] ?? -1, hi: ticks.at(-1) ?? 1, label: formatErrorTick };
}

function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(() => {
      setWidth(el.clientWidth);
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
    };
  }, []);
  return [ref, width] as const;
}

type Anchor = { x: number; y: number };

export function KernelRowsChart({
  scope,
  rows,
  scale,
  trailingLabel,
  tooltip,
  label,
  footer,
}: {
  /** Rows with the same `scope` and kernel index highlight together across charts. */
  scope: string;
  rows: readonly ChartRow[];
  scale: ChartScale;
  trailingLabel?: string | undefined;
  tooltip: (row: ChartRow) => ReactNode;
  label: string;
  footer?: ReactNode;
}) {
  const source = useId();
  const [box, width] = useWidth();
  const scroller = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [anchor, setAnchor] = useState<Anchor | null>(null);
  const focus = useKernelFocus(scope);

  const narrow = width > 0 && width < NARROW;
  const barCount = rows[0]?.bars.length ?? 1;
  const { thick, bars, rowH } = geometry(barCount, narrow);
  const hasTrailing = trailingLabel !== undefined;
  const gutter = narrow ? 0 : Math.round(Math.min(300, Math.max(168, width * 0.3)));
  const trailW = hasTrailing ? (narrow ? 0 : 76) : 0;
  const plotWidth = Math.max(0, width - gutter - trailW - (narrow ? 0 : 24));
  const ax = axis(scale, plotWidth);
  const x = (v: number) => percentAlong(v, ax.lo, ax.hi);
  const zero = x(0);

  const total = rows.length * rowH;
  const viewH = Math.min(total, MAX_VIEW);
  const first = Math.max(0, Math.floor(scrollTop / rowH) - OVERSCAN);
  const last = Math.min(rows.length, Math.ceil((scrollTop + viewH) / rowH) + OVERSCAN);
  const scrollable = total > MAX_VIEW;
  const atEnd = scrollTop + viewH >= total - 2;

  const columns: CSSProperties = narrow
    ? { gridTemplateColumns: "minmax(0,1fr)" }
    : { gridTemplateColumns: `${gutter}px minmax(0,1fr)${hasTrailing ? ` ${trailW}px` : ""}`, columnGap: 12 };

  const focused = focus?.index ?? null;
  const tipRow = focus?.source === source && anchor !== null ? rows.find((r) => r.index === focused) : undefined;

  const point = (index: number, a: Anchor) => {
    setAnchor(a);
    setKernelFocus({ scope, index, source });
  };

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp" && e.key !== "Home" && e.key !== "End" && e.key !== "Escape") return;
    e.preventDefault();
    if (e.key === "Escape") {
      setKernelFocus(null);
      return;
    }
    const at = rows.findIndex((r) => r.index === focused);
    const next = e.key === "Home" ? 0 : e.key === "End" ? rows.length - 1 : at === -1 ? 0 : Math.min(rows.length - 1, Math.max(0, at + (e.key === "ArrowDown" ? 1 : -1)));
    const row = rows[next];
    const el = scroller.current;
    if (!row || !el) return;
    const top = next * rowH;
    if (top < el.scrollTop) el.scrollTop = top;
    else if (top + rowH > el.scrollTop + el.clientHeight) el.scrollTop = top + rowH - el.clientHeight;
    const rect = el.getBoundingClientRect();
    point(row.index, { x: rect.left + gutter + plotWidth * 0.5, y: rect.top + Math.min(top - el.scrollTop, el.clientHeight - rowH) + rowH });
  };

  return (
    <div ref={box} className="min-w-0">
      {width > 0 && (
        <>
          <div className="grid h-6 items-end border-b border-border pb-1.5 text-[11px] leading-4 text-muted" style={columns}>
            {!narrow && <span className="pl-9">Kernel</span>}
            <div className="relative h-4 tabular-nums">
              {ax.ticks.map((t, i) => (
                <span
                  key={t}
                  className="absolute top-0 whitespace-nowrap"
                  style={{ left: `${x(t)}%`, transform: i === 0 ? "none" : i === ax.ticks.length - 1 ? "translateX(-100%)" : "translateX(-50%)" }}
                >
                  {ax.label(t)}
                </span>
              ))}
            </div>
            {hasTrailing && !narrow && <span className="text-right">{trailingLabel}</span>}
          </div>
          <div
            ref={scroller}
            role="list"
            aria-label={label}
            tabIndex={0}
            onKeyDown={onKey}
            onBlur={() => {
              if (focus?.source === source) setKernelFocus(null);
            }}
            onScroll={(e) => {
              setScrollTop(e.currentTarget.scrollTop);
            }}
            onPointerLeave={() => {
              setAnchor(null);
              if (focus?.source === source) setKernelFocus(null);
            }}
            className="relative overflow-y-auto overscroll-contain outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)] focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
            style={{
              height: viewH,
              maskImage: scrollable && !atEnd ? "linear-gradient(to bottom, black calc(100% - 28px), transparent)" : undefined,
            }}
          >
            <div className="relative" style={{ height: total }}>
              <div aria-hidden className="pointer-events-none absolute inset-0 grid" style={columns}>
                {!narrow && <span />}
                <div className="relative">
                  {ax.ticks.map((t) => (
                    <span key={t} className="absolute inset-y-0 w-px" style={{ left: `${x(t)}%`, background: t === 0 ? "var(--axis)" : "var(--grid)" }} />
                  ))}
                </div>
              </div>
              {rows.slice(first, last).map((row, i) => (
                <Row
                  key={row.index}
                  row={row}
                  top={(first + i) * rowH}
                  height={rowH}
                  columns={columns}
                  narrow={narrow}
                  thick={thick}
                  barsH={bars}
                  hasTrailing={hasTrailing}
                  active={row.index === focused}
                  x={x}
                  zero={zero}
                  onPoint={(a) => {
                    point(row.index, a);
                  }}
                />
              ))}
            </div>
          </div>
          {footer !== undefined && (
            <div className="grid pt-2 text-[11px] leading-4 text-muted" style={columns}>
              {!narrow && <span />}
              {footer}
            </div>
          )}
          {tipRow && anchor && <Tooltip anchor={anchor}>{tooltip(tipRow)}</Tooltip>}
        </>
      )}
    </div>
  );
}

function Row({
  row,
  top,
  height,
  columns,
  narrow,
  thick,
  barsH,
  hasTrailing,
  active,
  x,
  zero,
  onPoint,
}: {
  row: ChartRow;
  top: number;
  height: number;
  columns: CSSProperties;
  narrow: boolean;
  thick: number;
  barsH: number;
  hasTrailing: boolean;
  active: boolean;
  x: (v: number) => number;
  zero: number;
  onPoint: (a: Anchor) => void;
}) {
  const radius = Math.min(3, thick / 2);
  const label = (
    <span className="flex min-w-0 items-center gap-2">
      <span className="w-7 shrink-0 text-right text-[11px] text-faint tabular-nums">{row.index}</span>
      <KernelLabel name={row.name} className={active ? "text-fg" : undefined} />
    </span>
  );
  return (
    <div
      role="listitem"
      className={cx("absolute inset-x-0 grid rounded-[8px] transition-[top,background-color] duration-200 ease-out", active && "bg-surface-2/80")}
      style={{ top, height, ...columns, alignItems: narrow ? "start" : "center" }}
      onPointerEnter={(e) => {
        onPoint({ x: e.clientX, y: e.clientY });
      }}
      onPointerMove={(e) => {
        onPoint({ x: e.clientX, y: e.clientY });
      }}
    >
      {narrow ? (
        <span className="flex min-w-0 items-center justify-between gap-3 pt-2 pr-1">
          {label}
          {row.trailing !== null && <span className="shrink-0 text-[12px] text-muted tabular-nums">{row.trailing}</span>}
        </span>
      ) : (
        label
      )}
      <div className={cx("relative", narrow && "mt-1.5")} style={{ height: barsH }}>
        {row.bars.map((b, i) => {
          if (b.value === null) return null;
          const v = x(b.value);
          const negative = v < zero;
          return (
            <span
              key={b.key}
              className="absolute transition-[left,width,opacity] duration-300 ease-out"
              style={{
                top: i * (thick + 2),
                height: thick,
                left: `${Math.min(v, zero)}%`,
                width: `${Math.abs(v - zero)}%`,
                minWidth: 2,
                background: b.color,
                borderRadius: negative ? `${radius}px 0 0 ${radius}px` : `0 ${radius}px ${radius}px 0`,
                opacity: active ? 1 : 0.92,
              }}
            />
          );
        })}
        {row.marker !== null && (
          <span
            className="absolute -top-1 -bottom-1 w-[2px] -translate-x-1/2 rounded-full transition-[left] duration-300 ease-out"
            style={{ left: `${x(row.marker)}%`, background: "var(--fg-2)", boxShadow: "0 0 0 1.5px var(--surface)" }}
          />
        )}
      </div>
      {hasTrailing && !narrow && <span className="text-right text-[12px] text-muted tabular-nums">{row.trailing ?? "—"}</span>}
    </div>
  );
}

function Tooltip({ anchor, children }: { anchor: Anchor; children: ReactNode }) {
  const w = Math.min(340, window.innerWidth - 24);
  const preferLeft = anchor.x > window.innerWidth / 2;
  const left = Math.max(12, Math.min(preferLeft ? anchor.x - 16 - w : anchor.x + 16, window.innerWidth - w - 12));
  const up = anchor.y > window.innerHeight * 0.6;
  return (
    <div
      role="tooltip"
      className="pointer-events-none fixed z-50 rounded-[14px] border border-border bg-surface p-3.5 shadow-pop"
      style={{ left, width: w, top: anchor.y + (up ? -12 : 16), transform: up ? "translateY(-100%)" : undefined }}
    >
      {children}
    </div>
  );
}
