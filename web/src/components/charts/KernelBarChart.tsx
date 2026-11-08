import type { ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatNs } from "../../domain/format";
import type { KernelRow, Series } from "../../domain/stats";

const PLOT_HEIGHT = 260;
const BAR_PX = 12;

export function Legend({ series }: { series: readonly Series[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1">
      {series.map((s) => (
        <li key={s.key} className="inline-flex items-center gap-1.5 text-[12px] text-fg-2">
          <span className="size-2.5 rounded-[3px]" style={{ background: s.color }} />
          <span className="font-mono">{s.label}</span>
        </li>
      ))}
    </ul>
  );
}

/** Grouped columns, one group per kernel; scrolls sideways once groups would get thinner than their bars. */
export function KernelBarChart({
  rows,
  series,
  detail,
}: {
  rows: readonly KernelRow[];
  series: readonly Series[];
  detail?: (row: KernelRow, s: Series) => ReactNode;
}) {
  const groupPx = series.length * (BAR_PX + 2) + 14;
  const minWidth = rows.length * groupPx + 72;
  const data = rows.map((r) => ({ index: r.index, ...r.values }));
  const byIndex = new Map(rows.map((r) => [r.index, r]));

  return (
    <div className="overflow-x-auto overscroll-x-contain">
      <div style={{ minWidth, height: PLOT_HEIGHT + 28 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barGap={2} barCategoryGap="22%">
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis
              dataKey="index"
              tickLine={false}
              axisLine={{ stroke: "var(--axis)" }}
              tick={{ fill: "var(--muted)", fontSize: 11 }}
              tickFormatter={(v: number) => `#${v}`}
              interval="preserveStartEnd"
              minTickGap={8}
              height={28}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tick={{ fill: "var(--muted)", fontSize: 11 }}
              tickFormatter={(v: number) => (v === 0 ? "0" : formatNs(v))}
              width={64}
            />
            <Tooltip
              cursor={{ fill: "var(--surface-2)", opacity: 0.7 }}
              isAnimationActive={false}
              wrapperStyle={{ outline: "none", zIndex: 20 }}
              content={({ active, label }) => {
                const row = active && typeof label === "number" ? byIndex.get(label) : undefined;
                return row ? <ChartTooltip row={row} series={series} detail={detail} /> : null;
              }}
            />
            {series.map((s) => (
              <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ChartTooltip({
  row,
  series,
  detail,
}: {
  row: KernelRow;
  series: readonly Series[];
  detail: ((row: KernelRow, s: Series) => ReactNode) | undefined;
}) {
  return (
    <div className="w-[300px] rounded-lg border border-border bg-surface p-3 shadow-pop">
      <div className="text-[11px] font-medium text-muted tabular-nums">Kernel #{row.index}</div>
      <div className="mb-2 truncate font-mono text-[11px] text-fg-2">{row.name}</div>
      <ul className="flex flex-col gap-1">
        {series.map((s) => {
          const v = row.values[s.key];
          return (
            <li key={s.key} className="flex items-center gap-2 text-[12px]">
              <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: s.color }} />
              <span className="font-semibold tabular-nums">{v === null || v === undefined ? "—" : formatNs(v)}</span>
              <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-muted">{s.label}</span>
              {detail?.(row, s)}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export const CHART_BLOCK_HEIGHT = PLOT_HEIGHT + 28;
