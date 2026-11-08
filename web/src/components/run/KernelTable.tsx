import type { ReactNode } from "react";
import { formatNs } from "../../domain/format";
import { REAL_SERIES, type KernelRow, type Series } from "../../domain/stats";
import { Mono } from "../ui";

const TH = "px-3 py-2 text-[11px] font-medium tracking-wide text-muted uppercase whitespace-nowrap";

/** Table twin of KernelBarChart: every plotted value, readable without hovering. */
export function KernelValueTable({
  rows,
  series,
  extra,
  maxHeight,
}: {
  rows: readonly KernelRow[];
  series: readonly Series[];
  extra?: (row: KernelRow, s: Series) => ReactNode;
  maxHeight: number;
}) {
  return (
    <div className="overflow-auto overscroll-contain" style={{ maxHeight }}>
      <table className="w-full table-fixed border-collapse text-[12px]" style={{ minWidth: 300 + series.length * (extra ? 150 : 104) }}>
        <colgroup>
          <col className="w-12" />
          <col />
          {series.map((s) => (
            <col key={s.key} style={{ width: s.key === REAL_SERIES.key || !extra ? 104 : 150 }} />
          ))}
        </colgroup>
        <thead className="sticky top-0 z-10 bg-surface">
          <tr className="border-b border-border">
            <th className={`${TH} text-right`}>#</th>
            <th className={`${TH} text-left`}>Kernel</th>
            {series.map((s) => (
              <th key={s.key} className={`${TH} text-right`}>
                <span className="inline-flex max-w-full items-center gap-1.5">
                  <span className="size-2 shrink-0 rounded-[2px]" style={{ background: s.color }} />
                  <span className="truncate font-mono normal-case">{s.label}</span>
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.index} className="h-9 border-b border-border last:border-b-0">
              <td className="px-3 text-right text-muted tabular-nums">{r.index}</td>
              <td className="max-w-0 px-3">
                <Mono className="text-[11px] text-fg-2">{r.name}</Mono>
              </td>
              {series.map((s) => {
                const v = r.values[s.key];
                return (
                  <td key={s.key} className="px-3 text-right whitespace-nowrap tabular-nums">
                    {v === null || v === undefined ? <span className="text-faint">—</span> : formatNs(v)}
                    {extra?.(r, s)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
