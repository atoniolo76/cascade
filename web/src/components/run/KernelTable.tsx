import type { ReactNode } from "react";
import { cx } from "../cx";
import { KernelLabel } from "../charts/KernelLabel";
import { CopyButton } from "../ui";

export interface KernelColumn<T> {
  key: string;
  header: ReactNode;
  width: number;
  cell: (row: T) => ReactNode;
}

const TH = "h-9 px-3 text-[12px] font-normal whitespace-nowrap text-muted";

/** Kernel rows with a fixed # and name column and right-aligned value columns; the name column copies the recorded name. */
export function KernelTable<T extends { index: number; name: string }>({ rows, columns, maxHeight }: { rows: readonly T[]; columns: readonly KernelColumn<T>[]; maxHeight: number }) {
  return (
    <div className="overflow-auto overscroll-contain rounded-[12px] ring-1 ring-border" style={{ maxHeight }}>
      <table className="w-full table-fixed border-collapse text-[13px]" style={{ minWidth: 320 + columns.reduce((n, c) => n + c.width, 0) }}>
        <colgroup>
          <col className="w-12" />
          <col />
          {columns.map((c) => (
            <col key={c.key} style={{ width: c.width }} />
          ))}
        </colgroup>
        <thead className="sticky top-0 z-10 bg-surface-2">
          <tr className="border-b border-border">
            <th className={cx(TH, "text-right")}>#</th>
            <th className={cx(TH, "text-left")}>Kernel</th>
            {columns.map((c) => (
              <th key={c.key} className={cx(TH, "text-right")}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.index} className="group h-10 border-b border-border transition-colors duration-150 last:border-b-0 hover:bg-surface-2/50">
              <td className="px-3 text-right text-[12px] text-faint tabular-nums">{r.index}</td>
              <td className="max-w-0 px-3">
                <span className="flex min-w-0 items-center gap-1">
                  <KernelLabel name={r.name} />
                  <CopyButton value={r.name} label="kernel name" className="size-6 opacity-0 group-hover:opacity-100 focus-visible:opacity-100" />
                </span>
              </td>
              {columns.map((c) => (
                <td key={c.key} className="px-3 text-right whitespace-nowrap text-fg tabular-nums">
                  {c.cell(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A value with a thin bar under it scaled to the column's max, so a scan down the column reads magnitude. */
export function BarCell({ text, fraction, color }: { text: ReactNode; fraction: number | null; color: string }) {
  return (
    <span className="inline-flex w-full flex-col items-end gap-1">
      <span className="leading-4">{text}</span>
      <span aria-hidden className="h-[3px] w-full max-w-[96px] overflow-hidden rounded-full bg-surface-2">
        {fraction !== null && <span className="block h-full rounded-full" style={{ width: `${Math.max(2, fraction * 100)}%`, background: color }} />}
      </span>
    </span>
  );
}
