import { Scale, Timer } from "lucide-react";
import { useComparison } from "../../api/queries";
import { formatSigned } from "../../domain/format";
import { REAL_SERIES, comparisonRows, relativeError, type KernelRow, type Series } from "../../domain/stats";
import { CHART_BLOCK_HEIGHT, KernelBarChart, Legend } from "../charts/KernelBarChart";
import { EmptyState, ErrorNote, Panel, PanelHeader, Skeleton } from "../ui";
import { ICON } from "../styles";
import { CsvButton } from "./SimTimePanel";
import { KernelValueTable } from "./KernelTable";

function ErrorPct({ row, s }: { row: KernelRow; s: Series }) {
  if (s.key === REAL_SERIES.key) return null;
  const err = relativeError(row.values[s.key], row.values[REAL_SERIES.key] ?? null);
  if (err === null) return null;
  return <span className="ml-1.5 inline-block w-[52px] text-right text-[12px] text-muted tabular-nums">{formatSigned(err)}</span>;
}

function meanAbsError(rows: readonly KernelRow[], key: string): { mae: number; n: number } | null {
  const errs = rows.flatMap((r) => {
    const e = relativeError(r.values[key], r.values[REAL_SERIES.key] ?? null);
    return e === null ? [] : [Math.abs(e)];
  });
  if (errs.length === 0) return null;
  return { mae: errs.reduce((a, b) => a + b, 0) / errs.length, n: errs.length };
}

export function ComparisonPanel({ runId, live, series }: { runId: string; live: boolean; series: readonly Series[] }) {
  const comparison = useComparison(runId, { live });
  const data = comparison.data;
  const shownSeries = data ? [REAL_SERIES, ...series.filter((s) => data.configs.includes(s.key))] : [];
  const rows = data?.has_real_timing ? comparisonRows(data.rows) : [];

  return (
    <Panel className="min-w-0">
      <PanelHeader
        icon={<Scale {...ICON} />}
        title="Real vs. simulated"
        subtitle="Per-kernel time on the original hardware against each config, matched by kernel order"
        actions={data?.has_real_timing ? <CsvButton rows={rows} series={shownSeries} filename={`${runId}.real-vs-sim.csv`} /> : undefined}
      />
      <div className="p-5">
        {comparison.isError && data === undefined ? (
          <ErrorNote error={comparison.error} />
        ) : data === undefined ? (
          <Skeleton className="w-full" style={{ height: CHART_BLOCK_HEIGHT }} />
        ) : !data.has_real_timing ? (
          <EmptyState
            icon={<Timer {...ICON} />}
            title="Real hardware timing isn't captured for this trace yet"
            body="Once the profiler uploads real_timing.json next to the trace, per-kernel real vs. simulated time shows up here."
          />
        ) : (
          <ComparisonBody rows={rows} series={shownSeries} />
        )}
      </div>
    </Panel>
  );
}

function ComparisonBody({ rows, series }: { rows: KernelRow[]; series: Series[] }) {
  const configs = series.filter((s) => s.key !== REAL_SERIES.key);
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {configs.map((s) => {
          const m = meanAbsError(rows, s.key);
          return (
            <div key={s.key} className="rounded-[14px] bg-card px-4 py-3 ring-1 ring-border">
              <div className="flex items-center gap-2 text-[12px] text-muted">
                <span className="size-2.5 rounded-[3px]" style={{ background: s.color }} />
                <span className="truncate font-mono">{s.label}</span>
              </div>
              <div className="mt-1 text-[24px] leading-8 font-medium text-fg tabular-nums">{m === null ? "—" : `${(m.mae * 100).toFixed(1)}%`}</div>
              <div className="text-[12px] text-muted">{m === null ? "no kernels simulated yet" : `mean abs. error over ${m.n} kernels`}</div>
            </div>
          );
        })}
      </div>
      <Legend series={series} />
      <KernelBarChart rows={rows} series={series} detail={(row, s) => <ErrorPct row={row} s={s} />} />
      <div className="overflow-hidden rounded-[14px] border border-border">
        <KernelValueTable rows={rows} series={series} maxHeight={CHART_BLOCK_HEIGHT + 80} extra={(row, s) => <ErrorPct row={row} s={s} />} />
      </div>
    </div>
  );
}
