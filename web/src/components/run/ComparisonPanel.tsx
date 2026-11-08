import { useState, type ReactNode } from "react";
import { Scale, Timer } from "lucide-react";
import { useComparison } from "../../api/queries";
import type { ComparisonRow } from "../../api/types";
import { configAccuracy, errorRange, relativeError, type ConfigAccuracy } from "../../domain/accuracy";
import { formatNs, formatSigned, plural } from "../../domain/format";
import { kernelDisplayName } from "../../domain/kernelName";
import { comparisonCsv, type Series } from "../../domain/stats";
import { ChartLegend, ErrorPill, TooltipLine, type LegendItem } from "../charts/ChartParts";
import { KernelHeading } from "../charts/KernelLabel";
import { KernelRowsChart, type ChartRow } from "../charts/KernelRowsChart";
import { EmptyState, ErrorNote, Panel, PanelHeader, Skeleton } from "../ui";
import { ICON } from "../styles";
import { KernelTable, type KernelColumn } from "./KernelTable";
import { SegmentedToggle } from "./SegmentedToggle";
import { REAL, TABLE_MAX_HEIGHT, VIEWS, useHiddenKeys, type View } from "../charts/panelState";
import { CsvButton } from "./SimTimePanel";

type Order = "launch" | "error" | "slowest";
const ORDERS = [
  { value: "launch", label: "Launch order" },
  { value: "error", label: "Largest error" },
  { value: "slowest", label: "Slowest" },
] as const;

interface Row {
  index: number;
  name: string;
  real: number | null;
  sim: Record<string, number | null>;
}

const toRow = (r: ComparisonRow): Row => ({ index: r.kernel_index, name: r.kernel_name, real: r.real_ns, sim: r.sim_ns });

export function ComparisonPanel({ runId, live, series }: { runId: string; live: boolean; series: readonly Series[] }) {
  const comparison = useComparison(runId, { live });
  const [view, setView] = useState<View>("chart");
  const data = comparison.data;
  const configs = data ? series.filter((s) => data.configs.includes(s.key)) : [];
  const ready = data?.has_real_timing === true;

  return (
    <Panel className="min-w-0">
      <PanelHeader
        icon={<Scale {...ICON} />}
        title="Real vs. simulated"
        subtitle="Per-kernel error against the original hardware, matched by kernel order"
        actions={
          ready ? (
            <>
              <ComparisonCsvButton rows={data.rows} configs={configs} runId={runId} />
              <SegmentedToggle value={view} options={VIEWS} onChange={setView} label="View" />
            </>
          ) : undefined
        }
      />
      {comparison.isError && data === undefined ? (
        <div className="p-5">
          <ErrorNote error={comparison.error} />
        </div>
      ) : data === undefined ? (
        <ComparisonSkeleton />
      ) : !data.has_real_timing ? (
        <EmptyState
          icon={<Timer {...ICON} />}
          title="Real hardware timing isn't captured for this trace yet"
          body={
            <>
              Once the profiler uploads <code className="rounded-[5px] bg-surface-2 px-1 py-px font-mono text-[12.5px] text-fg-2">real_timing.json</code> next to the trace, per-kernel real vs.
              simulated time shows up here.
            </>
          }
        />
      ) : (
        <ComparisonBody scope={runId} rows={data.rows.map(toRow)} raw={data.rows} configs={configs} view={view} />
      )}
    </Panel>
  );
}

function ComparisonCsvButton({ rows, configs, runId }: { rows: readonly ComparisonRow[]; configs: readonly Series[]; runId: string }) {
  return (
    <CsvButton
      disabled={rows.length === 0}
      filename={`${runId}.real-vs-sim.csv`}
      csv={() =>
        comparisonCsv(
          rows,
          configs.map((c) => c.key),
        )
      }
    />
  );
}

function ComparisonSkeleton() {
  return (
    <div aria-hidden className="flex flex-col gap-5 p-5">
      <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-3">
        <Skeleton className="h-[132px] rounded-[14px]" />
        <Skeleton className="h-[132px] rounded-[14px]" />
      </div>
      <div className="flex flex-col gap-3">
        {[72, 54, 88, 40, 64].map((w, i) => (
          <span key={i} className="flex items-center gap-3">
            <Skeleton className="h-3 w-40 shrink-0" />
            <Skeleton className="h-2 rounded-full" style={{ width: `${w / 2}%` }} />
          </span>
        ))}
      </div>
    </div>
  );
}

function ComparisonBody({ scope, rows, raw, configs, view }: { scope: string; rows: readonly Row[]; raw: readonly ComparisonRow[]; configs: readonly Series[]; view: View }) {
  const [order, setOrder] = useState<Order>("launch");
  const [hidden, toggle] = useHiddenKeys();
  const visible = configs.filter((s) => !hidden.has(s.key));
  const err = (r: Row, key: string) => relativeError(r.sim[key], r.real);
  const worstOf = (r: Row) => Math.max(-1, ...visible.map((s) => Math.abs(err(r, s.key) ?? -1)));
  const sorted =
    order === "launch" ? rows : order === "error" ? rows.toSorted((a, b) => worstOf(b) - worstOf(a)) : rows.toSorted((a, b) => (b.real ?? -1) - (a.real ?? -1));

  const legend: LegendItem[] = configs.map((s) => ({ key: s.key, label: s.label, color: s.color, glyph: "bar", mono: true }));
  const chartRows: ChartRow[] = sorted.map((r) => ({
    index: r.index,
    name: r.name,
    bars: visible.map((s) => ({ key: s.key, color: s.color, value: err(r, s.key) })),
    marker: null,
    trailing: r.real === null ? null : formatNs(r.real),
  }));

  const columns: KernelColumn<Row>[] = [
    { key: REAL.key, width: 104, header: REAL.label, cell: (r) => (r.real === null ? <span className="text-faint">—</span> : formatNs(r.real)) },
    ...visible.map((s) => ({
      key: s.key,
      width: 176,
      header: <span className="font-mono">{s.label}</span>,
      cell: (r: Row) => {
        const v = r.sim[s.key] ?? null;
        return (
          <span className="flex items-center justify-end gap-2">
            {v === null ? <span className="text-faint">—</span> : formatNs(v)}
            <ErrorPill error={err(r, s.key)} />
          </span>
        );
      },
    })),
  ];

  return (
    <div className="flex flex-col gap-5 p-5">
      <div className="grid grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-3">
        {configs.map((s) => (
          <AccuracyCard key={s.key} series={s} accuracy={configAccuracy(raw, s.key)} />
        ))}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <ChartLegend items={legend} hidden={hidden} onToggle={toggle} />
        <span className="flex items-center gap-3">
          <span className="text-[12px] whitespace-nowrap text-muted tabular-nums">{plural(rows.length, "kernel")}</span>
          <SegmentedToggle size="sm" value={order} options={ORDERS} onChange={setOrder} label="Sort kernels" />
        </span>
      </div>
      {view === "chart" ? (
        <KernelRowsChart
          scope={scope}
          rows={chartRows}
          scale={{
            kind: "error",
            ...errorRange(
              raw,
              visible.map((s) => s.key),
            ),
          }}
          trailingLabel="Real"
          label="Simulated time error per kernel"
          footer={
            <span className="flex justify-between">
              <span>← Simulated faster</span>
              <span>Simulated slower →</span>
            </span>
          }
          tooltip={(row) => {
            const r = rows.find((x) => x.index === row.index);
            return (
              <>
                <KernelHeading index={row.index} name={row.name} />
                <ul className="flex flex-col gap-1">
                  {visible.map((s) => {
                    const v = r?.sim[s.key] ?? null;
                    return (
                      <TooltipLine key={s.key} color={s.color} glyph="bar" value={v === null ? "—" : formatNs(v)} label={s.label} aside={<ErrorPill error={r ? err(r, s.key) : null} />} />
                    );
                  })}
                  {r?.real != null && <TooltipLine color={REAL.color} glyph="tick" value={formatNs(r.real)} label={REAL.label} mono={false} />}
                </ul>
              </>
            );
          }}
        />
      ) : (
        <KernelTable rows={sorted} columns={columns} maxHeight={TABLE_MAX_HEIGHT} />
      )}
    </div>
  );
}

function bias(error: number): string {
  if (Math.abs(error) < 0.005) return "on target";
  return error < 0 ? "runs fast" : "runs slow";
}

function Stat({ label, children, title }: { label: string; children: ReactNode; title?: string }) {
  return (
    <div className="grid min-w-0 grid-cols-[76px_minmax(0,1fr)] items-baseline gap-x-3 @[600px]:grid-cols-1 @[600px]:gap-y-0.5" title={title}>
      <dt className="text-muted">{label}</dt>
      <dd className="flex min-w-0 items-baseline gap-1.5 text-fg-2 tabular-nums">{children}</dd>
    </div>
  );
}

function AccuracyCard({ series, accuracy }: { series: Series; accuracy: ConfigAccuracy | null }) {
  return (
    <div className="@container min-w-0 rounded-[14px] bg-card px-4 py-3.5 ring-1 ring-border">
      <div className="flex items-center gap-2 text-[12px] text-fg-2">
        <span aria-hidden className="h-[7px] w-3 shrink-0 rounded-[2px]" style={{ background: series.color }} />
        <span className="truncate font-mono">{series.label}</span>
      </div>
      {accuracy === null ? (
        <p className="mt-3 text-[13px] text-muted">No kernels simulated yet.</p>
      ) : (
        <div className="mt-2 flex flex-col gap-3 @[600px]:flex-row @[600px]:items-end @[600px]:gap-10">
          <div className="flex shrink-0 items-baseline gap-2">
            <span className="text-[26px] leading-8 font-medium tracking-tight text-fg">{(accuracy.mape * 100).toFixed(1)}%</span>
            <span className="text-[12px] text-muted">mean abs. error · {plural(accuracy.matched, "kernel")}</span>
          </div>
          <dl className="flex min-w-0 flex-col gap-1 text-[12px] leading-5 @[600px]:flex-1 @[600px]:flex-row @[600px]:gap-8">
            <Stat label="Bias">
              <span className="shrink-0">{formatSigned(accuracy.bias)}</span>
              <span className="truncate text-muted">sim {bias(accuracy.bias)}</span>
            </Stat>
            <Stat label="End to end" title={`${formatNs(accuracy.total.sim)} simulated vs. ${formatNs(accuracy.total.real)} real`}>
              <span className="shrink-0">{formatSigned(accuracy.total.error)}</span>
              <span className="truncate text-muted">
                {formatNs(accuracy.total.sim)} vs. {formatNs(accuracy.total.real)}
              </span>
            </Stat>
            <Stat label="Worst kernel" title={accuracy.worst.name}>
              <span className="shrink-0">{formatSigned(accuracy.worst.error)}</span>
              <span className="truncate font-mono text-[11.5px] text-muted">
                #{accuracy.worst.index} {kernelDisplayName(accuracy.worst.name)}
              </span>
            </Stat>
          </dl>
        </div>
      )}
    </div>
  );
}
