import { useState } from "react";
import { BarChart3, FileSpreadsheet, Timer } from "lucide-react";
import { saveText } from "../../api/client";
import { useComparison } from "../../api/queries";
import type { Comparison, SimSummary } from "../../api/types";
import { relativeError } from "../../domain/accuracy";
import { formatNs, plural } from "../../domain/format";
import type { KernelsBySim } from "../../domain/runLive";
import { kernelRowsCsv, simTimeRows, type KernelRow, type Series } from "../../domain/stats";
import { ChartLegend, ErrorPill, TooltipLine, type LegendItem } from "../charts/ChartParts";
import { KernelHeading } from "../charts/KernelLabel";
import { KernelRowsChart, type ChartRow } from "../charts/KernelRowsChart";
import { Button, EmptyState, Panel, PanelHeader } from "../ui";
import { ICON } from "../styles";
import { BarCell, KernelTable, type KernelColumn } from "./KernelTable";
import { SegmentedToggle } from "./SegmentedToggle";
import { REAL, TABLE_MAX_HEIGHT, VIEWS, useHiddenKeys, type View } from "../charts/panelState";

type Order = "launch" | "slowest";
const ORDERS = [
  { value: "launch", label: "Launch order" },
  { value: "slowest", label: "Slowest first" },
] as const;

export function CsvButton({ disabled, filename, csv }: { disabled: boolean; filename: string; csv: () => string }) {
  return (
    <Button
      size="sm"
      disabled={disabled}
      title="Download as CSV"
      onClick={() => {
        saveText({ text: csv(), filename, type: "text/csv" });
      }}
    >
      <FileSpreadsheet className="size-3.5" {...ICON} />
      CSV
    </Button>
  );
}

function realTimes(data: Comparison | undefined): ReadonlyMap<number, number> {
  const m = new Map<number, number>();
  if (data?.has_real_timing) for (const r of data.rows) if (r.real_ns !== null) m.set(r.kernel_index, r.real_ns);
  return m;
}

export function SimTimePanel({ runId, sims, kernels, series }: { runId: string; sims: readonly SimSummary[]; kernels: KernelsBySim; series: readonly Series[] }) {
  const [view, setView] = useState<View>("chart");
  const [order, setOrder] = useState<Order>("launch");
  const [hidden, toggle] = useHiddenKeys();
  const real = realTimes(useComparison(runId, { live: false }).data);
  const rows = simTimeRows({ sims, kernels, configs: series.map((s) => s.key) });

  const visible = series.filter((s) => !hidden.has(s.key));
  const showReal = real.size > 0 && !hidden.has(REAL.key);
  const peak = (r: KernelRow) => Math.max(...visible.map((s) => r.values[s.key] ?? -1));
  const sorted = order === "launch" ? rows : rows.toSorted((a, b) => peak(b) - peak(a));
  const max = Math.max(0, ...rows.map(peak), ...(showReal ? sorted.map((r) => real.get(r.index) ?? 0) : []));

  const legend: LegendItem[] = [
    ...series.map((s) => ({ key: s.key, label: s.label, color: s.color, glyph: "bar" as const, mono: true })),
    ...(real.size > 0 ? [{ key: REAL.key, label: REAL.label, color: REAL.color, glyph: "tick" as const, mono: false }] : []),
  ];

  const chartRows: ChartRow[] = sorted.map((r) => {
    const only = visible.length === 1 ? (r.values[visible[0]?.key ?? ""] ?? null) : null;
    return {
      index: r.index,
      name: r.name,
      bars: visible.map((s) => ({ key: s.key, color: s.color, value: r.values[s.key] ?? null })),
      marker: showReal ? (real.get(r.index) ?? null) : null,
      trailing: only === null ? null : formatNs(only),
    };
  });

  const columns: KernelColumn<KernelRow>[] = [
    ...(showReal ? [{ key: REAL.key, width: 112, header: REAL.label, cell: (r: KernelRow) => (real.has(r.index) ? formatNs(real.get(r.index) ?? 0) : "—") }] : []),
    ...visible.map((s) => ({
      key: s.key,
      width: 132,
      header: <span className="font-mono">{s.label}</span>,
      cell: (r: KernelRow) => {
        const v = r.values[s.key] ?? null;
        return <BarCell text={v === null ? <span className="text-faint">—</span> : formatNs(v)} fraction={v === null || max === 0 ? null : v / max} color={s.color} />;
      },
    })),
  ];

  return (
    <Panel className="min-w-0">
      <PanelHeader
        icon={<Timer {...ICON} />}
        title="Simulated time per kernel"
        subtitle="Median across replicas · gpu_sim_cycle ÷ core clock"
        actions={
          <>
            <CsvButton disabled={rows.length === 0} filename={`${runId}.sim-time.csv`} csv={() => kernelRowsCsv(rows, series)} />
            <SegmentedToggle value={view} options={VIEWS} onChange={setView} label="View" />
          </>
        }
      />
      {rows.length === 0 ? (
        <EmptyState icon={<BarChart3 {...ICON} />} title="No kernels simulated yet" body="Each kernel shows up here as soon as a sim finishes it." />
      ) : (
        <div className="flex flex-col gap-4 p-5">
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
            <ChartLegend items={legend} hidden={hidden} onToggle={toggle} />
            <span className="flex items-center gap-3">
              <span className="hidden text-[12px] whitespace-nowrap text-muted tabular-nums sm:inline">{plural(rows.length, "kernel")}</span>
              <SegmentedToggle size="sm" value={order} options={ORDERS} onChange={setOrder} label="Sort kernels" />
            </span>
          </div>
          {view === "chart" ? (
            <KernelRowsChart
              scope={runId}
              rows={chartRows}
              scale={{ kind: "time", max }}
              trailingLabel={visible.length === 1 ? "Time" : undefined}
              label="Simulated time per kernel"
              tooltip={(row) => (
                <>
                  <KernelHeading index={row.index} name={row.name} />
                  <ul className="flex flex-col gap-1">
                    {row.bars.map((b) => (
                      <TooltipLine
                        key={b.key}
                        color={b.color}
                        glyph="bar"
                        value={b.value === null ? "—" : formatNs(b.value)}
                        label={b.key}
                        aside={real.size > 0 ? <ErrorPill error={relativeError(b.value, real.get(row.index) ?? null)} /> : undefined}
                      />
                    ))}
                    {real.has(row.index) && <TooltipLine color={REAL.color} glyph="tick" value={formatNs(real.get(row.index) ?? 0)} label={REAL.label} mono={false} />}
                  </ul>
                </>
              )}
            />
          ) : (
            <KernelTable rows={sorted} columns={columns} maxHeight={TABLE_MAX_HEIGHT} />
          )}
        </div>
      )}
    </Panel>
  );
}
