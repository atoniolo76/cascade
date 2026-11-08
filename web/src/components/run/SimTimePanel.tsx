import { useState } from "react";
import { BarChart3, FileSpreadsheet, Timer } from "lucide-react";
import { saveText } from "../../api/client";
import type { SimSummary } from "../../api/types";
import type { KernelsBySim } from "../../domain/runLive";
import { kernelRowsCsv, simTimeRows, type KernelRow, type Series } from "../../domain/stats";
import { CHART_BLOCK_HEIGHT, KernelBarChart, Legend } from "../charts/KernelBarChart";
import { Button, EmptyState, Panel, PanelHeader } from "../ui";
import { ICON } from "../styles";
import { KernelValueTable } from "./KernelTable";
import { SegmentedToggle } from "./SegmentedToggle";

type View = "chart" | "table";
const VIEWS = [
  { value: "chart", label: "Chart" },
  { value: "table", label: "Table" },
] as const;

export function CsvButton({ rows, series, filename }: { rows: readonly KernelRow[]; series: readonly Series[]; filename: string }) {
  return (
    <Button
      size="sm"
      disabled={rows.length === 0}
      title="Download this table as CSV"
      onClick={() => {
        saveText({ text: kernelRowsCsv(rows, series), filename, type: "text/csv" });
      }}
    >
      <FileSpreadsheet className="size-3.5" {...ICON} />
      CSV
    </Button>
  );
}

export function SimTimePanel({ runId, sims, kernels, series }: { runId: string; sims: readonly SimSummary[]; kernels: KernelsBySim; series: readonly Series[] }) {
  const [view, setView] = useState<View>("chart");
  const rows = simTimeRows({ sims, kernels, configs: series.map((s) => s.key) });
  return (
    <Panel className="min-w-0">
      <PanelHeader
        icon={<Timer {...ICON} />}
        title="Simulated time per kernel"
        subtitle="Median across replicas · gpu_sim_cycle ÷ core clock"
        actions={
          <>
            <CsvButton rows={rows} series={series} filename={`${runId}.sim-time.csv`} />
            <SegmentedToggle value={view} options={VIEWS} onChange={setView} />
          </>
        }
      />
      <div className="flex flex-col gap-3 p-5">
        <Legend series={series} />
        {rows.length === 0 ? (
          <div style={{ height: CHART_BLOCK_HEIGHT }} className="grid place-items-center rounded-[14px] bg-card">
            <EmptyState icon={<BarChart3 {...ICON} />} title="No kernels simulated yet" body="Bars appear as each sim finishes a kernel." />
          </div>
        ) : view === "chart" ? (
          <KernelBarChart rows={rows} series={series} />
        ) : (
          <KernelValueTable rows={rows} series={series} maxHeight={CHART_BLOCK_HEIGHT} />
        )}
      </div>
    </Panel>
  );
}
