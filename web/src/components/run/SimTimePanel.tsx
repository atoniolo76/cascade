import { useState } from "react";
import { BarChart3 } from "lucide-react";
import type { SimSummary } from "../../api/types";
import type { KernelsBySim } from "../../domain/runLive";
import { simTimeRows, type Series } from "../../domain/stats";
import { CHART_BLOCK_HEIGHT, KernelBarChart, Legend } from "../charts/KernelBarChart";
import { Card, CardHeader, EmptyState } from "../ui";
import { KernelValueTable } from "./KernelTable";
import { SegmentedToggle } from "./SegmentedToggle";

type View = "chart" | "table";
const VIEWS = [
  { value: "chart", label: "Chart" },
  { value: "table", label: "Table" },
] as const;

export function SimTimePanel({ sims, kernels, series }: { sims: readonly SimSummary[]; kernels: KernelsBySim; series: readonly Series[] }) {
  const [view, setView] = useState<View>("chart");
  const rows = simTimeRows({ sims, kernels, configs: series.map((s) => s.key) });
  return (
    <Card className="min-w-0">
      <CardHeader
        title="Simulated time per kernel"
        subtitle="Median across replicas · gpu_sim_cycle ÷ core clock"
        actions={<SegmentedToggle value={view} options={VIEWS} onChange={setView} />}
      />
      <div className="flex flex-col gap-3 p-4">
        <Legend series={series} />
        {rows.length === 0 ? (
          <div style={{ height: CHART_BLOCK_HEIGHT }} className="grid place-items-center rounded-lg bg-surface-2/50">
            <EmptyState icon={<BarChart3 className="size-5" />} title="No kernels simulated yet" body="Bars appear as each sim finishes a kernel." />
          </div>
        ) : view === "chart" ? (
          <KernelBarChart rows={rows} series={series} />
        ) : (
          <KernelValueTable rows={rows} series={series} maxHeight={CHART_BLOCK_HEIGHT} />
        )}
      </div>
    </Card>
  );
}
