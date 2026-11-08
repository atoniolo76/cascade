import { useMemo } from "react";
import { Link } from "react-router";
import { ChartGantt, GitCompareArrows } from "lucide-react";
import { useComparison } from "../../api/queries";
import type { RunDetail } from "../../api/types";
import type { KernelsBySim } from "../../domain/runLive";
import type { Series } from "../../domain/stats";
import { buildRunTracks } from "../../domain/timeline";
import { Panel, PanelHeader } from "../ui";
import { ICON, buttonClass } from "../styles";
import { paths } from "../../paths";
import { TimelineView } from "./TimelineView";
import { trackMeta } from "./meta";

export function RunTimeline({ run, kernels, series, live }: { run: RunDetail; kernels: KernelsBySim; series: readonly Series[]; live: boolean }) {
  const comparison = useComparison(run.id, { live });
  const rows = comparison.data?.rows ?? null;
  const tracks = useMemo(
    () =>
      buildRunTracks({
        run,
        trace: run.trace,
        kernels,
        real: rows,
        colorOf: (c) => series.find((s) => s.key === c)?.color ?? "var(--series-ref)",
      }),
    [run, kernels, rows, series],
  );
  const model = run.trace.model;
  return (
    <Panel className="min-w-0">
      <PanelHeader
        icon={<ChartGantt {...ICON} />}
        title="Timeline"
        subtitle="Each config's kernels end to end in simulated time, median across replicas, against real hardware when it was measured"
        actions={
          <Link to={paths.compare([run.id])} className={buttonClass({ size: "sm" })}>
            <GitCompareArrows className="size-3.5" {...ICON} />
            Compare with other runs
          </Link>
        }
      />
      <TimelineView
        showDifferences={false}
        tracks={tracks}
        linkable
        meta={(t) => trackMeta(t, model)}
        maxTrackHeight="min(440px, 60dvh)"
      />
    </Panel>
  );
}
