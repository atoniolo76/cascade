import { Footprints } from "lucide-react";
import { useRuns, useSampleStepTimelines, useStepTimelines, useTrace, useTraces } from "../../api/queries";
import type { TraceSummary } from "../../api/types";
import { architectureTimingRows, latestSampleRuns, TIMING_METRICS } from "../../domain/stepTiming";
import type { Series } from "../../domain/stats";
import { ErrorNote, Panel, PanelHeader } from "../ui";
import { ICON } from "../styles";

export function StepTimingPanel({ runId, trace, live, series }: { runId: string; trace: TraceSummary; live: boolean; series: readonly Series[] }) {
  const timelines = useStepTimelines(runId, { live });
  const traces = useTraces();
  const runs = useRuns();
  const detail = useTrace(trace.id);
  const current = traces.data?.find((t) => t.id === trace.id) ?? trace;
  const sampleRuns = latestSampleRuns(current, traces.data ?? [], runs.data ?? []);
  const samples = useSampleStepTimelines(sampleRuns.map((r) => r.id));
  const sources = [{ trace: current, timelines: timelines.data ?? [] }, ...sampleRuns.flatMap((run, i) => {
    const other = traces.data?.find((t) => t.id === run.trace_id);
    return other ? [{ trace: other, timelines: samples[i]?.data ?? [] }] : [];
  })];
  const pending = detail.data?.metadata?.sample_pending_configs;
  const pendingConfigs = current.sample_set && Array.isArray(pending) ? pending.filter((c): c is string => typeof c === "string") : [];
  const rows = architectureTimingRows(current, sources, traces.data ?? [current], pendingConfigs);
  if (!(timelines.data?.length)) return timelines.isError ? <ErrorNote error={timelines.error} /> : null;
  const failed = samples.find((s) => s.isError);
  return (
    <Panel className="min-w-0">
      <PanelHeader
        icon={<Footprints {...ICON} />}
        title="Architecture timing"
        subtitle="Mean timings in ms · full requests and per-step prefill / decode · simulated kernels + corrected host time"
      />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[1060px] text-left text-[12px] tabular-nums">
          <thead className="border-b border-border text-muted">
            <tr>
              <th className="min-w-[285px] px-5 py-3 font-medium">Hardware / simulation</th>
              {TIMING_METRICS.map((m) => <th key={m.key} className="px-3 py-3 text-right font-medium whitespace-nowrap">{m.label}</th>)}
              <th className="px-5 py-3 text-right font-medium whitespace-nowrap">E2E error</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} className={row.kind === "measured" ? "border-b border-border bg-surface-2/50" : "border-b border-border last:border-0"}>
                <th scope="row" className="px-5 py-4 font-normal">
                  <div className="flex items-center gap-2 font-medium text-fg">
                    {row.kind === "simulated" && <span className="h-2 w-2 shrink-0 rounded-sm" style={{ background: series.find((s) => s.key === row.config)?.color ?? "var(--series-ref)" }} />}
                    <span className={row.kind === "measured" ? "" : "font-mono"}>{row.label}</span>
                  </div>
                  <div className="mt-1 max-w-[300px] text-[11px] leading-4 text-muted">{row.note}</div>
                </th>
                {row.kind === "pending" ? <td colSpan={8} className="px-5 py-4 text-muted">Awaiting native capture</td> : <>
                  {TIMING_METRICS.map((m) => <td key={m.key} className="px-3 py-4 text-right text-fg-2">{row.values[m.key]?.toFixed(2) ?? "—"}</td>)}
                  <td className="px-5 py-4 text-right font-medium text-fg">
                    {row.e2eError !== null ? <span className="rounded-md bg-surface-2 px-2 py-1">{row.e2eError >= 0 ? "+" : ""}{(row.e2eError * 100).toFixed(2)}%</span> : null}
                  </td>
                </>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="border-t border-border px-5 py-3 text-[12px] leading-5 text-muted">
        Error is shown only for each native trace's explicitly identified real-hardware config, never for counterfactual hardware.
        {current.sample_set && " Other native traces use their latest succeeded run and their own measured baseline."}
        {samples.some((s) => s.isPending) && " Loading other native traces…"}
        {failed && <ErrorNote error={failed.error} />}
      </div>
    </Panel>
  );
}
