import type { ReactNode } from "react";
import { Footprints } from "lucide-react";
import { useStepTimelines } from "../../api/queries";
import { relativeError } from "../../domain/accuracy";
import { formatInt, formatNs, formatPercent, plural } from "../../domain/format";
import type { Series } from "../../domain/stats";
import { stepTimingByConfig, type ConfigStepTiming, type Predicted } from "../../domain/stepTiming";
import { ErrorPill } from "../charts/ChartParts";
import { Panel, PanelHeader } from "../ui";
import { ICON } from "../styles";

const KIND_LABEL = { prefill: "Prefill wall", decode: "Decode wall" } as const;

/** Shown only once a sim has posted step timing, which needs the trace's corrected timeline and kernel mapping. */
export function StepTimingPanel({ runId, live, series }: { runId: string; live: boolean; series: readonly Series[] }) {
  const configs = stepTimingByConfig(useStepTimelines(runId, { live }).data ?? []);
  if (configs.length === 0) return null;
  return (
    <Panel className="min-w-0">
      <PanelHeader
        icon={<Footprints {...ICON} />}
        title="Step timing"
        subtitle="Predicted against real step wall time: simulated kernels plus the profiler's corrected host time"
      />
      <div className="grid grid-cols-[repeat(auto-fill,minmax(360px,1fr))] gap-3 p-5">
        {configs.map((c) => (
          <StepCard key={c.config} timing={c} series={series.find((s) => s.key === c.config)} />
        ))}
      </div>
    </Panel>
  );
}

function Row({ label, value, unit }: { label: string; value: Predicted; unit: (n: number) => string }) {
  return (
    <div className="grid grid-cols-[96px_minmax(0,1fr)_auto] items-baseline gap-x-3">
      <dt className="text-muted">{label}</dt>
      <dd className="flex min-w-0 items-baseline gap-1.5 tabular-nums">
        <span className="shrink-0 text-fg-2">{unit(value.predicted)}</span>
        {value.real !== null && <span className="truncate text-muted">vs. {unit(value.real)} real</span>}
      </dd>
      <dd>
        <ErrorPill error={relativeError(value.predicted, value.real)} />
      </dd>
    </div>
  );
}

function StepCard({ timing, series }: { timing: ConfigStepTiming; series: Series | undefined }) {
  const { stepError, coverage, ttft } = timing;
  let headline: ReactNode = <p className="mt-2 text-[13px] text-muted">No real steps to compare against.</p>;
  if (stepError !== null) {
    headline = (
      <div className="mt-2 flex items-baseline gap-2">
        <span className="text-[26px] leading-8 font-medium tracking-tight text-fg">{(stepError.mape * 100).toFixed(1)}%</span>
        <span className="text-[12px] text-muted">mean abs. step error · {plural(stepError.steps, "step")}</span>
      </div>
    );
  }
  return (
    <div className="min-w-0 rounded-[14px] bg-card px-4 py-3.5 ring-1 ring-border">
      <div className="flex items-center gap-2 text-[12px] text-fg-2">
        <span aria-hidden className="h-[7px] w-3 shrink-0 rounded-[2px]" style={{ background: series?.color ?? "var(--series-ref)" }} />
        <span className="truncate font-mono">{series?.label ?? timing.config}</span>
        {timing.replicas > 1 && (
          <span className="shrink-0 text-muted" title="Figures come from the lowest replica; every replica replays the same trace">
            · {timing.replicas} replicas
          </span>
        )}
      </div>
      {headline}
      <dl className="mt-3 flex flex-col gap-1 text-[12px] leading-5">
        {timing.kinds.map((k) => (
          <Row key={k.kind} label={KIND_LABEL[k.kind]} value={k} unit={formatNs} />
        ))}
        {ttft !== null && <Row label="TTFT" value={ttft} unit={(msValue) => formatNs(msValue * 1e6)} />}
      </dl>
      {coverage !== null && (
        <p className="mt-3 border-t border-border pt-2.5 text-[12px] leading-5 text-muted tabular-nums" title="Kernels without a simulated time keep their measured duration">
          Simulated {formatInt(coverage.simulatedKernels)} of {formatInt(coverage.kernels)} kernels ({formatPercent(coverage.kernelFraction, 1)}), {formatPercent(coverage.gpuTimeFraction, 1)} of real GPU time
        </p>
      )}
    </div>
  );
}
