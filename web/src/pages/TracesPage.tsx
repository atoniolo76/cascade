import { useState } from "react";
import { Link, useNavigate } from "react-router";
import { Check, CircleDot, Clock, Cpu, HardDrive, Hash, Layers, Send, Timer } from "lucide-react";
import { useTraces } from "../api/queries";
import type { TraceSummary } from "../api/types";
import { PageHeader } from "../components/AppShell";
import { QuickLaunch } from "../components/QuickLaunch";
import { SendTraceGuide } from "../components/SendTraceGuide";
import { TraceIcon, TraceState } from "../components/TraceState";
import { Button, Card, Dash, ErrorNote, Panel, RelTime, SectionTitle, Skeleton, Th } from "../components/ui";
import { ICON, ROW, THEAD } from "../components/styles";
import { formatBytes, formatInt } from "../domain/format";
import { paths } from "../paths";

const CELL = "h-[60px] px-4 align-middle";

export function TracesPage() {
  const traces = useTraces();
  const [guideOpen, setGuideOpen] = useState(false);
  const empty = traces.data?.length === 0;

  return (
    <>
      <PageHeader
        crumbs={[{ label: "Traces" }]}
        status={traces.data !== undefined && traces.data.length > 0 ? <span className="text-[15px] text-muted tabular-nums">{formatInt(traces.data.length)}</span> : undefined}
        actions={
          !empty && (
            <Button
              aria-expanded={guideOpen}
              onClick={() => {
                setGuideOpen((o) => !o);
              }}
            >
              <Send className="size-4" {...ICON} />
              Send a trace
            </Button>
          )
        }
      >
        <p className="max-w-2xl text-[15px] text-muted">Kernel traces captured on real hardware. Replay any ready trace in Accel-Sim on other GPU configs.</p>
      </PageHeader>

      {(empty || guideOpen) && (
        <Card className="mb-8 p-5 sm:p-7">
          <div className="mb-6">
            <SectionTitle title={empty ? "Send your first trace" : "Send a trace"} subtitle="The profiler streams kernel files into this app's bucket while it traces." />
          </div>
          <SendTraceGuide waiting={empty} />
        </Card>
      )}

      {traces.isError && traces.data === undefined ? (
        <ErrorNote error={traces.error} />
      ) : traces.data === undefined ? (
        <Panel className="overflow-hidden">
          <SkeletonRows />
        </Panel>
      ) : traces.data.length > 0 ? (
        <Panel className="overflow-hidden">
          <TraceTable traces={traces.data} />
        </Panel>
      ) : null}
    </>
  );
}

function TraceName({ t }: { t: TraceSummary }) {
  return (
    <span className="flex min-w-0 items-center gap-3">
      <TraceIcon />
      <span className="min-w-0">
        <span className="block truncate text-[14px] font-medium text-fg" title={t.id}>
          {t.model ?? t.id}
        </span>
        <span className="block truncate font-mono text-[12px] leading-4 text-muted" title={t.layers.join(", ")}>
          {t.layers.length > 0 ? t.layers.join(", ") : "layers pending"}
        </span>
      </span>
    </span>
  );
}

function TraceTable({ traces }: { traces: TraceSummary[] }) {
  const navigate = useNavigate();
  return (
    <>
      <table className="hidden w-full table-fixed border-collapse lg:table">
        <colgroup>
          <col />
          <col className="w-[200px]" />
          <col className="w-[200px]" />
          <col className="w-[96px]" />
          <col className="w-[104px]" />
          <col className="w-[104px]" />
          <col className="w-[108px]" />
          <col className="w-[88px]" />
        </colgroup>
        <thead className={THEAD}>
          <tr>
            <Th icon={<Layers {...ICON} />}>Trace</Th>
            <Th icon={<CircleDot {...ICON} />}>State</Th>
            <Th icon={<Cpu {...ICON} />}>Source GPU</Th>
            <Th icon={<Hash {...ICON} />} className="text-right">
              Kernels
            </Th>
            <Th icon={<HardDrive {...ICON} />} className="text-right">
              Size
            </Th>
            <Th icon={<Timer {...ICON} />}>Real time</Th>
            <Th icon={<Clock {...ICON} />}>Created</Th>
            <Th />
          </tr>
        </thead>
        <tbody>
          {traces.map((t) => (
            <tr
              key={t.id}
              onClick={() => {
                void navigate(paths.trace(t.id));
              }}
              className={`${ROW} cursor-pointer hover:bg-surface-2/50`}
            >
              <td className={CELL}>
                <Link
                  to={paths.trace(t.id)}
                  className="block min-w-0"
                  onClick={(e) => {
                    e.stopPropagation();
                  }}
                >
                  <TraceName t={t} />
                </Link>
              </td>
              <td className={CELL}>
                <TraceState trace={t} />
              </td>
              <td className={CELL}>
                <span className="block truncate text-[14px] text-fg-2" title={t.source_gpu ?? undefined}>
                  {t.source_gpu ?? <Dash />}
                </span>
                <span className="block font-mono text-[12px] leading-4 text-muted">{t.trace_gpu_arch}</span>
              </td>
              <td className={`${CELL} text-right text-[14px] text-fg-2 tabular-nums`}>{formatInt(t.kernel_count)}</td>
              <td className={`${CELL} text-right text-[14px] text-fg-2 tabular-nums`}>{formatBytes(t.bytes)}</td>
              <td className={CELL}>
                <RealTiming has={t.has_real_timing} />
              </td>
              <td className={`${CELL} text-[14px] text-muted`}>
                <RelTime iso={t.created_at} />
              </td>
              <td className={`${CELL} text-right`}>{t.state === "ready" && <QuickLaunch trace={t} size="sm" variant="secondary" />}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <ul className="divide-y divide-border lg:hidden">
        {traces.map((t) => (
          <li key={t.id} className="flex items-start gap-3 px-4 py-4">
            <Link to={paths.trace(t.id)} className="flex min-w-0 flex-1 flex-col gap-2.5">
              <TraceName t={t} />
              <div className="pl-9">
                <TraceState trace={t} />
              </div>
              <div className="flex flex-wrap gap-x-3 gap-y-1 pl-9 text-[13px] text-muted">
                {t.source_gpu !== null && <span className="text-fg-2">{t.source_gpu}</span>}
                <span className="tabular-nums">{formatInt(t.kernel_count)} kernels</span>
                <span className="tabular-nums">{formatBytes(t.bytes)}</span>
                <RelTime iso={t.created_at} />
              </div>
            </Link>
            {t.state === "ready" && <QuickLaunch trace={t} size="sm" variant="secondary" />}
          </li>
        ))}
      </ul>
    </>
  );
}

function RealTiming({ has }: { has: boolean }) {
  return has ? (
    <span className="inline-flex items-center gap-1.5 text-[14px] text-fg-2">
      <Check className="size-4" style={{ color: "var(--st-ok-dot)" }} strokeWidth={2} />
      Captured
    </span>
  ) : (
    <span className="text-[14px] text-faint">None</span>
  );
}

function SkeletonRows() {
  return (
    <div className="divide-y divide-border">
      <div className="h-11 bg-surface-2/60" />
      {Array.from({ length: 5 }, (_, i) => (
        <div key={i} className="flex h-[60px] items-center gap-6 px-4">
          <Skeleton className="size-6 rounded-[7px]" />
          <Skeleton className="h-3 w-[30%]" />
          <Skeleton className="h-3 w-20" />
          <Skeleton className="ml-auto h-3 w-12" />
        </div>
      ))}
    </div>
  );
}
