import { Link, useNavigate } from "react-router";
import { Check, Inbox } from "lucide-react";
import { useTraces } from "../api/queries";
import type { TraceSummary } from "../api/types";
import { PageHeader } from "../components/AppShell";
import { TraceState } from "../components/TraceState";
import { Card, EmptyState, ErrorNote, Mono, Skeleton } from "../components/ui";
import { formatAge, formatBytes, ms } from "../domain/format";
import { useNow } from "../hooks/useNow";
import { paths } from "../paths";

const HEAD = "px-3 py-2.5 text-left text-[11px] font-medium tracking-wide text-muted uppercase whitespace-nowrap";
const CELL = "px-3 h-12 align-middle";

export function TracesPage() {
  const traces = useTraces();
  return (
    <>
      <PageHeader title="Traces" subtitle="Kernel traces captured on real hardware, ready to replay in Accel-Sim." />
      {traces.isError && traces.data === undefined ? (
        <ErrorNote error={traces.error} />
      ) : (
        <Card className="overflow-hidden">
          {traces.data === undefined ? (
            <SkeletonRows />
          ) : traces.data.length === 0 ? (
            <EmptyState
              icon={<Inbox className="size-5" />}
              title="No traces yet"
              body="Point the profiler at this app's S3 endpoint. A trace shows up here as soon as its first file lands."
            />
          ) : (
            <TraceTable traces={traces.data} />
          )}
        </Card>
      )}
    </>
  );
}

function TraceTable({ traces }: { traces: TraceSummary[] }) {
  const now = useNow();
  const navigate = useNavigate();
  return (
    <>
      <table className="hidden w-full table-fixed border-collapse lg:table">
        <colgroup>
          <col />
          <col className="w-[272px]" />
          <col className="w-[16%]" />
          <col className="w-[13%]" />
          <col className="w-[148px]" />
          <col className="w-[64px]" />
          <col className="w-[84px]" />
          <col className="w-[64px]" />
          <col className="w-[60px]" />
        </colgroup>
        <thead className="border-b border-border bg-surface-2/50">
          <tr>
            <th className={HEAD}>Trace</th>
            <th className={HEAD}>State</th>
            <th className={HEAD}>Model</th>
            <th className={HEAD}>Layers</th>
            <th className={HEAD}>Source GPU</th>
            <th className={HEAD}>Arch</th>
            <th className={`${HEAD} text-right`}>Size</th>
            <th className={HEAD}>Real</th>
            <th className={`${HEAD} text-right`}>Age</th>
          </tr>
        </thead>
        <tbody>
          {traces.map((t) => (
            <tr
              key={t.id}
              onClick={() => {
                void navigate(paths.trace(t.id));
              }}
              className="cursor-pointer border-b border-border transition-colors duration-150 last:border-b-0 hover:bg-surface-2/60"
            >
              <td className={CELL}>
                <Link to={paths.trace(t.id)} className="block min-w-0 text-fg" onClick={(e) => { e.stopPropagation(); }}>
                  <Mono>{t.id}</Mono>
                </Link>
              </td>
              <td className={CELL}>
                <TraceState trace={t} />
              </td>
              <td className={CELL}>
                <span className="block truncate text-[13px]" title={t.model ?? undefined}>
                  {t.model ?? <Dash />}
                </span>
              </td>
              <td className={CELL}>
                {t.layers.length > 0 ? <Mono className="text-fg-2">{t.layers.join(", ")}</Mono> : <Dash />}
              </td>
              <td className={CELL}>
                <span className="block truncate text-[13px] text-fg-2" title={t.source_gpu ?? undefined}>
                  {t.source_gpu ?? <Dash />}
                </span>
              </td>
              <td className={CELL}>
                <span className="font-mono text-[12px] text-fg-2">{t.trace_gpu_arch}</span>
              </td>
              <td className={`${CELL} text-right text-[13px] text-fg-2 tabular-nums`}>{formatBytes(t.bytes)}</td>
              <td className={CELL}>
                <RealTiming has={t.has_real_timing} />
              </td>
              <td className={`${CELL} text-right text-[13px] text-muted tabular-nums`}>{formatAge(now - ms(t.created_at))}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <ul className="divide-y divide-border lg:hidden">
        {traces.map((t) => (
          <li key={t.id}>
            <Link to={paths.trace(t.id)} className="flex flex-col gap-2 px-4 py-3 transition-colors duration-150 hover:bg-surface-2/60">
              <div className="flex items-baseline gap-3">
                <Mono className="flex-1 text-fg">{t.id}</Mono>
                <span className="shrink-0 text-[12px] text-muted tabular-nums">{formatAge(now - ms(t.created_at))}</span>
              </div>
              <TraceState trace={t} />
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-[12px] text-muted">
                <span className="text-fg-2">{t.model ?? "unknown model"}</span>
                {t.source_gpu !== null && <span>{t.source_gpu}</span>}
                <span className="font-mono">{t.trace_gpu_arch}</span>
                <span className="tabular-nums">{formatBytes(t.bytes)}</span>
                {t.has_real_timing && <span>real timing</span>}
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </>
  );
}

function RealTiming({ has }: { has: boolean }) {
  return has ? (
    <span className="inline-flex items-center gap-1 text-[13px] text-fg-2">
      <Check className="size-3.5" style={{ color: "var(--st-ok-ink)" }} />
      Yes
    </span>
  ) : (
    <span className="text-[13px] text-faint">No</span>
  );
}

function Dash() {
  return <span className="text-faint">—</span>;
}

function SkeletonRows() {
  return (
    <div className="divide-y divide-border">
      <div className="h-10 bg-surface-2/50" />
      {Array.from({ length: 5 }, (_, i) => (
        <div key={i} className="flex h-12 items-center gap-6 px-3">
          <Skeleton className="h-3 w-[38%]" />
          <Skeleton className="h-5 w-20 rounded-full" />
          <Skeleton className="h-3 w-[14%]" />
          <Skeleton className="ml-auto h-3 w-12" />
        </div>
      ))}
    </div>
  );
}
