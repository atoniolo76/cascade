import { useState, type ReactNode } from "react";
import { Link, useParams } from "react-router";
import { Activity, Braces, Hash, Layers, Settings2 } from "lucide-react";
import { useRuns, useTrace, useTraces } from "../api/queries";
import type { KernelLaunch, TraceDetail } from "../api/types";
import { PageHeader } from "../components/AppShell";
import { QuickLaunch } from "../components/QuickLaunch";
import { RunsTable } from "../components/RunsTable";
import { ModelTile, TraceState } from "../components/TraceState";
import { LogoGlyph, LogoTile } from "../components/BrandLogo";
import { gpuLogo } from "../domain/logos";
import { Card, CopyButton, EmptyState, ErrorNote, Mono, Panel, PanelHeader, RelTime, Skeleton, Stat, Th, Button } from "../components/ui";
import { ICON, THEAD, buttonClass } from "../components/styles";
import { formatBytes, formatDims, formatInt } from "../domain/format";
import { paths } from "../paths";

const PAGE = 100;
/** Shown in the summary card above, or summarized separately. */
const HIDDEN_METADATA = new Set(["files", "model", "layers", "gpu", "trace_gpu_arch", "suggested_sim_config"]);

export function TraceDetailPage() {
  const { traceId = "" } = useParams();
  const trace = useTrace(traceId);
  const traces = useTraces();
  const order = traces.data?.map((t) => t.id) ?? [];
  const at = order.indexOf(traceId);
  const neighbor = (i: number) => {
    const id = at === -1 ? undefined : order[i];
    return id === undefined ? undefined : paths.trace(id);
  };

  return (
    <>
      <PageHeader
        crumbs={[{ label: "Traces", to: paths.traces }, {
            label: (
              <span className="flex min-w-0 items-center gap-2.5">
                {trace.data && <ModelTile model={trace.data.model} fallback={trace.data.id} />}
                <span className="truncate">{trace.data?.model ?? traceId}</span>
              </span>
            ),
          }]}
        status={trace.data ? <TraceState trace={trace.data} detail={false} /> : <Skeleton className="h-5 w-20" />}
        actions={
          trace.data && (
            <>
              <Link to={paths.newRun(trace.data.id)} aria-disabled={trace.data.state !== "ready"} className={buttonClass({})}>
                <Settings2 className="size-4" {...ICON} />
                Configure run
              </Link>
              <QuickLaunch trace={trace.data} />
            </>
          )
        }
        neighbors={{ prev: neighbor(at - 1), next: neighbor(at + 1), noun: "trace" }}
      >
        <div className="flex min-w-0 items-center gap-1">
          <Mono className="text-[13px] text-muted">{traceId}</Mono>
          <CopyButton value={traceId} label="trace id" />
        </div>
      </PageHeader>
      {trace.isError && trace.data === undefined ? (
        <ErrorNote error={trace.error} />
      ) : trace.data === undefined ? (
        <DetailSkeleton />
      ) : (
        <TraceBody trace={trace.data} />
      )}
    </>
  );
}

function TraceBody({ trace }: { trace: TraceDetail }) {
  const runs = useRuns();
  const mine = runs.data?.filter((r) => r.trace_id === trace.id);
  return (
    <div className="flex flex-col gap-8">
      <Card className="grid grid-cols-2 gap-x-8 gap-y-5 p-5 sm:grid-cols-3 sm:p-6 lg:grid-cols-6">
        <Stat
          label="Source GPU"
          value={
            <span className="flex min-w-0 items-center gap-2">
              <LogoTile logo={gpuLogo(trace.source_gpu)} fallback={trace.source_gpu ?? "?"} size="sm" />
              <span className="truncate">{trace.source_gpu ?? "—"}</span>
            </span>
          }
        />
        <Stat label="Arch" value={<span className="font-mono text-[14px]">{trace.trace_gpu_arch}</span>} />
        <Stat
          label="Suggested config"
          value={
            <span className="flex items-center gap-1.5 font-mono text-[14px]">
              {(() => {
                const logo = gpuLogo(trace.suggested_sim_config);
                return logo && <LogoGlyph logo={logo} className="text-muted" />;
              })()}
              {trace.suggested_sim_config ?? "—"}
            </span>
          }
        />
        <Stat label="Kernels" value={formatInt(trace.kernel_count)} />
        <Stat label="Size" value={formatBytes(trace.bytes)} />
        <Stat label="Created" value={<RelTime iso={trace.created_at} />} />
        <Stat
          className="col-span-2 sm:col-span-3 lg:col-span-6"
          label="Layers"
          value={
            trace.layers.length > 0 ? (
              <Mono className="text-[13px] text-fg-2">{trace.layers.join(", ")}</Mono>
            ) : (
              <span className="text-[14px] font-normal text-muted">{trace.metadata === null ? "Listed once metadata.json lands" : "Not listed in metadata.json"}</span>
            )
          }
        />
      </Card>

      <Panel className="overflow-hidden">
        <PanelHeader
          icon={<Activity {...ICON} />}
          title="Runs of this trace"
          subtitle={trace.has_real_timing ? "Real hardware timing captured, so each run compares against it" : "No real hardware timing for this trace yet"}
          actions={
            mine !== undefined && mine.length > 0 ? (
              <Link to={paths.runsForTrace(trace.id)} className="text-[14px] font-medium text-link hover:underline">
                Open in Runs
              </Link>
            ) : undefined
          }
        />
        {mine === undefined ? (
          <div className="p-5">
            <Skeleton className="h-10 w-full" />
          </div>
        ) : mine.length === 0 ? (
          <EmptyState
            title="No runs yet"
            body={trace.state === "ready" ? "Run it on the suggested config, or configure several GPUs and replicas." : "Runs can start once the trace is ready."}
          />
        ) : (
          <RunsTable runs={mine} showTrace={false} />
        )}
      </Panel>

      <div className="grid gap-8 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Panel className="min-w-0 self-start overflow-hidden">
          <PanelHeader icon={<Braces {...ICON} />} title="Metadata" subtitle="From the profiler's metadata.json" />
          {trace.metadata === null ? (
            <EmptyState title="No metadata yet" body="metadata.json is uploaded last, once tracing finishes." />
          ) : (
            <MetadataList metadata={trace.metadata} />
          )}
        </Panel>
        <KernelTable kernels={trace.kernels} />
      </div>
    </div>
  );
}

function humanize(key: string) {
  const s = key.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isPrimitive(v: unknown): v is string | number | boolean | null {
  return v === null || ["string", "number", "boolean"].includes(typeof v);
}

function Primitive({ value }: { value: string | number | boolean | null }) {
  if (value === null) return <span className="text-faint">—</span>;
  if (typeof value === "boolean") return <span>{value ? "true" : "false"}</span>;
  if (typeof value === "number") return <span className="tabular-nums">{formatInt(value)}</span>;
  return <span className="break-words">{value}</span>;
}

function MetadataValue({ value }: { value: unknown }): ReactNode {
  if (isPrimitive(value)) return <Primitive value={value} />;
  if (Array.isArray(value)) {
    const items: unknown[] = value;
    const [only] = items;
    if (items.length === 1 && isPrimitive(only)) return <Primitive value={only} />;
    if (items.every((v) => typeof v === "string" && v.length < 32)) return <span className="font-mono text-[12px]">{items.join(", ")}</span>;
    return (
      <ol className="flex flex-col gap-1.5">
        {items.map((v, i) => (
          <li key={i} className="flex gap-2">
            <span className="w-4 shrink-0 text-right text-faint tabular-nums">{i + 1}</span>
            <span className="min-w-0">{isPrimitive(v) ? <Primitive value={v} /> : <Json value={v} />}</span>
          </li>
        ))}
      </ol>
    );
  }
  if (isRecord(value)) {
    return (
      <dl className="grid grid-cols-[max-content_minmax(0,1fr)] gap-x-3 gap-y-1">
        {Object.entries(value).map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="font-mono text-[12px] text-muted">{k}</dt>
            <dd className="min-w-0 font-mono text-[12px]">{isPrimitive(v) ? <Primitive value={v} /> : <Json value={v} />}</dd>
          </div>
        ))}
      </dl>
    );
  }
  return <Json value={value} />;
}

function Json({ value }: { value: unknown }) {
  return <pre className="overflow-x-auto font-mono text-[11px] text-fg-2">{JSON.stringify(value, null, 2)}</pre>;
}

function MetadataList({ metadata }: { metadata: Record<string, unknown> }) {
  const entries = Object.entries(metadata).filter(([k]) => !HIDDEN_METADATA.has(k));
  const files = metadata.files;
  return (
    <dl className="divide-y divide-border">
      {entries.map(([key, value]) => (
        <div key={key} className="grid grid-cols-1 gap-1 px-5 py-3 sm:grid-cols-[148px_minmax(0,1fr)] sm:gap-4">
          <dt className="text-[13px] leading-5 text-muted">{humanize(key)}</dt>
          <dd className="min-w-0 text-[14px] leading-5 text-fg-2">
            <MetadataValue value={value} />
          </dd>
        </div>
      ))}
      {Array.isArray(files) && (
        <div className="grid grid-cols-1 gap-1 px-5 py-3 sm:grid-cols-[148px_minmax(0,1fr)] sm:gap-4">
          <dt className="text-[13px] leading-5 text-muted">Files</dt>
          <dd className="text-[14px] leading-5 text-fg-2 tabular-nums">{formatInt(files.length)} listed</dd>
        </div>
      )}
    </dl>
  );
}

function KernelTable({ kernels }: { kernels: KernelLaunch[] }) {
  const [shown, setShown] = useState(PAGE);
  const sorted = kernels.toSorted((a, b) => a.kernel_id - b.kernel_id);
  return (
    <Panel className="min-w-0 self-start overflow-hidden">
      <PanelHeader icon={<Layers {...ICON} />} title="Kernel launches" subtitle={`${formatInt(kernels.length)} launches on the original hardware`} />
      {kernels.length === 0 ? (
        <EmptyState title="No kernels yet" body="Launch dimensions appear once the stats file is uploaded." />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full table-fixed border-collapse sm:min-w-[640px]">
            <colgroup>
              <col className="w-[52px] sm:w-[64px]" />
              <col />
              <col className="hidden w-[108px] sm:table-column" />
              <col className="hidden w-[108px] sm:table-column" />
              <col className="w-[96px] sm:w-[112px]" />
            </colgroup>
            <thead className={THEAD}>
              <tr>
                <Th icon={<Hash {...ICON} />} className="text-right">
                  ID
                </Th>
                <Th>Kernel</Th>
                <Th className="hidden sm:table-cell">Grid</Th>
                <Th className="hidden sm:table-cell">Block</Th>
                <Th className="text-right">Instructions</Th>
              </tr>
            </thead>
            <tbody className="text-[13px]">
              {sorted.slice(0, shown).map((k) => (
                <tr key={k.kernel_id} className="h-11 border-b border-border last:border-b-0">
                  <td className="px-4 text-right text-muted tabular-nums">{k.kernel_id}</td>
                  <td className="px-4 py-2 sm:py-0">
                    <Mono>{k.name}</Mono>
                    <span className="block font-mono text-[11px] text-muted sm:hidden">
                      {formatDims(k.grid)} · {formatDims(k.block)}
                    </span>
                  </td>
                  <td className="hidden px-4 font-mono text-[12px] whitespace-nowrap text-fg-2 sm:table-cell">{formatDims(k.grid)}</td>
                  <td className="hidden px-4 font-mono text-[12px] whitespace-nowrap text-fg-2 sm:table-cell">{formatDims(k.block)}</td>
                  <td className="px-4 text-right tabular-nums">{k.instructions === null ? <span className="text-faint">—</span> : formatInt(k.instructions)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {sorted.length > shown && (
            <div className="flex justify-center border-t border-border p-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setShown((n) => n + PAGE);
                }}
              >
                Show {formatInt(Math.min(PAGE, sorted.length - shown))} more of {formatInt(sorted.length - shown)}
              </Button>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}

function DetailSkeleton() {
  return (
    <div className="flex flex-col gap-6">
      <Card className="grid grid-cols-2 gap-x-6 gap-y-4 p-4 sm:grid-cols-3 lg:grid-cols-6">
        {Array.from({ length: 6 }, (_, i) => (
          <div key={i} className="flex flex-col gap-1.5">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-4 w-28" />
          </div>
        ))}
      </Card>
      <Panel className="h-80">
        <span />
      </Panel>
    </div>
  );
}
