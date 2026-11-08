import { useState, type ReactNode } from "react";
import { Link, useParams } from "react-router";
import { Play } from "lucide-react";
import { useTrace } from "../api/queries";
import type { KernelLaunch, TraceDetail } from "../api/types";
import { PageHeader } from "../components/AppShell";
import { TraceState } from "../components/TraceState";
import { Button, Card, CardHeader, EmptyState, ErrorNote, Mono, Skeleton, Stat } from "../components/ui";
import { formatAge, formatBytes, formatDims, formatInt, ms } from "../domain/format";
import { useNow } from "../hooks/useNow";
import { paths } from "../paths";

const PAGE = 100;
/** Shown in the summary card above, or summarized separately. */
const HIDDEN_METADATA = new Set(["files", "model", "layers", "gpu", "trace_gpu_arch", "suggested_sim_config"]);

export function TraceDetailPage() {
  const { traceId = "" } = useParams();
  const trace = useTrace(traceId);

  return (
    <>
      <PageHeader
        eyebrow={
          <Link to={paths.traces} className="transition-colors duration-150 hover:text-fg">
            Traces
          </Link>
        }
        title={<span className="font-mono text-[16px] leading-6 font-medium break-all">{traceId}</span>}
        subtitle={trace.data ? <TraceState trace={trace.data} /> : <Skeleton className="h-5 w-24 rounded-full" />}
        actions={<NewRunAction trace={trace.data} />}
      />
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

function NewRunAction({ trace }: { trace: TraceDetail | undefined }) {
  const ready = trace?.state === "ready";
  if (!ready) {
    return (
      <span title={trace ? `Trace is ${trace.state}; runs need a ready trace` : undefined}>
        <Button variant="primary" disabled>
          <Play className="size-3.5" />
          New run
        </Button>
      </span>
    );
  }
  return (
    <Link
      to={paths.newRun(trace.id)}
      className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-accent px-3 text-[13px] font-medium text-accent-fg shadow-card transition-colors duration-150 hover:bg-accent-hover"
    >
      <Play className="size-3.5" />
      New run
    </Link>
  );
}

function TraceBody({ trace }: { trace: TraceDetail }) {
  const now = useNow();
  return (
    <div className="flex flex-col gap-6">
      <Card className="grid grid-cols-2 gap-x-6 gap-y-4 p-4 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Model" value={trace.model ?? "—"} />
        <Stat label="Source GPU" value={trace.source_gpu ?? "—"} />
        <Stat label="Arch" value={<span className="font-mono text-[12px]">{trace.trace_gpu_arch}</span>} />
        <Stat label="Suggested config" value={<span className="font-mono text-[12px]">{trace.suggested_sim_config ?? "—"}</span>} />
        <Stat label="Kernels · size" value={`${formatInt(trace.kernel_count)} · ${formatBytes(trace.bytes)}`} />
        <Stat label="Real timing · age" value={`${trace.has_real_timing ? "Captured" : "Not captured"} · ${formatAge(now - ms(trace.created_at))}`} />
        <Stat
          className="col-span-2 sm:col-span-3 lg:col-span-6"
          label="Layers"
          value={trace.layers.length > 0 ? <Mono className="text-fg-2">{trace.layers.join(", ")}</Mono> : "—"}
        />
      </Card>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Card className="min-w-0 self-start">
          <CardHeader title="Metadata" subtitle="From the profiler's metadata.json" />
          {trace.metadata === null ? (
            <EmptyState title="No metadata yet" body="metadata.json is uploaded last, once tracing finishes." />
          ) : (
            <MetadataList metadata={trace.metadata} />
          )}
        </Card>
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
        <div key={key} className="grid grid-cols-1 gap-1 px-4 py-2.5 sm:grid-cols-[148px_minmax(0,1fr)] sm:gap-4">
          <dt className="text-[12px] leading-5 text-muted">{humanize(key)}</dt>
          <dd className="min-w-0 text-[13px] leading-5">
            <MetadataValue value={value} />
          </dd>
        </div>
      ))}
      {Array.isArray(files) && (
        <div className="grid grid-cols-1 gap-1 px-4 py-2.5 sm:grid-cols-[148px_minmax(0,1fr)] sm:gap-4">
          <dt className="text-[12px] leading-5 text-muted">Files</dt>
          <dd className="text-[13px] leading-5 tabular-nums">{formatInt(files.length)} listed</dd>
        </div>
      )}
    </dl>
  );
}

const TH = "px-3 py-2.5 text-[11px] font-medium tracking-wide text-muted uppercase whitespace-nowrap";

function KernelTable({ kernels }: { kernels: KernelLaunch[] }) {
  const [shown, setShown] = useState(PAGE);
  const sorted = kernels.toSorted((a, b) => a.kernel_id - b.kernel_id);
  return (
    <Card className="min-w-0 self-start overflow-hidden">
      <CardHeader title="Kernel launches" subtitle={`${formatInt(kernels.length)} launches on the original hardware`} />
      {kernels.length === 0 ? (
        <EmptyState title="No kernels yet" body="Launch dimensions appear once the stats file is uploaded." />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] table-fixed border-collapse">
            <colgroup>
              <col className="w-[64px]" />
              <col />
              <col className="w-[108px]" />
              <col className="w-[108px]" />
              <col className="w-[112px]" />
            </colgroup>
            <thead className="border-b border-border bg-surface-2/50">
              <tr>
                <th className={`${TH} text-right`}>ID</th>
                <th className={`${TH} text-left`}>Kernel</th>
                <th className={`${TH} text-left`}>Grid</th>
                <th className={`${TH} text-left`}>Block</th>
                <th className={`${TH} text-right`}>Instructions</th>
              </tr>
            </thead>
            <tbody className="text-[13px]">
              {sorted.slice(0, shown).map((k) => (
                <tr key={k.kernel_id} className="h-10 border-b border-border last:border-b-0">
                  <td className="px-3 text-right text-muted tabular-nums">{k.kernel_id}</td>
                  <td className="px-3">
                    <Mono>{k.name}</Mono>
                  </td>
                  <td className="px-3 font-mono text-[12px] whitespace-nowrap text-fg-2">{formatDims(k.grid)}</td>
                  <td className="px-3 font-mono text-[12px] whitespace-nowrap text-fg-2">{formatDims(k.block)}</td>
                  <td className="px-3 text-right tabular-nums">{k.instructions === null ? <span className="text-faint">—</span> : formatInt(k.instructions)}</td>
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
    </Card>
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
      <Card className="h-80">
        <span />
      </Card>
    </div>
  );
}
