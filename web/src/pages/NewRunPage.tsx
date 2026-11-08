import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { Check, Cpu, Layers, Rocket } from "lucide-react";
import { useCreateRun, useGpuConfigs, useTraces } from "../api/queries";
import type { GpuConfig, RunEntry, TraceSummary } from "../api/types";
import { PageHeader } from "../components/AppShell";
import { TraceState } from "../components/TraceState";
import { Button, Card, Chip, ErrorNote, Panel, PanelHeader, Skeleton, Stepper } from "../components/ui";
import { ICON } from "../components/styles";
import { configSeries } from "../domain/stats";
import { LogoGlyph, LogoTile } from "../components/BrandLogo";
import { gpuLogo, logoByKey } from "../domain/logos";
import { initialConfigs } from "../domain/configs";
import { toast } from "../lib/toast";
import { cx } from "../components/cx";
import { formatBytes, formatInt, plural } from "../domain/format";
import { TRACE_STATE } from "../domain/status";
import { paths } from "../paths";

const MAX_COUNT = 64;
const PREVIEW_BLOCKS = 48;

export function NewRunPage() {
  const [params, setParams] = useSearchParams();
  const traces = useTraces();
  const configs = useGpuConfigs();

  const requested = params.get("trace");
  const trace =
    traces.data?.find((t) => t.id === requested) ?? (requested === null ? traces.data?.find((t) => t.state === "ready") : undefined);

  return (
    <>
      <PageHeader crumbs={[{ label: "Runs", to: paths.runs }, { label: "New run" }]}>
        <p className="max-w-2xl text-[15px] text-muted">Each simulation forks its own sailbox from the trace's checkpoint and replays every kernel.</p>
      </PageHeader>
      {traces.isError || configs.isError ? (
        <ErrorNote error={traces.error ?? configs.error} />
      ) : traces.data === undefined || configs.data === undefined ? (
        <FormSkeleton />
      ) : (
        <RunForm
          key={trace?.id ?? "none"}
          traces={traces.data}
          trace={trace}
          configs={configs.data}
          onPickTrace={(id) => {
            setParams({ trace: id }, { replace: true });
          }}
        />
      )}
    </>
  );
}

type Selection = Record<string, number>;

function initialSelection(trace: TraceSummary | undefined, configs: GpuConfig[]): Selection {
  return Object.fromEntries((trace ? initialConfigs(trace, configs) : []).map((c) => [c.name, 1]));
}

function RunForm({
  traces,
  trace,
  configs,
  onPickTrace,
}: {
  traces: TraceSummary[];
  trace: TraceSummary | undefined;
  configs: GpuConfig[];
  onPickTrace: (id: string) => void;
}) {
  const navigate = useNavigate();
  const create = useCreateRun();
  const [selection, setSelection] = useState<Selection>(() => initialSelection(trace, configs));

  const ordered = trace
    ? configs.toSorted((a, b) => Number(b.arch === trace.trace_gpu_arch) - Number(a.arch === trace.trace_gpu_arch) || a.name.localeCompare(b.name))
    : configs;
  const entries: RunEntry[] = ordered.flatMap((c) => {
    const count = selection[c.name];
    return count === undefined ? [] : [{ gpu_config: c.name, count }];
  });
  const total = entries.reduce((n, e) => n + e.count, 0);
  const canLaunch = trace?.state === "ready" && total > 0 && !create.isPending;

  const launch = () => {
    if (!trace) return;
    create.mutate(
      { trace_id: trace.id, entries },
      {
        onSuccess: (run) => {
          toast({ tone: "active", title: "Run launched", body: `${plural(total, "sim")} · ${run.id}` });
          void navigate(paths.run(run.id));
        },
      },
    );
  };

  return (
    <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="flex min-w-0 flex-col gap-8">
        <Panel>
          <PanelHeader icon={<Layers {...ICON} />} title="Trace" subtitle="Only ready traces can launch." />
          <div className="flex flex-col gap-3 p-5">
            <select
              value={trace?.id ?? ""}
              onChange={(e) => {
                onPickTrace(e.target.value);
              }}
              className="h-10 w-full min-w-0 truncate rounded-[12px] bg-surface px-3 font-mono text-[13px] text-fg ring-1 ring-border-strong outline-none ring-inset focus:ring-2 focus:ring-accent"
            >
              {trace === undefined && <option value="">Select a trace</option>}
              {traces.map((t) => (
                <option key={t.id} value={t.id} disabled={t.state !== "ready"}>
                  {t.state === "ready" ? t.id : `${t.id} (${TRACE_STATE[t.state].label.toLowerCase()})`}
                </option>
              ))}
            </select>
            {trace && (
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-muted">
                <TraceState trace={trace} detail={false} />
                <span className="text-fg-2">{trace.model ?? "unknown model"}</span>
                <span>{formatInt(trace.kernel_count)} kernels</span>
                <span>{formatBytes(trace.bytes)}</span>
                <span className="font-mono">{trace.trace_gpu_arch}</span>
              </div>
            )}
          </div>
        </Panel>

        <Panel className="overflow-hidden">
          <PanelHeader icon={<Cpu {...ICON} />} title="GPU configs" subtitle={trace ? `Accel-Sim tested configs matching ${trace.trace_gpu_arch}` : "Pick a trace first"} />
          <ul className="divide-y divide-border">
            {ordered.map((config) => (
              <ConfigRow
                key={config.name}
                config={config}
                trace={trace}
                count={selection[config.name]}
                onChange={(count) => {
                  setSelection((prev) => {
                    const rest = Object.fromEntries(Object.entries(prev).filter(([name]) => name !== config.name));
                    return count === undefined ? rest : { ...rest, [config.name]: count };
                  });
                }}
              />
            ))}
          </ul>
        </Panel>
      </div>

      <Card className="lg:sticky lg:top-6">
        <div className="flex flex-col gap-4 p-5">
          <div>
            <div className="flex items-center gap-2 text-[13px] text-muted">
              <LogoTile logo={logoByKey("sail")} fallback="Sail" size="sm" />
              Total sailboxes
            </div>
            <div className="text-[36px] leading-11 font-medium tracking-tight text-fg tabular-nums">{total}</div>
            <SailboxPreview entries={entries} />
          </div>
          <ul className="flex min-h-12 flex-col gap-1.5 text-[14px]">
            {entries.length === 0 ? (
              <li className="text-muted">No configs selected</li>
            ) : (
              entries.map((e) => (
                <li key={e.gpu_config} className="flex items-center justify-between gap-3">
                  <span className="flex min-w-0 items-center gap-1.5 font-mono text-[13px] text-fg">
                    {(() => {
                      const logo = gpuLogo(e.gpu_config);
                      return logo && <LogoGlyph logo={logo} className="size-3 text-muted" />;
                    })()}
                    <span className="truncate">{e.gpu_config}</span>
                  </span>
                  <span className="text-muted tabular-nums">× {e.count}</span>
                </li>
              ))
            )}
          </ul>
          {create.isError && <ErrorNote error={create.error} />}
          <Button variant="primary" className="w-full" disabled={!canLaunch} onClick={launch}>
            <Rocket className="size-4" {...ICON} />
            {create.isPending ? "Launching…" : total === 1 ? "Launch 1 simulation" : `Launch ${total} simulations`}
          </Button>
        </div>
      </Card>
    </div>
  );
}

function ConfigRow({
  config,
  trace,
  count,
  onChange,
}: {
  config: GpuConfig;
  trace: TraceSummary | undefined;
  count: number | undefined;
  onChange: (count: number | undefined) => void;
}) {
  const compatible = trace !== undefined && config.arch === trace.trace_gpu_arch;
  const selected = count !== undefined;
  const suggested = trace?.suggested_sim_config === config.name;
  return (
    <li className={cx("flex min-h-[72px] items-center gap-3.5 px-5 py-3 transition-colors duration-150", selected && "bg-accent-soft", !compatible && "opacity-55")}>
      <button
        type="button"
        role="checkbox"
        aria-checked={selected}
        aria-label={`Include ${config.name}`}
        disabled={!compatible}
        onClick={() => {
          onChange(selected ? undefined : 1);
        }}
        className={cx(
          "grid size-5 shrink-0 place-items-center rounded-[6px] transition-colors duration-150 disabled:cursor-not-allowed",
          selected ? "bg-accent text-accent-fg" : "bg-surface ring-1 ring-border-strong ring-inset",
        )}
      >
        {selected && <Check className="size-3" strokeWidth={3} />}
      </button>
      <LogoTile logo={gpuLogo(config.name)} fallback={config.name} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-mono text-[14px] font-medium text-fg">{config.name}</span>
          {suggested && compatible && <Chip className="h-6 px-2 text-[12px] text-accent-ink">Suggested</Chip>}
        </div>
        <div className="mt-0.5 truncate text-[13px] text-muted">
          {compatible || trace === undefined ? (
            <span className="tabular-nums">
              {config.arch} · {formatInt(config.core_clock_mhz)} MHz core clock
            </span>
          ) : (
            <span>
              {config.arch} config · trace contains {trace.trace_gpu_arch} instructions
            </span>
          )}
        </div>
      </div>
      {compatible && (
        <Stepper
          label={`${config.name} count`}
          value={count ?? 1}
          min={1}
          max={MAX_COUNT}
          disabled={!selected}
          onChange={(n) => {
            onChange(n);
          }}
        />
      )}
    </li>
  );
}

function SailboxPreview({ entries }: { entries: RunEntry[] }) {
  const series = configSeries(entries);
  const blocks = entries.flatMap((e, i) => Array<string>(e.count).fill(series[i]?.color ?? "var(--series-ref)"));
  if (blocks.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap items-center gap-[3px]" aria-hidden>
      {blocks.slice(0, PREVIEW_BLOCKS).map((color, i) => (
        <span key={i} className="size-[10px] rounded-[3px]" style={{ background: color }} />
      ))}
      {blocks.length > PREVIEW_BLOCKS && <span className="ml-1 text-[12px] text-muted tabular-nums">+{blocks.length - PREVIEW_BLOCKS}</span>}
    </div>
  );
}

function FormSkeleton() {
  return (
    <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_320px]">
      <Card className="flex flex-col gap-3 p-4">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-9 w-full" />
        {Array.from({ length: 3 }, (_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </Card>
      <Card className="flex flex-col gap-3 p-4">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-8 w-12" />
        <Skeleton className="h-9 w-full" />
      </Card>
    </div>
  );
}
