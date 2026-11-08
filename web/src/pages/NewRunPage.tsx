import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { Check, Rocket } from "lucide-react";
import { useCreateRun, useGpuConfigs, useTraces } from "../api/queries";
import type { GpuConfig, RunEntry, TraceSummary } from "../api/types";
import { PageHeader } from "../components/AppShell";
import { TraceState } from "../components/TraceState";
import { Button, Card, CardHeader, ErrorNote, Skeleton, Stepper } from "../components/ui";
import { cx } from "../components/cx";
import { formatBytes, formatInt } from "../domain/format";
import { TRACE_STATE } from "../domain/status";
import { paths } from "../paths";

const MAX_COUNT = 64;

export function NewRunPage() {
  const [params, setParams] = useSearchParams();
  const traces = useTraces();
  const configs = useGpuConfigs();

  const requested = params.get("trace");
  const trace =
    traces.data?.find((t) => t.id === requested) ?? (requested === null ? traces.data?.find((t) => t.state === "ready") : undefined);

  return (
    <>
      <PageHeader title="New run" subtitle="Each simulation forks its own sailbox from the trace's checkpoint." />
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
  if (!trace) return {};
  const compatible = configs.filter((c) => c.arch === trace.trace_gpu_arch);
  const pick = compatible.find((c) => c.name === trace.suggested_sim_config) ?? compatible[0];
  return pick ? { [pick.name]: 1 } : {};
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
          void navigate(paths.run(run.id));
        },
      },
    );
  };

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="flex min-w-0 flex-col gap-6">
        <Card>
          <CardHeader title="Trace" subtitle="Only ready traces can launch." />
          <div className="flex flex-col gap-3 p-4">
            <select
              value={trace?.id ?? ""}
              onChange={(e) => {
                onPickTrace(e.target.value);
              }}
              className="h-9 w-full min-w-0 truncate rounded-lg bg-surface px-3 font-mono text-[12px] ring-1 ring-border-strong outline-none ring-inset focus:ring-2 focus:ring-accent"
            >
              {trace === undefined && <option value="">Select a trace</option>}
              {traces.map((t) => (
                <option key={t.id} value={t.id} disabled={t.state !== "ready"}>
                  {t.state === "ready" ? t.id : `${t.id} (${TRACE_STATE[t.state].label.toLowerCase()})`}
                </option>
              ))}
            </select>
            {trace && (
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-muted">
                <TraceState trace={trace} />
                <span className="text-fg-2">{trace.model ?? "unknown model"}</span>
                <span>{formatInt(trace.kernel_count)} kernels</span>
                <span>{formatBytes(trace.bytes)}</span>
                <span className="font-mono">{trace.trace_gpu_arch}</span>
              </div>
            )}
          </div>
        </Card>

        <Card className="overflow-hidden">
          <CardHeader title="GPU configs" subtitle={trace ? `Accel-Sim tested configs matching ${trace.trace_gpu_arch}` : "Pick a trace first"} />
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
        </Card>
      </div>

      <Card className="lg:sticky lg:top-20">
        <div className="flex flex-col gap-4 p-4">
          <div>
            <div className="text-[12px] font-medium text-muted">Total sailboxes</div>
            <div className="text-[32px] leading-10 font-semibold tracking-tight">{total}</div>
          </div>
          <ul className="flex min-h-12 flex-col gap-1.5 text-[13px]">
            {entries.length === 0 ? (
              <li className="text-muted">No configs selected</li>
            ) : (
              entries.map((e) => (
                <li key={e.gpu_config} className="flex items-center justify-between gap-3">
                  <span className="truncate font-mono text-[12px]">{e.gpu_config}</span>
                  <span className="text-muted tabular-nums">× {e.count}</span>
                </li>
              ))
            )}
          </ul>
          {create.isError && <ErrorNote error={create.error} />}
          <Button variant="primary" className="h-9 w-full" disabled={!canLaunch} onClick={launch}>
            <Rocket className="size-4" />
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
    <li className={cx("flex min-h-16 items-center gap-3 px-4 py-3 transition-colors duration-150", selected && "bg-accent-soft/40", !compatible && "opacity-60")}>
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
          "grid size-[18px] shrink-0 place-items-center rounded-[5px] transition-colors duration-150 disabled:cursor-not-allowed",
          selected ? "bg-accent text-accent-fg" : "bg-surface ring-1 ring-border-strong ring-inset",
        )}
      >
        {selected && <Check className="size-3" strokeWidth={3} />}
      </button>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-mono text-[13px] font-medium">{config.name}</span>
          {suggested && compatible && (
            <span className="rounded-md bg-accent-soft px-1.5 py-0.5 text-[11px] leading-none font-medium text-accent-ink">Suggested</span>
          )}
        </div>
        <div className="mt-0.5 truncate text-[12px] text-muted">
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

function FormSkeleton() {
  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
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
