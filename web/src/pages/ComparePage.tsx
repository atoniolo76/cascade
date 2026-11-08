import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { Check, ChartGantt, GitCompareArrows, Plus, X } from "lucide-react";
import { useRuns, useRunsForTimeline, useTraces } from "../api/queries";
import type { RunSummary, TraceSummary } from "../api/types";
import { PageHeader } from "../components/AppShell";
import { ConfigChips } from "../components/RunsTable";
import { Status } from "../components/meters";
import { ModelTile, SampleBadge } from "../components/TraceState";
import { TimelineView } from "../components/timeline/TimelineView";
import { trackMeta } from "../components/timeline/meta";
import { Button, Card, EmptyState, ErrorNote, Panel, PanelHeader, RelTime, Skeleton } from "../components/ui";
import { ICON, buttonClass } from "../components/styles";
import { cx } from "../components/cx";
import { RUN_STATE } from "../domain/status";
import { buildTimeline, groupKernels, type RunTimelineInput, type Track } from "../domain/timeline";
import { paths } from "../paths";

const MAX_RUNS = 6;
const SERIES_SLOTS = 8;

function parseIds(raw: string | null): string[] {
  return [...new Set((raw ?? "").split(",").filter(Boolean))];
}

const shortRun = (id: string) => id.replace(/^run-/, "").slice(0, 6);

export function ComparePage() {
  const [params, setParams] = useSearchParams();
  const ids = parseIds(params.get("runs"));
  const runs = useRuns();
  const traces = useTraces();
  const summaries = new Map(runs.data?.map((r) => [r.id, r]));
  const traceById = new Map(traces.data?.map((t) => [t.id, t]));
  const data = useRunsForTimeline(ids, (id) => summaries.get(id)?.state === "active");

  const setIds = (next: readonly string[]) => {
    const p = new URLSearchParams(params);
    if (next.length === 0) p.delete("runs");
    else p.set("runs", next.join(","));
    setParams(p, { replace: true });
  };

  const inputs: RunTimelineInput[] = [];
  let colorSlot = 0;
  for (const i of ids.keys()) {
    const run = data.runs[i]?.data;
    const kernels = data.kernels[i]?.data;
    if (!run || !kernels) continue;
    const colors = new Map<string, string>();
    for (const e of run.entries) {
      if (colors.has(e.gpu_config)) continue;
      colors.set(e.gpu_config, colorSlot < SERIES_SLOTS ? `var(--series-${colorSlot + 1})` : "var(--series-ref)");
      colorSlot++;
    }
    inputs.push({
      run,
      trace: run.trace,
      kernels: groupKernels(kernels),
      real: data.comparisons[i]?.data?.rows ?? null,
      colorOf: (c) => colors.get(c) ?? "var(--series-ref)",
    });
  }
  const loading = ids.length > 0 && inputs.length < ids.length && !data.runs.some((q) => q.isError);
  const error = [...data.runs, ...data.kernels].find((q) => q.isError)?.error;
  const traceIds = new Set(inputs.map((x) => x.run.trace_id));
  const linkable = traceIds.size <= 1;
  const multi = inputs.length > 1;

  const timeline = buildTimeline(inputs).tracks;
  const tracks: Track[] = multi ? timeline.map((t) => (t.source.kind === "real" ? t : { ...t, label: `${t.label} · ${shortRun(t.source.runId)}` })) : timeline;

  const modelOf = (runId: string) => inputs.find((x) => x.run.id === runId)?.trace.model ?? null;

  return (
    <div>
      <PageHeader crumbs={[{ label: "Runs", to: paths.runs }, { label: "Compare" }]}>
        <p className="max-w-2xl text-[14px] text-muted">
          Lay runs side by side as timelines. Each run adds its configs, and real hardware appears once per trace.
        </p>
      </PageHeader>

      <div className="flex flex-col gap-6">
        <RunPicker ids={ids} runs={runs.data} traces={traceById} onChange={setIds} />

        {error !== undefined ? (
          <ErrorNote error={error} />
        ) : ids.length === 0 ? (
          <Card>
            <EmptyState
              icon={<GitCompareArrows {...ICON} />}
              title="Pick runs to compare"
              body="Runs of the same trace line up kernel by kernel. Add two or more to see where each config gains or loses time."
            />
          </Card>
        ) : (
          <Panel className="min-w-0">
            <PanelHeader
              icon={<ChartGantt {...ICON} />}
              title="Timeline"
              subtitle={linkable ? "Median across replicas per config; expand a config for its replicas" : "Runs of different traces, laid out in time only"}
            />
            {loading ? (
              <div className="flex flex-col gap-3 p-5">
                <Skeleton className="h-9 w-full" />
                <Skeleton className="h-[260px] w-full" />
              </div>
            ) : (
              <TimelineView
                tracks={tracks}
                linkable={linkable}
                unlinkedReason="These runs replay different traces, so kernel #i is not the same kernel across tracks. Kernel links and Align by kernel are off."
                meta={(t) => trackMeta(t, modelOf(t.source.runId))}
                maxTrackHeight="max(320px, calc(100dvh - 420px))"
                autoFocus
              />
            )}
          </Panel>
        )}
      </div>
    </div>
  );
}

function RunOption({ run, trace, selected, disabled, onToggle }: { run: RunSummary; trace: TraceSummary | undefined; selected: boolean; disabled: boolean; onToggle: () => void }) {
  const state = RUN_STATE[run.state];
  return (
    <li>
      <button
        type="button"
        role="menuitemcheckbox"
        aria-checked={selected}
        disabled={disabled && !selected}
        onClick={onToggle}
        className="grid w-full grid-cols-[20px_24px_minmax(0,1fr)] items-center gap-3 rounded-[12px] px-2.5 py-2 text-left transition-colors duration-150 hover:bg-surface-2 disabled:opacity-40"
      >
        <span className={cx("grid size-[18px] place-items-center rounded-[6px] ring-1 ring-inset", selected ? "bg-btn text-btn-fg ring-btn" : "ring-border-strong")}>
          {selected && <Check className="size-3" strokeWidth={2.5} />}
        </span>
        <ModelTile model={trace?.model ?? null} fallback={run.trace_id} />
        <span className="flex min-w-0 flex-col gap-1">
          <span className="flex min-w-0 items-center gap-2">
            <span className="font-mono text-[13px] font-medium text-fg">{run.id}</span>
            {run.sample && <SampleBadge />}
            <Status tone={state.tone} live={state.live} label={state.label} className="text-[12px]" />
            <span className="ml-auto shrink-0 text-[12px] text-muted">
              <RelTime iso={run.created_at} />
            </span>
          </span>
          <span className="truncate text-[12px] text-muted">{trace?.model ?? run.trace_id}</span>
          <ConfigChips entries={run.entries} />
        </span>
      </button>
    </li>
  );
}

function RunPicker({ ids, runs, traces, onChange }: { ids: string[]; runs: RunSummary[] | undefined; traces: Map<string, TraceSummary>; onChange: (ids: string[]) => void }) {
  const [open, setOpen] = useState(ids.length === 0);
  const box = useRef<HTMLDivElement>(null);
  const byId = new Map(runs?.map((r) => [r.id, r]));
  const selected = ids.flatMap((id) => {
    const r = byId.get(id);
    return r ? [r] : [];
  });
  const anchorTrace = selected[0]?.trace_id;
  const same = runs?.filter((r) => anchorTrace !== undefined && r.trace_id === anchorTrace) ?? [];
  const other = runs?.filter((r) => anchorTrace === undefined || r.trace_id !== anchorTrace) ?? [];
  const toggle = (id: string) => {
    onChange(ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]);
  };

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (box.current && e.target instanceof Node && !box.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <Card className="p-3 sm:p-4">
      <div className="flex flex-wrap items-center gap-2">
        {selected.map((r) => {
          const t = traces.get(r.trace_id);
          const state = RUN_STATE[r.state];
          return (
            <span key={r.id} className="inline-flex h-10 max-w-full items-center gap-2 rounded-[12px] bg-surface pr-1 pl-2 ring-1 ring-border">
              <ModelTile model={t?.model ?? null} fallback={r.trace_id} size="sm" />
              <Link to={paths.run(r.id)} className="font-mono text-[13px] font-medium text-fg hover:underline">
                {r.id}
              </Link>
              {r.sample && <SampleBadge />}
              <span className="hidden md:inline-flex">
                <ConfigChips entries={r.entries} />
              </span>
              <Status tone={state.tone} live={state.live} label={state.label} className="text-[12px]" />
              <button
                type="button"
                aria-label={`Remove ${r.id}`}
                onClick={() => {
                  toggle(r.id);
                }}
                className="grid size-7 place-items-center rounded-[8px] text-faint hover:bg-surface-2 hover:text-fg"
              >
                <X className="size-3.5" {...ICON} />
              </button>
            </span>
          );
        })}
        <div ref={box} className="relative">
          <Button
            aria-haspopup="menu"
            aria-expanded={open}
            onClick={() => {
              setOpen(!open);
            }}
            className="h-10"
          >
            <Plus className="size-4" {...ICON} />
            Add run
          </Button>
          {open && (
            <div role="menu" aria-label="Runs" className="absolute top-12 left-0 z-40 max-h-[min(520px,70dvh)] w-[min(520px,calc(100vw-48px))] animate-toast-in overflow-y-auto rounded-[16px] bg-surface p-2 shadow-pop ring-1 ring-border">
              {runs === undefined ? (
                <Skeleton className="h-24 w-full" />
              ) : (
                <>
                  {same.length > 0 && (
                    <>
                      <p className="px-2.5 pt-1.5 pb-1 text-[11px] font-medium tracking-[0.04em] text-faint uppercase">Same trace</p>
                      <ul>
                        {same.map((r) => (
                          <RunOption key={r.id} run={r} trace={traces.get(r.trace_id)} selected={ids.includes(r.id)} disabled={ids.length >= MAX_RUNS} onToggle={() => { toggle(r.id); }} />
                        ))}
                      </ul>
                    </>
                  )}
                  {other.length > 0 && (
                    <>
                      <p className="px-2.5 pt-2.5 pb-1 text-[11px] font-medium tracking-[0.04em] text-faint uppercase">{anchorTrace === undefined ? "All runs" : "Other traces · time only"}</p>
                      <ul>
                        {other.map((r) => (
                          <RunOption key={r.id} run={r} trace={traces.get(r.trace_id)} selected={ids.includes(r.id)} disabled={ids.length >= MAX_RUNS} onToggle={() => { toggle(r.id); }} />
                        ))}
                      </ul>
                    </>
                  )}
                </>
              )}
            </div>
          )}
        </div>
        {selected.length > 0 && (
          <Link to={paths.runs} className={cx(buttonClass({ variant: "ghost" }), "ml-auto h-10 text-muted")}>
            All runs
          </Link>
        )}
      </div>
    </Card>
  );
}
