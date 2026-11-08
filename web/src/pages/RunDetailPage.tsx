import { useEffect, useEffectEvent, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import { Download, Square } from "lucide-react";
import { downloadFile } from "../api/client";
import { useCancelRun, useRuns } from "../api/queries";
import type { RunDetail, SimState } from "../api/types";
import { PageHeader } from "../components/AppShell";
import { BlockMeter, Status } from "../components/meters";
import { ComparisonPanel } from "../components/run/ComparisonPanel";
import { SimGrid } from "../components/run/SimGrid";
import { SimPanel } from "../components/run/SimPanel";
import { SimTimePanel } from "../components/run/SimTimePanel";
import { ConfigChips } from "../components/RunsTable";
import { TraceChip } from "../components/TraceState";
import { Button, Card, CopyButton, EmptyState, ErrorNote, Panel, RelTime, Skeleton } from "../components/ui";
import { ICON, buttonClass } from "../components/styles";
import { cx } from "../components/cx";
import { formatDuration, formatInt, formatPercent, ms } from "../domain/format";
import type { RunLive } from "../domain/runLive";
import { OVER_HINT, aggregateProgress, configSeries, estimateRemaining, orderedSims } from "../domain/stats";
import { RUN_STATE, SIM_STATE, TONE_VARS, deriveRunState, type Tone } from "../domain/status";
import { useNow } from "../hooks/useNow";
import { useRunStream, type Connection } from "../hooks/useRunStream";
import { toast } from "../lib/toast";
import { paths } from "../paths";

const CONNECTION: Record<Connection, { label: string; tone: Tone; live: boolean }> = {
  connecting: { label: "Connecting", tone: "pending", live: false },
  live: { label: "Live", tone: "ok", live: true },
  reconnecting: { label: "Reconnecting", tone: "lost", live: true },
  ended: { label: "Stream closed", tone: "off", live: false },
  missing: { label: "Not found", tone: "bad", live: false },
};

type Live = Extract<RunLive, { kind: "live" }>;

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const { state, dispatch, connection } = useRunStream(runId);
  useSimTransitionToasts(runId, state.kind === "live" ? state.run : undefined);

  if (state.kind === "waiting") {
    return connection === "missing" ? (
      <Card>
        <EmptyState
          title="Run not found"
          body={`No run with id ${runId}.`}
          action={
            <Link to={paths.runs} className={buttonClass({})}>
              Back to runs
            </Link>
          }
        />
      </Card>
    ) : (
      <RunSkeleton runId={runId} />
    );
  }

  return (
    <RunView
      live={state}
      connection={connection}
      onRun={(run) => {
        dispatch({ type: "run", run });
      }}
    />
  );
}

/** Toasts for sims that fail or go quiet while you watch; the snapshot that opens the page sets the baseline. */
function useSimTransitionToasts(runId: string, run: RunDetail | undefined) {
  const seen = useRef<{ runId: string; states: Map<string, SimState> } | null>(null);
  useEffect(() => {
    if (!run) return;
    const prev = seen.current?.runId === runId ? seen.current.states : null;
    if (prev) {
      for (const sim of run.sims) {
        const was = prev.get(sim.id);
        if (was === undefined || was === sim.state) continue;
        const name = `${sim.gpu_config} replica ${sim.replica}`;
        const action = { label: "Open sim", to: paths.sim(runId, sim.id) };
        if (sim.state === "failed") toast({ tone: "bad", title: "Sim failed", body: sim.error === null ? name : `${name}: ${sim.error}`, action });
        if (sim.state === "lost") toast({ tone: "lost", title: "Sim lost its heartbeat", body: name, action });
      }
    }
    seen.current = { runId, states: new Map(run.sims.map((s) => [s.id, s.state])) };
  }, [runId, run]);
}

function isTyping(target: EventTarget | null) {
  return target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));
}

function RunView({ live, connection, onRun }: { live: Live; connection: Connection; onRun: (run: RunDetail) => void }) {
  const { run } = live;
  const runState = deriveRunState(run.sims.map((s) => s.state));
  const series = configSeries(run.entries);
  const sims = orderedSims(run.sims, run.entries);
  const runs = useRuns();
  const [params, setParams] = useSearchParams();
  const selectedId = params.get("sim");
  const index = selectedId === null ? -1 : sims.findIndex((s) => s.id === selectedId);
  const selected = sims[index];

  const order = runs.data?.map((r) => r.id) ?? [];
  const at = order.indexOf(run.id);
  const neighbor = (i: number) => {
    const id = at === -1 ? undefined : order[i];
    return id === undefined ? undefined : paths.run(id);
  };

  const select = (id: string | null) => {
    setParams(
      (p) => {
        const next = new URLSearchParams(p);
        if (id === null) next.delete("sim");
        else next.set("sim", id);
        return next;
      },
      { replace: true },
    );
  };

  const step = (delta: 1 | -1) => {
    if (sims.length === 0) return;
    const from = index === -1 ? (delta === 1 ? -1 : 0) : index;
    const next = sims[(from + delta + sims.length) % sims.length];
    if (next) select(next.id);
  };

  const onKey = useEffectEvent((e: KeyboardEvent) => {
    if (e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target)) return;
    const open = index !== -1;
    if (e.key === "j" || (open && (e.key === "ArrowDown" || e.key === "ArrowRight"))) {
      e.preventDefault();
      step(1);
    } else if (e.key === "k" || (open && (e.key === "ArrowUp" || e.key === "ArrowLeft"))) {
      e.preventDefault();
      step(-1);
    } else if (e.key === "Escape" && open) {
      select(null);
    }
  });

  useEffect(() => {
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  return (
    <>
      <div className={cx("transition-[padding] duration-200", selected && "lg:pr-[560px]")}>
        <PageHeader
          crumbs={[
            { label: "Runs", to: paths.runs },
            {
              label: (
                <span className="flex items-center gap-1">
                  <span className="font-mono text-[18px]">{run.id}</span>
                  <CopyButton value={run.id} label="run id" />
                </span>
              ),
            },
          ]}
          status={
            <span className="flex items-center gap-3">
              <Status tone={RUN_STATE[runState].tone} live={RUN_STATE[runState].live} label={RUN_STATE[runState].label} />
              {(runState === "active" || connection !== "live") && <ConnectionBadge connection={connection} />}
            </span>
          }
          actions={<RunActions run={run} active={runState === "active"} onRun={onRun} />}
          neighbors={{ prev: neighbor(at - 1), next: neighbor(at + 1), noun: "run" }}
        >
          <div className="w-fit max-w-full rounded-[16px] bg-card px-4 py-2.5 text-[15px] leading-8 text-fg-2 ring-1 ring-border">
            <span className="mr-2">Replaying</span>
            <TraceChip trace={run.trace} />
            <span className="mx-2">on</span>
            <span className="inline-flex align-middle">
              <ConfigChips entries={run.entries} />
            </span>
            <span className="ml-2 text-muted">
              · started <RelTime iso={run.created_at} />
            </span>
          </div>
        </PageHeader>

        <div className="flex flex-col gap-8">
          <RunSummaryCard run={run} runState={runState} />

          <SimGrid sims={sims} series={series} selectedId={selected?.id ?? null} onSelect={select} />

          <div className="grid gap-8 2xl:grid-cols-2">
            <SimTimePanel runId={run.id} sims={run.sims} kernels={live.kernels} series={series} />
            <ComparisonPanel runId={run.id} live={runState === "active"} series={series} />
          </div>
        </div>
      </div>
      {selected && (
        <SimPanel
          sim={selected}
          color={series.find((s) => s.key === selected.gpu_config)?.color ?? "var(--series-ref)"}
          log={live.logs[selected.id]}
          position={{ index, total: sims.length }}
          onStep={step}
          onClose={() => {
            select(null);
          }}
        />
      )}
    </>
  );
}

function ConnectionBadge({ connection }: { connection: Connection }) {
  const c = CONNECTION[connection];
  return (
    <span className="inline-flex items-center gap-1.5 text-[13px] font-normal text-muted">
      <span className={cx("size-1.5 rounded-full", c.live && "animate-pulse-soft")} style={{ background: TONE_VARS[c.tone].dot }} />
      {c.label}
    </span>
  );
}

function RunActions({ run, active, onRun }: { run: RunDetail; active: boolean; onRun: (run: RunDetail) => void }) {
  const cancel = useCancelRun(run.id);
  const [confirming, setConfirming] = useState(false);
  const [downloadError, setDownloadError] = useState<unknown>(null);

  useEffect(() => {
    if (!confirming) return;
    const t = setTimeout(() => {
      setConfirming(false);
    }, 4000);
    return () => {
      clearTimeout(t);
    };
  }, [confirming]);

  return (
    <div className="flex flex-col items-end gap-2">
      <div className="flex items-center gap-2">
        <Button
          title="Every sim's kernel stats, the trace, and the real vs. simulated comparison, as JSON"
          onClick={() => {
            setDownloadError(null);
            downloadFile({ path: `/api/runs/${encodeURIComponent(run.id)}/results`, filename: `${run.id}.results.json` }).catch(setDownloadError);
          }}
        >
          <Download className="size-4" {...ICON} />
          Results
        </Button>
        {active && (
          <Button
            variant="danger"
            className="min-w-[124px]"
            disabled={cancel.isPending}
            onClick={() => {
              if (!confirming) {
                setConfirming(true);
                return;
              }
              setConfirming(false);
              cancel.mutate(undefined, { onSuccess: onRun });
            }}
          >
            <Square className="size-3" fill="currentColor" {...ICON} />
            {cancel.isPending ? "Cancelling…" : confirming ? "Confirm cancel" : "Cancel run"}
          </Button>
        )}
      </div>
      {(cancel.isError || downloadError !== null) && <ErrorNote error={cancel.error ?? downloadError} />}
    </div>
  );
}

function SummaryItem({ label, children, className }: { label: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={cx("min-w-0", className)}>
      <div className="mb-1.5 text-[13px] text-muted">{label}</div>
      {children}
    </div>
  );
}

function RunSummaryCard({ run, runState }: { run: RunDetail; runState: ReturnType<typeof deriveRunState> }) {
  const now = useNow();
  const terminal = runState !== "active";
  const { done, total, fraction, over } = aggregateProgress(run.sims);
  const finishedAt = run.sims.reduce((t, s) => (s.finished_at === null ? t : Math.max(t, ms(s.finished_at))), 0);
  const elapsed = (terminal && finishedAt > 0 ? finishedAt : now) - ms(run.created_at);
  const eta = estimateRemaining(run.sims, now, (s) => SIM_STATE[s.state].terminal);
  const finishedSims = run.sims.filter((s) => SIM_STATE[s.state].terminal).length;

  return (
    <Card className="grid grid-cols-2 gap-x-8 gap-y-5 p-5 sm:p-6 lg:grid-cols-[minmax(0,2.2fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)]">
      <SummaryItem label="Kernels simulated" className="col-span-2 lg:col-span-1">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="text-[28px] leading-8 font-medium tracking-tight text-fg tabular-nums">{formatPercent(fraction)}</span>
          <span className="flex flex-col gap-1">
            <BlockMeter value={fraction} blocks={24} tone={terminal ? RUN_STATE[runState].tone : "active"} label="Kernels simulated" />
            <span className="text-[12px] leading-4 text-muted tabular-nums">
              {formatInt(done)} of {formatInt(total)} kernels{over && ` · ${OVER_HINT}`}
            </span>
          </span>
        </div>
      </SummaryItem>
      <SummaryItem label="Sims finished">
        <span className="text-[22px] leading-8 font-medium text-fg tabular-nums">
          {finishedSims}
          <span className="text-muted"> / {run.sims.length}</span>
        </span>
      </SummaryItem>
      <SummaryItem label={terminal ? "Took" : "Elapsed"}>
        <span className="text-[22px] leading-8 font-medium text-fg tabular-nums">{formatDuration(elapsed)}</span>
      </SummaryItem>
      <SummaryItem label="Remaining">
        <span className="text-[22px] leading-8 font-medium text-fg tabular-nums" title="Projected from each running sim's pace; the run ends with its slowest sim">
          {eta.kind === "done" ? <span className="text-muted">Done</span> : eta.kind === "unknown" ? <span className="text-[15px] text-muted">Estimating…</span> : `~${formatDuration(eta.ms)}`}
        </span>
      </SummaryItem>
    </Card>
  );
}

function RunSkeleton({ runId }: { runId: string }) {
  return (
    <div>
      <PageHeader crumbs={[{ label: "Runs", to: paths.runs }, { label: <span className="font-mono text-[18px]">{runId}</span> }]}>
        <Skeleton className="h-11 w-[480px] max-w-full rounded-[16px]" />
      </PageHeader>
      <div className="flex flex-col gap-8">
        <Card className="h-[104px] p-5">
          <Skeleton className="h-full" />
        </Card>
        <Panel className="flex h-[260px] flex-col gap-4 p-5">
          <Skeleton className="h-5 w-24" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </Panel>
      </div>
    </div>
  );
}
