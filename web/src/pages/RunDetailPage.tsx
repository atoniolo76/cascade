import { useEffect, useState } from "react";
import { Link, useParams } from "react-router";
import { Download, Square } from "lucide-react";
import { downloadFile } from "../api/client";
import { useCancelRun } from "../api/queries";
import type { RunDetail, SimSummary } from "../api/types";
import { PageHeader } from "../components/AppShell";
import { SimCountsBar, SimCountsText } from "../components/SimCounts";
import { ComparisonPanel } from "../components/run/ComparisonPanel";
import { LogDrawer } from "../components/run/LogDrawer";
import { SimCard } from "../components/run/SimCard";
import { SimTimePanel } from "../components/run/SimTimePanel";
import { Button, Card, EmptyState, ErrorNote, Mono, ProgressBar, Skeleton, StatusPill } from "../components/ui";
import { cx } from "../components/cx";
import { formatDuration, formatInt, formatPercent, ms } from "../domain/format";
import type { RunLive } from "../domain/runLive";
import { configSeries, aggregateProgress, type Series } from "../domain/stats";
import { RUN_STATE, SIM_STATE, TONE_VARS, countSimStates, deriveRunState, type Tone } from "../domain/status";
import { useNow } from "../hooks/useNow";
import { useRunStream, type Connection } from "../hooks/useRunStream";
import { paths } from "../paths";

const CONNECTION: Record<Connection, { label: string; tone: Tone; live: boolean }> = {
  connecting: { label: "Connecting", tone: "pending", live: false },
  live: { label: "Live", tone: "ok", live: true },
  reconnecting: { label: "Reconnecting", tone: "lost", live: true },
  ended: { label: "Stream closed", tone: "off", live: false },
  missing: { label: "Not found", tone: "bad", live: false },
};

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const { state, dispatch, connection } = useRunStream(runId);
  const [logSimId, setLogSimId] = useState<string | null>(null);

  if (state.kind === "waiting") {
    return connection === "missing" ? (
      <Card>
        <EmptyState
          title="Run not found"
          body={`No run with id ${runId}.`}
          action={
            <Link to={paths.runs} className="text-[13px] font-medium text-accent-ink hover:underline">
              Back to runs
            </Link>
          }
        />
      </Card>
    ) : (
      <RunSkeleton runId={runId} />
    );
  }

  const logSim = logSimId === null ? undefined : state.run.sims.find((s) => s.id === logSimId);

  return (
    <>
      <RunView
        live={state}
        connection={connection}
        onRun={(run) => {
          dispatch({ type: "run", run });
        }}
        onOpenLog={setLogSimId}
      />
      {logSim && (
        <LogDrawer
          sim={logSim}
          log={state.logs[logSim.id]}
          onClose={() => {
            setLogSimId(null);
          }}
        />
      )}
    </>
  );
}

function RunView({
  live,
  connection,
  onRun,
  onOpenLog,
}: {
  live: Extract<RunLive, { kind: "live" }>;
  connection: Connection;
  onRun: (run: RunDetail) => void;
  onOpenLog: (simId: string) => void;
}) {
  const { run } = live;
  const states = run.sims.map((s) => s.state);
  const runState = deriveRunState(states);
  const series = configSeries(run.entries);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        eyebrow={
          <Link to={paths.runs} className="transition-colors duration-150 hover:text-fg">
            Runs
          </Link>
        }
        title={
          <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="font-mono text-[18px] whitespace-nowrap">{run.id}</span>
            <StatusPill tone={RUN_STATE[runState].tone} live={RUN_STATE[runState].live} label={RUN_STATE[runState].label} />
            {(runState === "active" || connection !== "live") && <ConnectionBadge connection={connection} />}
          </span>
        }
        subtitle={
          <Link to={paths.trace(run.trace_id)} className="block max-w-full transition-colors duration-150 hover:text-fg">
            <Mono>{run.trace_id}</Mono>
          </Link>
        }
        actions={<RunActions run={run} active={runState === "active"} onRun={onRun} />}
      />

      <RunSummaryStrip run={run} terminal={runState !== "active"} />

      <div className="flex flex-col gap-6">
        {series.map((s) => (
          <ConfigGroup key={s.key} series={s} sims={run.sims.filter((sim) => sim.gpu_config === s.key)} live={live} onOpenLog={onOpenLog} />
        ))}
      </div>

      <div className="grid gap-6 2xl:grid-cols-2">
        <SimTimePanel sims={run.sims} kernels={live.kernels} series={series} />
        <ComparisonPanel runId={run.id} live={runState === "active"} series={series} />
      </div>
    </div>
  );
}

function ConnectionBadge({ connection }: { connection: Connection }) {
  const c = CONNECTION[connection];
  return (
    <span className="inline-flex items-center gap-1.5 text-[12px] font-normal text-muted">
      <span className={cx("size-1.5 rounded-full", c.live && "animate-pulse-soft")} style={{ background: TONE_VARS[c.tone].dot, color: TONE_VARS[c.tone].dot }} />
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
          onClick={() => {
            setDownloadError(null);
            downloadFile({ path: `/api/runs/${encodeURIComponent(run.id)}/results`, filename: `${run.id}.results.json` }).catch(setDownloadError);
          }}
        >
          <Download className="size-3.5" />
          Download results
        </Button>
        <Button
          variant="danger"
          className="w-[132px]"
          disabled={!active || cancel.isPending}
          onClick={() => {
            if (!confirming) {
              setConfirming(true);
              return;
            }
            setConfirming(false);
            cancel.mutate(undefined, { onSuccess: onRun });
          }}
        >
          <Square className="size-3" fill="currentColor" />
          {cancel.isPending ? "Cancelling…" : confirming ? "Confirm cancel" : "Cancel run"}
        </Button>
      </div>
      {(cancel.isError || downloadError !== null) && <ErrorNote error={cancel.error ?? downloadError} />}
    </div>
  );
}

function RunSummaryStrip({ run, terminal }: { run: RunDetail; terminal: boolean }) {
  const now = useNow();
  const { done, total } = aggregateProgress(run.sims);
  const fraction = total > 0 ? done / total : 0;
  const finishedAt = run.sims.reduce((t, s) => (s.finished_at === null ? t : Math.max(t, ms(s.finished_at))), 0);
  const elapsed = (terminal && finishedAt > 0 ? finishedAt : now) - ms(run.created_at);
  const counts = countSimStates(run.sims.map((s) => s.state));

  return (
    <Card className="grid grid-cols-1 gap-x-8 gap-y-4 p-4 sm:grid-cols-2 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,2fr)]">
      <div className="min-w-0">
        <div className="mb-1 flex items-baseline justify-between gap-3">
          <span className="text-[11px] font-medium tracking-wide text-muted uppercase">Progress</span>
          <span className="text-[12px] text-muted tabular-nums">
            {formatInt(done)} / {formatInt(total)} kernels
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="w-14 shrink-0 text-[20px] leading-7 font-semibold tabular-nums">{formatPercent(fraction)}</span>
          <ProgressBar value={fraction} tone={terminal ? RUN_STATE[deriveRunState(run.sims.map((s) => s.state))].tone : "active"} className="flex-1" />
        </div>
      </div>
      <div className="min-w-0">
        <div className="mb-1 text-[11px] font-medium tracking-wide text-muted uppercase">Elapsed</div>
        <div className="text-[20px] leading-7 font-semibold tabular-nums">{formatDuration(elapsed)}</div>
      </div>
      <div className="min-w-0 sm:col-span-2 lg:col-span-1">
        <div className="mb-1 flex items-baseline justify-between gap-3">
          <span className="text-[11px] font-medium tracking-wide text-muted uppercase">Sims</span>
          <span className="text-[12px] text-muted tabular-nums">{run.sims.length} sailboxes</span>
        </div>
        <SimCountsBar counts={counts} className="mt-2.5 mb-2" />
        <div className="text-[12px] text-fg-2">
          <SimCountsText counts={counts} />
        </div>
      </div>
    </Card>
  );
}

function ConfigGroup({
  series,
  sims,
  live,
  onOpenLog,
}: {
  series: Series;
  sims: SimSummary[];
  live: Extract<RunLive, { kind: "live" }>;
  onOpenLog: (simId: string) => void;
}) {
  const sorted = sims.toSorted((a, b) => a.replica - b.replica);
  const { done, total } = aggregateProgress(sims);
  const finished = sims.filter((s) => SIM_STATE[s.state].terminal).length;
  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="size-2.5 rounded-[3px]" style={{ background: series.color }} />
        <h2 className="font-mono text-[14px] font-semibold">{series.label}</h2>
        <span className="text-[12px] text-muted tabular-nums">
          {sims.length} {sims.length === 1 ? "replica" : "replicas"} · {finished} finished · {formatInt(done)} / {formatInt(total)} kernels
        </span>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3 min-[1680px]:grid-cols-4">
        {sorted.map((sim) => (
          <SimCard
            key={sim.id}
            sim={sim}
            log={live.logs[sim.id]}
            onOpenLog={() => {
              onOpenLog(sim.id);
            }}
          />
        ))}
      </div>
    </section>
  );
}

function RunSkeleton({ runId }: { runId: string }) {
  return (
    <div className="flex flex-col gap-6">
      <PageHeader eyebrow="Runs" title={<span className="font-mono text-[18px]">{runId}</span>} subtitle={<Skeleton className="h-4 w-80 max-w-full" />} />
      <Card className="h-[92px] p-4">
        <Skeleton className="h-full" />
      </Card>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {Array.from({ length: 3 }, (_, i) => (
          <Card key={i} className="flex h-[404px] flex-col gap-3 p-4">
            <Skeleton className="h-5 w-24 rounded-full" />
            <Skeleton className="h-1.5 w-full rounded-full" />
            <Skeleton className="h-10 w-full" />
            <Skeleton className="flex-1" />
          </Card>
        ))}
      </div>
    </div>
  );
}
