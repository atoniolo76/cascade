import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Download, FileText, RefreshCw, X } from "lucide-react";
import { apiText, downloadFile } from "../../api/client";
import type { SimState, SimSummary } from "../../api/types";
import { formatAge, formatCompact, formatDuration, formatExact, formatInt, formatPercent, ms } from "../../domain/format";
import { linesReceivedSince, type SimLog } from "../../domain/runLive";
import { OVER_HINT, kernelProgress } from "../../domain/stats";
import { SIM_STATE } from "../../domain/status";
import { useNow } from "../../hooks/useNow";
import { BlockMeter, Status } from "../meters";
import { Button, CopyButton, ErrorNote, IconButton, Skeleton, Swatch } from "../ui";
import { ICON } from "../styles";
import { LogLines } from "./LogLines";
import { LogoTile } from "../BrandLogo";
import { gpuLogo, logoByKey } from "../../domain/logos";

const MAX_RENDERED = 5000;
const STALE_HEARTBEAT_MS = 30_000;
const enc = encodeURIComponent;

const IDLE_TEXT: Record<SimState, string> = {
  pending: "Waiting for a sailbox",
  launching: "Forking a sailbox from the checkpoint",
  running: "Starting the simulator",
  lost: "No heartbeat; waiting for it to resume",
  succeeded: "All kernels simulated",
  failed: "Simulation failed",
  cancelled: "Cancelled",
};

type Fetched = { kind: "loading" } | { kind: "error"; error: unknown } | { kind: "ready"; lines: string[]; since: number };

/** Full log: fetched once per sim, then followed with the live lines that arrive afterwards. */
function useFullLog(simId: string, log: SimLog | undefined) {
  const [fetched, setFetched] = useState<Fetched>({ kind: "loading" });
  const [reloads, setReloads] = useState(0);
  const logRef = useRef(log);

  useEffect(() => {
    logRef.current = log;
  }, [log]);

  useEffect(() => {
    let cancelled = false;
    apiText(`/api/sims/${enc(simId)}/log`)
      .then((text) => {
        if (cancelled) return;
        const lines = text.endsWith("\n") ? text.slice(0, -1).split("\n") : text.split("\n");
        setFetched({ kind: "ready", lines: text === "" ? [] : lines, since: logRef.current?.received ?? 0 });
      })
      .catch((error: unknown) => {
        if (!cancelled) setFetched({ kind: "error", error });
      });
    return () => {
      cancelled = true;
    };
  }, [simId, reloads]);

  const all = fetched.kind === "ready" ? fetched.lines.concat(linesReceivedSince(log, fetched.since)) : [];
  return {
    fetched,
    all,
    reload: () => {
      setFetched({ kind: "loading" });
      setReloads((n) => n + 1);
    },
  };
}

export function SimPanel({
  sim,
  color,
  log,
  position,
  onStep,
  onClose,
}: {
  sim: SimSummary;
  color: string;
  log: SimLog | undefined;
  position: { index: number; total: number };
  onStep: (delta: 1 | -1) => void;
  onClose: () => void;
}) {
  const now = useNow();
  const meta = SIM_STATE[sim.state];
  const progress = kernelProgress(sim);
  const heartbeatAge = sim.last_heartbeat_at === null ? null : now - ms(sim.last_heartbeat_at);
  const stale = sim.state === "lost" || (heartbeatAge !== null && heartbeatAge > STALE_HEARTBEAT_MS && !meta.terminal);

  return (
    <div className="fixed inset-0 z-40 flex justify-end lg:pointer-events-none" role="dialog" aria-label={`${sim.gpu_config} replica ${sim.replica}`}>
      <button type="button" aria-label="Close sim panel" tabIndex={-1} onClick={onClose} className="absolute inset-0 bg-black/25 lg:hidden" />
      <aside className="pointer-events-auto relative flex h-full w-full max-w-[600px] animate-panel-in flex-col bg-surface shadow-pop lg:m-2 lg:h-[calc(100%-16px)] lg:rounded-[22px] lg:ring-1 lg:ring-border">
        <div className="flex items-start gap-3 border-b border-border px-5 pt-4 pb-3.5">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
              <LogoTile logo={gpuLogo(sim.gpu_config)} fallback={sim.gpu_config} />
              <span className="font-mono text-[15px] font-medium text-fg">{sim.gpu_config}</span>
              <Swatch color={color} />
              <span className="text-[15px] text-muted">Replica {sim.replica}</span>
              <Status tone={meta.tone} live={meta.live} label={meta.label} />
            </div>
            <div className="mt-0.5 flex items-center gap-0.5">
              <span className="truncate font-mono text-[12px] text-muted">{sim.id}</span>
              <CopyButton value={sim.id} label="sim id" />
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            <span className="mr-1 hidden text-[12px] text-muted tabular-nums sm:inline">
              {position.index + 1} / {position.total}
            </span>
            <IconButton label="Previous sim (k)" onClick={() => { onStep(-1); }}>
              <ChevronUp className="size-[18px]" {...ICON} />
            </IconButton>
            <IconButton label="Next sim (j)" onClick={() => { onStep(1); }}>
              <ChevronDown className="size-[18px]" {...ICON} />
            </IconButton>
            <IconButton label="Close (Esc)" onClick={onClose}>
              <X className="size-[18px]" {...ICON} />
            </IconButton>
          </div>
        </div>

        <div className="flex flex-col gap-4 border-b border-border px-5 py-4">
          <div>
            <div className="mb-2 flex items-baseline justify-between text-[13px]">
              <span className="text-muted">
                Kernel <span className="font-medium text-fg tabular-nums">{formatInt(sim.kernels_done)}</span>
                <span className="tabular-nums"> / {formatInt(sim.kernels_total)}</span>
              </span>
              <span className="text-muted tabular-nums">{progress.over ? OVER_HINT : formatPercent(progress.fraction)}</span>
            </div>
            <BlockMeter value={progress.fraction} blocks={24} tone={meta.tone === "pending" ? "active" : meta.tone} label="Kernel progress" />
            <div className="mt-2.5 min-h-5 text-[13px]">
              {sim.error !== null ? (
                <span className="block break-words" style={{ color: "var(--st-bad-ink)" }}>
                  {sim.error}
                </span>
              ) : sim.current_kernel !== null && !meta.terminal ? (
                <span className="block truncate font-mono text-[12px] text-fg-2" title={sim.current_kernel}>
                  {sim.current_kernel}
                </span>
              ) : (
                <span className="block truncate text-muted">{IDLE_TEXT[sim.state]}</span>
              )}
            </div>
          </div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
            <Metric label="Cycles" value={sim.latest.gpu_tot_sim_cycle === null ? null : formatCompact(sim.latest.gpu_tot_sim_cycle)} />
            <Metric label="IPC" value={sim.latest.gpu_tot_ipc === null ? null : sim.latest.gpu_tot_ipc.toFixed(1)} />
            <Metric label="Sim rate" value={sim.latest.sim_rate_ips === null ? null : `${formatCompact(sim.latest.sim_rate_ips)}/s`} />
            {meta.terminal ? (
              <Metric label="Runtime" value={sim.started_at !== null && sim.finished_at !== null ? formatDuration(ms(sim.finished_at) - ms(sim.started_at)) : null} />
            ) : (
              <div className="min-w-0" title={sim.last_heartbeat_at === null ? "No heartbeat yet" : formatExact(sim.last_heartbeat_at)}>
                <dt className="text-[12px] leading-4 text-muted">Heartbeat</dt>
                <dd className="truncate text-[16px] leading-6 font-medium tabular-nums" style={{ color: stale ? "var(--st-lost-ink)" : "var(--fg)" }}>
                  {heartbeatAge === null ? <span className="text-faint">—</span> : `${formatAge(heartbeatAge)} ago`}
                </dd>
              </div>
            )}
          </dl>
          {sim.launcher_ref !== null && (
            <div className="flex items-center gap-1.5 text-[13px] text-muted">
              <LogoTile logo={logoByKey("sail")} fallback="Sail" size="sm" />
              Sailbox <span className="truncate font-mono text-[12px] text-fg-2">{sim.launcher_ref}</span>
              <CopyButton value={sim.launcher_ref} label="sailbox id" />
            </div>
          )}
        </div>

        <SimLogView key={sim.id} sim={sim} log={log} />
      </aside>
    </div>
  );
}

function SimLogView({ sim, log }: { sim: SimSummary; log: SimLog | undefined }) {
  const { fetched, all, reload } = useFullLog(sim.id, log);
  const [downloadError, setDownloadError] = useState<unknown>(null);
  const shown = all.length > MAX_RENDERED ? all.slice(-MAX_RENDERED) : all;

  const download = (kind: "results" | "log") => {
    setDownloadError(null);
    const req =
      kind === "results"
        ? { path: `/api/sims/${enc(sim.id)}/results`, filename: `${sim.id}.results.json` }
        : { path: `/api/sims/${enc(sim.id)}/log`, filename: `${sim.id}.log` };
    downloadFile(req).catch(setDownloadError);
  };

  return (
    <>
      <div className="flex h-11 items-center gap-1 border-b border-border px-3 text-[12px] text-muted tabular-nums">
        <span className="ml-2 min-w-0 flex-1 truncate">
          {fetched.kind === "ready"
            ? all.length > MAX_RENDERED
              ? `Last ${formatInt(MAX_RENDERED)} of ${formatInt(all.length)} lines`
              : `${formatInt(all.length)} log lines`
            : fetched.kind === "loading"
              ? "Loading log…"
              : "Couldn't load the log"}
        </span>
        <Button variant="ghost" size="sm" title="Reload the full log" onClick={reload}>
          <RefreshCw className="size-3.5" {...ICON} />
          <span className="hidden sm:inline">Reload</span>
        </Button>
        <Button variant="ghost" size="sm" title="Download the raw Accel-Sim log" onClick={() => { download("log"); }}>
          <FileText className="size-3.5" {...ICON} />
          Log
        </Button>
        <Button variant="ghost" size="sm" title="Download this sim's results as JSON" onClick={() => { download("results"); }}>
          <Download className="size-3.5" {...ICON} />
          Results
        </Button>
      </div>
      {downloadError !== null && (
        <div className="px-5 pt-3">
          <ErrorNote error={downloadError} />
        </div>
      )}
      <div className="flex min-h-0 flex-1 flex-col bg-card px-5 py-3 lg:rounded-b-[22px]">
        {fetched.kind === "loading" ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 12 }, (_, i) => (
              <Skeleton key={i} className="h-3" />
            ))}
          </div>
        ) : fetched.kind === "error" ? (
          <ErrorNote error={fetched.error} />
        ) : (
          <LogLines lines={shown} empty="The log is empty." wrap outerClassName="flex-1" className="h-full text-[12px] leading-5" />
        )}
      </div>
    </>
  );
}

function Metric({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="min-w-0">
      <dt className="text-[12px] leading-4 text-muted">{label}</dt>
      <dd className="truncate text-[16px] leading-6 font-medium text-fg tabular-nums">{value ?? <span className="text-faint">—</span>}</dd>
    </div>
  );
}
