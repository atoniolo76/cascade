import { useState } from "react";
import { Download, FileText, HeartPulse, Maximize2, Timer } from "lucide-react";
import { downloadFile } from "../../api/client";
import type { SimState, SimSummary } from "../../api/types";
import { formatAge, formatCompact, formatDuration, formatPercent, ms } from "../../domain/format";
import type { SimLog } from "../../domain/runLive";
import { SIM_STATE } from "../../domain/status";
import { useNow } from "../../hooks/useNow";
import { Button, Card, ProgressBar, StatusPill } from "../ui";
import { LogLines } from "./LogLines";

const TAIL_LINES = 120;
const STALE_HEARTBEAT_MS = 30_000;

const IDLE_TEXT: Record<SimState, string> = {
  pending: "Waiting for a sailbox",
  launching: "Forking sailbox from checkpoint",
  running: "Starting simulator",
  lost: "No heartbeat; waiting for it to resume",
  succeeded: "All kernels simulated",
  failed: "Simulation failed",
  cancelled: "Cancelled",
};

const enc = encodeURIComponent;

export function SimCard({ sim, log, onOpenLog }: { sim: SimSummary; log: SimLog | undefined; onOpenLog: () => void }) {
  const now = useNow();
  const meta = SIM_STATE[sim.state];
  const progress = sim.kernels_total > 0 ? sim.kernels_done / sim.kernels_total : 0;
  const heartbeatAge = sim.last_heartbeat_at === null ? null : now - ms(sim.last_heartbeat_at);
  const stale = sim.state === "lost" || (heartbeatAge !== null && heartbeatAge > STALE_HEARTBEAT_MS && !meta.terminal);
  const lines = (log?.lines ?? sim.log_tail).slice(-TAIL_LINES);
  const [downloadError, setDownloadError] = useState<unknown>(null);

  const download = (kind: "results" | "log") => {
    setDownloadError(null);
    const req =
      kind === "results"
        ? { path: `/api/sims/${enc(sim.id)}/results`, filename: `${sim.id}.results.json` }
        : { path: `/api/sims/${enc(sim.id)}/log`, filename: `${sim.id}.log` };
    downloadFile(req).catch(setDownloadError);
  };

  return (
    <Card className="flex h-[404px] min-w-0 flex-col">
      <div className="flex items-center gap-2 px-4 pt-3.5">
        <StatusPill tone={meta.tone} live={meta.live} label={meta.label} />
        <span className="text-[13px] font-medium">Replica {sim.replica}</span>
        {meta.terminal ? (
          <span className="ml-auto inline-flex items-center gap-1 text-[12px] text-muted tabular-nums" title="Wall-clock runtime">
            <Timer className="size-3.5" />
            {sim.started_at !== null && sim.finished_at !== null ? formatDuration(ms(sim.finished_at) - ms(sim.started_at)) : "—"}
          </span>
        ) : (
          <span
            className="ml-auto inline-flex items-center gap-1 text-[12px] tabular-nums"
            style={{ color: stale ? "var(--st-lost-ink)" : "var(--muted)" }}
            title={sim.last_heartbeat_at === null ? "No heartbeat yet" : `Last heartbeat ${sim.last_heartbeat_at}`}
          >
            <HeartPulse className="size-3.5" />
            {heartbeatAge === null ? "—" : `${formatAge(heartbeatAge)} ago`}
          </span>
        )}
      </div>
      <div className="truncate px-4 font-mono text-[11px] leading-5 text-faint" title={sim.launcher_ref ?? undefined}>
        {sim.id}
        {sim.launcher_ref !== null && ` · ${sim.launcher_ref}`}
      </div>

      <div className="mt-3 px-4">
        <div className="mb-1.5 flex items-baseline justify-between text-[12px]">
          <span className="text-muted">
            Kernel <span className="font-medium text-fg tabular-nums">{sim.kernels_done}</span>
            <span className="tabular-nums"> / {sim.kernels_total}</span>
          </span>
          <span className="text-muted tabular-nums">{formatPercent(progress)}</span>
        </div>
        <ProgressBar value={progress} tone={meta.tone === "pending" ? "active" : meta.tone} />
        <div className="mt-2 h-4 text-[11px] leading-4">
          {sim.current_kernel !== null && !meta.terminal ? (
            <span className="block truncate font-mono text-fg-2" title={sim.current_kernel}>
              {sim.current_kernel}
            </span>
          ) : sim.error !== null ? (
            <span className="block truncate" style={{ color: "var(--st-bad-ink)" }} title={sim.error}>
              {sim.error}
            </span>
          ) : (
            <span className="block truncate text-muted">{IDLE_TEXT[sim.state]}</span>
          )}
        </div>
      </div>

      <dl className="mt-3 grid grid-cols-3 gap-2 px-4">
        <Metric label="Cycles" value={sim.latest.gpu_tot_sim_cycle === null ? null : formatCompact(sim.latest.gpu_tot_sim_cycle)} />
        <Metric label="IPC" value={sim.latest.gpu_tot_ipc === null ? null : sim.latest.gpu_tot_ipc.toFixed(1)} />
        <Metric label="Sim rate" value={sim.latest.sim_rate_ips === null ? null : `${formatCompact(sim.latest.sim_rate_ips)}/s`} />
      </dl>

      <div className="mx-4 mt-3 min-h-0 flex-1 rounded-lg bg-surface-2 px-2.5 py-2">
        <LogLines lines={lines} empty="No log output yet" className="h-[128px]" />
      </div>

      <div className="flex items-center gap-1 px-2.5 py-2">
        <Button variant="ghost" size="sm" onClick={onOpenLog}>
          <Maximize2 className="size-3.5" />
          Full log
        </Button>
        {downloadError !== null && (
          <span className="min-w-0 truncate text-[11px]" style={{ color: "var(--st-bad-ink)" }} title={downloadError instanceof Error ? downloadError.message : undefined}>
            Download failed
          </span>
        )}
        <div className="ml-auto flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={() => { download("results"); }} title="Download this sim's results as JSON">
            <Download className="size-3.5" />
            Results
          </Button>
          <Button variant="ghost" size="sm" onClick={() => { download("log"); }} title="Download the raw Accel-Sim log">
            <FileText className="size-3.5" />
            Log
          </Button>
        </div>
      </div>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] leading-4 text-muted">{label}</dt>
      <dd className="truncate text-[15px] leading-6 font-semibold tabular-nums">{value ?? <span className="text-faint">—</span>}</dd>
    </div>
  );
}
