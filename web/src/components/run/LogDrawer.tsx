import { useEffect, useRef, useState } from "react";
import { Download, RefreshCw, X } from "lucide-react";
import { apiText, downloadFile } from "../../api/client";
import type { SimSummary } from "../../api/types";
import { formatInt } from "../../domain/format";
import { linesReceivedSince, type SimLog } from "../../domain/runLive";
import { SIM_STATE } from "../../domain/status";
import { Button, ErrorNote, Skeleton, StatusPill } from "../ui";
import { LogLines } from "./LogLines";

const MAX_RENDERED = 5000;

type Fetched = { kind: "loading" } | { kind: "error"; error: unknown } | { kind: "ready"; lines: string[]; since: number };

/** Full log: fetched once on open, then followed with the live lines that arrive afterwards. */
export function LogDrawer({ sim, log, onClose }: { sim: SimSummary; log: SimLog | undefined; onClose: () => void }) {
  const [fetched, setFetched] = useState<Fetched>({ kind: "loading" });
  const [reloads, setReloads] = useState(0);
  const logRef = useRef(log);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    logRef.current = log;
  }, [log]);

  useEffect(() => {
    let cancelled = false;
    apiText(`/api/sims/${encodeURIComponent(sim.id)}/log`)
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
  }, [sim.id, reloads]);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  const all = fetched.kind === "ready" ? fetched.lines.concat(linesReceivedSince(log, fetched.since)) : [];
  const shown = all.length > MAX_RENDERED ? all.slice(-MAX_RENDERED) : all;
  const meta = SIM_STATE[sim.state];

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-label={`Log for ${sim.id}`}>
      <button type="button" aria-label="Close log" tabIndex={-1} onClick={onClose} className="absolute inset-0 bg-black/30 backdrop-blur-[2px] dark:bg-black/50" />
      <div className="relative flex h-full w-full max-w-[880px] flex-col border-l border-border bg-surface shadow-pop">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-border px-4 py-3">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="font-mono text-[13px] font-medium">{sim.gpu_config}</span>
              <span className="text-[13px] text-muted">Replica {sim.replica}</span>
              <StatusPill tone={meta.tone} live={meta.live} label={meta.label} />
            </div>
            <div className="truncate font-mono text-[11px] text-faint">{sim.id}</div>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setFetched({ kind: "loading" });
                setReloads((n) => n + 1);
              }}
            >
              <RefreshCw className="size-3.5" />
              Reload
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                void downloadFile({ path: `/api/sims/${encodeURIComponent(sim.id)}/log`, filename: `${sim.id}.log` });
              }}
            >
              <Download className="size-3.5" />
              Download
            </Button>
            <button
              ref={closeRef}
              type="button"
              aria-label="Close"
              onClick={onClose}
              className="grid size-8 place-items-center rounded-lg text-muted transition-colors duration-150 hover:bg-surface-2 hover:text-fg"
            >
              <X className="size-4" />
            </button>
          </div>
        </div>
        <div className="flex h-8 items-center border-b border-border px-4 text-[12px] text-muted tabular-nums">
          {fetched.kind === "ready"
            ? all.length > MAX_RENDERED
              ? `Showing the last ${formatInt(MAX_RENDERED)} of ${formatInt(all.length)} lines. Download for the full log.`
              : `${formatInt(all.length)} lines`
            : fetched.kind === "loading"
              ? "Loading log…"
              : "Couldn't load the log"}
        </div>
        <div className="flex min-h-0 flex-1 flex-col bg-surface-2/40 px-4 py-3">
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
      </div>
    </div>
  );
}
