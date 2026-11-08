import { useEffect, useRef, type CSSProperties, type ReactNode } from "react";
import { Link, Outlet } from "react-router";
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useRuns, useTraces } from "../api/queries";
import { RUN_STATE } from "../domain/status";
import { RAIL_WIDTH, useRailState } from "../lib/rail";
import { dismissToast, toast, useToasts } from "../lib/toast";
import { paths } from "../paths";
import { Status } from "./meters";
import { Rail } from "./Rail";
import { IconButton } from "./ui";
import { ICON } from "./styles";
import { cx } from "./cx";

function useTransitionToasts() {
  const traces = useTraces();
  const runs = useRuns();
  const traceStates = useRef<Map<string, string> | null>(null);
  const runStates = useRef<Map<string, string> | null>(null);

  useEffect(() => {
    if (!traces.data) return;
    const prev = traceStates.current;
    if (prev) {
      for (const t of traces.data) {
        const was = prev.get(t.id);
        if (was === undefined || was === t.state) continue;
        const name = t.model ?? t.id;
        if (t.state === "ready") toast({ tone: "ok", title: "Trace ready", body: name, action: { label: "Launch a run", to: paths.trace(t.id) } });
        if (t.state === "failed") toast({ tone: "bad", title: "Trace failed", body: t.error ?? name, action: { label: "Open trace", to: paths.trace(t.id) } });
      }
    }
    traceStates.current = new Map(traces.data.map((t) => [t.id, t.state]));
  }, [traces.data]);

  useEffect(() => {
    if (!runs.data) return;
    const prev = runStates.current;
    if (prev) {
      for (const r of runs.data) {
        if (prev.get(r.id) !== "active" || r.state === "active") continue;
        toast({ tone: RUN_STATE[r.state].tone, title: `Run ${RUN_STATE[r.state].label.toLowerCase()}`, body: r.id, action: { label: "Open run", to: paths.run(r.id) } });
      }
    }
    runStates.current = new Map(runs.data.map((r) => [r.id, r.state]));
  }, [runs.data]);
}

function Toaster() {
  const toasts = useToasts();
  return (
    <div aria-live="polite" className="pointer-events-none fixed bottom-4 left-4 z-[60] flex w-[min(360px,calc(100vw-32px))] flex-col gap-2 md:left-[calc(var(--rail-w)+16px)]">
      {toasts.map((t) => (
        <div key={t.id} role="status" className="pointer-events-auto flex animate-toast-in items-start gap-3 rounded-[16px] bg-surface p-3.5 shadow-pop ring-1 ring-border">
          <div className="min-w-0 flex-1">
            <Status tone={t.tone} label={t.title} />
            {t.body !== undefined && <p className="mt-1 truncate text-[13px] text-fg-2" title={t.body}>{t.body}</p>}
            {t.action !== undefined && (
              <Link to={t.action.to} onClick={() => { dismissToast(t.id); }} className="mt-1.5 inline-block text-[13px] font-medium text-link hover:underline">
                {t.action.label}
              </Link>
            )}
          </div>
          <button type="button" aria-label="Dismiss" onClick={() => { dismissToast(t.id); }} className="grid size-7 shrink-0 place-items-center rounded-[8px] text-faint hover:bg-surface-2 hover:text-fg">
            <X className="size-4" {...ICON} />
          </button>
        </div>
      ))}
    </div>
  );
}

export function AppShell() {
  useTransitionToasts();
  const rail = useRailState();
  return (
    <div
      style={{ "--rail-w": `${RAIL_WIDTH[rail]}px` } as CSSProperties}
      className="min-h-dvh bg-window transition-[padding-left] duration-[180ms] ease-out md:pl-(--rail-w)"
    >
      <Rail state={rail} />
      <main className="min-h-[calc(100dvh-56px)] border-t border-border bg-panel md:m-2 md:ml-0 md:min-h-[calc(100dvh-16px)] md:rounded-[22px] md:border">
        <div className="mx-auto max-w-[1320px] px-4 pt-5 pb-20 sm:px-8 sm:pt-7 lg:px-12">
          <Outlet />
        </div>
      </main>
      <Toaster />
    </div>
  );
}

export interface Crumb {
  label: ReactNode;
  to?: string;
}

export interface Neighbors {
  prev?: string | undefined;
  next?: string | undefined;
  noun: string;
}

export function PageHeader({
  crumbs,
  status,
  actions,
  neighbors,
  children,
}: {
  crumbs: readonly Crumb[];
  status?: ReactNode;
  actions?: ReactNode;
  neighbors?: Neighbors;
  children?: ReactNode;
}) {
  return (
    <header className="mb-7 flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
        <h1 className="flex min-w-0 flex-1 basis-[280px] flex-wrap items-center gap-x-2.5 gap-y-1 text-[20px] leading-8 tracking-[-0.01em]">
          {crumbs.map((c, i) => {
            const last = i === crumbs.length - 1;
            return (
              <span key={i} className={cx("flex min-w-0 items-center gap-2.5", last && "max-w-full")}>
                {c.to !== undefined && !last ? (
                  <Link to={c.to} className="whitespace-nowrap text-muted transition-colors duration-150 hover:text-fg">
                    {c.label}
                  </Link>
                ) : (
                  <span className={cx("min-w-0", last ? "font-medium text-fg" : "text-muted")}>{c.label}</span>
                )}
                {!last && <span aria-hidden className="text-faint">/</span>}
              </span>
            );
          })}
          {status !== undefined && <span className="ml-1.5 flex items-center">{status}</span>}
        </h1>
        {(actions !== undefined || neighbors !== undefined) && (
          <div className="flex flex-wrap items-center gap-2">
            {actions}
            {neighbors !== undefined && <NeighborButtons neighbors={neighbors} />}
          </div>
        )}
      </div>
      {children}
    </header>
  );
}

function NeighborButtons({ neighbors }: { neighbors: Neighbors }) {
  return (
    <div className="flex gap-2">
      {(["prev", "next"] as const).map((dir) => {
        const to = neighbors[dir];
        const label = `${dir === "prev" ? "Newer" : "Older"} ${neighbors.noun}`;
        const icon = dir === "prev" ? <ChevronLeft className="size-[18px]" {...ICON} /> : <ChevronRight className="size-[18px]" {...ICON} />;
        return to === undefined ? (
          <IconButton key={dir} label={label} disabled>
            {icon}
          </IconButton>
        ) : (
          <Link
            key={dir}
            to={to}
            aria-label={label}
            title={label}
            className="grid size-9 place-items-center rounded-[12px] bg-surface text-muted ring-1 ring-border transition-colors duration-150 ring-inset hover:text-fg hover:ring-border-strong"
          >
            {icon}
          </Link>
        );
      })}
    </div>
  );
}
