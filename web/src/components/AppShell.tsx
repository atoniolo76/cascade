import { useEffect, useRef, type ReactNode } from "react";
import { Link, Outlet, useLocation } from "react-router";
import { Activity, BookOpen, ChevronLeft, ChevronRight, GitCompareArrows, Layers, LogOut, Monitor, Moon, Plus, Sun, X } from "lucide-react";
import { tokenStore } from "../api/client";
import { useHealth, useRuns, useTraces } from "../api/queries";
import { DOCS_URL } from "../config";
import { RUN_STATE } from "../domain/status";
import { dismissToast, toast, useToasts } from "../lib/toast";
import { themeStore, useThemePref, type ThemePref } from "../lib/theme";
import { paths } from "../paths";
import { BrandTile } from "./Logo";
import { LogoGlyph } from "./BrandLogo";
import { logoByKey } from "../domain/logos";
import { Status } from "./meters";
import { IconButton } from "./ui";
import { ICON } from "./styles";
import { cx } from "./cx";

const DOCS_ON_GITHUB = new URL(DOCS_URL).hostname === "github.com" ? logoByKey("github") : undefined;

type NavId = "traces" | "runs" | "compare" | "new";

function activeNav(pathname: string): NavId | null {
  if (pathname === "/runs/new") return "new";
  if (pathname.startsWith("/compare")) return "compare";
  if (pathname.startsWith("/runs")) return "runs";
  if (pathname === "/" || pathname.startsWith("/traces")) return "traces";
  return null;
}

const THEME_META: Record<ThemePref, { label: string; icon: ReactNode }> = {
  system: { label: "Theme: system", icon: <Monitor {...ICON} /> },
  light: { label: "Theme: light", icon: <Sun {...ICON} /> },
  dark: { label: "Theme: dark", icon: <Moon {...ICON} /> },
};

const RAIL_ITEM =
  "group relative grid size-10 shrink-0 place-items-center rounded-[12px] text-muted transition-[color,background-color,box-shadow] duration-150 hover:text-fg [&>svg]:size-[19px]";
const RAIL_ACTIVE = "bg-surface text-fg shadow-tile ring-1 ring-border";

function Tip({ children }: { children: string }) {
  return (
    <span className="pointer-events-none absolute top-1/2 left-full z-50 ml-2.5 hidden -translate-y-1/2 rounded-[8px] bg-btn px-2 py-1 text-[12px] leading-4 font-medium whitespace-nowrap text-btn-fg opacity-0 transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100 md:block">
      {children}
    </span>
  );
}

function RailLink({ to, label, active, children }: { to: string; label: string; active: boolean; children: ReactNode }) {
  return (
    <Link to={to} aria-label={label} aria-current={active ? "page" : undefined} className={cx(RAIL_ITEM, active && RAIL_ACTIVE)}>
      {children}
      <Tip>{label}</Tip>
    </Link>
  );
}

function RailDivider() {
  return <span aria-hidden className="mx-auto h-5 w-px shrink-0 bg-border-strong md:h-px md:w-5" />;
}

function WorkspaceName() {
  const workspace = useHealth().data?.workspace;
  if (workspace == null) return null;
  return (
    <span
      title={`Workspace: ${workspace}`}
      className="max-w-[40vw] truncate rounded-[8px] bg-surface px-2 py-1 font-mono text-[11px] leading-4 text-muted ring-1 ring-border md:-mt-3 md:mb-3 md:max-w-[60px] md:px-1.5"
    >
      {workspace}
    </span>
  );
}

function Rail() {
  const { pathname } = useLocation();
  const nav = activeNav(pathname);
  const theme = useThemePref();
  return (
    <nav
      aria-label="Main"
      className="sticky top-0 z-30 flex h-14 items-center gap-2 bg-window/90 px-3 backdrop-blur-md md:fixed md:inset-y-0 md:left-0 md:h-auto md:w-[72px] md:flex-col md:gap-2.5 md:bg-transparent md:px-0 md:py-4 md:backdrop-blur-none"
    >
      <Link to={paths.traces} aria-label="Cascade home" className="mr-1 md:mr-0 md:mb-5">
        <BrandTile />
      </Link>
      <WorkspaceName />
      <RailLink to={paths.traces} label="Traces" active={nav === "traces"}>
        <Layers {...ICON} />
      </RailLink>
      <RailLink to={paths.runs} label="Runs" active={nav === "runs"}>
        <Activity {...ICON} />
      </RailLink>
      <RailLink to={paths.compare()} label="Compare runs" active={nav === "compare"}>
        <GitCompareArrows {...ICON} />
      </RailLink>
      <RailDivider />
      <RailLink to={paths.newRun()} label="New run" active={nav === "new"}>
        <Plus {...ICON} />
      </RailLink>
      <div className="ml-auto flex items-center gap-1 md:mt-auto md:ml-0 md:flex-col md:gap-2.5">
        <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" aria-label="Docs (opens in a new tab)" className={RAIL_ITEM}>
          {DOCS_ON_GITHUB ? <LogoGlyph logo={DOCS_ON_GITHUB} className="size-[18px]" /> : <BookOpen {...ICON} />}
          <Tip>{DOCS_ON_GITHUB ? "Docs on GitHub" : "Docs"}</Tip>
        </a>
        <button type="button" aria-label={THEME_META[theme].label} onClick={themeStore.cycle} className={RAIL_ITEM}>
          {THEME_META[theme].icon}
          <Tip>{THEME_META[theme].label}</Tip>
        </button>
        <RailDivider />
        <button type="button" aria-label="Forget token" onClick={tokenStore.clear} className={RAIL_ITEM}>
          <LogOut {...ICON} />
          <Tip>Forget token</Tip>
        </button>
      </div>
    </nav>
  );
}

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
    <div aria-live="polite" className="pointer-events-none fixed bottom-4 left-4 z-[60] flex w-[min(360px,calc(100vw-32px))] flex-col gap-2 md:left-[88px]">
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
  return (
    <div className="min-h-dvh bg-window md:pl-[72px]">
      <Rail />
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
