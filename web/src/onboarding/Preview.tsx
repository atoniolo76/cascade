import type { ReactNode } from "react";
import { Activity, Cpu, Layers, Play } from "lucide-react";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { BrandIcon, GlyphIcon, WorkspaceIcon } from "./icons";
import { activity, configDef, errorPct, footer, simStates, traceDef, type ActivityIcon, type ActivityRow, type OnboardingState, type SimState } from "./store";

/** "Your workspace, so far": a small copy of the app, drawn only from the onboarding state. */
export function Preview({ state }: { state: OnboardingState }) {
  return (
    <div className="ob-glow relative flex h-full min-h-[560px] flex-col overflow-hidden rounded-[18px] border border-border">
      <p className="px-8 pt-7 pb-4 text-[13px] text-muted">Your workspace, so far</p>
      <div className="ml-8 flex min-h-0 flex-1 flex-col overflow-hidden rounded-tl-[14px] border-t border-l border-border bg-window shadow-pop">
        <div className="relative flex h-10 shrink-0 items-center border-b border-border bg-panel px-4">
          <span className="flex gap-1.5" aria-hidden>
            {[0, 1, 2].map((i) => (
              <span key={i} className="size-2.5 rounded-full bg-block-off" />
            ))}
          </span>
          <span className="absolute left-1/2 flex -translate-x-1/2 items-center gap-1.5 rounded-[8px] bg-surface-2 px-2.5 py-0.5 font-mono text-[11.5px] text-muted">
            {state.workspace} · Sail
          </span>
        </div>
        <div className="flex min-h-0 flex-1">
          <Sidebar state={state} />
          <main className="min-w-0 flex-1 overflow-y-auto p-4">
            <div className="rounded-[14px] border border-border bg-surface shadow-card">
              <div className="flex h-11 items-center gap-2 border-b border-border px-4 text-fg">
                <Activity className="size-4 text-muted" {...ICON} />
                <h3 className="text-[14px] font-medium">Activity</h3>
              </div>
              <ol className="px-4 pt-3.5 pb-1">
                {activity(state).map((row, i, rows) => (
                  <Row key={row.key} row={row} last={i === rows.length - 1} />
                ))}
              </ol>
              <RunCard state={state} />
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}

function Sidebar({ state }: { state: OnboardingState }) {
  const { run } = state;
  return (
    <aside className="flex w-[188px] shrink-0 flex-col border-r border-border bg-panel px-3 py-3">
      <div className="flex items-center gap-2 px-1">
        <WorkspaceIcon size="md" />
        <span className="truncate text-[13px] font-medium text-fg">{state.workspace}</span>
      </div>
      <Section title="Traces" icon={<Layers {...ICON} />} placeholder="A trace">
        {state.trace === null ? [] : [<Item key={state.trace} icon={<BrandIcon brand="meta" size="sm" />}>{traceDef(state.trace).name}</Item>]}
      </Section>
      <Section title="GPUs" icon={<Cpu {...ICON} />} placeholder="A GPU config">
        {state.entries.map((e) => (
          <Item key={e.config} icon={<BrandIcon brand="nvidia" size="sm" />} aside={<span className="font-mono text-[11.5px] text-muted">×{e.count}</span>}>
            <span className="font-mono text-[11.5px]">{e.config}</span>
          </Item>
        ))}
      </Section>
      <Section title="Runs" icon={<Play {...ICON} />} placeholder="Your first run">
        {run.status === "idle"
          ? []
          : [
              <Item
                key="run"
                icon={<GlyphIcon Icon={Play} size="sm" tone={run.status === "done" ? "ok" : "accent"} />}
                aside={<span className={cx("size-1.5 rounded-full", run.status === "running" ? "animate-pulse-soft bg-accent" : "bg-(--st-ok-dot)")} />}
              >
                First run
              </Item>,
            ]}
      </Section>
      <div className="mt-auto flex items-center gap-2 border-t border-border px-1 pt-3">
        <span className="grid size-5 place-items-center rounded-full bg-accent-soft text-[10px] font-medium text-accent-ink">Y</span>
        <span key={footer(state)} className="ob-row-in text-[12px] text-muted">
          {footer(state)}
        </span>
      </div>
    </aside>
  );
}

function Section({ title, icon, placeholder, children }: { title: string; icon: ReactNode; placeholder: string; children: ReactNode[] }) {
  return (
    <div className="mt-5">
      <p className="flex items-center gap-1.5 px-1 text-[12px] text-muted [&>svg]:size-3.5 [&>svg]:text-faint">
        {icon}
        {title}
      </p>
      <ul className="mt-1.5 space-y-0.5">
        {children.length === 0 ? (
          <li className="flex h-7 items-center gap-2 px-1 text-[12.5px] text-faint">
            <span className="size-[18px] rounded-[5px] border border-dashed border-border-strong" />
            {placeholder}
          </li>
        ) : (
          children
        )}
      </ul>
    </div>
  );
}

function Item({ icon, aside, children }: { icon: ReactNode; aside?: ReactNode; children: ReactNode }) {
  return (
    <li className="ob-row-in flex h-7 items-center gap-2 rounded-[8px] px-1 text-[12.5px] text-fg-2">
      {icon}
      <span className="min-w-0 flex-1 truncate">{children}</span>
      {aside}
    </li>
  );
}

function RowIcon({ icon }: { icon: ActivityIcon }) {
  switch (icon.kind) {
    case "workspace":
      return <WorkspaceIcon size="md" />;
    case "trace":
      return <BrandIcon brand="meta" size="md" />;
    case "config":
      return <BrandIcon brand="nvidia" size="md" />;
    case "run":
      return <GlyphIcon Icon={Play} size="md" tone="ok" />;
  }
}

function Row({ row, last }: { row: ActivityRow; last: boolean }) {
  return (
    <li className={cx("ob-row-in ob-fade relative flex items-start gap-3 pb-3.5", row.pending && "opacity-50")}>
      {!last && <span className="absolute top-7 bottom-0.5 left-[11.5px] w-px bg-border" aria-hidden />}
      <RowIcon icon={row.icon} />
      <p className="min-w-0 pt-0.5 text-[13px] leading-5 text-muted">
        {row.lead !== "" && `${row.lead} `}
        <span className="font-medium text-fg">{row.subject}</span>
        {row.tail}
        {row.detail !== null && (
          <>
            {" · "}
            <span className={cx(/\d/.test(row.detail) && "font-mono text-[12px]")}>{row.detail}</span>
          </>
        )}
      </p>
    </li>
  );
}

const TILE: Record<SimState, { label: string; ink: string; bg: string; dot: string }> = {
  forking: { label: "Forking", ink: "var(--st-pending-ink)", bg: "var(--st-pending-bg)", dot: "var(--st-pending-dot)" },
  running: { label: "Simulating", ink: "var(--st-active-ink)", bg: "var(--st-active-bg)", dot: "var(--st-active-dot)" },
  done: { label: "Done", ink: "var(--st-ok-ink)", bg: "var(--st-ok-bg)", dot: "var(--st-ok-dot)" },
};

/** The run's simulations as tiles, one per forked sandbox, like the run page's grid. */
function RunCard({ state }: { state: OnboardingState }) {
  const { run, entries, trace } = state;
  if (run.status === "idle" || trace === null) return null;
  const configs = entries.flatMap((e) => Array.from({ length: e.count }, (_, i) => ({ config: e.config, replica: i })));
  const shown = configs.slice(0, 12);
  const states = simStates(configs.length, run.progress);
  const e2e = entries.some((e) => configDef(e.config).e2e);
  return (
    <div className="ob-row-in mx-4 mb-4 ml-[52px] rounded-[12px] border border-border bg-window/60 p-3">
      <div className="grid grid-cols-2 gap-1.5 xl:grid-cols-3">
        {shown.map((c, i) => {
          const t = TILE[states[i] ?? "forking"];
          return (
            <div key={`${c.config}-${c.replica}`} className="ob-fade rounded-[8px] px-2 py-1.5" style={{ background: t.bg }}>
              <div className="truncate font-mono text-[11px] text-fg-2">
                {c.config} <span className="text-faint">#{c.replica}</span>
              </div>
              <div className="flex items-center gap-1.5 text-[11px]" style={{ color: t.ink }}>
                <span className={cx("size-1.5 rounded-full", states[i] === "running" && "animate-pulse-soft")} style={{ background: t.dot }} />
                {t.label}
              </div>
            </div>
          );
        })}
      </div>
      {configs.length > shown.length && <p className="mt-2 font-mono text-[11px] text-muted">+{configs.length - shown.length} more</p>}
      {run.status === "done" && (
        <p className="ob-row-in mt-2.5 text-[12px] leading-4 text-(--st-ok-ink)">
          {e2e ? (
            <>
              end-to-end <span className="font-mono">{errorPct(traceDef(trace).recorded)}</span> vs the real run
            </>
          ) : (
            "per-kernel times ready"
          )}
        </p>
      )}
    </div>
  );
}
