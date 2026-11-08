import type { ReactNode } from "react";
import { Activity, Cpu, Layers, Route } from "lucide-react";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { SourceIcon, WorkspaceIcon } from "./icons";
import { activity, sourceDef, type ActivityRow, type OnboardingState } from "./store";

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
          <span className="absolute left-1/2 -translate-x-1/2 rounded-[8px] bg-surface-2 px-2.5 py-0.5 font-mono text-[11.5px] text-muted">
            {state.workspace}.cascadesim.app
          </span>
        </div>
        <div className="flex min-h-0 flex-1">
          <Sidebar state={state} />
          <main className="min-w-0 flex-1 p-4">
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
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}

function Sidebar({ state }: { state: OnboardingState }) {
  return (
    <aside className="flex w-[188px] shrink-0 flex-col border-r border-border bg-panel px-3 py-3">
      <div className="flex items-center gap-2 px-1">
        <WorkspaceIcon size="md" />
        <span className="truncate text-[13px] font-medium text-fg">{state.workspace}</span>
      </div>
      <Section title="Traces" icon={<Layers {...ICON} />} placeholder="A trace">
        {state.sources.map((s) => (
          <Item key={s.id} pending={s.status === "connecting"} icon={<SourceIcon id={s.id} size="sm" />}>
            {s.id === "upload" ? s.detail : sourceDef(s.id).name}
          </Item>
        ))}
      </Section>
      <Section title="Fleet" icon={<Cpu {...ICON} />} placeholder="Your first GPU" />
      <Section title="Router" icon={<Route {...ICON} />} placeholder="A policy" />
      <div className="mt-auto flex items-center gap-2 border-t border-border px-1 pt-3">
        <span className="grid size-5 place-items-center rounded-full bg-accent-soft text-[10px] font-medium text-accent-ink">Y</span>
        <span className="text-[12px] text-muted">Just you</span>
      </div>
    </aside>
  );
}

function Section({ title, icon, placeholder, children }: { title: string; icon: ReactNode; placeholder: string; children?: ReactNode[] }) {
  const empty = children === undefined || children.length === 0;
  return (
    <div className="mt-5">
      <p className="flex items-center gap-1.5 px-1 text-[12px] text-muted [&>svg]:size-3.5 [&>svg]:text-faint">
        {icon}
        {title}
      </p>
      <ul className="mt-1.5 space-y-0.5">
        {empty ? (
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

function Item({ pending, icon, children }: { pending: boolean; icon: ReactNode; children: ReactNode }) {
  return (
    <li className={cx("ob-row-in ob-fade flex h-7 items-center gap-2 rounded-[8px] px-1 text-[12.5px] text-fg-2", pending && "opacity-40")}>
      {icon}
      <span className="truncate">{children}</span>
    </li>
  );
}

function Row({ row, last }: { row: ActivityRow; last: boolean }) {
  return (
    <li className={cx("ob-row-in ob-fade relative flex items-start gap-3 pb-3.5", row.pending && "opacity-40")}>
      {!last && <span className="absolute top-7 bottom-0.5 left-[11.5px] w-px bg-border" aria-hidden />}
      {row.icon.kind === "workspace" ? <WorkspaceIcon size="md" /> : <SourceIcon id={row.icon.id} size="md" />}
      <p className="min-w-0 pt-0.5 text-[13px] leading-5 text-muted">
        You {row.verb} <span className="font-medium text-fg">{row.subject}</span>
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
