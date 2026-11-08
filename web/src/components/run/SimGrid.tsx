import { Boxes, CircleAlert } from "lucide-react";
import type { SimState, SimSummary } from "../../api/types";
import { formatInt, formatPercent } from "../../domain/format";
import type { Series } from "../../domain/stats";
import { SIM_BLOCK_ORDER, SIM_STATE, TONE_VARS, countSimStates } from "../../domain/status";
import { cx } from "../cx";
import { Status } from "../meters";
import { Panel, PanelHeader, Swatch } from "../ui";
import { ICON } from "../styles";

function progressOf(sim: SimSummary): number {
  if (sim.state === "succeeded") return 1;
  return sim.kernels_total > 0 ? sim.kernels_done / sim.kernels_total : 0;
}

function tileLabel(sim: SimSummary): string {
  const meta = SIM_STATE[sim.state];
  return `${sim.gpu_config} replica ${sim.replica}: ${meta.label.toLowerCase()}, ${sim.kernels_done} of ${sim.kernels_total} kernels`;
}

function SimTile({ sim, selected, onSelect }: { sim: SimSummary; selected: boolean; onSelect: () => void }) {
  const meta = SIM_STATE[sim.state];
  const v = TONE_VARS[meta.tone];
  const pending = sim.state === "pending";
  return (
    <button
      type="button"
      aria-label={tileLabel(sim)}
      aria-pressed={selected}
      title={`Replica ${sim.replica} · ${meta.label} · ${sim.kernels_done}/${sim.kernels_total} kernels`}
      onClick={onSelect}
      className={cx(
        "relative size-[30px] shrink-0 overflow-hidden rounded-[8px] transition-[box-shadow,transform] duration-150 hover:-translate-y-px",
        selected ? "ring-2 ring-fg ring-offset-2 ring-offset-surface" : "ring-1 ring-fg/5 ring-inset",
        sim.state === "launching" && "animate-pulse-soft",
      )}
      style={{ background: pending ? "var(--block-off)" : v.bg }}
    >
      <span
        aria-hidden
        className="absolute inset-x-0 bottom-0 transition-[height,background-color] duration-500 ease-out"
        style={{ height: `${progressOf(sim) * 100}%`, background: v.dot }}
      />
      {(sim.state === "lost" || sim.state === "failed") && (
        <span aria-hidden className={cx("absolute inset-0 rounded-[8px]", sim.state === "lost" && "animate-pulse-soft")} style={{ boxShadow: `inset 0 0 0 2px ${v.dot}` }} />
      )}
    </button>
  );
}

function ConfigRow({ series, sims, selectedId, onSelect }: { series: Series; sims: SimSummary[]; selectedId: string | null; onSelect: (id: string) => void }) {
  const finished = sims.filter((s) => SIM_STATE[s.state].terminal).length;
  const done = sims.reduce((n, s) => n + s.kernels_done, 0);
  const total = sims.reduce((n, s) => n + s.kernels_total, 0);
  return (
    <div className="grid gap-x-6 gap-y-3 py-4 md:grid-cols-[220px_minmax(0,1fr)] md:items-center">
      <div className="flex min-w-0 items-center gap-2.5">
        <Swatch color={series.color} />
        <span className="min-w-0">
          <span className="block truncate font-mono text-[14px] font-medium text-fg">{series.label}</span>
          <span className="block text-[12px] leading-4 text-muted tabular-nums">
            {finished}/{sims.length} finished · {total > 0 ? formatPercent(done / total) : "0%"} of kernels
          </span>
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {sims.map((sim) => (
          <SimTile
            key={sim.id}
            sim={sim}
            selected={sim.id === selectedId}
            onSelect={() => {
              onSelect(sim.id);
            }}
          />
        ))}
      </div>
    </div>
  );
}

function Legend({ counts }: { counts: Partial<Record<SimState, number>> }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1.5">
      {SIM_BLOCK_ORDER.map((s) => {
        const n = counts[s] ?? 0;
        if (n === 0) return null;
        const meta = SIM_STATE[s];
        return (
          <li key={s} className="text-[13px]">
            <Status tone={meta.tone} live={meta.live} label={`${formatInt(n)} ${meta.label.toLowerCase()}`} />
          </li>
        );
      })}
    </ul>
  );
}

export function SimGrid({
  sims,
  series,
  selectedId,
  onSelect,
}: {
  sims: readonly SimSummary[];
  series: readonly Series[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const counts = countSimStates(sims.map((s) => s.state));
  const troubled = sims.filter((s) => s.state === "failed" || s.state === "lost");
  return (
    <Panel>
      <PanelHeader
        icon={<Boxes {...ICON} />}
        title="Sims"
        subtitle={
          <span>
            One tile per sailbox, filled by kernel progress. Click a tile, or press <Kbd>j</Kbd> <Kbd>k</Kbd> to step through.
          </span>
        }
        actions={<Legend counts={counts} />}
      />
      <div className="divide-y divide-border px-5">
        {series.map((s) => (
          <ConfigRow key={s.key} series={s} sims={sims.filter((sim) => sim.gpu_config === s.key)} selectedId={selectedId} onSelect={onSelect} />
        ))}
      </div>
      {troubled.length > 0 && (
        <div className="border-t border-border px-5 py-4">
          <div className="mb-2 flex items-center gap-2 text-[13px] font-medium text-fg">
            <CircleAlert className="size-4 text-muted" {...ICON} />
            Needs a look
          </div>
          <ul className="flex flex-col">
            {troubled.map((s) => {
              const meta = SIM_STATE[s.state];
              return (
                <li key={s.id}>
                  <button
                    type="button"
                    onClick={() => {
                      onSelect(s.id);
                    }}
                    className="grid w-full grid-cols-[96px_minmax(0,1fr)] items-center gap-3 rounded-[10px] px-2 py-1.5 text-left transition-colors duration-150 hover:bg-surface-2 sm:grid-cols-[96px_200px_minmax(0,1fr)]"
                  >
                    <Status tone={meta.tone} label={meta.label} />
                    <span className="truncate font-mono text-[13px] text-fg">
                      {s.gpu_config} #{s.replica}
                    </span>
                    <span className="col-span-2 truncate text-[13px] text-muted sm:col-span-1" title={s.error ?? undefined}>
                      {s.error ?? (s.state === "lost" ? "No heartbeat; waiting for it to resume" : "")}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </Panel>
  );
}

export function Kbd({ children }: { children: string }) {
  return (
    <kbd className="mx-0.5 inline-grid h-5 min-w-5 place-items-center rounded-[6px] bg-surface px-1 font-sans text-[11px] font-medium text-fg-2 ring-1 ring-border-strong">
      {children}
    </kbd>
  );
}
