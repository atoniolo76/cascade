import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { Stepper } from "../components/ui";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { gpuDef, ROUTERS, type Engine, type FleetEntry, type RouterDef, type RouterId } from "./store";

/** Step 3: three router cards, then the engine settings behind a disclosure. */
export function StepRouter({
  fleet,
  router,
  engine,
  choose,
  setEngine,
}: {
  fleet: FleetEntry[];
  router: RouterId | null;
  engine: Engine;
  choose: (r: RouterId) => void;
  setEngine: (e: Partial<Engine>) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <div className="mt-8 grid gap-2.5" role="radiogroup" aria-label="Router">
        {ROUTERS.map((r) => (
          <RouterCard key={r.id} def={r} fleet={fleet} selected={router === r.id} onClick={() => { choose(r.id); }} />
        ))}
      </div>
      <div className="mt-3 rounded-[14px] border border-border">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => { setOpen((o) => !o); }}
          className="flex h-11 w-full items-center gap-2 rounded-[14px] px-3.5 text-left text-[13.5px] text-fg-2 transition-colors duration-150 hover:text-fg"
        >
          <ChevronRight className={cx("size-4 text-muted transition-transform duration-200 ease-out", open && "rotate-90")} {...ICON} />
          Engine settings
          <span className="ml-auto font-mono text-[12px] text-muted">
            {engine.maxRequests} · {engine.batchSize} · TP{engine.tensorParallel}
          </span>
        </button>
        {open && (
          <div className="ob-row-in grid gap-2.5 border-t border-border px-3.5 py-3.5">
            <Setting label="Max requests" value={engine.maxRequests} min={1} max={256} onChange={(n) => { setEngine({ maxRequests: n }); }} />
            <Setting label="Batch size" value={engine.batchSize} min={1} max={256} onChange={(n) => { setEngine({ batchSize: n }); }} />
            <Setting label="Tensor parallel" value={engine.tensorParallel} min={1} max={8} onChange={(n) => { setEngine({ tensorParallel: n }); }} />
          </div>
        )}
      </div>
    </>
  );
}

function RouterCard({ def, fleet, selected, onClick }: { def: RouterDef; fleet: FleetEntry[]; selected: boolean; onClick: () => void }) {
  const suited = fleet.filter((f) => def.suits.includes(f.gpu));
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      onClick={onClick}
      className={cx(
        "ob-fade flex items-center gap-4 rounded-[14px] border bg-surface px-4 py-3.5 text-left",
        selected ? "border-accent bg-accent-soft" : "border-border hover:border-border-strong hover:bg-surface-2",
      )}
    >
      <span className="min-w-0 flex-1">
        <span className="block text-[14px] leading-5 font-medium text-fg">{def.name}</span>
        <span className="block text-[13px] leading-5 text-muted">{def.blurb}</span>
      </span>
      <span className="flex shrink-0 flex-col items-end gap-1">
        <span className="flex gap-1">
          {fleet.map((f) => {
            const fits = def.suits.includes(f.gpu);
            return (
              <span
                key={f.gpu}
                className={cx(
                  "inline-flex h-5 items-center rounded-[6px] px-1.5 font-mono text-[11px] ring-1 ring-inset",
                  fits ? "bg-surface text-fg-2 ring-border-strong" : "text-faint line-through ring-border",
                )}
              >
                {gpuDef(f.gpu).short}
              </span>
            );
          })}
        </span>
        <span className="font-mono text-[11px] text-muted">
          {suited.length} of {fleet.length} suited
        </span>
      </span>
    </button>
  );
}

function Setting({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (n: number) => void }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-[13.5px] text-fg-2">{label}</span>
      <Stepper value={value} onChange={onChange} min={min} max={max} label={label} />
    </div>
  );
}
