import { Plus } from "lucide-react";
import { Stepper } from "../components/ui";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { BrandIcon } from "./icons";
import { configDef, MAX_REPLICAS, traceDef, type ConfigDef, type ConfigId, type Entry, type TraceId } from "./store";

/** Step 2: the Accel-Sim configs this trace has results for, each with replicas. */
export function StepGpus({ trace, entries, setCount }: { trace: TraceId; entries: Entry[]; setCount: (c: ConfigId, n: number) => void }) {
  return (
    <>
      <ul className="mt-8 overflow-hidden rounded-[14px] border border-border bg-surface">
        {traceDef(trace).configs.map((id) => (
          <ConfigRow
            key={id}
            def={configDef(id)}
            count={entries.find((e) => e.config === id)?.count ?? 0}
            setCount={(n) => { setCount(id, n); }}
          />
        ))}
      </ul>
      <p className="mt-3 text-[13px] leading-5 text-muted">Each simulation runs in its own Sail sandbox, forked from your workspace. Samples replay their recorded Accel-Sim results.</p>
    </>
  );
}

function ConfigRow({ def, count, setCount }: { def: ConfigDef; count: number; setCount: (n: number) => void }) {
  return (
    <li className={cx("ob-fade flex h-[60px] items-center gap-3 border-b border-border px-3.5 last:border-b-0", count > 0 && "bg-accent-soft/50")}>
      <BrandIcon brand="nvidia" size="lg" />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-mono text-[13.5px] leading-5 font-medium text-fg">{def.id}</span>
        <span className="block truncate text-[12px] leading-4 text-muted">
          {def.gpu} · <span className="font-mono text-[11.5px]">{def.arch} · {def.clockMhz} MHz</span>
        </span>
      </span>
      <span
        className={cx(
          "inline-flex h-5 shrink-0 items-center rounded-[6px] px-1.5 text-[11.5px] font-medium",
          def.e2e ? "bg-(--st-ok-bg) text-(--st-ok-ink)" : "bg-surface-2 text-muted",
        )}
      >
        {def.e2e ? "End-to-end" : "Kernel times"}
      </span>
      <span className="flex w-[124px] shrink-0 justify-end">
        {count === 0 ? (
          <button
            type="button"
            onClick={() => { setCount(1); }}
            aria-label={`Add ${def.id}`}
            className="inline-flex h-9 items-center gap-1 rounded-[10px] px-2.5 text-[13px] font-medium text-fg-2 transition-colors duration-150 hover:bg-surface-2 hover:text-fg"
          >
            <Plus className="size-3.5" {...ICON} />
            Add
          </button>
        ) : (
          <span className="ob-pop">
            <Stepper value={count} onChange={setCount} min={0} max={MAX_REPLICAS} label={`${def.id} replicas`} />
          </span>
        )}
      </span>
    </li>
  );
}
