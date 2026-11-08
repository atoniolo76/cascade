import { Plus } from "lucide-react";
import { Stepper } from "../components/ui";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { GpuIcon } from "./icons";
import { GPUS, MAX_PER_GPU, type FleetEntry, type GpuDef, type GpuId } from "./store";

/** Step 2: a row per GPU type, each with Add that turns into a count stepper. */
export function StepFleet({ fleet, setCount }: { fleet: FleetEntry[]; setCount: (gpu: GpuId, count: number) => void }) {
  return (
    <>
      <ul className="mt-8 overflow-hidden rounded-[14px] border border-border bg-surface">
        {GPUS.map((g) => (
          <GpuRow key={g.id} gpu={g} count={fleet.find((f) => f.gpu === g.id)?.count ?? 0} setCount={(n) => { setCount(g.id, n); }} />
        ))}
      </ul>
      <p className="mt-3 flex items-center gap-2 text-[13px] text-muted">
        <ProfileTag profile="estimated" />
        Estimated profiles are marked and shown hatched in every chart.
      </p>
    </>
  );
}

function GpuRow({ gpu, count, setCount }: { gpu: GpuDef; count: number; setCount: (n: number) => void }) {
  return (
    <li className={cx("ob-fade flex h-[60px] items-center gap-3 border-b border-border px-3.5 last:border-b-0", count > 0 && "bg-accent-soft/50")}>
      <GpuIcon id={gpu.id} size="lg" />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-mono text-[13.5px] leading-5 font-medium text-fg">{gpu.name}</span>
        <span className="block truncate font-mono text-[11.5px] leading-4 text-muted">{gpu.memory}</span>
      </span>
      <ProfileTag profile={gpu.profile} />
      <span className="flex w-[124px] shrink-0 justify-end">
        {count === 0 ? (
          <button
            type="button"
            onClick={() => { setCount(1); }}
            aria-label={`Add ${gpu.name}`}
            className="inline-flex h-9 items-center gap-1 rounded-[10px] px-2.5 text-[13px] font-medium text-fg-2 transition-colors duration-150 hover:bg-surface-2 hover:text-fg"
          >
            <Plus className="size-3.5" {...ICON} />
            Add
          </button>
        ) : (
          <span className="ob-pop">
            <Stepper value={count} onChange={setCount} min={0} max={MAX_PER_GPU} label={`${gpu.name} count`} />
          </span>
        )}
      </span>
    </li>
  );
}

/** Measured in the ok green, estimated in the accent violet and hatched, like the charts draw them. */
export function ProfileTag({ profile }: { profile: GpuDef["profile"] }) {
  const measured = profile === "measured";
  return (
    <span
      className={cx(
        "inline-flex h-5 shrink-0 items-center rounded-[6px] px-1.5 text-[11.5px] font-medium",
        measured ? "bg-(--st-ok-bg) text-(--st-ok-ink)" : "bg-accent-soft text-accent-ink",
      )}
      style={measured ? undefined : { backgroundImage: "repeating-linear-gradient(135deg, transparent 0 3px, var(--accent-soft) 3px 5px)" }}
    >
      {measured ? "Measured" : "Estimated"}
    </span>
  );
}
