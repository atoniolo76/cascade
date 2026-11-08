import { Check, Cpu, Upload, type LucideIcon } from "lucide-react";
import { LogoTile } from "../components/BrandLogo";
import { Mark } from "../components/Logo";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { logoByKey } from "../domain/logos";
import { gpuDef, type GpuId, type SourceId } from "./store";

const BRAND: Partial<Record<SourceId, string>> = { modal: "modal", vllm: "vllm", aiperf: "nvidia", openai: "openai" };

const OWN = {
  sm: { tile: "size-[18px] rounded-[5px]", mark: "size-3" },
  md: { tile: "size-6 rounded-[7px]", mark: "size-3.5" },
  lg: { tile: "size-8 rounded-[9px]", mark: "size-[18px]" },
} as const;

/** The source's real logo on the app's white logo tile; the sample wears the Cascade mark. */
export function SourceIcon({ id, size }: { id: SourceId; size: keyof typeof OWN }) {
  const brand = BRAND[id];
  if (brand !== undefined) return <LogoTile logo={logoByKey(brand)} fallback={brand} size={size} />;
  const s = OWN[size];
  return (
    <span className={cx("grid shrink-0 place-items-center bg-surface text-fg ring-1 ring-border ring-inset", s.tile)}>
      {id === "sample" ? <Mark className={s.mark} /> : <Upload className={s.mark} {...ICON} />}
    </span>
  );
}

export function CheckDot({ className }: { className?: string }) {
  return (
    <span className={cx("ob-pop grid size-5 place-items-center rounded-full bg-(--st-ok-dot) text-white", className)}>
      <Check className="size-3" strokeWidth={2.25} />
    </span>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" fill="none" className={cx("ob-spin", className)} aria-hidden>
      <circle cx="8" cy="8" r="6" stroke="var(--border-strong)" strokeWidth={1.5} />
      <path d="M8 2a6 6 0 0 1 6 6" stroke="var(--accent)" strokeWidth={1.5} strokeLinecap="round" />
    </svg>
  );
}

/** The Cascade mark on a tile, standing in for the workspace in the preview. */
export function WorkspaceIcon({ size }: { size: keyof typeof OWN }) {
  const s = OWN[size];
  return (
    <span className={cx("grid shrink-0 place-items-center bg-surface ring-1 ring-border ring-inset", s.tile)}>
      <Mark className={cx("text-fg", s.mark)} />
    </span>
  );
}

/** NVIDIA parts wear the NVIDIA logo; the TPU gets a plain chip, since there is no Google logo in the set. */
export function GpuIcon({ id, size }: { id: GpuId; size: keyof typeof OWN }) {
  if (gpuDef(id).vendor === "nvidia") return <LogoTile logo={logoByKey("nvidia")} fallback="nvidia" size={size} />;
  return <GlyphIcon Icon={Cpu} size={size} />;
}

/** A stroke icon on the same tile as the logos, so every row in the preview lines up. */
export function GlyphIcon({ Icon, size, tone = "plain" }: { Icon: LucideIcon; size: keyof typeof OWN; tone?: "plain" | "accent" | "ok" }) {
  const s = OWN[size];
  const look = {
    plain: "bg-surface text-fg ring-border",
    accent: "bg-accent-soft text-accent-ink ring-transparent",
    ok: "bg-(--st-ok-bg) text-(--st-ok-ink) ring-transparent",
  }[tone];
  return (
    <span className={cx("grid shrink-0 place-items-center ring-1 ring-inset", look, s.tile)}>
      <Icon className={s.mark} {...ICON} />
    </span>
  );
}

/** A grey dot for queued checklist items, sized like CheckDot. */
export function QueuedDot() {
  return (
    <span className="grid size-5 place-items-center">
      <span className="size-1.5 rounded-full bg-block-off" />
    </span>
  );
}
