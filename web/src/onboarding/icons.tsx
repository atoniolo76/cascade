import { Check, type LucideIcon } from "lucide-react";
import { LogoTile } from "../components/BrandLogo";
import { Mark } from "../components/Logo";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { logoByKey } from "../domain/logos";

const OWN = {
  sm: { tile: "size-[18px] rounded-[5px]", mark: "size-3" },
  md: { tile: "size-6 rounded-[7px]", mark: "size-3.5" },
  lg: { tile: "size-8 rounded-[9px]", mark: "size-[18px]" },
} as const;

export type IconSize = keyof typeof OWN;

/** A real brand logo on the app's white logo tile. */
export function BrandIcon({ brand, size }: { brand: "nvidia" | "modal" | "meta" | "sail"; size: IconSize }) {
  return <LogoTile logo={logoByKey(brand)} fallback={brand} size={size} />;
}

/** The Cascade mark on a tile, standing in for the workspace in the preview. */
export function WorkspaceIcon({ size }: { size: IconSize }) {
  const s = OWN[size];
  return (
    <span className={cx("grid shrink-0 place-items-center bg-surface ring-1 ring-border ring-inset", s.tile)}>
      <Mark className={cx("text-fg", s.mark)} />
    </span>
  );
}

/** A stroke icon on the same tile as the logos, so every row in the preview lines up. */
export function GlyphIcon({ Icon, size, tone = "plain" }: { Icon: LucideIcon; size: IconSize; tone?: "plain" | "accent" | "ok" }) {
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

/** A grey dot for queued checklist items, sized like CheckDot. */
export function QueuedDot() {
  return (
    <span className="grid size-5 place-items-center">
      <span className="size-1.5 rounded-full bg-block-off" />
    </span>
  );
}
