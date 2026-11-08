import { Check, Upload } from "lucide-react";
import { LogoTile } from "../components/BrandLogo";
import { Mark } from "../components/Logo";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { logoByKey } from "../domain/logos";
import type { SourceId } from "./store";

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
    <span className={cx("grid shrink-0 place-items-center bg-surface shadow-tile ring-1 ring-border ring-inset", s.tile)}>
      <Mark className={cx("text-fg", s.mark)} />
    </span>
  );
}
