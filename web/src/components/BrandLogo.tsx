import type { CSSProperties } from "react";
import type { Logo } from "../domain/logos";
import { cx } from "./cx";

function maskStyle(url: string): CSSProperties {
  const image = `url("${url}")`;
  return {
    maskImage: image,
    WebkitMaskImage: image,
    maskSize: "contain",
    WebkitMaskSize: "contain",
    maskRepeat: "no-repeat",
    WebkitMaskRepeat: "no-repeat",
    maskPosition: "center",
    WebkitMaskPosition: "center",
    backgroundColor: "currentColor",
  };
}

/** The single-color logo in currentColor, for places where brand color would clash. */
export function LogoGlyph({ logo, className }: { logo: Logo; className?: string }) {
  return <span role="img" aria-label={logo.name} className={cx("inline-block size-3.5 shrink-0", className)} style={maskStyle(logo.mono)} />;
}

const TILE = {
  sm: { tile: "size-[18px] rounded-[5px]", mark: "size-3", text: "text-[10px]" },
  md: { tile: "size-6 rounded-[7px]", mark: "size-3.5", text: "text-[11px]" },
  lg: { tile: "size-8 rounded-[9px]", mark: "size-[18px]", text: "text-[13px]" },
} as const;

/**
 * A rounded tile holding the color logo. The tile stays white in both themes so every brand color reads,
 * and a name with no logo gets its initial instead.
 */
export function LogoTile({ logo, fallback, size = "md", className }: { logo: Logo | undefined; fallback: string; size?: keyof typeof TILE; className?: string }) {
  const s = TILE[size];
  if (logo === undefined) {
    return (
      <span aria-hidden className={cx("grid shrink-0 place-items-center bg-surface-2 font-medium text-muted ring-1 ring-border ring-inset", s.tile, s.text, className)}>
        {fallback.replace(/[^a-z0-9]/gi, "").charAt(0).toUpperCase() || "?"}
      </span>
    );
  }
  return (
    <span title={logo.name} className={cx("grid shrink-0 place-items-center bg-white ring-1 ring-black/8 ring-inset", s.tile, className)}>
      {logo.color !== null && !logo.tileMono ? (
        <img src={logo.color} alt={logo.name} className={cx("object-contain", s.mark)} draggable={false} />
      ) : (
        <span role="img" aria-label={logo.name} className={cx("inline-block text-[#0b0b0a]", s.mark)} style={maskStyle(logo.mono)} />
      )}
    </span>
  );
}
