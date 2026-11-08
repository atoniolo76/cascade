import { useId } from "react";
import { cx } from "./cx";

/** Tier clip paths from public/brand/mark.svg. */
const TIERS = [
  { shift: -4, fill: "currentColor", d: "M-6.00 -4.00 L-3.50 -4.00 L-1.00 -4.00 L1.50 -4.00 L4.00 -4.00 L6.50 -4.00 L9.00 -4.00 L11.50 -4.00 L14.00 -4.00 L16.50 -4.00 L19.00 -4.00 L21.50 -4.00 L24.00 -4.00 L26.50 -4.00 L29.00 -4.00 L31.50 -4.00 L34.00 -4.00 L36.50 -4.00 L39.00 -4.00 L41.50 -4.00 L44.00 -4.00 L46.50 -4.00 L49.00 -4.00 L51.50 -4.00 L54.00 -4.00 L54.00 18.39 L51.50 17.02 L49.00 15.77 L46.50 14.63 L44.00 13.62 L41.50 12.72 L39.00 11.95 L36.50 11.29 L34.00 10.75 L31.50 10.34 L29.00 10.04 L26.50 9.86 L24.00 9.80 L21.50 9.86 L19.00 10.04 L16.50 10.34 L14.00 10.75 L11.50 11.29 L9.00 11.95 L6.50 12.72 L4.00 13.62 L1.50 14.63 L-1.00 15.77 L-3.50 17.02 L-6.00 18.39Z" },
  { shift: 0, fill: "#6D4AE0", d: "M-6.00 21.79 L-3.50 20.42 L-1.00 19.17 L1.50 18.03 L4.00 17.02 L6.50 16.12 L9.00 15.35 L11.50 14.69 L14.00 14.15 L16.50 13.74 L19.00 13.44 L21.50 13.26 L24.00 13.20 L26.50 13.26 L29.00 13.44 L31.50 13.74 L34.00 14.15 L36.50 14.69 L39.00 15.35 L41.50 16.12 L44.00 17.02 L46.50 18.03 L49.00 19.17 L51.50 20.42 L54.00 21.79 L54.00 32.39 L51.50 31.02 L49.00 29.77 L46.50 28.63 L44.00 27.62 L41.50 26.72 L39.00 25.95 L36.50 25.29 L34.00 24.75 L31.50 24.34 L29.00 24.04 L26.50 23.86 L24.00 23.80 L21.50 23.86 L19.00 24.04 L16.50 24.34 L14.00 24.75 L11.50 25.29 L9.00 25.95 L6.50 26.72 L4.00 27.62 L1.50 28.63 L-1.00 29.77 L-3.50 31.02 L-6.00 32.39Z" },
  { shift: 4, fill: "#A35CE8", d: "M-6.00 35.79 L-3.50 34.42 L-1.00 33.17 L1.50 32.03 L4.00 31.02 L6.50 30.12 L9.00 29.35 L11.50 28.69 L14.00 28.15 L16.50 27.74 L19.00 27.44 L21.50 27.26 L24.00 27.20 L26.50 27.26 L29.00 27.44 L31.50 27.74 L34.00 28.15 L36.50 28.69 L39.00 29.35 L41.50 30.12 L44.00 31.02 L46.50 32.03 L49.00 33.17 L51.50 34.42 L54.00 35.79 L54.00 52.00 L51.50 52.00 L49.00 52.00 L46.50 52.00 L44.00 52.00 L41.50 52.00 L39.00 52.00 L36.50 52.00 L34.00 52.00 L31.50 52.00 L29.00 52.00 L26.50 52.00 L24.00 52.00 L21.50 52.00 L19.00 52.00 L16.50 52.00 L14.00 52.00 L11.50 52.00 L9.00 52.00 L6.50 52.00 L4.00 52.00 L1.50 52.00 L-1.00 52.00 L-3.50 52.00 L-6.00 52.00Z" },
] as const;

/** The tiered orb. The first tier takes currentColor; clip ids are unique per instance. */
export function Mark({ className }: { className?: string }) {
  const prefix = `mark${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  return (
    <svg viewBox="0 0 48 48" className={cx("shrink-0", className)} aria-hidden>
      <defs>
        {TIERS.map((t, i) => (
          <clipPath key={i} id={`${prefix}-t${i}`}>
            <path d={t.d} />
          </clipPath>
        ))}
      </defs>
      {TIERS.map((t, i) => (
        <g key={i} transform={`translate(${t.shift} 0)`}>
          <circle cx="24" cy="24" r="21" fill={t.fill} clipPath={`url(#${prefix}-t${i})`} />
        </g>
      ))}
    </svg>
  );
}

export function BrandTile({ className }: { className?: string }) {
  return (
    <span className={cx("grid size-10 shrink-0 place-items-center rounded-[12px] bg-surface shadow-tile ring-1 ring-border", className)}>
      <Mark className="size-[24px] text-fg" />
    </span>
  );
}
