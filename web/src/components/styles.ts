import { cx } from "./cx";

export const ICON = { strokeWidth: 1.5 } as const;

export const THEAD = "border-b border-border bg-surface-2/60";
export const ROW = "border-b border-border transition-colors duration-150 last:border-b-0";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-btn text-btn-fg hover:bg-btn-hover shadow-tile",
  secondary: "bg-surface text-fg ring-1 ring-border-strong ring-inset hover:bg-surface-2",
  ghost: "text-fg-2 hover:bg-surface-2 hover:text-fg",
  danger: "bg-surface text-[var(--st-bad-ink)] ring-1 ring-border-strong ring-inset hover:bg-[var(--st-bad-bg)]",
};

export const BUTTON_SIZES = {
  md: "h-9 px-3.5 text-[14px] rounded-[12px]",
  sm: "h-8 px-3 text-[13px] rounded-[10px]",
} as const;

export function buttonClass({ variant = "secondary", size = "md" }: { variant?: ButtonVariant; size?: keyof typeof BUTTON_SIZES }) {
  return cx(
    "inline-flex shrink-0 items-center justify-center gap-1.5 font-medium whitespace-nowrap transition-[background-color,color,box-shadow,opacity] duration-150 select-none disabled:pointer-events-none disabled:opacity-45 aria-disabled:pointer-events-none aria-disabled:opacity-45",
    BUTTON_SIZES[size],
    BUTTON_VARIANTS[variant],
  );
}
