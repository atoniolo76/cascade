import type { ButtonHTMLAttributes, CSSProperties, ReactNode } from "react";
import { Minus, Plus } from "lucide-react";
import { TONE_VARS, type Tone } from "../domain/status";
import { cx } from "./cx";

export function Dot({ tone, live = false }: { tone: Tone; live?: boolean }) {
  const v = TONE_VARS[tone];
  return (
    <span
      aria-hidden
      className={cx("inline-block size-1.5 shrink-0 rounded-full", live && "animate-pulse-soft")}
      style={{ background: v.dot, color: v.dot }}
    />
  );
}

export function StatusPill({ tone, live = false, label, className }: { tone: Tone; live?: boolean; label: string; className?: string }) {
  const v = TONE_VARS[tone];
  return (
    <span
      className={cx(
        "inline-flex h-5 shrink-0 items-center gap-1.5 rounded-full px-2 text-[12px] leading-none font-medium whitespace-nowrap transition-colors duration-200",
        tone === "off" && "ring-1 ring-border ring-inset",
        className,
      )}
      style={{ background: v.bg, color: v.ink }}
    >
      <Dot tone={tone} live={live} />
      {label}
    </span>
  );
}

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-accent text-accent-fg hover:bg-accent-hover shadow-card",
  secondary: "bg-surface text-fg ring-1 ring-border-strong ring-inset hover:bg-surface-2 shadow-card",
  ghost: "text-fg-2 hover:bg-surface-2 hover:text-fg",
  danger: "bg-surface text-[var(--st-bad-ink)] ring-1 ring-border-strong ring-inset hover:bg-[var(--st-bad-bg)]",
};

export function Button({
  variant = "secondary",
  size = "md",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: "sm" | "md" }) {
  return (
    <button
      type="button"
      {...props}
      className={cx(
        "inline-flex shrink-0 items-center justify-center gap-1.5 rounded-lg font-medium whitespace-nowrap transition-[background-color,color,box-shadow,opacity] duration-150 select-none disabled:pointer-events-none disabled:opacity-45",
        size === "md" ? "h-8 px-3 text-[13px]" : "h-7 px-2.5 text-[12px]",
        BUTTON_VARIANTS[variant],
        className,
      )}
    />
  );
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cx("rounded-xl border border-border bg-surface shadow-card", className)}>{children}</div>;
}

export function CardHeader({ title, subtitle, actions }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex min-h-14 flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-border px-4 py-3">
      <div className="min-w-0">
        <h2 className="text-[14px] leading-5 font-semibold">{title}</h2>
        {subtitle !== undefined && <p className="text-[12px] leading-4 text-muted">{subtitle}</p>}
      </div>
      {actions !== undefined && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Mono({ children, className }: { children: string; className?: string }) {
  return (
    <span title={children} className={cx("block min-w-0 truncate font-mono text-[12px]", className)}>
      {children}
    </span>
  );
}

export function Skeleton({ className, style }: { className?: string; style?: CSSProperties }) {
  return <span aria-hidden style={style} className={cx("block animate-shimmer rounded-md bg-surface-2", className)} />;
}

export function ProgressBar({ value, tone = "active", className }: { value: number; tone?: Tone; className?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className={cx("h-1.5 overflow-hidden rounded-full bg-surface-2", className)} role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
      <div
        className="h-full rounded-full transition-[width,background-color] duration-300 ease-out"
        style={{ width: `${pct}%`, background: TONE_VARS[tone].dot }}
      />
    </div>
  );
}

export function Stepper({ value, onChange, min, max, disabled = false, label }: { value: number; onChange: (n: number) => void; min: number; max: number; disabled?: boolean; label: string }) {
  const clamp = (n: number) => Math.max(min, Math.min(max, n));
  return (
    <div className={cx("inline-flex h-8 items-center rounded-lg ring-1 ring-border-strong ring-inset", disabled && "opacity-45")}>
      <button
        type="button"
        aria-label={`Decrease ${label}`}
        disabled={disabled || value <= min}
        onClick={() => {
          onChange(clamp(value - 1));
        }}
        className="grid h-8 w-8 place-items-center rounded-l-lg text-muted transition-colors duration-150 hover:bg-surface-2 hover:text-fg disabled:pointer-events-none disabled:opacity-40"
      >
        <Minus className="size-3.5" />
      </button>
      <input
        aria-label={label}
        inputMode="numeric"
        disabled={disabled}
        value={value}
        onChange={(e) => {
          const n = Number.parseInt(e.target.value, 10);
          if (!Number.isNaN(n)) onChange(clamp(n));
        }}
        className="h-8 w-10 bg-transparent text-center text-[13px] font-medium tabular-nums outline-none"
      />
      <button
        type="button"
        aria-label={`Increase ${label}`}
        disabled={disabled || value >= max}
        onClick={() => {
          onChange(clamp(value + 1));
        }}
        className="grid h-8 w-8 place-items-center rounded-r-lg text-muted transition-colors duration-150 hover:bg-surface-2 hover:text-fg disabled:pointer-events-none disabled:opacity-40"
      >
        <Plus className="size-3.5" />
      </button>
    </div>
  );
}

export function EmptyState({ icon, title, body, action }: { icon?: ReactNode; title: string; body?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
      {icon !== undefined && <div className="mb-1 grid size-10 place-items-center rounded-xl bg-surface-2 text-muted">{icon}</div>}
      <p className="text-[14px] font-medium">{title}</p>
      {body !== undefined && <p className="max-w-sm text-[13px] text-muted">{body}</p>}
      {action !== undefined && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : "Something went wrong";
  return (
    <div className="rounded-lg px-3 py-2 text-[13px]" style={{ background: "var(--st-bad-bg)", color: "var(--st-bad-ink)" }}>
      {message}
    </div>
  );
}

export function Stat({ label, value, className }: { label: string; value: ReactNode; className?: string }) {
  return (
    <div className={cx("min-w-0", className)}>
      <div className="text-[11px] leading-4 font-medium tracking-wide text-muted uppercase">{label}</div>
      <div className="truncate text-[13px] leading-5 font-medium tabular-nums">{value}</div>
    </div>
  );
}
