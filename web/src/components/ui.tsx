import { useEffect, useState, type ButtonHTMLAttributes, type ComponentProps, type CSSProperties, type ReactNode } from "react";
import { Link } from "react-router";
import { Check, Copy, Minus, Plus } from "lucide-react";
import { formatAge, formatExact, ms } from "../domain/format";
import { TONE_VARS, type Tone } from "../domain/status";
import { useNow } from "../hooks/useNow";
import { cx } from "./cx";
import { BUTTON_SIZES, ICON, buttonClass, type ButtonVariant } from "./styles";

export function Button({
  variant = "secondary",
  size = "md",
  className,
  ...props
}: ComponentProps<"button"> & { variant?: ButtonVariant; size?: keyof typeof BUTTON_SIZES }) {
  return <button type="button" {...props} className={cx(buttonClass({ variant, size }), className)} />;
}

export function IconButton({ label, className, children, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      {...props}
      className={cx(
        "grid size-9 shrink-0 place-items-center rounded-[12px] bg-surface text-muted ring-1 ring-border ring-inset transition-colors duration-150 hover:text-fg hover:ring-border-strong disabled:pointer-events-none disabled:opacity-40",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cx("rounded-[18px] border border-border bg-card", className)}>{children}</div>;
}

export function Panel({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cx("rounded-[18px] border border-border bg-surface shadow-card", className)}>{children}</div>;
}

export function SectionTitle({ icon, title, aside, subtitle }: { icon?: ReactNode; title: ReactNode; aside?: ReactNode; subtitle?: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        {icon !== undefined && <span className="text-muted [&>svg]:size-[18px]">{icon}</span>}
        <h2 className="text-[16px] leading-6 font-medium text-fg">{title}</h2>
        {aside}
      </div>
      {subtitle !== undefined && <p className="text-[13px] leading-5 text-muted">{subtitle}</p>}
    </div>
  );
}

export function PanelHeader({ icon, title, subtitle, actions }: { icon?: ReactNode; title: ReactNode; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex min-h-16 flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-border px-5 py-3.5">
      <SectionTitle icon={icon} title={title} subtitle={subtitle} />
      {actions !== undefined && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Chip({ children, className, title }: { children: ReactNode; className?: string; title?: string }) {
  return (
    <span
      title={title}
      className={cx("inline-flex h-7 max-w-full items-center gap-1.5 rounded-[9px] bg-surface px-2.5 text-[13px] whitespace-nowrap text-fg-2 ring-1 ring-border ring-inset", className)}
    >
      {children}
    </span>
  );
}

export function Swatch({ color }: { color: string }) {
  return <span aria-hidden className="size-2.5 shrink-0 rounded-[3px]" style={{ background: color }} />;
}

export function EntityIcon({ tone = "active", children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  const v = TONE_VARS[tone];
  return (
    <span aria-hidden className={cx("grid size-6 shrink-0 place-items-center rounded-[7px] [&>svg]:size-3.5", className)} style={{ background: v.bg, color: v.ink }}>
      {children}
    </span>
  );
}

export function EntityChip({
  icon,
  children,
  to,
  title,
  mono = false,
  className,
}: {
  icon: ReactNode;
  children: ReactNode;
  to?: string | undefined;
  title?: string | undefined;
  mono?: boolean;
  className?: string;
}) {
  const label = (
    <span
      className={cx(
        "min-w-0 truncate underline decoration-border-strong decoration-[1.5px] underline-offset-[5px] transition-colors duration-150",
        mono && "font-mono text-[0.92em]",
        to !== undefined && "group-hover:decoration-fg-2",
      )}
    >
      {children}
    </span>
  );
  const cls = cx("group inline-flex max-w-full min-w-0 items-center gap-1.5 align-middle text-fg", className);
  return to === undefined ? (
    <span className={cls} title={title}>
      {icon}
      {label}
    </span>
  ) : (
    <Link to={to} className={cls} title={title}>
      {icon}
      {label}
    </Link>
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

export function CopyButton({ value, label, className }: { value: string; label: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => {
      setCopied(false);
    }, 1400);
    return () => {
      clearTimeout(t);
    };
  }, [copied]);
  return (
    <button
      type="button"
      aria-label={copied ? "Copied" : `Copy ${label}`}
      title={copied ? "Copied" : `Copy ${label}`}
      onClick={(e) => {
        e.stopPropagation();
        e.preventDefault();
        void navigator.clipboard.writeText(value).then(() => {
          setCopied(true);
        });
      }}
      className={cx(
        "inline-grid size-7 shrink-0 place-items-center rounded-[8px] text-faint transition-colors duration-150 hover:bg-surface-2 hover:text-fg",
        copied && "text-[var(--st-ok-ink)] hover:text-[var(--st-ok-ink)]",
        className,
      )}
    >
      {copied ? <Check className="size-3.5" strokeWidth={2} /> : <Copy className="size-3.5" {...ICON} />}
    </button>
  );
}

export function RelTime({ iso, className }: { iso: string; className?: string }) {
  const now = useNow();
  return (
    <time dateTime={iso} title={formatExact(iso)} className={cx("whitespace-nowrap tabular-nums", className)}>
      {formatAge(now - ms(iso))} ago
    </time>
  );
}

export function Stepper({ value, onChange, min, max, disabled = false, label }: { value: number; onChange: (n: number) => void; min: number; max: number; disabled?: boolean; label: string }) {
  const clamp = (n: number) => Math.max(min, Math.min(max, n));
  const step = "grid h-9 w-9 place-items-center text-muted transition-colors duration-150 hover:bg-surface-2 hover:text-fg disabled:pointer-events-none disabled:opacity-40";
  return (
    <div className={cx("inline-flex h-9 items-center overflow-hidden rounded-[12px] bg-surface ring-1 ring-border-strong ring-inset", disabled && "opacity-45")}>
      <button
        type="button"
        aria-label={`Decrease ${label}`}
        disabled={disabled || value <= min}
        onClick={() => {
          onChange(clamp(value - 1));
        }}
        className={step}
      >
        <Minus className="size-3.5" {...ICON} />
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
        className="h-9 w-10 bg-transparent text-center text-[14px] font-medium text-fg tabular-nums outline-none"
      />
      <button
        type="button"
        aria-label={`Increase ${label}`}
        disabled={disabled || value >= max}
        onClick={() => {
          onChange(clamp(value + 1));
        }}
        className={step}
      >
        <Plus className="size-3.5" {...ICON} />
      </button>
    </div>
  );
}

export function EmptyState({ icon, title, body, action }: { icon?: ReactNode; title: string; body?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-14 text-center">
      {icon !== undefined && <div className="mb-1 grid size-11 place-items-center rounded-[14px] bg-surface text-muted ring-1 ring-border [&>svg]:size-5">{icon}</div>}
      <p className="text-[15px] font-medium text-fg">{title}</p>
      {body !== undefined && <p className="max-w-md text-[14px] text-muted">{body}</p>}
      {action !== undefined && <div className="mt-3">{action}</div>}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : "Something went wrong";
  return (
    <div className="rounded-[12px] px-3.5 py-2.5 text-[13px]" style={{ background: "var(--st-bad-bg)", color: "var(--st-bad-ink)" }}>
      {message}
    </div>
  );
}

export function Stat({ label, value, className }: { label: string; value: ReactNode; className?: string }) {
  return (
    <div className={cx("min-w-0", className)}>
      <div className="text-[13px] leading-5 text-muted">{label}</div>
      <div className="truncate text-[15px] leading-6 font-medium text-fg tabular-nums">{value}</div>
    </div>
  );
}

export function Th({ icon, children, className }: { icon?: ReactNode; children?: ReactNode; className?: string }) {
  return (
    <th className={cx("h-11 px-4 text-left text-[13px] font-normal whitespace-nowrap text-muted", className)}>
      <span className="inline-flex items-center gap-2 [&>svg]:size-4 [&>svg]:shrink-0 [&>svg]:text-faint">
        {icon}
        {children}
      </span>
    </th>
  );
}

export function Dash() {
  return <span className="text-faint">—</span>;
}
