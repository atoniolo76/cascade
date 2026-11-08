import { cx } from "../cx";

const SIZES = {
  md: { track: "h-8 rounded-[10px]", item: "h-7 rounded-[8px] px-3 text-[13px]" },
  sm: { track: "h-7 rounded-[9px]", item: "h-6 rounded-[7px] px-2.5 text-[12px]" },
} as const;

export function SegmentedToggle<T extends string>({
  value,
  options,
  onChange,
  size = "md",
  label,
}: {
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (v: T) => void;
  size?: keyof typeof SIZES;
  label?: string;
}) {
  const s = SIZES[size];
  return (
    <div role="group" aria-label={label} className={cx("inline-flex shrink-0 items-center bg-surface-2 p-0.5", s.track)}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === value}
          onClick={() => {
            onChange(o.value);
          }}
          className={cx(
            "font-medium whitespace-nowrap transition-[background-color,color,box-shadow] duration-150",
            s.item,
            o.value === value ? "bg-surface text-fg shadow-tile ring-1 ring-border" : "text-muted hover:text-fg",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
