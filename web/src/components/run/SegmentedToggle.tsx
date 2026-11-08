import { cx } from "../cx";

export function SegmentedToggle<T extends string>({ value, options, onChange }: { value: T; options: readonly { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex h-8 items-center rounded-[10px] bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === value}
          onClick={() => {
            onChange(o.value);
          }}
          className={cx(
            "h-7 rounded-[8px] px-3 text-[13px] font-medium transition-[background-color,color,box-shadow] duration-150",
            o.value === value ? "bg-surface text-fg shadow-tile ring-1 ring-border" : "text-muted hover:text-fg",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
