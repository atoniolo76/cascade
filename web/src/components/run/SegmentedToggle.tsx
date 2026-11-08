import { cx } from "../cx";

export function SegmentedToggle<T extends string>({ value, options, onChange }: { value: T; options: readonly { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex h-7 items-center rounded-lg bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === value}
          onClick={() => {
            onChange(o.value);
          }}
          className={cx(
            "h-6 rounded-md px-2.5 text-[12px] font-medium transition-[background-color,color,box-shadow] duration-150",
            o.value === value ? "bg-surface text-fg shadow-card ring-1 ring-border" : "text-muted hover:text-fg",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
