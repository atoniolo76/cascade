import { ArrowDown } from "lucide-react";
import { useStickToBottom } from "../../hooks/useStickToBottom";
import { cx } from "../cx";

export function LogLines({
  lines,
  className,
  outerClassName,
  empty,
  wrap = false,
}: {
  lines: readonly string[];
  className?: string;
  outerClassName?: string;
  empty: string;
  wrap?: boolean;
}) {
  const { ref, onScroll, pinned, jump } = useStickToBottom<HTMLDivElement>(lines);
  return (
    <div className={cx("relative min-h-0", outerClassName)}>
      <div ref={ref} onScroll={onScroll} className={cx("overflow-y-auto overscroll-contain font-mono text-[11px] leading-4", className)}>
        {lines.length === 0 ? (
          <div className="text-faint">{empty}</div>
        ) : (
          lines.map((line, i) => (
            <div key={i} title={wrap ? undefined : line} className={cx("text-fg-2", wrap ? "break-all whitespace-pre-wrap" : "truncate whitespace-pre")}>
              {line === "" ? " " : line}
            </div>
          ))
        )}
      </div>
      <button
        type="button"
        onClick={jump}
        tabIndex={pinned ? -1 : 0}
        className={cx(
          "absolute right-2 bottom-2 inline-flex h-6 items-center gap-1 rounded-md bg-surface px-2 text-[11px] font-medium text-fg-2 shadow-pop ring-1 ring-border transition-opacity duration-150",
          pinned ? "pointer-events-none opacity-0" : "opacity-100",
        )}
      >
        <ArrowDown className="size-3" />
        Latest
      </button>
    </div>
  );
}
