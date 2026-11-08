import { simplifyKernelName } from "../../domain/kernelName";
import { cx } from "../cx";

/** The readable kernel name, namespaces dimmed so the function stands out; the recorded name is on hover. */
export function KernelLabel({ name, className }: { name: string; className?: string | undefined }) {
  const k = simplifyKernelName(name);
  return (
    <span title={k.full} className={cx("flex min-w-0 font-mono text-[12px] leading-5 text-fg-2", className)}>
      {k.scope !== "" && (
        <span className="min-w-[2ch] truncate text-faint" style={{ flexShrink: 4 }}>
          {k.scope}
        </span>
      )}
      <span className="min-w-0 truncate">{k.base}</span>
    </span>
  );
}

/** Tooltip header: the readable name, and the recorded one below it when they differ. */
export function KernelHeading({ index, name }: { index: number; name: string }) {
  const k = simplifyKernelName(name);
  return (
    <div className="mb-2.5 flex flex-col gap-0.5">
      <div className="flex items-baseline gap-2">
        <span className="shrink-0 text-[11px] font-medium text-muted tabular-nums">#{index}</span>
        <span className="min-w-0 font-mono text-[12px] leading-[18px] break-all text-fg">
          <span className="text-muted">{k.scope}</span>
          {k.base}
        </span>
      </div>
      {k.full !== `${k.scope}${k.base}` && <p className="line-clamp-2 font-mono text-[10.5px] leading-[15px] break-all text-faint">{k.full}</p>}
    </div>
  );
}
