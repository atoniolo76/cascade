import type { TraceSummary } from "../api/types";
import { formatBytes, formatInt } from "../domain/format";
import { TRACE_STATE } from "../domain/status";
import { StatusPill } from "./ui";

export function TraceState({ trace }: { trace: TraceSummary }) {
  const meta = TRACE_STATE[trace.state];
  return (
    <div className="flex min-w-0 items-center gap-2">
      <StatusPill tone={meta.tone} live={meta.live} label={meta.label} />
      {trace.state === "uploading" && (
        <span className="truncate text-[12px] text-muted tabular-nums">
          {formatInt(trace.kernel_count)} kernels · {formatBytes(trace.bytes)}
        </span>
      )}
      {trace.state === "failed" && trace.error !== null && (
        <span title={trace.error} className="min-w-0 truncate text-[12px]" style={{ color: "var(--st-bad-ink)" }}>
          {trace.error}
        </span>
      )}
    </div>
  );
}
