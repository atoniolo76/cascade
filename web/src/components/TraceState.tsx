import { Layers } from "lucide-react";
import type { TraceSummary } from "../api/types";
import { formatBytes, plural } from "../domain/format";
import { TRACE_STATE } from "../domain/status";
import { Status } from "./meters";
import { EntityChip, EntityIcon } from "./ui";
import { ICON } from "./styles";
import { paths } from "../paths";

export function TraceState({ trace, detail = true }: { trace: TraceSummary; detail?: boolean }) {
  const meta = TRACE_STATE[trace.state];
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <Status tone={meta.tone} live={meta.live} label={meta.label} />
      {detail && trace.state === "uploading" && (
        <span className="truncate text-[12px] leading-4 text-muted tabular-nums">
          {plural(trace.kernel_count, "kernel")} · {formatBytes(trace.bytes)}
        </span>
      )}
      {detail && trace.state === "processing" && <span className="truncate text-[12px] leading-4 text-muted">Post-processing traces</span>}
      {detail && trace.state === "failed" && trace.error !== null && (
        <span title={trace.error} className="truncate text-[12px] leading-4" style={{ color: "var(--st-bad-ink)" }}>
          {trace.error}
        </span>
      )}
    </div>
  );
}

export function TraceIcon() {
  return (
    <EntityIcon tone="active">
      <Layers {...ICON} />
    </EntityIcon>
  );
}

export function TraceChip({ trace, link = true }: { trace: Pick<TraceSummary, "id" | "model">; link?: boolean }) {
  return (
    <EntityChip
      icon={<Layers className="size-4 shrink-0 text-accent-ink" {...ICON} />}
      to={link ? paths.trace(trace.id) : undefined}
      title={trace.id}
    >
      {trace.model ?? trace.id}
    </EntityChip>
  );
}
