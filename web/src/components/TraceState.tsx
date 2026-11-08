import type { TraceSummary } from "../api/types";
import { formatBytes, plural } from "../domain/format";
import { TRACE_STATE } from "../domain/status";
import { Status } from "./meters";
import { EntityChip } from "./ui";
import { LogoTile } from "./BrandLogo";
import { modelLogo } from "../domain/logos";
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

export function ModelTile({ model, fallback, size = "md" }: { model: string | null; fallback: string; size?: "sm" | "md" | "lg" }) {
  return <LogoTile logo={modelLogo(model)} fallback={model ?? fallback} size={size} />;
}

export function TraceChip({ trace, link = true }: { trace: Pick<TraceSummary, "id" | "model">; link?: boolean }) {
  return (
    <EntityChip
      icon={<ModelTile model={trace.model} fallback={trace.id} size="sm" />}
      to={link ? paths.trace(trace.id) : undefined}
      title={trace.id}
    >
      {trace.model ?? trace.id}
    </EntityChip>
  );
}
