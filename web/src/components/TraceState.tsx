import { FlaskConical, Repeat } from "lucide-react";
import type { TraceSummary } from "../api/types";
import { formatBytes, plural } from "../domain/format";
import { TRACE_STATE } from "../domain/status";
import { Status } from "./meters";
import { EntityChip } from "./ui";
import { LogoTile } from "./BrandLogo";
import { modelLogo } from "../domain/logos";
import { paths } from "../paths";
import { REPLAY_RUN_LINE, SAMPLE_TOOLTIP } from "../domain/samples";

export function TraceState({
  trace,
  detail = true,
}: {
  trace: TraceSummary;
  detail?: boolean;
}) {
  const meta = TRACE_STATE[trace.state];
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <Status tone={meta.tone} live={meta.live} label={meta.label} />
      {detail && trace.state === "uploading" && (
        <span className="truncate text-[12px] leading-4 text-muted tabular-nums">
          {plural(trace.kernel_count, "kernel")} · {formatBytes(trace.bytes)}
        </span>
      )}
      {detail && trace.state === "processing" && (
        <span className="truncate text-[12px] leading-4 text-muted">
          Post-processing traces
        </span>
      )}
      {detail && trace.state === "failed" && trace.error !== null && (
        <span
          title={trace.error}
          className="truncate text-[12px] leading-4"
          style={{ color: "var(--st-bad-ink)" }}
        >
          {trace.error}
        </span>
      )}
    </div>
  );
}

export function ModelTile({
  model,
  fallback,
  size = "md",
}: {
  model: string | null;
  fallback: string;
  size?: "sm" | "md" | "lg";
}) {
  return (
    <LogoTile
      logo={modelLogo(model)}
      fallback={model ?? fallback}
      size={size}
    />
  );
}

export function TraceChip({
  trace,
  link = true,
}: {
  trace: Pick<TraceSummary, "id" | "model">;
  link?: boolean;
}) {
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

/** The trace carries the profiler's corrected timeline and a kernel mapping, so its runs predict step timing. */
export function StepsBadge() {
  return (
    <span
      title="Step timing: corrected_timeline.json and kernel_mapping.json are in this trace"
      className="inline-flex h-5 shrink-0 items-center rounded-[6px] bg-surface-2 px-1.5 text-[11px] font-medium text-fg-2"
    >
      Steps
    </span>
  );
}

/** Marks the bundled sample trace and run, which hold recorded results rather than this workspace's own. */
export function SampleBadge() {
  return (
    <span
      title={SAMPLE_TOOLTIP}
      className="inline-flex h-5 shrink-0 cursor-help items-center gap-1 rounded-[6px] bg-accent-soft px-1.5 text-[11px] font-medium text-accent-ink ring-1 ring-accent/25 ring-inset"
    >
      <FlaskConical className="size-3" strokeWidth={2} aria-hidden="true" />
      Sample
      <span className="sr-only">: {SAMPLE_TOOLTIP}</span>
    </span>
  );
}

/** Marks a run launched on a sample trace, which replays recorded results rather than simulating. */
export function ReplayBadge() {
  return (
    <span
      title={REPLAY_RUN_LINE}
      className="inline-flex h-5 shrink-0 cursor-help items-center gap-1 rounded-[6px] bg-surface-2 px-1.5 text-[11px] font-medium text-fg-2 ring-1 ring-border ring-inset"
    >
      <Repeat className="size-3" strokeWidth={2} aria-hidden="true" />
      Replay
      <span className="sr-only">: {REPLAY_RUN_LINE}</span>
    </span>
  );
}
