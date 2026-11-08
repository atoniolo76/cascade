import { formatInt, formatNs } from "../../domain/format";
import type { Hover } from "./controller";

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] leading-4 text-muted">{label}</div>
      <div className="truncate text-[13px] leading-5 font-semibold text-fg tabular-nums">{value}</div>
    </div>
  );
}

export function HoverCard({ hover }: { hover: Hover }) {
  const { clip, track } = hover;
  return (
    <div className="w-[280px] rounded-[14px] border border-border bg-surface p-3.5 shadow-pop">
      <div className="flex items-center gap-2 text-[11px] font-medium text-muted tabular-nums">
        <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: track.color }} />
        <span className="truncate font-mono">{track.label}</span>
        <span className="ml-auto shrink-0">Kernel #{clip.kernelIndex}</span>
      </div>
      <div className="mt-1 mb-2.5 line-clamp-2 font-mono text-[11px] leading-4 break-all text-fg-2">{clip.name}</div>
      {clip.pending ? (
        <p className="text-[12px] text-muted">Still simulating. Its length is a guess from the same kernel on other tracks.</p>
      ) : (
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <Field label="Duration" value={formatNs(clip.end - clip.start)} />
          <Field label="Starts at" value={formatNs(clip.start)} />
          <Field label="Cycles" value={clip.cycles === null ? "—" : formatInt(clip.cycles)} />
          <Field label="IPC" value={clip.ipc === null ? "—" : clip.ipc.toFixed(2)} />
        </div>
      )}
    </div>
  );
}
