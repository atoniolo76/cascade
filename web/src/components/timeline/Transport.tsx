import type { ReactNode, RefObject } from "react";
import { ChevronDown, ChevronLeft, ChevronRight, Keyboard, Maximize2, Minus, Pause, Play, Plus, Rows2, Rows3, SkipBack, SkipForward } from "lucide-react";
import { SPEEDS, type Mode } from "../../domain/timeline";
import { cx } from "../cx";
import { ICON } from "../styles";
import { SegmentedToggle } from "../run/SegmentedToggle";
import type { Coarse, TimelineController } from "./controller";
import { Shortcuts } from "./Shortcuts";

const MODES = [
  { value: "time", label: "Time" },
  { value: "align", label: "Align by kernel" },
] as const;

function TBtn({ label, onClick, children, active = false, className }: { label: string; onClick: () => void; children: ReactNode; active?: boolean; className?: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={active || undefined}
      onClick={onClick}
      className={cx(
        "grid size-8 shrink-0 place-items-center rounded-[9px] text-fg-2 transition-colors duration-150 hover:bg-surface-2 hover:text-fg [&>svg]:size-4",
        active && "bg-surface-2 text-fg",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function Transport({
  ctl,
  coarse,
  mode,
  onMode,
  modeLocked,
  compact,
  onCompact,
  showKeys,
  onShowKeys,
  timecodeRef,
  kernelRef,
  zoomRef,
}: {
  ctl: TimelineController;
  coarse: Coarse;
  mode: Mode;
  onMode: (m: Mode) => void;
  modeLocked: boolean;
  compact: boolean;
  onCompact: (c: boolean) => void;
  showKeys: boolean;
  onShowKeys: (v: boolean) => void;
  timecodeRef: RefObject<HTMLSpanElement | null>;
  kernelRef: RefObject<HTMLSpanElement | null>;
  zoomRef: RefObject<HTMLInputElement | null>;
}) {
  const playing = coarse.rate !== 0;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-border px-3 py-2.5 sm:px-4">
      <div className="flex items-center gap-0.5 rounded-[12px] bg-card p-0.5 ring-1 ring-border">
        <TBtn label="Jump to start (Home)" onClick={() => { ctl.toStart(); }}>
          <SkipBack {...ICON} />
        </TBtn>
        <TBtn label="Previous kernel (←)" onClick={() => { ctl.step(-1); }}>
          <ChevronLeft {...ICON} />
        </TBtn>
        <button
          type="button"
          aria-label={playing ? "Pause (Space)" : "Play (Space)"}
          title={playing ? "Pause (Space)" : "Play (Space)"}
          onClick={() => {
            ctl.togglePlay();
          }}
          className="grid size-8 place-items-center rounded-[9px] bg-btn text-btn-fg shadow-tile transition-colors duration-150 hover:bg-btn-hover"
        >
          {playing ? <Pause className="size-3.5" fill="currentColor" strokeWidth={0} /> : <Play className="ml-0.5 size-3.5" fill="currentColor" strokeWidth={0} />}
        </button>
        <TBtn label="Next kernel (→)" onClick={() => { ctl.step(1); }}>
          <ChevronRight {...ICON} />
        </TBtn>
        <TBtn label="Jump to end (End)" onClick={() => { ctl.toEnd(); }}>
          <SkipForward {...ICON} />
        </TBtn>
      </div>

      <div className="flex h-9 min-w-0 items-center gap-3 rounded-[12px] bg-card px-3 ring-1 ring-border">
        <span ref={timecodeRef} data-testid="timecode" className="min-w-[112px] font-mono text-[14px] font-medium text-fg tabular-nums">
          0 µs
        </span>
        <span aria-hidden className="h-4 w-px bg-border-strong" />
        <span className="flex items-baseline gap-1.5 font-mono text-[12px] text-muted tabular-nums">
          <span className="hidden text-faint sm:inline">kernel</span>
          <span ref={kernelRef} data-testid="kernel-readout">—</span>
        </span>
        {coarse.rate !== 0 && Math.abs(coarse.rate) !== 1 && (
          <span className="rounded-[6px] bg-accent-soft px-1.5 font-mono text-[11px] font-medium text-accent-ink tabular-nums">
            {coarse.rate < 0 ? "◀ " : ""}
            {Math.abs(coarse.rate)}×
          </span>
        )}
        {coarse.rate < 0 && Math.abs(coarse.rate) === 1 && <span className="rounded-[6px] bg-accent-soft px-1.5 font-mono text-[11px] font-medium text-accent-ink">◀ 1×</span>}
      </div>

      <label className="relative flex h-9 items-center rounded-[12px] bg-card ring-1 ring-border">
        <span className="sr-only">Playback speed</span>
        <select
          value={coarse.speed}
          onChange={(e) => {
            ctl.setSpeed(Number(e.target.value));
          }}
          className="h-9 cursor-pointer appearance-none rounded-[12px] bg-transparent pr-8 pl-3 font-mono text-[12px] text-fg-2 outline-none"
          title="Playback maps simulated time to wall time"
        >
          {SPEEDS.map((s, i) => (
            <option key={s.label} value={i}>
              {mode === "time" ? s.timeLabel : s.alignLabel}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute right-2.5 size-3.5 text-muted" {...ICON} />
      </label>

      <div className="ml-auto flex flex-wrap items-center gap-2">
        <span title={modeLocked ? "Align by kernel needs runs of the same trace" : undefined} className={cx(modeLocked && "pointer-events-none opacity-45")}>
          <SegmentedToggle value={mode} options={MODES} onChange={onMode} />
        </span>
        <div className="hidden items-center gap-1.5 rounded-[12px] bg-card px-1.5 ring-1 ring-border md:flex">
          <TBtn label="Zoom out (−)" onClick={() => { ctl.zoomBy(1 / 0.6); }}>
            <Minus {...ICON} />
          </TBtn>
          <input
            ref={zoomRef}
            type="range"
            min={0}
            max={1000}
            defaultValue={0}
            aria-label="Zoom"
            onChange={(e) => {
              ctl.setZoomLevel(Number(e.target.value) / 1000);
            }}
            className="h-1 w-24 cursor-pointer accent-[var(--accent)]"
          />
          <TBtn label="Zoom in (+)" onClick={() => { ctl.zoomBy(0.6); }}>
            <Plus {...ICON} />
          </TBtn>
        </div>
        <TBtn label="Zoom to fit (F)" onClick={() => { ctl.fit(); }} className="ring-1 ring-border bg-card">
          <Maximize2 {...ICON} />
        </TBtn>
        <TBtn label={compact ? "Taller tracks" : "Compact tracks"} onClick={() => { onCompact(!compact); }} className="ring-1 ring-border bg-card">
          {compact ? <Rows2 {...ICON} /> : <Rows3 {...ICON} />}
        </TBtn>
        <div className="relative">
          <TBtn label="Keyboard shortcuts (?)" active={showKeys} onClick={() => { onShowKeys(!showKeys); }} className="ring-1 ring-border bg-card">
            <Keyboard {...ICON} />
          </TBtn>
          {showKeys && (
            <Shortcuts
              onClose={() => {
                onShowKeys(false);
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
