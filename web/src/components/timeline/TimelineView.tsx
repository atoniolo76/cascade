import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type CSSProperties, type KeyboardEvent } from "react";
import { Link2Off } from "lucide-react";
import { alignScale, defaultSpeed, parentOf, type Mode, type Track } from "../../domain/timeline";
import type { TrackMeta } from "./meta";
import { ICON } from "../styles";
import { onThemeChange } from "./colors";
import { RULER_H, TimelineController, rowMetrics, type Row, type Scene } from "./controller";
import { HoverCard } from "./HoverCard";
import { Inspector } from "./Inspector";
import { TrackHeaders } from "./TrackHeaders";
import { Transport } from "./Transport";

const MINIMAP_H = 40;

function toggle(set: ReadonlySet<string>, id: string): Set<string> {
  const next = new Set(set);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
}

export function TimelineView({
  tracks,
  linkable,
  unlinkedReason,
  meta,
  maxTrackHeight,
  autoFocus = false,
}: {
  tracks: readonly Track[];
  linkable: boolean;
  unlinkedReason?: string | undefined;
  meta: (track: Track) => TrackMeta;
  maxTrackHeight: string;
  autoFocus?: boolean;
}) {
  const [mode, setMode] = useState<Mode>("time");
  const [compact, setCompact] = useState(false);
  const [hidden, setHidden] = useState<ReadonlySet<string>>(new Set());
  const [solo, setSolo] = useState<ReadonlySet<string>>(new Set());
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set());
  const [focusId, setFocusId] = useState<string | null>(null);
  const [ctl] = useState(() => new TimelineController(setFocusId));
  const [showKeys, setShowKeys] = useState(false);
  const effectiveMode: Mode = linkable ? mode : "time";

  const rows = useMemo<Row[]>(() => {
    const out: Row[] = [];
    for (const track of tracks) {
      const parent = parentOf(track);
      if (parent !== null && !expanded.has(parent)) continue;
      const active = solo.size > 0 ? solo.has(track.id) || (parent !== null && solo.has(parent)) : !hidden.has(track.id) && (parent === null || !hidden.has(parent));
      out.push({ track, active, depth: parent === null ? 0 : 1 });
    }
    return out;
  }, [tracks, expanded, hidden, solo]);

  const scene = useMemo<Scene>(() => {
    const live = rows.filter((r) => r.active);
    const align = alignScale(live.map((r) => r.track));
    const end = effectiveMode === "time" ? live.reduce((m, r) => Math.max(m, r.track.clips.at(-1)?.end ?? 0), 0) : align.slots;
    const focus = focusId !== null && rows.some((r) => r.track.id === focusId) ? focusId : (live[0]?.track.id ?? null);
    return { rows, mode: effectiveMode, align, end, linkable, compact, focusId: focus };
  }, [rows, effectiveMode, linkable, compact, focusId]);

  const root = useRef<HTMLDivElement>(null);
  const ruler = useRef<HTMLCanvasElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const trackBox = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const minimap = useRef<HTMLCanvasElement>(null);
  const minimapBox = useRef<HTMLDivElement>(null);
  const minimapWindow = useRef<HTMLDivElement>(null);
  const minimapPlayhead = useRef<HTMLDivElement>(null);
  const playhead = useRef<HTMLDivElement>(null);
  const timecode = useRef<HTMLSpanElement>(null);
  const kernelReadout = useRef<HTMLSpanElement>(null);
  const hoverCard = useRef<HTMLDivElement>(null);
  const probe = useRef<HTMLSpanElement>(null);
  const zoom = useRef<HTMLInputElement>(null);

  useLayoutEffect(() => {
    const [r, ru, t, tb, sc, mm, mb, mw, mp, ph, tc, kr, hc, pr] = [root, ruler, canvas, trackBox, scroller, minimap, minimapBox, minimapWindow, minimapPlayhead, playhead, timecode, kernelReadout, hoverCard, probe].map((ref) => ref.current);
    if (!r || !(ru instanceof HTMLCanvasElement) || !(t instanceof HTMLCanvasElement) || !tb || !sc || !(mm instanceof HTMLCanvasElement) || !mb || !mw || !mp || !ph || !tc || !kr || !hc || !pr) return;
    ctl.attach({ root: r, ruler: ru, tracks: t, trackBox: tb, scroller: sc, minimap: mm, minimapBox: mb, minimapWindow: mw, minimapPlayhead: mp, playhead: ph, timecode: tc, kernelReadout: kr, hoverCard: hc, probe: pr, zoom: zoom.current });
    const offTheme = onThemeChange(() => {
      ctl.themeChanged();
    });
    return () => {
      offTheme();
      ctl.detach();
    };
  }, [ctl]);

  useLayoutEffect(() => {
    ctl.setScene(scene);
  }, [ctl, scene]);

  const speedChosen = useRef(false);
  useEffect(() => {
    if (speedChosen.current || scene.end <= 0) return;
    speedChosen.current = true;
    ctl.setSpeed(defaultSpeed(scene.end, scene.mode));
  }, [ctl, scene.end, scene.mode]);

  useEffect(() => {
    if (autoFocus) root.current?.focus({ preventScroll: true });
  }, [autoFocus]);

  const coarse = useSyncExternalStore(ctl.subscribe, ctl.getCoarse);

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const actions: Record<string, () => void> = {
      " ": () => {
        ctl.togglePlay();
      },
      j: () => {
        ctl.shuttle("j");
      },
      k: () => {
        ctl.shuttle("k");
      },
      l: () => {
        ctl.shuttle("l");
      },
      ArrowLeft: () => {
        ctl.step(-1);
      },
      ArrowRight: () => {
        ctl.step(1);
      },
      Home: () => {
        ctl.toStart();
      },
      End: () => {
        ctl.toEnd();
      },
      "+": () => {
        ctl.zoomBy(0.6);
      },
      "=": () => {
        ctl.zoomBy(0.6);
      },
      "-": () => {
        ctl.zoomBy(1 / 0.6);
      },
      _: () => {
        ctl.zoomBy(1 / 0.6);
      },
      f: () => {
        ctl.fit();
      },
      i: () => {
        ctl.markIn();
      },
      o: () => {
        ctl.markOut();
      },
      "?": () => {
        setShowKeys((v) => !v);
      },
      Escape: () => {
        if (showKeys) setShowKeys(false);
        else if (coarse.selected) ctl.select(null);
        else if (coarse.range) ctl.clearRange();
      },
    };
    const action = actions[key];
    if (!action) return;
    e.preventDefault();
    e.stopPropagation();
    action();
  };

  const { h, gap } = rowMetrics(compact);
  const contentH = rows.length === 0 ? 120 : 12 + rows.length * (h + gap) - gap;
  const hasReplicas = (id: string) => tracks.some((t) => parentOf(t) === id);

  return (
    <div
      ref={root}
      data-testid="timeline"
      tabIndex={0}
      onKeyDown={onKeyDown}
      aria-label="Timeline. Space plays, J K L shuttle, arrows step kernels, question mark lists shortcuts."
      className="relative min-w-0 rounded-b-[18px] outline-none [--hdr:60px] focus-visible:ring-2 focus-visible:ring-[var(--ring)] sm:[--hdr:264px]"
    >
      <span ref={probe} aria-hidden className="pointer-events-none absolute size-0 opacity-0" />
      <Transport
        ctl={ctl}
        coarse={coarse}
        mode={effectiveMode}
        onMode={setMode}
        modeLocked={!linkable}
        compact={compact}
        onCompact={setCompact}
        showKeys={showKeys}
        onShowKeys={setShowKeys}
        timecodeRef={timecode}
        kernelRef={kernelReadout}
        zoomRef={zoom}
      />
      {!linkable && unlinkedReason !== undefined && (
        <div className="flex items-center gap-2 border-b border-border bg-[var(--st-lost-bg)] px-5 py-2 text-[13px] text-[var(--st-lost-ink)]">
          <Link2Off className="size-4 shrink-0" {...ICON} />
          {unlinkedReason}
        </div>
      )}

      <div className="relative grid grid-cols-[var(--hdr)_minmax(0,1fr)] border-b border-border">
        <div className="flex items-end border-r border-b border-border bg-card px-3 pb-1.5 text-[11px] font-medium tracking-[0.04em] text-faint uppercase" style={{ height: RULER_H }}>
          <span className="hidden sm:inline">{rows.length} tracks</span>
        </div>
        <div className="relative border-b border-border bg-card" style={{ height: RULER_H }}>
          <canvas ref={ruler} className="block cursor-ew-resize touch-none" aria-label="Time ruler. Click or drag to scrub." />
        </div>

        <div ref={scroller} className="col-span-2 grid grid-cols-[var(--hdr)_minmax(0,1fr)] overflow-x-hidden overflow-y-auto overscroll-y-contain" style={{ maxHeight: maxTrackHeight }}>
          <TrackHeaders
            rows={rows}
            meta={meta}
            compact={compact}
            focusId={scene.focusId}
            expanded={expanded}
            hasReplicas={hasReplicas}
            onFocus={setFocusId}
            onToggleHidden={(id) => {
              setHidden((s) => toggle(s, id));
            }}
            onToggleSolo={(id) => {
              setSolo((s) => toggle(s, id));
            }}
            onToggleExpand={(id) => {
              setExpanded((s) => toggle(s, id));
            }}
            solo={solo}
            hidden={hidden}
            height={contentH}
          />
          <div ref={trackBox} className="relative" style={{ height: contentH }}>
            <canvas ref={canvas} className="sticky top-0 block cursor-default touch-none" />
            {rows.length === 0 && <p className="absolute inset-0 grid place-items-center text-[13px] text-muted">No kernels simulated yet</p>}
          </div>
        </div>

        <div className="pointer-events-none absolute top-0 right-0 bottom-0 left-[var(--hdr)] overflow-hidden">
          <div ref={playhead} className="absolute top-0 bottom-0 left-0 w-0 will-change-transform" style={{ visibility: "hidden" }}>
            <div className="absolute top-0 bottom-0 -left-px w-[2px] bg-accent/90" />
            <svg aria-hidden width="14" height="16" viewBox="0 0 14 16" className="absolute top-[3px] -left-[7px] text-accent drop-shadow-sm">
              <path d="M1 2.5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v6.6a2 2 0 0 1-.7 1.5l-4 3.5a2 2 0 0 1-2.6 0l-4-3.5A2 2 0 0 1 1 9.1z" fill="currentColor" />
            </svg>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-[var(--hdr)_minmax(0,1fr)] border-b border-border">
        <div className="flex items-center border-r border-border px-3 text-[11px] font-medium tracking-[0.04em] text-faint uppercase" style={{ height: MINIMAP_H }}>
          <span className="hidden sm:inline">Overview</span>
        </div>
        <div ref={minimapBox} className="relative touch-none overflow-hidden bg-card select-none" style={{ height: MINIMAP_H }} aria-label="Overview. Drag the window to pan, drag its edges to zoom.">
          <canvas ref={minimap} className="pointer-events-none absolute inset-0 block" />
          <div ref={minimapPlayhead} className="pointer-events-none absolute top-0 bottom-0 left-0 w-px bg-accent" />
          <div ref={minimapWindow} data-edge="move" className="absolute top-1 bottom-1 left-0 cursor-grab rounded-[7px] bg-accent/8 ring-1 ring-accent/70 active:cursor-grabbing" style={{ width: 40 }}>
            <span data-edge="start" className="absolute inset-y-0 -left-1 flex w-3 cursor-ew-resize items-center justify-center">
              <span data-edge="start" className="h-4 w-[3px] rounded-full bg-accent" />
            </span>
            <span data-edge="end" className="absolute inset-y-0 -right-1 flex w-3 cursor-ew-resize items-center justify-center">
              <span data-edge="end" className="h-4 w-[3px] rounded-full bg-accent" />
            </span>
          </div>
        </div>
      </div>

      <Inspector ctl={ctl} meta={meta} linkable={linkable} />

      <div ref={hoverCard} className="pointer-events-none absolute top-0 left-0 z-20 will-change-transform" style={{ visibility: coarse.hover ? "visible" : "hidden" } satisfies CSSProperties}>
        {coarse.hover && <HoverCard hover={coarse.hover} />}
      </div>
    </div>
  );
}

