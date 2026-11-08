import {
  SPEEDS,
  boundaries,
  clampView,
  clipSpan,
  fitView,
  formatTimecode,
  readTrack,
  shuttle,
  stepBoundary,
  ticks,
  zoomAround,
  type AlignScale,
  type Clip,
  type Mode,
  type Track,
  type View,
} from "../../domain/timeline";
import { formatInt, formatNs } from "../../domain/format";
import { kernelLabel } from "../../domain/kernelLabel";

const clipLabel = (name: string) => kernelLabel(name).split("::").at(-1) ?? name;
import { ColorResolver, inkOn, withAlpha } from "./colors";

export interface Row {
  track: Track;
  /** Hidden or outside a solo: drawn faint and left out of the readout. */
  active: boolean;
  depth: 0 | 1;
}

export interface Selection {
  trackId: string;
  kernelIndex: number;
}

export interface Range {
  from: number;
  to: number;
}

export interface Hover {
  track: Track;
  clip: Clip;
}

export interface Coarse {
  rate: number;
  speed: number;
  selected: Selection | null;
  range: Range | null;
  hover: Hover | null;
}

export interface Scene {
  rows: Row[];
  mode: Mode;
  align: AlignScale;
  end: number;
  linkable: boolean;
  compact: boolean;
  focusId: string | null;
}

export interface Elements {
  root: HTMLElement;
  ruler: HTMLCanvasElement;
  tracks: HTMLCanvasElement;
  trackBox: HTMLElement;
  scroller: HTMLElement;
  minimap: HTMLCanvasElement;
  minimapBox: HTMLElement;
  minimapWindow: HTMLElement;
  minimapPlayhead: HTMLElement;
  playhead: HTMLElement;
  timecode: HTMLElement;
  kernelReadout: HTMLElement;
  hoverCard: HTMLElement;
  probe: HTMLElement;
  zoom: HTMLInputElement | null;
}

export const RULER_H = 30;
const PAD_Y = 6;

export function rowMetrics(compact: boolean): { h: number; gap: number } {
  return compact ? { h: 22, gap: 6 } : { h: 38, gap: 10 };
}

const LABEL_FONT = '500 11px "JetBrains Mono Variable", ui-monospace, monospace';
const SUB_FONT = '400 10px "JetBrains Mono Variable", ui-monospace, monospace';
const RULER_FONT = '400 10px "JetBrains Mono Variable", ui-monospace, monospace';

type Drag =
  | { kind: "scrub"; id: number }
  | { kind: "pan"; id: number; x0: number; view: View; moved: boolean; hit: Hover | null }
  | { kind: "minimap"; id: number; edge: "move" | "start" | "end"; x0: number; view: View };

/**
 * Owns everything that changes per frame (view, playhead, playback, hover) and paints it imperatively,
 * so React re-renders only when the coarse snapshot changes.
 */
export class TimelineController {
  private el: Elements | null = null;
  private colors: ColorResolver | null = null;
  private scene: Scene = { rows: [], mode: "time", align: { slots: 0, longest: new Float64Array() }, end: 0, linkable: true, compact: false, focusId: null };
  private view: View = { start: 0, end: 1 };
  private follow = true;
  private width = 0;
  private trackH = 0;
  private dpr = 1;
  private frame = 0;
  private dirty = { tracks: true, ruler: true, minimap: true };
  private lastTick = 0;
  private drag: Drag | null = null;
  private hatch = new Map<string, CanvasPattern | null>();
  private cleanup: (() => void)[] = [];
  private listeners = new Set<() => void>();
  private readListeners = new Set<() => void>();
  private readVersion = 0;
  private readTimer: ReturnType<typeof setTimeout> | undefined;
  private lastRead = 0;

  playhead = 0;
  private onFocus: (trackId: string) => void;

  constructor(onFocus: (trackId: string) => void) {
    this.onFocus = onFocus;
  }
  private coarse: Coarse = { rate: 0, speed: 1, selected: null, range: null, hover: null };

  getCoarse = (): Coarse => this.coarse;
  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };
  getReadVersion = (): number => this.readVersion;
  subscribeRead = (fn: () => void) => {
    this.readListeners.add(fn);
    return () => this.readListeners.delete(fn);
  };

  get currentScene(): Scene {
    return this.scene;
  }

  get axisPerPx(): number {
    return this.width > 0 ? (this.view.end - this.view.start) / this.width : 1;
  }

  private setCoarse(patch: Partial<Coarse>) {
    this.coarse = { ...this.coarse, ...patch };
    for (const fn of this.listeners) fn();
  }

  private bumpRead() {
    const now = performance.now();
    const fire = () => {
      this.lastRead = performance.now();
      this.readVersion++;
      for (const fn of this.readListeners) fn();
    };
    clearTimeout(this.readTimer);
    if (now - this.lastRead > 90) fire();
    else this.readTimer = setTimeout(fire, 90 - (now - this.lastRead));
  }

  attach(el: Elements) {
    this.el = el;
    this.colors = new ColorResolver(el.probe);
    const on = <K extends keyof HTMLElementEventMap>(target: HTMLElement, type: K, fn: (e: HTMLElementEventMap[K]) => void, opts?: AddEventListenerOptions) => {
      target.addEventListener(type, fn, opts);
      this.cleanup.push(() => {
        target.removeEventListener(type, fn, opts);
      });
    };
    const ro = new ResizeObserver(() => {
      this.measure();
    });
    ro.observe(el.trackBox);
    ro.observe(el.scroller);
    this.cleanup.push(() => {
      ro.disconnect();
    });

    on(el.ruler, "pointerdown", (e) => {
      if (e.button !== 0) return;
      el.ruler.setPointerCapture(e.pointerId);
      this.drag = { kind: "scrub", id: e.pointerId };
      this.pause();
      this.seek(this.posAt(e, el.ruler));
    });
    on(el.ruler, "pointermove", (e) => {
      if (this.drag?.kind === "scrub" && this.drag.id === e.pointerId) this.seek(this.posAt(e, el.ruler));
    });
    const endDrag = (e: PointerEvent) => {
      const d = this.drag;
      if (!d || d.id !== e.pointerId) return;
      this.drag = null;
      if (d.kind === "pan" && !d.moved) this.click(d.hit, this.posAt(e, el.tracks));
      el.root.style.cursor = "";
    };
    on(el.ruler, "pointerup", endDrag);
    on(el.ruler, "pointercancel", endDrag);

    on(el.tracks, "pointerdown", (e) => {
      if (e.button !== 0 && e.button !== 1) return;
      el.tracks.setPointerCapture(e.pointerId);
      this.drag = { kind: "pan", id: e.pointerId, x0: e.clientX, view: { ...this.view }, moved: e.button === 1, hit: this.hitTest(e) };
    });
    on(el.tracks, "pointermove", (e) => {
      const d = this.drag;
      if (d?.kind === "pan" && d.id === e.pointerId) {
        const dx = e.clientX - d.x0;
        if (!d.moved && Math.abs(dx) < 4) return;
        d.moved = true;
        el.root.style.cursor = "grabbing";
        const shift = (dx / Math.max(1, this.width)) * (d.view.end - d.view.start);
        this.setView({ start: d.view.start - shift, end: d.view.end - shift });
        this.setHover(null, e);
        return;
      }
      if (!this.drag) this.setHover(this.hitTest(e), e);
    });
    on(el.tracks, "pointerup", endDrag);
    on(el.tracks, "pointercancel", endDrag);
    on(el.tracks, "pointerleave", (e) => {
      if (!this.drag) this.setHover(null, e);
    });

    const wheel = (e: WheelEvent, target: HTMLElement, vertical: boolean) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const px = e.clientX - target.getBoundingClientRect().left;
        this.zoomAt(px, Math.exp(e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0025)));
        return;
      }
      const dx = e.shiftKey ? e.deltaY || e.deltaX : Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : vertical ? e.deltaY : 0;
      if (dx === 0) return;
      e.preventDefault();
      this.panPx(dx * (e.deltaMode === 1 ? 16 : 1));
    };
    on(el.tracks, "wheel", (e) => {
      wheel(e, el.tracks, false);
    }, { passive: false });
    on(el.ruler, "wheel", (e) => {
      wheel(e, el.ruler, true);
    }, { passive: false });
    on(el.minimapBox, "wheel", (e) => {
      wheel(e, el.minimapBox, true);
    }, { passive: false });

    on(el.minimapBox, "pointerdown", (e) => {
      if (e.button !== 0) return;
      const target = e.target instanceof HTMLElement ? e.target.dataset.edge : undefined;
      el.minimapBox.setPointerCapture(e.pointerId);
      if (target === "start" || target === "end" || target === "move") {
        this.drag = { kind: "minimap", id: e.pointerId, edge: target, x0: e.clientX, view: { ...this.view } };
        return;
      }
      const span = this.view.end - this.view.start;
      const at = this.minimapPos(e);
      this.setView({ start: at - span / 2, end: at + span / 2 });
      this.drag = { kind: "minimap", id: e.pointerId, edge: "move", x0: e.clientX, view: { ...this.view } };
    });
    on(el.minimapBox, "pointermove", (e) => {
      const d = this.drag;
      if (d?.kind !== "minimap" || d.id !== e.pointerId) return;
      const full = this.fullSpan();
      const shift = ((e.clientX - d.x0) / Math.max(1, el.minimapBox.clientWidth)) * (full.end - full.start);
      const minSpan = this.minSpan();
      if (d.edge === "move") this.setView({ start: d.view.start + shift, end: d.view.end + shift });
      else if (d.edge === "start") this.setView({ start: Math.min(d.view.start + shift, d.view.end - minSpan), end: d.view.end });
      else this.setView({ start: d.view.start, end: Math.max(d.view.end + shift, d.view.start + minSpan) });
    });
    on(el.minimapBox, "pointerup", endDrag);
    on(el.minimapBox, "pointercancel", endDrag);

    on(el.scroller, "scroll", () => {
      this.invalidate("tracks");
    }, { passive: true });

    this.measure();
  }

  detach() {
    for (const fn of this.cleanup) fn();
    this.cleanup = [];
    cancelAnimationFrame(this.frame);
    this.frame = 0;
    clearTimeout(this.readTimer);
    this.el = null;
  }

  themeChanged() {
    this.colors?.reset();
    this.hatch.clear();
    this.invalidate();
  }

  setScene(next: Scene) {
    const prev = this.scene;
    this.scene = next;
    if (prev.mode !== next.mode) {
      this.playhead = this.convert(this.playhead, prev, next);
      this.follow = true;
      this.coarse = { ...this.coarse, range: null };
    }
    if (prev.compact !== next.compact || prev.rows.length !== next.rows.length) this.measure();
    if (this.follow || prev.mode !== next.mode) this.view = fitView(next.end);
    else this.view = clampView(this.view, next.end, this.minSpan());
    if (this.coarse.speed === 1 && prev.end === 0 && next.end > 0) this.coarse = { ...this.coarse };
    this.invalidate();
    this.bumpRead();
  }

  private convert(pos: number, from: Scene, to: Scene): number {
    const ref = (from.focusId === null ? undefined : from.rows.find((r) => r.track.id === from.focusId)) ?? from.rows.find((r) => r.active);
    if (!ref) return 0;
    const reading = readTrack(ref.track, pos, from.mode, from.align);
    if (!reading) return 0;
    if (to.mode === "time") return reading.elapsed;
    const [x0, x1] = clipSpan(reading.clip, "align", to.align);
    return x0 + reading.fraction * (x1 - x0);
  }

  private measure() {
    const el = this.el;
    if (!el) return;
    this.width = el.trackBox.clientWidth;
    this.trackH = el.scroller.clientHeight;
    this.dpr = window.devicePixelRatio || 1;
    this.size(el.ruler, this.width, RULER_H);
    this.size(el.tracks, this.width, this.trackH);
    this.size(el.minimap, el.minimapBox.clientWidth, el.minimapBox.clientHeight);
    this.invalidate();
  }

  private size(c: HTMLCanvasElement, w: number, h: number) {
    const W = Math.max(1, Math.round(w * this.dpr));
    const H = Math.max(1, Math.round(h * this.dpr));
    if (c.width !== W) c.width = W;
    if (c.height !== H) c.height = H;
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
  }

  private minSpan(): number {
    return this.scene.mode === "time" ? 40 : 2;
  }

  private fullSpan(): View {
    return fitView(this.scene.end);
  }

  private setView(v: View) {
    this.follow = false;
    this.view = clampView(v, this.scene.end, this.minSpan());
    this.invalidate();
  }

  fit() {
    const r = this.coarse.range;
    this.follow = r === null;
    if (r) {
      const pad = (r.to - r.from) * 0.04;
      this.view = clampView({ start: r.from - pad, end: r.to + pad }, this.scene.end, this.minSpan());
    } else this.view = fitView(this.scene.end);
    this.invalidate();
  }

  zoomAt(px: number, factor: number) {
    const anchor = this.view.start + (px / Math.max(1, this.width)) * (this.view.end - this.view.start);
    this.setView(zoomAround(this.view, factor, anchor));
  }

  zoomBy(factor: number) {
    const x = this.toX(this.playhead);
    this.zoomAt(x >= 0 && x <= this.width ? x : this.width / 2, factor);
  }

  /** 0 = fit, 1 = tightest zoom, on a log scale. */
  get zoomLevel(): number {
    const full = this.fullSpan();
    const max = full.end - full.start;
    const min = this.minSpan();
    const span = this.view.end - this.view.start;
    if (max <= min) return 0;
    return Math.max(0, Math.min(1, Math.log(max / span) / Math.log(max / min)));
  }

  setZoomLevel(level: number) {
    const full = this.fullSpan();
    const max = full.end - full.start;
    const span = max / (max / this.minSpan()) ** level;
    const anchor = this.playhead;
    const f = (anchor - this.view.start) / (this.view.end - this.view.start);
    const lead = Number.isFinite(f) && f >= 0 && f <= 1 ? f : 0.5;
    this.setView({ start: anchor - span * lead, end: anchor + span * (1 - lead) });
  }

  panPx(dx: number) {
    const shift = dx * this.axisPerPx;
    this.setView({ start: this.view.start + shift, end: this.view.end + shift });
  }

  private toX(pos: number): number {
    return ((pos - this.view.start) / (this.view.end - this.view.start)) * this.width;
  }

  private posAt(e: PointerEvent | MouseEvent, target: HTMLElement): number {
    const x = e.clientX - target.getBoundingClientRect().left;
    return this.view.start + (x / Math.max(1, this.width)) * (this.view.end - this.view.start);
  }

  private minimapPos(e: PointerEvent): number {
    const box = this.el?.minimapBox;
    if (!box) return 0;
    const full = this.fullSpan();
    const x = e.clientX - box.getBoundingClientRect().left;
    return full.start + (x / Math.max(1, box.clientWidth)) * (full.end - full.start);
  }

  seek(pos: number) {
    const max = this.scene.end;
    this.playhead = Math.max(0, Math.min(max, pos));
    this.invalidate("overlay");
    this.bumpRead();
  }

  private reveal() {
    const span = this.view.end - this.view.start;
    if (this.playhead < this.view.start || this.playhead > this.view.end) {
      this.setView({ start: this.playhead - span * 0.1, end: this.playhead + span * 0.9 });
    }
  }

  private focusTrack(): Track | undefined {
    const { rows, focusId } = this.scene;
    return (focusId === null ? undefined : rows.find((r) => r.track.id === focusId)?.track) ?? rows.find((r) => r.active)?.track;
  }

  step(dir: 1 | -1) {
    const track = this.focusTrack();
    if (!track) return;
    const next = stepBoundary(boundaries(track, this.scene.mode), this.playhead, dir);
    if (next === null) return;
    this.pause();
    this.seek(next);
    this.reveal();
  }

  toStart() {
    this.pause();
    this.seek(this.coarse.range?.from ?? 0);
    this.reveal();
  }

  toEnd() {
    this.pause();
    this.seek(this.coarse.range?.to ?? this.scene.end);
    this.reveal();
  }

  togglePlay() {
    if (this.coarse.rate !== 0) {
      this.pause();
      return;
    }
    const lo = this.coarse.range?.from ?? 0;
    const hi = this.coarse.range?.to ?? this.scene.end;
    if (this.playhead >= hi - 1e-9 || this.playhead < lo) this.seek(lo);
    this.play(1);
  }

  shuttle(key: "j" | "k" | "l") {
    const rate = shuttle(this.coarse.rate, key);
    if (rate === 0) this.pause();
    else this.play(rate);
  }

  private play(rate: number) {
    this.lastTick = performance.now();
    this.setCoarse({ rate });
    this.invalidate("overlay");
  }

  pause() {
    if (this.coarse.rate !== 0) this.setCoarse({ rate: 0 });
  }

  setSpeed(speed: number) {
    this.setCoarse({ speed });
  }

  markIn() {
    const to = this.coarse.range?.to ?? this.scene.end;
    this.setRange({ from: this.playhead, to: Math.max(to, this.playhead) });
  }

  markOut() {
    const from = this.coarse.range?.from ?? 0;
    this.setRange({ from: Math.min(from, this.playhead), to: this.playhead });
  }

  private setRange(range: Range | null) {
    this.setCoarse({ range: range && range.to - range.from > 0 ? range : null });
    this.fit();
    this.bumpRead();
  }

  clearRange() {
    this.setRange(null);
  }

  select(sel: Selection | null) {
    this.setCoarse({ selected: sel });
    this.invalidate("tracks");
    this.bumpRead();
  }

  private click(hit: Hover | null, pos: number) {
    if (hit) {
      const [x0] = clipSpan(hit.clip, this.scene.mode, this.scene.align);
      this.onFocus(hit.track.id);
      this.select({ trackId: hit.track.id, kernelIndex: hit.clip.kernelIndex });
      this.pause();
      this.seek(x0);
      return;
    }
    this.select(null);
    this.pause();
    this.seek(pos);
  }


  private rowTop(i: number): number {
    const { h, gap } = rowMetrics(this.scene.compact);
    return PAD_Y + i * (h + gap);
  }

  private hitTest(e: PointerEvent): Hover | null {
    const el = this.el;
    if (!el) return null;
    const rect = el.tracks.getBoundingClientRect();
    const y = e.clientY - rect.top + el.scroller.scrollTop;
    const { h, gap } = rowMetrics(this.scene.compact);
    const i = Math.floor((y - PAD_Y + gap / 2) / (h + gap));
    const row = this.scene.rows[i];
    if (!row) return null;
    const top = this.rowTop(i);
    if (y < top - 2 || y > top + h + 2) return null;
    const pos = this.posAt(e, el.tracks);
    const slop = 3 * this.axisPerPx;
    const clips = row.track.clips;
    const { mode, align } = this.scene;
    let best: Clip | null = null;
    let bestDist = Infinity;
    let i0 = this.firstClip(clips, pos - slop);
    for (; i0 < clips.length; i0++) {
      const c = clips[i0];
      if (!c) break;
      const [x0, span1] = clipSpan(c, mode, align);
      const x1 = mode === "align" ? Math.max(span1, x0 + 1) : span1;
      if (x0 > pos + slop) break;
      const d = pos < x0 ? x0 - pos : pos > x1 ? pos - x1 : 0;
      if (d < bestDist) {
        bestDist = d;
        best = c;
      }
    }
    return best && bestDist <= slop ? { track: row.track, clip: best } : null;
  }

  /** First clip whose span could reach `pos`; clip starts rise with clip order on both axes. */
  private firstClip(clips: readonly Clip[], pos: number): number {
    const key = this.scene.mode === "time" ? (c: Clip) => c.start : (c: Clip) => c.kernelIndex;
    let lo = 0;
    let hi = clips.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      const c = clips[mid];
      if (c && key(c) <= pos) lo = mid + 1;
      else hi = mid;
    }
    return Math.max(0, lo - 1);
  }

  private setHover(hover: Hover | null, e: PointerEvent) {
    const prev = this.coarse.hover;
    const card = this.el?.hoverCard;
    if (card && hover && this.el) {
      const box = this.el.root.getBoundingClientRect();
      const x = e.clientX - box.left;
      const y = e.clientY - box.top;
      const flip = x + 300 > box.width;
      card.style.transform = `translate(${flip ? x - 296 : x + 16}px, ${y + 18}px)`;
    }
    if (prev?.clip === hover?.clip && prev?.track === hover?.track) return;
    this.setCoarse({ hover });
    this.invalidate("tracks");
  }

  invalidate(part: "tracks" | "ruler" | "minimap" | "overlay" | "all" = "all") {
    if (part === "all") this.dirty = { tracks: true, ruler: true, minimap: true };
    else if (part !== "overlay") this.dirty[part] = true;
    if (this.frame === 0 && this.el) this.frame = requestAnimationFrame(this.paint);
  }

  private paint = (now: number) => {
    this.frame = 0;
    if (!this.el) return;
    if (this.coarse.rate !== 0) this.advance(now);
    const viewKey = `${this.view.start}:${this.view.end}`;
    if (viewKey !== this.lastViewKey) {
      this.lastViewKey = viewKey;
      this.dirty.tracks = true;
      this.dirty.ruler = true;
    }
    if (this.dirty.tracks) this.drawTracks();
    if (this.dirty.ruler) this.drawRuler();
    if (this.dirty.minimap) this.drawMinimap();
    this.dirty = { tracks: false, ruler: false, minimap: false };
    this.drawOverlay();
    if (this.coarse.rate !== 0) this.frame = requestAnimationFrame(this.paint);
  };
  private lastViewKey = "";

  private advance(now: number) {
    const dt = Math.min(0.1, (now - this.lastTick) / 1000);
    this.lastTick = now;
    const s = SPEEDS[this.coarse.speed] ?? SPEEDS[1];
    if (!s) return;
    const perSec = this.scene.mode === "time" ? s.time : s.align;
    const lo = this.coarse.range?.from ?? 0;
    const hi = this.coarse.range?.to ?? this.scene.end;
    let next = this.playhead + dt * perSec * this.coarse.rate;
    if (next >= hi) {
      if (this.coarse.range) next = lo;
      else {
        next = hi;
        this.pause();
      }
    } else if (next <= lo) {
      next = lo;
      this.pause();
    }
    this.playhead = next;
    const span = this.view.end - this.view.start;
    if (next > this.view.end - span * 0.02 && this.coarse.rate > 0) this.setView({ start: next - span * 0.1, end: next + span * 0.9 });
    else if (next < this.view.start + span * 0.02 && this.coarse.rate < 0) this.setView({ start: next - span * 0.9, end: next + span * 0.1 });
    this.bumpRead();
  }

  private ctx(c: HTMLCanvasElement): CanvasRenderingContext2D | null {
    const ctx = c.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    return ctx;
  }

  private highlightFor(track: Track): number | null {
    const h = this.coarse.hover ?? null;
    const s = this.coarse.selected;
    const pick = h ? { trackId: h.track.id, kernelIndex: h.clip.kernelIndex } : s;
    if (!pick) return null;
    return this.scene.linkable || pick.trackId === track.id ? pick.kernelIndex : null;
  }

  private pattern(ctx: CanvasRenderingContext2D, color: string): CanvasPattern | null {
    const key = color;
    const hit = this.hatch.get(key);
    if (hit !== undefined) return hit;
    const size = Math.round(8 * this.dpr);
    const c = document.createElement("canvas");
    c.width = size;
    c.height = size;
    const p = c.getContext("2d");
    if (!p) return null;
    p.fillStyle = withAlpha(color, 0.14);
    p.fillRect(0, 0, size, size);
    p.strokeStyle = withAlpha(color, 0.55);
    p.lineWidth = 1.6 * this.dpr;
    p.beginPath();
    for (const o of [-size, 0, size]) {
      p.moveTo(o, size);
      p.lineTo(o + size, 0);
    }
    p.stroke();
    const pat = ctx.createPattern(c, "repeat");
    pat?.setTransform(new DOMMatrix([1 / this.dpr, 0, 0, 1 / this.dpr, 0, 0]));
    this.hatch.set(key, pat);
    return pat;
  }

  private drawTracks() {
    const el = this.el;
    const colors = this.colors;
    if (!el || !colors) return;
    const ctx = this.ctx(el.tracks);
    if (!ctx) return;
    const ink = colors.ink;
    const W = this.width;
    const H = this.trackH;
    ctx.clearRect(0, 0, W, H);
    const { rows, mode, align, compact, focusId } = this.scene;
    const { h, gap } = rowMetrics(compact);
    const scroll = el.scroller.scrollTop;
    const first = Math.max(0, Math.floor((scroll - PAD_Y) / (h + gap)) - 1);
    const last = Math.min(rows.length - 1, Math.ceil((scroll + H) / (h + gap)) + 1);
    const span = this.view.end - this.view.start;
    const x = (p: number) => ((p - this.view.start) / span) * W;

    for (let i = first; i <= last; i++) {
      const row = rows[i];
      if (!row) continue;
      const y = this.rowTop(i) - scroll;
      ctx.fillStyle = row.track.id === focusId ? ink.laneFocus : ink.lane;
      ctx.beginPath();
      ctx.roundRect(0, y - 2, W, h + 4, 7);
      ctx.fill();
    }

    const t = ticks(this.view, W, mode);
    ctx.strokeStyle = ink.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const m of t.major) {
      const gx = Math.round(x(m.at)) + 0.5;
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, H);
    }
    ctx.stroke();

    const endX = x(this.scene.end);
    if (endX < W) {
      ctx.fillStyle = withAlpha(ink.dim, 0.6);
      ctx.fillRect(Math.max(0, endX), 0, W - Math.max(0, endX), H);
    }

    const bands: { y: number; span: [number, number] | null; color: string; active: boolean }[] = [];
    for (let i = first; i <= last; i++) {
      const row = rows[i];
      if (!row) continue;
      const y = this.rowTop(i) - scroll;
      const hl = this.highlightFor(row.track);
      const color = colors.resolve(row.track.color);
      const base = row.active ? 1 : 0.18;
      const dimmed = hl !== null && row.active ? 0.4 : 1;
      let hlClip: Clip | null = null;
      const clips = row.track.clips;
      let runA = -1;
      let runB = -1;
      const flush = () => {
        if (runA < 0) return;
        ctx.globalAlpha = base * dimmed * 0.88;
        ctx.fillStyle = color;
        ctx.fillRect(runA, y, Math.max(1, runB - runA), h);
        runA = -1;
      };
      for (let k = this.firstClip(clips, this.view.start); k < clips.length; k++) {
        const c = clips[k];
        if (!c) break;
        const [s0, s1] = clipSpan(c, mode, align);
        if (s0 > this.view.end) break;
        if (s1 < this.view.start) continue;
        if (c.kernelIndex === hl && !c.pending) hlClip = c;
        const x0 = x(s0);
        const x1 = x(s1);
        if (c.pending) {
          flush();
          this.drawPending(ctx, c, x0, x1, y, h, color, base);
          continue;
        }
        if (x1 - x0 < 3) {
          if (runA >= 0 && x0 - runB > 1.5) flush();
          if (runA < 0) runA = x0;
          runB = Math.max(runB, x1);
          continue;
        }
        flush();
        this.drawClip(ctx, c, x0, x1, y, h, color, base * dimmed, false, compact);
      }
      flush();
      if (hlClip) {
        const [s0, s1] = clipSpan(hlClip, mode, align);
        this.drawClip(ctx, hlClip, x(s0), Math.max(x(s1), x(s0) + 2), y, h, color, base, true, compact);
      }
      ctx.globalAlpha = 1;
      bands.push({
        y,
        span: hlClip ? [x(clipSpan(hlClip, mode, align)[0]), x(clipSpan(hlClip, mode, align)[1])] : null,
        color,
        active: row.active,
      });
    }

    if (this.scene.linkable) {
      for (let i = 0; i + 1 < bands.length; i++) {
        const a = bands[i];
        const b = bands[i + 1];
        if (!a?.span || !b?.span || !a.active || !b.active) continue;
        const ay = a.y + h + 1;
        const by = b.y - 1;
        ctx.fillStyle = withAlpha(ink.accent, 0.2);
        ctx.beginPath();
        ctx.moveTo(a.span[0], ay);
        ctx.lineTo(a.span[1], ay);
        ctx.lineTo(b.span[1], by);
        ctx.lineTo(b.span[0], by);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = withAlpha(ink.accent, 0.7);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(a.span[0], ay);
        ctx.lineTo(b.span[0], by);
        ctx.moveTo(a.span[1], ay);
        ctx.lineTo(b.span[1], by);
        ctx.stroke();
      }
    }

    const r = this.coarse.range;
    if (r) {
      ctx.fillStyle = withAlpha(ink.dim, 0.62);
      const rx0 = Math.max(0, x(r.from));
      const rx1 = Math.min(W, x(r.to));
      if (rx0 > 0) ctx.fillRect(0, 0, rx0, H);
      if (rx1 < W) ctx.fillRect(rx1, 0, W - rx1, H);
    }
  }

  private drawClip(ctx: CanvasRenderingContext2D, c: Clip, x0: number, x1: number, y: number, h: number, color: string, alpha: number, highlight: boolean, compact: boolean) {
    const w = x1 - x0;
    const gap = w > 8 ? 1 : w > 4 ? 0.5 : 0;
    const left = Math.max(x0 + gap, -4);
    const right = Math.min(x1 - gap, this.width + 4);
    const cw = right - left;
    if (cw <= 0) return;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    if (cw > 6) {
      ctx.beginPath();
      ctx.roundRect(left, y, cw, h, compact ? 3 : 4);
      ctx.fill();
    } else ctx.fillRect(left, y, cw, h);
    ctx.globalAlpha = 1;
    if (highlight) {
      const ink = this.colors?.ink;
      ctx.strokeStyle = ink?.fg ?? "#000";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(left - 1, y - 1, cw + 2, h + 2, compact ? 4 : 5);
      ctx.stroke();
    }
    if (cw < 34 || alpha < 0.3) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(Math.max(left, 0), y, Math.min(cw, this.width) - 8, h);
    ctx.clip();
    ctx.globalAlpha = Math.min(1, alpha + 0.2);
    ctx.fillStyle = inkOn(color);
    ctx.textBaseline = "middle";
    const tx = Math.max(left, 0) + 6;
    if (compact || cw < 64) {
      ctx.font = LABEL_FONT;
      ctx.fillText(clipLabel(c.name), tx, y + h / 2 + 0.5);
    } else {
      ctx.font = LABEL_FONT;
      ctx.fillText(clipLabel(c.name), tx, y + h / 2 - 6);
      ctx.font = SUB_FONT;
      ctx.globalAlpha = Math.min(1, alpha) * 0.78;
      ctx.fillText(`#${c.kernelIndex} · ${formatNs(c.end - c.start)}`, tx, y + h / 2 + 7.5);
    }
    ctx.restore();
  }

  private drawPending(ctx: CanvasRenderingContext2D, c: Clip, x0: number, x1: number, y: number, h: number, color: string, alpha: number) {
    const left = Math.max(x0 + 1, -4);
    const right = Math.min(x1 - 1, this.width + 4);
    const w = right - left;
    if (w <= 0) return;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = this.pattern(ctx, color) ?? withAlpha(color, 0.2);
    ctx.beginPath();
    ctx.roundRect(left, y, w, h, 4);
    ctx.fill();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = withAlpha(color, 0.8);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(left + 0.5, y + 0.5, w - 1, h - 1, 4);
    ctx.stroke();
    ctx.setLineDash([]);
    if (w > 70 && this.colors) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(left, y, w - 6, h);
      ctx.clip();
      ctx.font = LABEL_FONT;
      ctx.fillStyle = this.colors.ink.muted;
      ctx.textBaseline = "middle";
      ctx.fillText(`#${c.kernelIndex} simulating…`, Math.max(left, 0) + 6, y + h / 2 + 0.5);
      ctx.restore();
    }
    ctx.globalAlpha = 1;
  }

  private drawRuler() {
    const el = this.el;
    const colors = this.colors;
    if (!el || !colors) return;
    const ctx = this.ctx(el.ruler);
    if (!ctx) return;
    const ink = colors.ink;
    const W = this.width;
    ctx.clearRect(0, 0, W, RULER_H);
    const span = this.view.end - this.view.start;
    const x = (p: number) => ((p - this.view.start) / span) * W;
    const r = this.coarse.range;
    if (r) {
      const a = x(r.from);
      const b = x(r.to);
      ctx.fillStyle = ink.accentSoft;
      ctx.fillRect(a, 0, b - a, RULER_H);
      ctx.fillStyle = ink.accent;
      for (const [px, dir] of [
        [a, 1],
        [b, -1],
      ] as const) {
        ctx.fillRect(Math.round(px) - (dir === 1 ? 0 : 2), 2, 2, RULER_H - 4);
        ctx.fillRect(Math.round(px) - (dir === 1 ? 0 : 6), 2, 6, 2);
        ctx.fillRect(Math.round(px) - (dir === 1 ? 0 : 6), RULER_H - 4, 6, 2);
      }
    }
    const t = ticks(this.view, W, this.scene.mode);
    ctx.strokeStyle = ink.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const m of t.minor) {
      const gx = Math.round(x(m)) + 0.5;
      ctx.moveTo(gx, RULER_H - 5);
      ctx.lineTo(gx, RULER_H);
    }
    ctx.stroke();
    ctx.strokeStyle = ink.faint;
    ctx.beginPath();
    for (const m of t.major) {
      const gx = Math.round(x(m.at)) + 0.5;
      ctx.moveTo(gx, RULER_H - 11);
      ctx.lineTo(gx, RULER_H);
    }
    ctx.stroke();
    ctx.font = RULER_FONT;
    ctx.fillStyle = ink.muted;
    ctx.textBaseline = "alphabetic";
    for (const m of t.major) ctx.fillText(m.label, Math.round(x(m.at)) + 4, 13);
    ctx.fillStyle = ink.axis;
    ctx.fillRect(0, RULER_H - 1, W, 1);
  }

  private drawMinimap() {
    const el = this.el;
    const colors = this.colors;
    if (!el || !colors) return;
    const ctx = this.ctx(el.minimap);
    if (!ctx) return;
    const W = el.minimapBox.clientWidth;
    const H = el.minimapBox.clientHeight;
    ctx.clearRect(0, 0, W, H);
    const rows = this.scene.rows;
    if (rows.length === 0) return;
    const full = this.fullSpan();
    const span = full.end - full.start;
    const pad = 5;
    const lane = Math.min(5, (H - pad * 2) / rows.length);
    const gap = lane >= 3 ? 1 : 0;
    const top = (H - rows.length * lane) / 2;
    rows.forEach((row, i) => {
      const color = colors.resolve(row.track.color);
      ctx.fillStyle = color;
      ctx.globalAlpha = row.active ? 0.9 : 0.2;
      let a = -1;
      let b = -1;
      const y = top + i * lane;
      for (const c of row.track.clips) {
        const [s0, s1] = clipSpan(c, this.scene.mode, this.scene.align);
        const x0 = ((s0 - full.start) / span) * W;
        const x1 = ((s1 - full.start) / span) * W;
        if (a >= 0 && x0 - b > 1) {
          ctx.fillRect(a, y, Math.max(1, b - a), Math.max(0.5, lane - gap));
          a = -1;
        }
        if (a < 0) a = x0;
        b = Math.max(b, x1);
      }
      if (a >= 0) ctx.fillRect(a, y, Math.max(1, b - a), Math.max(0.5, lane - gap));
    });
    ctx.globalAlpha = 1;
  }

  private drawOverlay() {
    const el = this.el;
    if (!el) return;
    const px = this.toX(this.playhead);
    const visible = px >= -1 && px <= this.width + 1;
    el.playhead.style.transform = `translateX(${px}px)`;
    el.playhead.style.visibility = visible ? "visible" : "hidden";
    const full = this.fullSpan();
    const fw = el.minimapBox.clientWidth;
    const fx = (p: number) => ((p - full.start) / (full.end - full.start)) * fw;
    const a = Math.max(0, fx(this.view.start));
    const b = Math.min(fw, fx(this.view.end));
    el.minimapWindow.style.transform = `translateX(${a}px)`;
    el.minimapWindow.style.width = `${Math.max(8, b - a)}px`;
    el.minimapPlayhead.style.transform = `translateX(${fx(this.playhead)}px)`;
    if (el.zoom) el.zoom.value = String(Math.round(this.zoomLevel * 1000));
    el.timecode.textContent =
      this.scene.mode === "time" ? formatTimecode(this.playhead, this.axisPerPx) : `#${this.playhead.toFixed(this.axisPerPx < 0.05 ? 2 : 1)}`;
    const track = this.focusTrack();
    const reading = track ? readTrack(track, this.playhead, this.scene.mode, this.scene.align) : null;
    const total = track ? track.clips.filter((c) => !c.pending).length : 0;
    el.kernelReadout.textContent = reading
      ? `#${reading.clip.kernelIndex} · ${formatInt(Math.min(total, reading.clip.kernelIndex + 1))}/${formatInt(total)}`
      : "—";
  }
}
