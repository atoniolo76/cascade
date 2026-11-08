import { themeStore } from "../../lib/theme";

export interface Ink {
  fg: string;
  muted: string;
  faint: string;
  grid: string;
  axis: string;
  lane: string;
  laneFocus: string;
  surface: string;
  accent: string;
  accentSoft: string;
  dim: string;
}

/** Canvas can't read `var(--x)` or `light-dark()`, so a hidden probe element resolves them to rgb() per theme. */
export class ColorResolver {
  private cache = new Map<string, string>();
  private probe: HTMLElement;
  ink: Ink;

  constructor(probe: HTMLElement) {
    this.probe = probe;
    this.ink = this.readInk();
  }

  resolve(css: string): string {
    const hit = this.cache.get(css);
    if (hit !== undefined) return hit;
    this.probe.style.color = css;
    const value = getComputedStyle(this.probe).color;
    this.cache.set(css, value);
    return value;
  }

  reset() {
    this.cache.clear();
    this.ink = this.readInk();
  }

  private readInk(): Ink {
    const r = (name: string) => this.resolve(`var(--${name})`);
    return {
      fg: r("fg"),
      muted: r("muted"),
      faint: r("faint"),
      grid: r("grid"),
      axis: r("axis"),
      lane: r("card"),
      laneFocus: r("accent-soft"),
      surface: r("surface"),
      accent: r("accent"),
      accentSoft: r("accent-soft"),
      dim: r("panel"),
    };
  }
}

export function onThemeChange(fn: () => void): () => void {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const unsub = themeStore.subscribe(fn);
  mq.addEventListener("change", fn);
  return () => {
    unsub();
    mq.removeEventListener("change", fn);
  };
}

function channels(rgb: string): [number, number, number, number] {
  const m = rgb.match(/[\d.]+/g)?.map(Number) ?? [];
  return [m[0] ?? 0, m[1] ?? 0, m[2] ?? 0, m[3] ?? 1];
}

export function withAlpha(rgb: string, alpha: number): string {
  const [r, g, b, a] = channels(rgb);
  return `rgba(${r}, ${g}, ${b}, ${a * alpha})`;
}

/** Dark ink on light fills, white on dark ones; WCAG relative luminance with the 0.18 crossover. */
export function inkOn(rgb: string): string {
  const lin = (c: number) => {
    const s = c / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  const [r, g, b] = channels(rgb);
  const l = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return l > 0.18 ? "rgba(17, 17, 16, 0.92)" : "rgba(255, 255, 255, 0.96)";
}
