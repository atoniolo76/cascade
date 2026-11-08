import { cx } from "../components/cx";
import { LANE_CELLS, type Hotspot } from "./store";

/** A stable load in 0..1 for one cell, so the heatmap looks the same on every render. */
function load(lane: number, cell: number): number {
  const x = Math.sin(lane * 12.9898 + cell * 78.233) * 43758.5453;
  const noise = x - Math.floor(x);
  const wave = 0.5 + 0.35 * Math.sin((cell / LANE_CELLS) * Math.PI * 2 + lane);
  return Math.min(1, Math.max(0.08, wave * 0.7 + noise * 0.3));
}

/**
 * One lane per GPU, one cell per slice of the replay, shaded by load. Hotspots are outlined in the
 * warning color. Estimated lanes are hatched, as everywhere else that shows a simulated profile.
 */
export function MiniLanes({ lanes, hotspots, estimated, className }: { lanes: number; hotspots: readonly Hotspot[]; estimated: readonly boolean[]; className?: string }) {
  return (
    <div className={cx("grid gap-1", className)}>
      {Array.from({ length: lanes }, (_, lane) => (
        <div key={lane} className="flex items-center gap-2">
          <span className="w-8 shrink-0 font-mono text-[10.5px] text-faint">gpu{lane}</span>
          <div className="grid flex-1 gap-[2px]" style={{ gridTemplateColumns: `repeat(${LANE_CELLS}, minmax(0, 1fr))` }}>
            {Array.from({ length: LANE_CELLS }, (_, cell) => {
              const hot = hotspots.some((h) => h.gpu === lane && h.cell === cell);
              const l = hot ? 1 : load(lane, cell);
              return (
                <span
                  key={cell}
                  className={cx("h-3 rounded-[2px]", hot && "ring-[1.5px] ring-(--st-lost-dot) ring-offset-1 ring-offset-surface")}
                  style={{
                    backgroundColor: hot ? "var(--st-lost-dot)" : `color-mix(in oklab, var(--accent) ${Math.round(l * 85)}%, var(--surface-2))`,
                    backgroundImage: estimated[lane] === true && !hot ? "repeating-linear-gradient(135deg, transparent 0 2px, rgb(255 255 255 / 0.35) 2px 3px)" : undefined,
                  }}
                />
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
