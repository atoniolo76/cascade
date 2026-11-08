import { ChevronRight, Eye, EyeOff } from "lucide-react";
import { formatNs, formatSigned } from "../../domain/format";
import type { Track } from "../../domain/timeline";
import { SIM_STATE } from "../../domain/status";
import { LogoTile } from "../BrandLogo";
import { cx } from "../cx";
import { ICON } from "../styles";
import { rowMetrics, type Row } from "./controller";
import type { TrackMeta } from "./meta";

function reference(rows: readonly Row[]): Track | undefined {
  const live = rows.filter((r) => r.active && r.depth === 0);
  return (live.find((r) => r.track.source.kind === "real") ?? live[0])?.track;
}

function Delta({ track, reference: ref }: { track: Track; reference: Track | undefined }) {
  if (track.clips.at(-1)?.pending) return <span className="text-[var(--st-active-ink)]">growing</span>;
  if (!ref || ref === track || ref.total === 0) return <span className="text-faint">{ref === track ? "reference" : ""}</span>;
  const d = (track.total - ref.total) / ref.total;
  return (
    <span title={`${formatSigned(d)} total time vs ${ref.label}`} className="text-muted">
      {formatSigned(d)}
    </span>
  );
}

const ICON_BTN =
  "grid size-6 shrink-0 place-items-center rounded-[7px] text-faint transition-colors duration-150 hover:bg-surface-2 hover:text-fg [&>svg]:size-3.5";

export function TrackHeaders({
  rows,
  showDifferences,
  meta,
  compact,
  focusId,
  expanded,
  hasReplicas,
  solo,
  hidden,
  height,
  onFocus,
  onToggleHidden,
  onToggleSolo,
  onToggleExpand,
}: {
  rows: readonly Row[];
  showDifferences: boolean;
  meta: (track: Track) => TrackMeta;
  compact: boolean;
  focusId: string | null;
  expanded: ReadonlySet<string>;
  hasReplicas: (id: string) => boolean;
  solo: ReadonlySet<string>;
  hidden: ReadonlySet<string>;
  height: number;
  onFocus: (id: string) => void;
  onToggleHidden: (id: string) => void;
  onToggleSolo: (id: string) => void;
  onToggleExpand: (id: string) => void;
}) {
  const { h, gap } = rowMetrics(compact);
  const ref = reference(rows);
  return (
    <div className="sticky left-0 z-10 flex flex-col border-r border-border bg-surface pt-[6px]" style={{ height, rowGap: gap }}>
      {rows.map(({ track, active, depth }) => {
        const m = meta(track);
        const focused = track.id === focusId;
        const isSolo = solo.has(track.id);
        const isHidden = hidden.has(track.id);
        const canExpand = depth === 0 && hasReplicas(track.id);
        const open = expanded.has(track.id);
        const sub = track.source.kind === "sim" ? (track.source.state === "succeeded" ? undefined : SIM_STATE[track.source.state].label) : m.sub;
        return (
          <div
            key={track.id}
            data-track={track.id}
            onClick={() => {
              onFocus(track.id);
            }}
            className={cx(
              "group relative flex shrink-0 cursor-pointer items-center gap-1.5 pr-1.5 transition-[background-color,opacity] duration-150",
              depth === 1 ? "pl-2 sm:pl-6" : "pl-2",
              focused ? "bg-accent-soft" : "hover:bg-surface-2/70",
              !active && "opacity-45",
            )}
            style={{ height: h }}
          >
            <span aria-hidden className="absolute inset-y-1 left-0 w-[3px] rounded-r-full" style={{ background: track.color, opacity: depth === 1 ? 0.55 : 1 }} />
            {canExpand ? (
              <button
                type="button"
                aria-label={open ? `Collapse ${track.label} replicas` : `Expand ${track.label} replicas`}
                aria-expanded={open}
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleExpand(track.id);
                }}
                className={cx(ICON_BTN, "hidden sm:grid")}
              >
                <ChevronRight className={cx("transition-transform duration-150", open && "rotate-90")} {...ICON} />
              </button>
            ) : (
              depth === 0 && <span aria-hidden className="hidden w-6 shrink-0 sm:block" />
            )}
            {depth === 0 ? (
              <LogoTile logo={m.logo} fallback={track.label} size={compact ? "sm" : "md"} className="mx-auto sm:mx-0" />
            ) : (
              <span aria-hidden className="mx-auto grid size-[18px] shrink-0 place-items-center rounded-[5px] bg-surface-2 font-mono text-[10px] text-muted sm:mx-0">
                {track.source.kind === "sim" ? track.source.replica : ""}
              </span>
            )}
            <span className="hidden min-w-0 flex-1 flex-col sm:flex">
              <span className={cx("truncate font-mono text-[12.5px] leading-4 font-medium", focused ? "text-fg" : "text-fg-2")} title={track.label}>
                {track.label}
              </span>
              {!compact && (
                <span className="flex min-w-0 items-center gap-1.5 text-[11px] leading-4 tabular-nums">
                  <span className="shrink-0 text-fg-2">{formatNs(track.total)}</span>
                  <span className="shrink-0">
                    {showDifferences && <Delta track={track} reference={ref} />}
                  </span>
                  {sub !== undefined && <span className="min-w-0 truncate text-faint">· {sub}</span>}
                </span>
              )}
            </span>
            {compact && <span className="hidden shrink-0 font-mono text-[11px] text-muted tabular-nums sm:inline">{formatNs(track.total)}</span>}
            <span className="hidden items-center sm:flex">
              <button
                type="button"
                aria-label={isHidden ? `Show ${track.label}` : `Hide ${track.label}`}
                aria-pressed={isHidden}
                title={isHidden ? "Show track" : "Hide track"}
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleHidden(track.id);
                }}
                className={cx(ICON_BTN, isHidden && "text-fg")}
              >
                {isHidden ? <EyeOff {...ICON} /> : <Eye {...ICON} />}
              </button>
              <button
                type="button"
                aria-label={`Solo ${track.label}`}
                aria-pressed={isSolo}
                title="Solo: show only soloed tracks"
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleSolo(track.id);
                }}
                className={cx(ICON_BTN, "font-mono text-[11px] font-semibold", isSolo && "bg-[var(--st-lost-bg)] text-[var(--st-lost-ink)] hover:bg-[var(--st-lost-bg)] hover:text-[var(--st-lost-ink)]")}
              >
                S
              </button>
            </span>
          </div>
        );
      })}
    </div>
  );
}
