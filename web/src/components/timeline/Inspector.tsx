import { useSyncExternalStore } from "react";
import { formatInt, formatNs, formatPercent, formatSigned } from "../../domain/format";
import { clipForKernel, kernelRange, rangeTime, readTrack, type Track } from "../../domain/timeline";
import { kernelLabel } from "../../domain/kernelLabel";
import { LogoTile } from "../BrandLogo";
import { cx } from "../cx";
import type { TimelineController } from "./controller";
import type { TrackMeta } from "./meta";

function lead(ns: number): string {
  if (Math.abs(ns) < 0.5) return "level";
  return `${formatNs(Math.abs(ns))} ${ns < 0 ? "ahead" : "behind"}`;
}

const TH = "h-8 px-3 text-left text-[11px] font-medium whitespace-nowrap text-muted";
const TD = "h-9 px-3 text-[12px] whitespace-nowrap tabular-nums text-fg-2";

export function Inspector({ ctl, meta, linkable }: { ctl: TimelineController; meta: (t: Track) => TrackMeta; linkable: boolean }) {
  useSyncExternalStore(ctl.subscribeRead, ctl.getReadVersion);
  const coarse = useSyncExternalStore(ctl.subscribe, ctl.getCoarse);
  const scene = ctl.currentScene;
  const pos = ctl.playhead;
  const rows = scene.rows.filter((r) => r.active);
  const focus = rows.find((r) => r.track.id === scene.focusId)?.track ?? rows[0]?.track;
  if (!focus) return null;
  const focusReading = readTrack(focus, pos, scene.mode, scene.align);
  const k = focusReading && !focusReading.before ? focusReading.clip.kernelIndex : null;
  const focusClip = k === null ? undefined : clipForKernel(focus, k);

  const shown = rows.filter((r) => r.depth === 0 || r.track === focus);
  const replicasHidden = rows.length - shown.length;
  const readings = shown.map(({ track }) => {
    const reading = readTrack(track, pos, scene.mode, scene.align);
    const other = k === null ? undefined : clipForKernel(track, k);
    const diff = linkable && focusClip && other && !other.pending && !focusClip.pending && track !== focus ? other.start - focusClip.start : null;
    return { track, reading, diff };
  });

  let headline: { track: Track; diff: number } | null = null;
  for (const r of readings) if (r.diff !== null && (headline === null || Math.abs(r.diff) > Math.abs(headline.diff))) headline = { track: r.track, diff: r.diff };

  const span = coarse.range ? kernelRange(focus, coarse.range.from, coarse.range.to, scene.mode, scene.align) : null;
  const focusRange = span ? rangeTime(focus, span) : null;

  return (
    <section aria-label="Inspector" className="px-3 py-4 sm:px-5">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 className="text-[13px] font-medium text-fg">At the playhead</h3>
        <p data-testid="lead-sentence" className="min-w-0 text-[13px] text-fg-2">
          {headline !== null && k !== null ? (
            Math.abs(headline.diff) < 0.5 ? (
              <>
                Every track reaches kernel <span className="font-mono">#{k}</span> at the same time
              </>
            ) : (
              <>
                <span className="font-mono font-medium text-fg">{headline.track.label}</span> is{" "}
                <span className="font-medium text-fg tabular-nums">{formatNs(Math.abs(headline.diff))}</span> {headline.diff < 0 ? "ahead of" : "behind"}{" "}
                <span className="font-mono font-medium text-fg">{focus.label}</span> at kernel <span className="font-mono">#{k}</span>
              </>
            )
          ) : !linkable ? (
            <span className="text-muted">Different traces, so kernels aren't compared across tracks</span>
          ) : (
            <span className="text-muted">Move the playhead over a kernel to compare tracks</span>
          )}
        </p>
      </div>
      <div className="max-h-[320px] overflow-auto rounded-[14px] border border-border">
        <table className="w-full border-collapse">
          <thead className="sticky top-0 z-[1] bg-card">
            <tr className="border-b border-border">
              <th className={TH}>Track</th>
              <th className={TH}>Kernel</th>
              <th className={TH}>Through</th>
              <th className={cx(TH, "hidden sm:table-cell")}>Elapsed</th>
              <th className={cx(TH, "hidden md:table-cell")}>Cycles</th>
              <th className={cx(TH, "hidden md:table-cell")}>IPC</th>
              <th className={TH} title={`Start of this kernel vs ${focus.label}`}>
                vs <span className="font-mono">{focus.label}</span>
              </th>
              {span && focusRange && (
                <th className={cx(TH, "hidden lg:table-cell")}>
                  Kernels #{span[0]}–#{span[1]}
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {readings.map(({ track, reading, diff }) => {
              const clip = reading?.clip;
              const r = span ? rangeTime(track, span) : null;
              const m = meta(track);
              return (
                <tr key={track.id} className={cx("border-b border-border last:border-b-0", track === focus && "bg-accent-soft")}>
                  <td className={cx(TD, "max-w-[220px]")}>
                    <span className="flex min-w-0 items-center gap-2">
                      <span aria-hidden className="h-3 w-[3px] shrink-0 rounded-full" style={{ background: track.color }} />
                      {track.source.kind !== "sim" && <LogoTile logo={m.logo} fallback={track.label} size="sm" />}
                      <span className="truncate font-mono text-fg">{track.label}</span>
                    </span>
                  </td>
                  <td className={cx(TD, "max-w-[260px]")}>
                    {clip ? (
                      <span className="flex min-w-0 items-baseline gap-2">
                        <span className="font-mono text-fg">#{clip.kernelIndex}</span>
                        <span className="hidden truncate font-mono text-[11px] text-muted sm:inline" title={clip.name}>
                          {clip.pending ? "simulating…" : kernelLabel(clip.name)}
                        </span>
                      </span>
                    ) : (
                      <span className="text-faint">—</span>
                    )}
                  </td>
                  <td className={TD}>
                    {reading ? (
                      <span className="flex items-center gap-2">
                        <span className="relative h-1.5 w-12 overflow-hidden rounded-full bg-block-off">
                          <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${reading.fraction * 100}%`, background: track.color }} />
                        </span>
                        <span className="w-9 text-right">{reading.done ? "done" : reading.before ? "—" : formatPercent(reading.fraction)}</span>
                      </span>
                    ) : (
                      <span className="text-faint">—</span>
                    )}
                  </td>
                  <td className={cx(TD, "hidden sm:table-cell")}>{reading ? formatNs(reading.elapsed) : "—"}</td>
                  <td className={cx(TD, "hidden md:table-cell")}>{clip?.cycles != null ? formatInt(Math.round(clip.cycles)) : "—"}</td>
                  <td className={cx(TD, "hidden md:table-cell")}>{clip?.ipc != null ? clip.ipc.toFixed(2) : "—"}</td>
                  <td className={TD}>{track === focus ? <span className="text-faint">reference</span> : diff === null ? <span className="text-faint">—</span> : lead(diff)}</td>
                  {span && focusRange && (
                    <td className={cx(TD, "hidden lg:table-cell")}>
                      {r && r.kernels > 0 ? (
                        <span>
                          {formatNs(r.ns)}
                          {track !== focus && focusRange.ns > 0 && <span className="ml-1.5 text-muted">{formatSigned((r.ns - focusRange.ns) / focusRange.ns)}</span>}
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {replicasHidden > 0 && <p className="mt-2 text-[12px] text-muted">{replicasHidden} expanded replicas are on the timeline; click one's header to inspect it here.</p>}
    </section>
  );
}
