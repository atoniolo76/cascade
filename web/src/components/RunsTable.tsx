import { Link, useNavigate } from "react-router";
import { Activity, Boxes, CircleDot, Clock, Cpu, Layers } from "lucide-react";
import type { RunSummary, TraceSummary } from "../api/types";
import { plural } from "../domain/format";
import { RUN_STATE, finishedSims } from "../domain/status";
import { paths } from "../paths";
import { SimStateBlocks, Status } from "./meters";
import { ReplayBadge, SampleBadge, TraceChip } from "./TraceState";
import { LogoGlyph } from "./BrandLogo";
import { gpuLogo } from "../domain/logos";
import { Chip, EntityIcon, RelTime, Th } from "./ui";
import { ICON, ROW, THEAD } from "./styles";

const CELL = "h-[60px] px-4 align-middle";

export function RunIcon({ run }: { run: Pick<RunSummary, "state"> }) {
  return (
    <EntityIcon tone={RUN_STATE[run.state].tone}>
      <Activity {...ICON} />
    </EntityIcon>
  );
}

export function ConfigChips({ entries }: { entries: RunSummary["entries"] }) {
  return (
    <span className="flex min-w-0 flex-wrap gap-1.5">
      {entries.map((e) => {
        const logo = gpuLogo(e.gpu_config);
        return (
          <Chip key={e.gpu_config} className="h-6 px-2 font-mono text-[12px]">
            {logo && <LogoGlyph logo={logo} className="size-3 text-muted" />}
            {e.gpu_config}
            <span className="text-muted">×{e.count}</span>
          </Chip>
        );
      })}
    </span>
  );
}

function SimProgress({ run }: { run: RunSummary }) {
  const { done, total } = finishedSims(run.sim_counts);
  return (
    <span className="flex min-w-0 flex-col gap-1">
      <SimStateBlocks counts={run.sim_counts} slots={14} />
      <span className="text-[12px] leading-4 text-muted tabular-nums">
        {done === total ? plural(total, "sim") : `${done} of ${plural(total, "sim")} finished`}
      </span>
    </span>
  );
}

export function RunsTable({ runs, traces, showTrace = true }: { runs: readonly RunSummary[]; traces?: readonly TraceSummary[] | undefined; showTrace?: boolean }) {
  const navigate = useNavigate();
  const traceById = new Map(traces?.map((t) => [t.id, t]));
  return (
    <>
      <table className="hidden w-full table-fixed border-collapse lg:table">
        <colgroup>
          <col className="w-[200px]" />
          <col className="w-[148px]" />
          {showTrace && <col />}
          <col className={showTrace ? "w-[24%]" : undefined} />
          <col className="w-[236px]" />
          <col className="w-[112px]" />
        </colgroup>
        <thead className={THEAD}>
          <tr>
            <Th icon={<Activity {...ICON} />}>Run</Th>
            <Th icon={<CircleDot {...ICON} />}>State</Th>
            {showTrace && <Th icon={<Layers {...ICON} />}>Trace</Th>}
            <Th icon={<Cpu {...ICON} />}>Configs</Th>
            <Th icon={<Boxes {...ICON} />}>Sims</Th>
            <Th icon={<Clock {...ICON} />}>Started</Th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const meta = RUN_STATE[r.state];
            return (
              <tr
                key={r.id}
                onClick={() => {
                  void navigate(paths.run(r.id));
                }}
                className={`${ROW} cursor-pointer hover:bg-surface-2/50`}
              >
                <td className={CELL}>
                  <Link
                    to={paths.run(r.id)}
                    onClick={(e) => {
                      e.stopPropagation();
                    }}
                    className="flex min-w-0 items-center gap-3"
                  >
                    <RunIcon run={r} />
                    <span title={r.id} className="truncate font-mono text-[13px] font-medium text-fg">{r.id}</span>
                    {r.sample && <SampleBadge />}
                    {r.replay && <ReplayBadge />}
                  </Link>
                </td>
                <td className={CELL}>
                  <Status tone={meta.tone} live={meta.live} label={meta.label} />
                </td>
                {showTrace && (
                  <td className={CELL} onClick={(e) => { e.stopPropagation(); }}>
                    <TraceChip trace={traceById.get(r.trace_id) ?? { id: r.trace_id, model: null }} />
                  </td>
                )}
                <td className={CELL}>
                  <ConfigChips entries={r.entries} />
                </td>
                <td className={CELL}>
                  <SimProgress run={r} />
                </td>
                <td className={`${CELL} text-[14px] text-muted`}>
                  <RelTime iso={r.created_at} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <ul className="divide-y divide-border lg:hidden">
        {runs.map((r) => {
          const meta = RUN_STATE[r.state];
          return (
            <li key={r.id}>
              <Link to={paths.run(r.id)} className="flex flex-col gap-3 px-4 py-4 transition-colors duration-150 hover:bg-surface-2/50">
                <div className="flex items-center gap-3">
                  <RunIcon run={r} />
                  <span className="min-w-0 flex-1 truncate font-mono text-[13px] font-medium text-fg">{r.id}</span>
                  {r.sample && <SampleBadge />}
                  {r.replay && <ReplayBadge />}
                  <Status tone={meta.tone} live={meta.live} label={meta.label} />
                </div>
                <div className="flex flex-col gap-2.5 pl-9">
                  {showTrace && (
                    <span className="truncate text-[14px] text-fg-2">{traceById.get(r.trace_id)?.model ?? r.trace_id}</span>
                  )}
                  <ConfigChips entries={r.entries} />
                  <SimProgress run={r} />
                  <RelTime iso={r.created_at} className="text-[13px] text-muted" />
                </div>
              </Link>
            </li>
          );
        })}
      </ul>
    </>
  );
}
