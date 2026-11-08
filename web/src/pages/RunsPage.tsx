import { Link, useNavigate } from "react-router";
import { Layers } from "lucide-react";
import { useRuns } from "../api/queries";
import type { RunSummary } from "../api/types";
import { PageHeader } from "../components/AppShell";
import { SimCountsBar, SimCountsText } from "../components/SimCounts";
import { Card, EmptyState, ErrorNote, Mono, Skeleton, StatusPill } from "../components/ui";
import { entriesLabel, formatAge, ms } from "../domain/format";
import { RUN_STATE } from "../domain/status";
import { useNow } from "../hooks/useNow";
import { paths } from "../paths";

const HEAD = "px-3 py-2.5 text-left text-[11px] font-medium tracking-wide text-muted uppercase whitespace-nowrap";
const CELL = "px-3 h-14 align-middle";

export function RunsPage() {
  const runs = useRuns();
  return (
    <>
      <PageHeader title="Runs" subtitle="Simulation runs, one sailbox per sim." />
      {runs.isError && runs.data === undefined ? (
        <ErrorNote error={runs.error} />
      ) : (
        <Card className="overflow-hidden">
          {runs.data === undefined ? (
            <div className="divide-y divide-border">
              <div className="h-10 bg-surface-2/50" />
              {Array.from({ length: 4 }, (_, i) => (
                <div key={i} className="flex h-14 items-center gap-6 px-3">
                  <Skeleton className="h-3 w-24" />
                  <Skeleton className="h-5 w-20 rounded-full" />
                  <Skeleton className="h-3 w-[30%]" />
                  <Skeleton className="ml-auto h-1.5 w-28 rounded-full" />
                </div>
              ))}
            </div>
          ) : runs.data.length === 0 ? (
            <EmptyState
              icon={<Layers className="size-5" />}
              title="No runs yet"
              body="Open a ready trace and start a run to simulate it on other GPUs."
              action={
                <Link to={paths.newRun()} className="text-[13px] font-medium text-accent-ink hover:underline">
                  Start a run
                </Link>
              }
            />
          ) : (
            <RunTable runs={runs.data} />
          )}
        </Card>
      )}
    </>
  );
}

function RunTable({ runs }: { runs: RunSummary[] }) {
  const now = useNow();
  const navigate = useNavigate();
  return (
    <>
      <table className="hidden w-full table-fixed border-collapse lg:table">
        <colgroup>
          <col className="w-[150px]" />
          <col className="w-[124px]" />
          <col />
          <col className="w-[22%]" />
          <col className="w-[26%]" />
          <col className="w-[60px]" />
        </colgroup>
        <thead className="border-b border-border bg-surface-2/50">
          <tr>
            <th className={HEAD}>Run</th>
            <th className={HEAD}>State</th>
            <th className={HEAD}>Trace</th>
            <th className={HEAD}>Configs</th>
            <th className={HEAD}>Sims</th>
            <th className={`${HEAD} text-right`}>Age</th>
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
                className="cursor-pointer border-b border-border transition-colors duration-150 last:border-b-0 hover:bg-surface-2/60"
              >
                <td className={CELL}>
                  <Link to={paths.run(r.id)} onClick={(e) => { e.stopPropagation(); }} className="block">
                    <Mono className="font-medium text-fg">{r.id}</Mono>
                  </Link>
                </td>
                <td className={CELL}>
                  <StatusPill tone={meta.tone} live={meta.live} label={meta.label} />
                </td>
                <td className={CELL}>
                  <Mono className="text-fg-2">{r.trace_id}</Mono>
                </td>
                <td className={CELL}>
                  <span className="block truncate font-mono text-[12px] text-fg-2" title={entriesLabel(r.entries)}>
                    {entriesLabel(r.entries)}
                  </span>
                </td>
                <td className={CELL}>
                  <div className="flex flex-col gap-1.5">
                    <SimCountsBar counts={r.sim_counts} className="w-full max-w-[200px]" />
                    <span className="truncate text-[12px] text-muted">
                      <SimCountsText counts={r.sim_counts} />
                    </span>
                  </div>
                </td>
                <td className={`${CELL} text-right text-[13px] text-muted tabular-nums`}>{formatAge(now - ms(r.created_at))}</td>
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
              <Link to={paths.run(r.id)} className="flex flex-col gap-2 px-4 py-3 transition-colors duration-150 hover:bg-surface-2/60">
                <div className="flex items-center gap-3">
                  <Mono className="flex-1 font-medium text-fg">{r.id}</Mono>
                  <StatusPill tone={meta.tone} live={meta.live} label={meta.label} />
                  <span className="w-8 shrink-0 text-right text-[12px] text-muted tabular-nums">{formatAge(now - ms(r.created_at))}</span>
                </div>
                <Mono className="text-muted">{r.trace_id}</Mono>
                <span className="truncate font-mono text-[12px] text-fg-2">{entriesLabel(r.entries)}</span>
                <SimCountsBar counts={r.sim_counts} />
                <span className="text-[12px] text-muted">
                  <SimCountsText counts={r.sim_counts} />
                </span>
              </Link>
            </li>
          );
        })}
      </ul>
    </>
  );
}
