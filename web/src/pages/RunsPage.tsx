import { Link, useSearchParams } from "react-router";
import { Activity, Plus, X } from "lucide-react";
import { useRuns, useTraces } from "../api/queries";
import type { RunState } from "../api/types";
import { PageHeader } from "../components/AppShell";
import { RunsTable } from "../components/RunsTable";
import { EmptyState, ErrorNote, Panel, Skeleton } from "../components/ui";
import { ICON, buttonClass } from "../components/styles";
import { cx } from "../components/cx";
import { RUN_STATE } from "../domain/status";
import { paths } from "../paths";

type Filter = "all" | RunState;
const FILTERS: readonly Filter[] = ["all", "active", "succeeded", "partial", "failed", "cancelled"];

function isFilter(v: string | null): v is Filter {
  return v !== null && (FILTERS as readonly string[]).includes(v);
}

export function RunsPage() {
  const runs = useRuns();
  const traces = useTraces();
  const [params, setParams] = useSearchParams();
  const traceId = params.get("trace");
  const stateParam = params.get("state");
  const filter: Filter = isFilter(stateParam) ? stateParam : "all";
  const trace = traceId === null ? undefined : traces.data?.find((t) => t.id === traceId);

  const forTrace = runs.data?.filter((r) => traceId === null || r.trace_id === traceId);
  const shown = forTrace?.filter((r) => filter === "all" || r.state === filter);

  const setParam = (key: string, value: string | null) => {
    const next = new URLSearchParams(params);
    if (value === null) next.delete(key);
    else next.set(key, value);
    setParams(next, { replace: true });
  };

  return (
    <>
      <PageHeader
        crumbs={[{ label: "Runs" }]}
        status={runs.data !== undefined && runs.data.length > 0 ? <span className="text-[15px] text-muted tabular-nums">{runs.data.length}</span> : undefined}
        actions={
          <Link to={paths.newRun(traceId ?? undefined)} className={buttonClass({ variant: "primary" })}>
            <Plus className="size-4" {...ICON} />
            New run
          </Link>
        }
      >
        <p className="max-w-2xl text-[15px] text-muted">Each run replays one trace on a set of GPU configs, one sailbox per sim.</p>
      </PageHeader>

      {forTrace !== undefined && runs.data !== undefined && runs.data.length > 0 && (
        <div className="mb-4 flex flex-wrap items-center gap-2">
          {FILTERS.map((f) => {
            const n = f === "all" ? forTrace.length : forTrace.filter((r) => r.state === f).length;
            if (f !== "all" && n === 0) return null;
            return (
              <button
                key={f}
                type="button"
                aria-pressed={filter === f}
                onClick={() => {
                  setParam("state", f === "all" ? null : f);
                }}
                className={cx(
                  "inline-flex h-8 items-center gap-1.5 rounded-[10px] px-3 text-[13px] font-medium transition-colors duration-150",
                  filter === f ? "bg-surface text-fg shadow-tile ring-1 ring-border" : "text-muted hover:text-fg",
                )}
              >
                {f === "all" ? "All" : RUN_STATE[f].label}
                <span className="text-faint tabular-nums">{n}</span>
              </button>
            );
          })}
          {traceId !== null && (
            <span className="ml-auto inline-flex h-8 max-w-full items-center gap-1.5 rounded-[10px] bg-surface pr-1 pl-3 text-[13px] text-fg-2 ring-1 ring-border">
              <span className="text-muted">Trace</span>
              <span className="truncate font-medium text-fg">{trace?.model ?? traceId}</span>
              <button
                type="button"
                aria-label="Clear trace filter"
                onClick={() => {
                  setParam("trace", null);
                }}
                className="grid size-6 place-items-center rounded-[7px] text-faint hover:bg-surface-2 hover:text-fg"
              >
                <X className="size-3.5" {...ICON} />
              </button>
            </span>
          )}
        </div>
      )}

      {runs.isError && runs.data === undefined ? (
        <ErrorNote error={runs.error} />
      ) : (
        <Panel className="overflow-hidden">
          {shown === undefined ? (
            <div className="divide-y divide-border">
              <div className="h-11 bg-surface-2/60" />
              {Array.from({ length: 4 }, (_, i) => (
                <div key={i} className="flex h-[60px] items-center gap-6 px-4">
                  <Skeleton className="size-6 rounded-[7px]" />
                  <Skeleton className="h-3 w-24" />
                  <Skeleton className="h-3 w-[30%]" />
                  <Skeleton className="ml-auto h-2.5 w-36" />
                </div>
              ))}
            </div>
          ) : shown.length === 0 ? (
            <EmptyState
              icon={<Activity {...ICON} />}
              title={runs.data?.length === 0 ? "No runs yet" : "No runs match"}
              body={runs.data?.length === 0 ? "Pick a ready trace and press Run to simulate it on its suggested GPU config." : "Try another state or clear the trace filter."}
              action={
                runs.data?.length === 0 ? (
                  <Link to={paths.traces} className={buttonClass({})}>
                    Browse traces
                  </Link>
                ) : undefined
              }
            />
          ) : (
            <RunsTable runs={shown} traces={traces.data} showTrace={traceId === null} />
          )}
        </Panel>
      )}
    </>
  );
}
