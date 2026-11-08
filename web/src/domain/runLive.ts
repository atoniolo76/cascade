import type { KernelStat, RunDetail, RunEvent, SimSummary } from "../api/types";

const LOG_KEEP = 400;

/** `received` counts lines that arrived as live `log` events; it never resets, so it orders live lines across snapshots. */
export interface SimLog {
  lines: string[];
  received: number;
}

export type KernelsBySim = Record<string, Record<number, KernelStat>>;

export type RunLive =
  | { kind: "waiting" }
  | { kind: "live"; run: RunDetail; logs: Record<string, SimLog>; kernels: KernelsBySim };

export type RunAction = RunEvent | { type: "run"; run: RunDetail };

export const initialRunLive: RunLive = { kind: "waiting" };

function upsertSim(sims: SimSummary[], sim: SimSummary): SimSummary[] {
  const i = sims.findIndex((s) => s.id === sim.id);
  if (i === -1) return [...sims, sim];
  const next = sims.slice();
  next[i] = sim;
  return next;
}

function mergeKernels(kernels: KernelsBySim, stats: readonly KernelStat[]): KernelsBySim {
  if (stats.length === 0) return kernels;
  const next = { ...kernels };
  for (const stat of stats) next[stat.sim_id] = { ...next[stat.sim_id], [stat.kernel_index]: stat };
  return next;
}

function logsFromSnapshot(run: RunDetail, prev: Record<string, SimLog>): Record<string, SimLog> {
  const logs: Record<string, SimLog> = {};
  for (const sim of run.sims) logs[sim.id] = { lines: sim.log_tail.slice(-LOG_KEEP), received: prev[sim.id]?.received ?? 0 };
  return logs;
}

export function runLiveReducer(state: RunLive, action: RunAction): RunLive {
  if (action.type === "snapshot") {
    const prevLogs = state.kind === "live" ? state.logs : {};
    return { kind: "live", run: action.run, logs: logsFromSnapshot(action.run, prevLogs), kernels: mergeKernels({}, action.kernels) };
  }
  if (state.kind === "waiting") return state;
  switch (action.type) {
    case "run":
      return { ...state, run: action.run };
    case "sim": {
      const known = action.sim.id in state.logs;
      return {
        ...state,
        run: { ...state.run, sims: upsertSim(state.run.sims, action.sim) },
        logs: known ? state.logs : { ...state.logs, [action.sim.id]: { lines: action.sim.log_tail, received: 0 } },
      };
    }
    case "log": {
      const prev = state.logs[action.sim_id] ?? { lines: [], received: 0 };
      const lines = prev.lines.concat(action.lines).slice(-LOG_KEEP);
      return { ...state, logs: { ...state.logs, [action.sim_id]: { lines, received: prev.received + action.lines.length } } };
    }
    case "kernel":
      return { ...state, kernels: mergeKernels(state.kernels, [action.stat]) };
    default: {
      const exhaustive: never = action;
      return exhaustive;
    }
  }
}

/** Live lines that arrived after `since` (a value of SimLog.received), as far back as the kept window reaches. */
export function linesReceivedSince(log: SimLog | undefined, since: number): string[] {
  if (!log) return [];
  const n = Math.min(log.received - since, log.lines.length);
  return n > 0 ? log.lines.slice(log.lines.length - n) : [];
}
