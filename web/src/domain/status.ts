import type { RunState, SimState, TraceState } from "../api/types";

export type Tone = "pending" | "active" | "lost" | "ok" | "bad" | "off";

interface StateMeta {
  label: string;
  tone: Tone;
  live: boolean;
}

interface SimStateMeta extends StateMeta {
  terminal: boolean;
}

export const SIM_STATE: Record<SimState, SimStateMeta> = {
  pending: { label: "Pending", tone: "pending", live: false, terminal: false },
  launching: { label: "Launching", tone: "active", live: true, terminal: false },
  running: { label: "Running", tone: "active", live: true, terminal: false },
  lost: { label: "Lost", tone: "lost", live: false, terminal: false },
  succeeded: { label: "Succeeded", tone: "ok", live: false, terminal: true },
  failed: { label: "Failed", tone: "bad", live: false, terminal: true },
  cancelled: { label: "Cancelled", tone: "off", live: false, terminal: true },
};

export const SIM_STATE_ORDER: readonly SimState[] = [
  "running",
  "launching",
  "lost",
  "pending",
  "succeeded",
  "failed",
  "cancelled",
];

export const TRACE_STATE: Record<TraceState, StateMeta> = {
  uploading: { label: "Uploading", tone: "active", live: true },
  processing: { label: "Processing", tone: "active", live: true },
  ready: { label: "Ready", tone: "ok", live: false },
  failed: { label: "Failed", tone: "bad", live: false },
};

export const RUN_STATE: Record<RunState, StateMeta> = {
  active: { label: "Active", tone: "active", live: true },
  succeeded: { label: "Succeeded", tone: "ok", live: false },
  partial: { label: "Partial", tone: "lost", live: false },
  failed: { label: "Failed", tone: "bad", live: false },
  cancelled: { label: "Cancelled", tone: "off", live: false },
};

export const TONE_VARS: Record<Tone, { ink: string; bg: string; dot: string }> = {
  pending: { ink: "var(--st-pending-ink)", bg: "var(--st-pending-bg)", dot: "var(--st-pending-dot)" },
  active: { ink: "var(--st-active-ink)", bg: "var(--st-active-bg)", dot: "var(--st-active-dot)" },
  lost: { ink: "var(--st-lost-ink)", bg: "var(--st-lost-bg)", dot: "var(--st-lost-dot)" },
  ok: { ink: "var(--st-ok-ink)", bg: "var(--st-ok-bg)", dot: "var(--st-ok-dot)" },
  bad: { ink: "var(--st-bad-ink)", bg: "var(--st-bad-bg)", dot: "var(--st-bad-dot)" },
  off: { ink: "var(--st-off-ink)", bg: "var(--st-off-bg)", dot: "var(--st-off-dot)" },
};

/** Mirror of run_state() in cascade/models.py. */
export function deriveRunState(states: readonly SimState[]): RunState {
  if (states.length === 0 || states.some((s) => !SIM_STATE[s].terminal)) return "active";
  if (states.every((s) => s === "succeeded")) return "succeeded";
  if (states.every((s) => s === "failed")) return "failed";
  if (states.includes("cancelled") && !states.includes("succeeded")) return "cancelled";
  return "partial";
}

export function countSimStates(states: readonly SimState[]): Partial<Record<SimState, number>> {
  const counts: Partial<Record<SimState, number>> = {};
  for (const s of states) counts[s] = (counts[s] ?? 0) + 1;
  return counts;
}
