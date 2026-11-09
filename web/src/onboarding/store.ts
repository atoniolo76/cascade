/**
 * The whole onboarding flow lives in this one state. The form on the left dispatches actions and the
 * preview on the right is a pure render of the state, which is what keeps the two in sync.
 *
 * Everything here is what Cascade really does: the two OPT-125m samples every workspace opens with,
 * the Accel-Sim configs they have recorded results for, and the numbers from their READMEs under
 * samples/. The run is scripted (the samples replay recorded results, so the shape is the same).
 */

export type Step = 1 | 2 | 3 | "done";
export const STEPS = [1, 2, 3] as const;

/* ---------- Step 1: trace ---------- */

export type TraceId = "opt125m-h100" | "opt125m-a100";

/** What the run reports once it finishes, from the sample's README. */
export interface Recorded {
  realE2eMs: number;
  simE2eMs: number;
}

export interface TraceDef {
  id: TraceId;
  name: string;
  /** The GPU it was profiled on. */
  gpu: string;
  kernels: number;
  launches: number;
  /** The configs this sample has recorded Accel-Sim results for, so a run on it can replay them. */
  configs: readonly ConfigId[];
  recorded: Recorded;
}

export const TRACES: readonly TraceDef[] = [
  {
    id: "opt125m-h100",
    name: "OPT-125m on H100",
    gpu: "H100 80GB HBM3",
    kernels: 712,
    launches: 17088,
    configs: ["SM90_H100", "SM90_H200"],
    recorded: { realE2eMs: 253.44, simE2eMs: 261.11 },
  },
  {
    id: "opt125m-a100",
    name: "OPT-125m on A100",
    gpu: "A100",
    kernels: 624,
    launches: 14100,
    configs: ["SM80_A100"],
    recorded: { realE2eMs: 222.08, simE2eMs: 229.58 },
  },
];

const TRACE_BY_ID = Object.fromEntries(TRACES.map((t) => [t.id, t])) as Record<TraceId, TraceDef>;
export const traceDef = (id: TraceId): TraceDef => TRACE_BY_ID[id];

/* ---------- Step 2: GPU configs ---------- */

export type ConfigId = "SM90_H100" | "SM90_H200" | "SM80_A100";

export interface ConfigDef {
  id: ConfigId;
  gpu: string;
  arch: string;
  clockMhz: number;
  /** H200 has kernel times only; the sample has no end-to-end comparison for it. */
  e2e: boolean;
}

export const CONFIGS: readonly ConfigDef[] = [
  { id: "SM90_H100", gpu: "H100", arch: "sm_90", clockMhz: 1980, e2e: true },
  { id: "SM90_H200", gpu: "H200", arch: "sm_90", clockMhz: 1980, e2e: false },
  { id: "SM80_A100", gpu: "A100", arch: "sm_80", clockMhz: 1410, e2e: true },
];

const CONFIG_BY_ID = Object.fromEntries(CONFIGS.map((c) => [c.id, c])) as Record<ConfigId, ConfigDef>;
export const configDef = (id: ConfigId): ConfigDef => CONFIG_BY_ID[id];

/** The API caps each config's replicas at 64. */
export const MAX_REPLICAS = 64;

export interface Entry {
  config: ConfigId;
  count: number;
}

export const simCount = (entries: readonly Entry[]): number => entries.reduce((n, e) => n + e.count, 0);

/** "2× SM90_H100, 1× SM90_H200" */
export const entriesLabel = (entries: readonly Entry[]): string => entries.map((e) => `${e.count}× ${e.config}`).join(", ");

/* ---------- Step 3: the run ---------- */

export interface Run {
  status: "idle" | "running" | "done";
  /** 0..1 */
  progress: number;
}

/** The scripted run's step; with a 100 ms tick it finishes in 3 s. */
export const RUN_STEP = 1 / 30;

const IDLE_RUN: Run = { status: "idle", progress: 0 };

/** Forking the sandboxes takes the first stretch of the run, then the sims finish one by one. */
const FORKED_AT = 0.25;
const SIMS_DONE_AT = 0.85;

export type SimState = "forking" | "running" | "done";

/** Each simulation's state at this point in the run, in launch order. */
export function simStates(sims: number, progress: number): SimState[] {
  return Array.from({ length: sims }, (_, i) => {
    if (progress < FORKED_AT) return "forking";
    const finishesAt = FORKED_AT + ((SIMS_DONE_AT - FORKED_AT) * (i + 1)) / sims;
    return progress >= finishesAt ? "done" : "running";
  });
}

/** The run's sub-line: what Cascade is doing at this point. */
export function runPhase(sims: number, progress: number): string {
  if (progress < FORKED_AT) return `Forking ${sims} Sail sandbox${sims === 1 ? "" : "es"} from the workspace checkpoint`;
  if (progress < SIMS_DONE_AT) {
    const done = simStates(sims, progress).filter((s) => s === "done").length;
    return `Replaying recorded Accel-Sim results · ${done} of ${sims} done`;
  }
  return "Comparing to the real run";
}

/** Signed percent error of simulated against real, one decimal: "+3.0%". */
export function errorPct(r: Recorded): string {
  const pct = ((r.simE2eMs - r.realE2eMs) / r.realE2eMs) * 100;
  return `${pct >= 0 ? "+" : "−"}${Math.abs(pct).toFixed(1)}%`;
}

/* ---------- State ---------- */

export interface OnboardingState {
  step: Step;
  /** Which way the last step change went, so the form slides the right way. */
  direction: 1 | -1;
  workspace: string;
  trace: TraceId | null;
  /** In the order the user added them. */
  entries: Entry[];
  run: Run;
}

export const initialState: OnboardingState = {
  step: 1,
  direction: 1,
  workspace: "cascade-demo",
  trace: null,
  entries: [],
  run: IDLE_RUN,
};

export type Action =
  | { type: "chooseTrace"; trace: TraceId }
  | { type: "setCount"; config: ConfigId; count: number }
  | { type: "tick" }
  | { type: "next" }
  | { type: "back" };

const ORDER: Step[] = [1, 2, 3, "done"];

export function reducer(state: OnboardingState, action: Action): OnboardingState {
  switch (action.type) {
    case "chooseTrace":
      if (state.trace === action.trace) return state;
      // Configs belong to the trace's recorded results, so a new trace starts its GPU list over.
      return { ...state, trace: action.trace, entries: [] };
    case "setCount": {
      if (state.trace === null || !traceDef(state.trace).configs.includes(action.config)) return state;
      const count = Math.max(0, Math.min(MAX_REPLICAS, Math.round(action.count)));
      const has = state.entries.some((e) => e.config === action.config);
      const entries =
        count === 0
          ? state.entries.filter((e) => e.config !== action.config)
          : has
            ? state.entries.map((e) => (e.config === action.config ? { ...e, count } : e))
            : [...state.entries, { config: action.config, count }];
      return { ...state, entries };
    }
    case "tick": {
      if (state.run.status !== "running") return state;
      const progress = Math.min(1, state.run.progress + RUN_STEP);
      if (progress < 1 - 1e-9) return { ...state, run: { ...state.run, progress } };
      // A finished run lands on the done state by itself.
      return { ...state, run: { status: "done", progress: 1 }, step: "done", direction: 1 };
    }
    case "next": {
      if (!canAdvance(state)) return state;
      const step = ORDER[ORDER.indexOf(state.step) + 1] ?? state.step;
      // Reaching step 3 launches the run.
      const run: Run = step === 3 ? { status: "running", progress: 0 } : state.run;
      return { ...state, step, direction: 1, run };
    }
    case "back": {
      const i = ORDER.indexOf(state.step);
      if (i <= 0 || state.step === "done") return state;
      // Leaving step 3 drops the run; it relaunches with whatever the user changes.
      return { ...state, step: ORDER[i - 1] ?? state.step, direction: -1, run: IDLE_RUN };
    }
  }
}

export function canAdvance(state: OnboardingState): boolean {
  switch (state.step) {
    case 1:
      return state.trace !== null;
    case 2:
      return simCount(state.entries) > 0;
    case 3:
      return state.run.status === "done";
    case "done":
      return false;
  }
}

/* ---------- Derived views ---------- */

export type ActivityIcon = { kind: "workspace" } | { kind: "trace" } | { kind: "config" } | { kind: "run" };

export interface ActivityRow {
  key: string;
  icon: ActivityIcon;
  /** Text before the bold subject; empty when the subject leads. */
  lead: string;
  subject: string;
  /** Text after the subject, before the detail: " finished". */
  tail?: string | undefined;
  detail: string | null;
  pending: boolean;
}

const n = (x: number): string => x.toLocaleString("en-US");

/** The timeline in the preview, derived from the state rather than kept beside it. */
export function activity(state: OnboardingState): ActivityRow[] {
  const rows: ActivityRow[] = [
    { key: "workspace", icon: { kind: "workspace" }, lead: "You created", subject: state.workspace, detail: null, pending: false },
  ];
  if (state.trace !== null) {
    const t = traceDef(state.trace);
    rows.push({ key: `trace-${t.id}`, icon: { kind: "trace" }, lead: "You opened", subject: t.name, detail: `${n(t.kernels)} kernels`, pending: false });
  }
  for (const e of state.entries) {
    rows.push({ key: `config-${e.config}`, icon: { kind: "config" }, lead: "You added", subject: `${e.count}× ${e.config}`, detail: `${configDef(e.config).clockMhz} MHz`, pending: false });
  }
  if (state.run.status !== "idle" && state.trace !== null) {
    const sims = simCount(state.entries);
    const done = state.run.status === "done";
    rows.push({
      key: "run",
      icon: { kind: "run" },
      lead: done ? "" : "You launched",
      subject: done ? "Run" : "a run",
      tail: done ? " finished" : undefined,
      detail: done ? finishLine(state) : `${sims} simulation${sims === 1 ? "" : "s"}`,
      pending: !done,
    });
  }
  return rows;
}

/** What the finished run says, from the recorded results of the configs the user picked. */
function finishLine(state: OnboardingState): string {
  if (state.trace === null) return "";
  const t = traceDef(state.trace);
  return state.entries.some((e) => configDef(e.config).e2e) ? `${errorPct(t.recorded)} e2e vs real` : "kernel times only";
}

/** The sidebar footer: who is here, then what the workspace holds. */
export function footer(state: OnboardingState): string {
  if (state.run.status === "done") return "1 run";
  const sims = simCount(state.entries);
  return sims === 0 ? "Just you" : `${sims} simulation${sims === 1 ? "" : "s"} queued`;
}

export interface CheckItem {
  key: string;
  label: string;
  detail: string;
  state: "done" | "active" | "queued";
}

/** Step 3's live checklist. */
export function checklist(state: OnboardingState): CheckItem[] {
  if (state.trace === null) return [];
  const t = traceDef(state.trace);
  const sims = simCount(state.entries);
  const items: CheckItem[] = [
    { key: "trace", label: "Trace loaded", detail: `${t.name} · ${n(t.kernels)} kernels`, state: "done" },
    { key: "configs", label: "GPUs picked", detail: entriesLabel(state.entries), state: "done" },
  ];
  if (state.run.status === "done") {
    items.push({ key: "run", label: "Run finished", detail: resultLine(state), state: "done" });
  } else {
    items.push({ key: "run", label: "Simulating", detail: runPhase(sims, state.run.progress), state: state.run.status === "running" ? "active" : "queued" });
  }
  return items;
}

/** "sim e2e 261.11 ms vs 253.44 ms real · +3.0%", or kernel times when only H200 ran. */
export function resultLine(state: OnboardingState): string {
  if (state.trace === null) return "";
  const t = traceDef(state.trace);
  if (!state.entries.some((e) => configDef(e.config).e2e)) return "Per-kernel times ready · no end-to-end comparison for H200";
  const r = t.recorded;
  return `sim e2e ${r.simE2eMs.toFixed(2)} ms vs ${r.realE2eMs.toFixed(2)} ms real · ${errorPct(r)}`;
}

/** The one-line stand-in for the preview on narrow screens. */
export function summary(state: OnboardingState): string {
  const parts = [state.workspace, state.trace === null ? "no trace yet" : traceDef(state.trace).name];
  const sims = simCount(state.entries);
  if (sims > 0) parts.push(`${sims} sim${sims === 1 ? "" : "s"}`);
  return parts.join(" · ");
}
