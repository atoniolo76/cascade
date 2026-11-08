/**
 * The whole onboarding flow lives in this one state. The form on the left dispatches actions and the
 * preview on the right is a pure render of the state, which is what keeps the two in sync.
 */

export type Step = 1 | 2 | 3 | 4 | "done";
export const STEPS = [1, 2, 3, 4] as const;

export type SourceId = "modal" | "vllm" | "aiperf" | "openai" | "sample" | "upload";

export interface SourceDef {
  id: SourceId;
  name: string;
  /** The status line once connected, before the detail: "Connected" or "Loaded". */
  done: string;
  /** What the activity row says: "You connected Modal". */
  verb: string;
  /** The name as it reads inside an activity row. */
  subject: string;
  /** Shown after "Connected ·". Upload replaces it with the file name. */
  detail: string;
  /** The short version for the activity row, when it differs from `detail`. */
  activityDetail?: string;
}

export const SOURCES: readonly SourceDef[] = [
  { id: "modal", name: "Modal", done: "Connected", verb: "connected", subject: "Modal", detail: "cascade-demo" },
  { id: "vllm", name: "vLLM endpoint", done: "Connected", verb: "connected", subject: "vLLM", detail: "localhost:8000" },
  { id: "aiperf", name: "AIPerf log", done: "Connected", verb: "imported", subject: "AIPerf log", detail: "aiperf_a100x4_c10.json" },
  { id: "openai", name: "OpenAI-compatible URL", done: "Connected", verb: "connected", subject: "OpenAI-compatible URL", detail: "api.openai.com/v1" },
  {
    id: "sample",
    name: "Sample trace",
    done: "Loaded",
    verb: "loaded",
    subject: "sample trace",
    detail: "1,200 requests · 94.2 s",
    activityDetail: "1,200 requests",
  },
  { id: "upload", name: "Upload JSON/Parquet", done: "Uploaded", verb: "uploaded", subject: "a trace", detail: "" },
];

const BY_ID = Object.fromEntries(SOURCES.map((s) => [s.id, s])) as Record<SourceId, SourceDef>;
export const sourceDef = (id: SourceId): SourceDef => BY_ID[id];

export interface Source {
  id: SourceId;
  /** Connecting renders faded in the preview until it is confirmed. */
  status: "connecting" | "connected";
  detail: string;
}

export type GpuId = "a100" | "h100" | "l40s" | "h200" | "tpu-v6e";
export interface FleetEntry {
  gpu: GpuId;
  count: number;
}

export type RouterId = "round-robin" | "least-loaded" | "kv-aware";

export interface Engine {
  maxRequests: number;
  batchSize: number;
  tensorParallel: number;
}

export interface ReplayResult {
  p99TtftMs: number;
  hotspots: { gpu: number; cell: number }[];
  twinErrorPct: number;
}

export interface Replay {
  status: "idle" | "running" | "done";
  /** 0..1 */
  progress: number;
  result: ReplayResult | null;
}

export interface OnboardingState {
  step: Step;
  /** Which way the last step change went, so the form slides the right way. */
  direction: 1 | -1;
  workspace: string;
  /** In the order the user connected them; the activity timeline reads this order. */
  sources: Source[];
  fleet: FleetEntry[];
  router: RouterId | null;
  engine: Engine;
  replay: Replay;
}

export const initialState: OnboardingState = {
  step: 1,
  direction: 1,
  workspace: "cascade-demo",
  sources: [],
  fleet: [],
  router: null,
  engine: { maxRequests: 64, batchSize: 32, tensorParallel: 1 },
  replay: { status: "idle", progress: 0, result: null },
};

export type Action =
  | { type: "connect"; id: SourceId; detail?: string | undefined }
  | { type: "confirm"; id: SourceId }
  | { type: "disconnect"; id: SourceId }
  | { type: "next" }
  | { type: "back" };

const ORDER: Step[] = [1, 2, 3, 4, "done"];

export function reducer(state: OnboardingState, action: Action): OnboardingState {
  switch (action.type) {
    case "connect": {
      if (state.sources.some((s) => s.id === action.id)) return state;
      const detail = action.detail ?? sourceDef(action.id).detail;
      return { ...state, sources: [...state.sources, { id: action.id, status: "connecting", detail }] };
    }
    case "confirm":
      // A source disconnected while it was still connecting stays gone.
      return {
        ...state,
        sources: state.sources.map((s) => (s.id === action.id ? { ...s, status: "connected" } : s)),
      };
    case "disconnect":
      return { ...state, sources: state.sources.filter((s) => s.id !== action.id) };
    case "next": {
      if (!canAdvance(state)) return state;
      const i = ORDER.indexOf(state.step);
      return { ...state, step: ORDER[i + 1] ?? state.step, direction: 1 };
    }
    case "back": {
      const i = ORDER.indexOf(state.step);
      if (i <= 0 || state.step === "done") return state;
      return { ...state, step: ORDER[i - 1] ?? state.step, direction: -1 };
    }
  }
}

export function canAdvance(state: OnboardingState): boolean {
  switch (state.step) {
    case 1:
      return state.sources.some((s) => s.status === "connected");
    case 4:
      return state.replay.status === "done";
    case "done":
      return false;
    default:
      return true;
  }
}

export interface ActivityRow {
  key: string;
  icon: { kind: "workspace" } | { kind: "source"; id: SourceId };
  verb: string;
  subject: string;
  detail: string | null;
  pending: boolean;
}

/** The timeline in the preview, derived from the state rather than kept beside it. */
export function activity(state: OnboardingState): ActivityRow[] {
  const rows: ActivityRow[] = [
    { key: "workspace", icon: { kind: "workspace" }, verb: "created", subject: state.workspace, detail: null, pending: false },
  ];
  for (const s of state.sources) {
    const def = sourceDef(s.id);
    const upload = s.id === "upload";
    rows.push({
      key: `source-${s.id}`,
      icon: { kind: "source", id: s.id },
      verb: def.verb,
      subject: upload ? s.detail : def.subject,
      detail: upload ? null : (def.activityDetail ?? s.detail),
      pending: s.status === "connecting",
    });
  }
  return rows;
}

/** The one-line stand-in for the preview on narrow screens. */
export function summary(state: OnboardingState): string {
  const n = state.sources.filter((s) => s.status === "connected").length;
  const parts = [state.workspace, n === 0 ? "no traces yet" : `${n} trace source${n === 1 ? "" : "s"}`];
  const gpus = state.fleet.reduce((sum, f) => sum + f.count, 0);
  if (gpus > 0) parts.push(`${gpus} GPU${gpus === 1 ? "" : "s"}`);
  if (state.router !== null) parts.push(state.router);
  return parts.join(" · ");
}
