/**
 * The whole onboarding flow lives in this one state. The form on the left dispatches actions and the
 * preview on the right is a pure render of the state, which is what keeps the two in sync.
 */

export type Step = 1 | 2 | 3 | 4 | "done";
export const STEPS = [1, 2, 3, 4] as const;

/* ---------- Step 1: trace sources ---------- */

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

const SOURCE_BY_ID = Object.fromEntries(SOURCES.map((s) => [s.id, s])) as Record<SourceId, SourceDef>;
export const sourceDef = (id: SourceId): SourceDef => SOURCE_BY_ID[id];

export interface Source {
  id: SourceId;
  /** Connecting renders faded in the preview until it is confirmed. */
  status: "connecting" | "connected";
  detail: string;
}

/** The replay runs against the sample trace for now, whatever the user connected. */
export const TRACE_REQUESTS = 1200;

/* ---------- Step 2: fleet ---------- */

export type GpuId = "a100" | "h100" | "l40s" | "h200" | "tpu-v6e";

export interface GpuDef {
  id: GpuId;
  name: string;
  /** The short name used in counts: "2× H100". */
  short: string;
  memory: string;
  /** Measured profiles come from real runs; estimated ones are extrapolated and drawn hatched. */
  profile: "measured" | "estimated";
  vendor: "nvidia" | "google";
}

export const GPUS: readonly GpuDef[] = [
  { id: "a100", name: "A100 80 GB", short: "A100", memory: "80 GB HBM2e", profile: "measured", vendor: "nvidia" },
  { id: "h100", name: "H100 SXM", short: "H100", memory: "80 GB HBM3", profile: "measured", vendor: "nvidia" },
  { id: "l40s", name: "L40S 48 GB", short: "L40S", memory: "48 GB GDDR6", profile: "estimated", vendor: "nvidia" },
  { id: "h200", name: "H200", short: "H200", memory: "141 GB HBM3e", profile: "estimated", vendor: "nvidia" },
  { id: "tpu-v6e", name: "TPU v6e", short: "TPU v6e", memory: "32 GB HBM", profile: "estimated", vendor: "google" },
];

const GPU_BY_ID = Object.fromEntries(GPUS.map((g) => [g.id, g])) as Record<GpuId, GpuDef>;
export const gpuDef = (id: GpuId): GpuDef => GPU_BY_ID[id];

export const MAX_PER_GPU = 8;

export interface FleetEntry {
  gpu: GpuId;
  count: number;
}

export const fleetSize = (fleet: readonly FleetEntry[]): number => fleet.reduce((n, f) => n + f.count, 0);

/** "2× H100, 2× L40S" */
export const fleetLabel = (fleet: readonly FleetEntry[]): string => fleet.map((f) => `${f.count}× ${gpuDef(f.gpu).short}`).join(", ");

/* ---------- Step 3: router and engine ---------- */

export type RouterId = "round-robin" | "least-loaded" | "kv-aware";

export interface RouterDef {
  id: RouterId;
  name: string;
  blurb: string;
  /** The GPU types this router suits, shown as a strip against the user's fleet. */
  suits: readonly GpuId[];
}

export const ROUTERS: readonly RouterDef[] = [
  { id: "round-robin", name: "Round-robin", blurb: "Simple and fair. Ignores load.", suits: ["a100", "h100", "h200"] },
  { id: "least-loaded", name: "Least-loaded", blurb: "Sends each request to the shortest queue.", suits: ["a100", "h100", "l40s", "h200", "tpu-v6e"] },
  { id: "kv-aware", name: "KV-aware", blurb: "Prefers the GPU that already holds the prompt's cache.", suits: ["a100", "h100", "h200"] },
];

const ROUTER_BY_ID = Object.fromEntries(ROUTERS.map((r) => [r.id, r])) as Record<RouterId, RouterDef>;
export const routerDef = (id: RouterId): RouterDef => ROUTER_BY_ID[id];

export interface Engine {
  maxRequests: number;
  batchSize: number;
  tensorParallel: number;
}

export const DEFAULT_ENGINE: Engine = { maxRequests: 64, batchSize: 32, tensorParallel: 1 };

const isDefaultEngine = (e: Engine): boolean =>
  e.maxRequests === DEFAULT_ENGINE.maxRequests && e.batchSize === DEFAULT_ENGINE.batchSize && e.tensorParallel === DEFAULT_ENGINE.tensorParallel;

/* ---------- Step 4: replay ---------- */

/** Cells per lane in the preview heatmap. */
export const LANE_CELLS = 24;

export interface Hotspot {
  gpu: number;
  cell: number;
}

export interface ReplayResult {
  p99TtftMs: number;
  hotspots: Hotspot[];
  twinErrorPct: number;
  /** p99 change against an all-A100 fleet of the same size, in percent. */
  vsBaselinePct: number;
}

export interface Replay {
  status: "idle" | "running" | "done";
  /** 0..1 */
  progress: number;
  result: ReplayResult | null;
}

/** The scripted replay's step; with a 100 ms tick it finishes in 2.5 s. */
export const REPLAY_STEP = 0.04;

const IDLE_REPLAY: Replay = { status: "idle", progress: 0, result: null };

/** Scripted against the sample trace until the real twin is wired in behind `replay.status`. */
export function scriptedResult(fleet: readonly FleetEntry[]): ReplayResult {
  const lanes = Math.max(1, fleetSize(fleet));
  const spots: Hotspot[] = [
    { gpu: Math.min(2, lanes - 1), cell: 17 },
    { gpu: Math.min(1, lanes - 1), cell: 9 },
  ];
  const hotspots = spots.filter((h, i) => spots.findIndex((o) => o.gpu === h.gpu) === i);
  return { p99TtftMs: 1105, hotspots, twinErrorPct: 6.8, vsBaselinePct: -18 };
}

/** The replay's sub-line: what the twin is doing at this point in the run. */
export function replayPhase(progress: number): string {
  if (progress < 0.6) {
    const request = Math.max(1, Math.round((progress / 0.6) * TRACE_REQUESTS));
    const seconds = Math.round((progress / 0.6) * 94);
    const clock = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    return `Replaying request ${request.toLocaleString("en-US")} of ${TRACE_REQUESTS.toLocaleString("en-US")} · ${clock}`;
  }
  return progress < 0.85 ? "Detecting hotspots" : "Comparing to the real run";
}

/* ---------- State ---------- */

export interface OnboardingState {
  step: Step;
  /** Which way the last step change went, so the form slides the right way. */
  direction: 1 | -1;
  workspace: string;
  /** In the order the user connected them; the activity timeline reads this order. */
  sources: Source[];
  /** In the order the user added them. */
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
  engine: DEFAULT_ENGINE,
  replay: IDLE_REPLAY,
};

export type Action =
  | { type: "connect"; id: SourceId; detail?: string | undefined }
  | { type: "confirm"; id: SourceId }
  | { type: "disconnect"; id: SourceId }
  | { type: "setCount"; gpu: GpuId; count: number }
  | { type: "chooseRouter"; router: RouterId }
  | { type: "setEngine"; engine: Partial<Engine> }
  | { type: "tick" }
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
      return { ...state, sources: state.sources.map((s) => (s.id === action.id ? { ...s, status: "connected" } : s)) };
    case "disconnect":
      return { ...state, sources: state.sources.filter((s) => s.id !== action.id) };
    case "setCount": {
      const count = Math.max(0, Math.min(MAX_PER_GPU, Math.round(action.count)));
      const has = state.fleet.some((f) => f.gpu === action.gpu);
      const fleet =
        count === 0
          ? state.fleet.filter((f) => f.gpu !== action.gpu)
          : has
            ? state.fleet.map((f) => (f.gpu === action.gpu ? { ...f, count } : f))
            : [...state.fleet, { gpu: action.gpu, count }];
      return { ...state, fleet };
    }
    case "chooseRouter":
      return { ...state, router: action.router };
    case "setEngine":
      return { ...state, engine: { ...state.engine, ...action.engine } };
    case "tick": {
      if (state.replay.status !== "running") return state;
      const progress = Math.min(1, state.replay.progress + REPLAY_STEP);
      return {
        ...state,
        replay: progress < 1 ? { ...state.replay, progress } : { status: "done", progress: 1, result: scriptedResult(state.fleet) },
        // A finished replay lands on the done state by itself, the way the reference does.
        ...(progress < 1 ? {} : { step: "done" as const, direction: 1 as const }),
      };
    }
    case "next": {
      if (!canAdvance(state)) return state;
      const step = ORDER[ORDER.indexOf(state.step) + 1] ?? state.step;
      // Reaching step 4 starts the replay.
      const replay: Replay = step === 4 ? { status: "running", progress: 0, result: null } : state.replay;
      return { ...state, step, direction: 1, replay };
    }
    case "back": {
      const i = ORDER.indexOf(state.step);
      if (i <= 0 || state.step === "done") return state;
      // Leaving step 4 drops the replay; it reruns on the fleet and router the user goes back to change.
      return { ...state, step: ORDER[i - 1] ?? state.step, direction: -1, replay: IDLE_REPLAY };
    }
  }
}

export function canAdvance(state: OnboardingState): boolean {
  switch (state.step) {
    case 1:
      return state.sources.some((s) => s.status === "connected");
    case 2:
      return fleetSize(state.fleet) > 0;
    case 3:
      return state.router !== null;
    case 4:
      return state.replay.status === "done";
    case "done":
      return false;
  }
}

/* ---------- Derived views ---------- */

export type ActivityIcon =
  | { kind: "workspace" }
  | { kind: "source"; id: SourceId }
  | { kind: "gpu"; id: GpuId }
  | { kind: "router" }
  | { kind: "replay" };

export interface ActivityRow {
  key: string;
  icon: ActivityIcon;
  /** Text before the bold subject; "You connected" or empty when the subject leads. */
  lead: string;
  subject: string;
  /** Text after the subject, before the detail: " finished". */
  tail?: string;
  detail: string | null;
  pending: boolean;
}

/** The timeline in the preview, derived from the state rather than kept beside it. */
export function activity(state: OnboardingState): ActivityRow[] {
  const rows: ActivityRow[] = [
    { key: "workspace", icon: { kind: "workspace" }, lead: "You created", subject: state.workspace, detail: null, pending: false },
  ];
  for (const s of state.sources) {
    const def = sourceDef(s.id);
    const upload = s.id === "upload";
    rows.push({
      key: `source-${s.id}`,
      icon: { kind: "source", id: s.id },
      lead: `You ${def.verb}`,
      subject: upload ? s.detail : def.subject,
      detail: upload ? null : (def.activityDetail ?? s.detail),
      pending: s.status === "connecting",
    });
  }
  for (const f of state.fleet) {
    const g = gpuDef(f.gpu);
    rows.push({ key: `gpu-${f.gpu}`, icon: { kind: "gpu", id: f.gpu }, lead: "You added", subject: `${f.count}× ${g.short}`, detail: g.profile, pending: false });
  }
  if (state.router !== null) {
    rows.push({
      key: "router",
      icon: { kind: "router" },
      lead: "You chose",
      subject: routerDef(state.router).name.toLowerCase(),
      detail: isDefaultEngine(state.engine) ? "engine defaults" : "custom engine",
      pending: false,
    });
  }
  if (state.replay.status !== "idle") {
    const result = state.replay.result;
    rows.push({
      key: "replay",
      icon: { kind: "replay" },
      lead: "",
      subject: "Replay",
      tail: result === null ? " running" : " finished",
      detail: result === null ? null : hotspotLine(result),
      pending: result === null,
    });
  }
  return rows;
}

function hotspotLine(result: ReplayResult): string {
  const where = result.hotspots.map((h) => `gpu${h.gpu}`);
  const n = result.hotspots.length;
  return `${n} hotspot${n === 1 ? "" : "s"} on ${where.join(" and ")}`;
}

/** The sidebar footer: who is here, then what the workspace holds. */
export function footer(state: OnboardingState): string {
  if (state.replay.status === "done") return "1 replay";
  const n = fleetSize(state.fleet);
  return n === 0 ? "Just you" : `${n} GPU${n === 1 ? "" : "s"}`;
}

export interface CheckItem {
  key: string;
  label: string;
  detail: string;
  state: "done" | "active" | "queued";
}

/** Step 4's live checklist. */
export function checklist(state: OnboardingState): CheckItem[] {
  const { replay } = state;
  const done = replay.status === "done";
  const items: CheckItem[] = [
    { key: "trace", label: "Trace loaded", detail: `${TRACE_REQUESTS.toLocaleString("en-US")} requests`, state: "done" },
    { key: "fleet", label: "Fleet built", detail: fleetLabel(state.fleet), state: "done" },
    { key: "router", label: "Router set", detail: state.router === null ? "" : routerDef(state.router).name.toLowerCase(), state: "done" },
  ];
  if (done && replay.result !== null) {
    const r = replay.result;
    const n = r.hotspots.length;
    items.push({
      key: "replay",
      label: "Replay done",
      detail: `p99 TTFT ${r.p99TtftMs.toLocaleString("en-US")} ms · ${n} hotspot${n === 1 ? "" : "s"} · twin error ±${r.twinErrorPct}%`,
      state: "done",
    });
  } else {
    items.push({ key: "replay", label: "Replaying", detail: replayPhase(replay.progress), state: replay.status === "running" ? "active" : "queued" });
  }
  return items;
}

/** The one-line stand-in for the preview on narrow screens. */
export function summary(state: OnboardingState): string {
  const n = state.sources.filter((s) => s.status === "connected").length;
  const parts = [state.workspace, n === 0 ? "no traces yet" : `${n} trace source${n === 1 ? "" : "s"}`];
  const gpus = fleetSize(state.fleet);
  if (gpus > 0) parts.push(`${gpus} GPU${gpus === 1 ? "" : "s"}`);
  if (state.router !== null) parts.push(state.router);
  return parts.join(" · ");
}
