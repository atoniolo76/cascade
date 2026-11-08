// Mirror of cascade/models.py. Keep the two in sync.

export type TraceState = "uploading" | "processing" | "ready" | "failed";

export type SimState =
  | "pending"
  | "launching"
  | "running"
  | "lost"
  | "succeeded"
  | "failed"
  | "cancelled";

export type RunState = "active" | "succeeded" | "partial" | "failed" | "cancelled";

export type IsoDate = string;

export interface GpuConfig {
  name: string;
  arch: string;
  core_clock_mhz: number;
}

export interface KernelLaunch {
  kernel_id: number;
  name: string;
  grid: [number, number, number];
  block: [number, number, number];
  instructions: number | null;
}

export interface TraceSummary {
  id: string;
  state: TraceState;
  error: string | null;
  created_at: IsoDate;
  updated_at: IsoDate;
  model: string | null;
  layers: string[];
  source_gpu: string | null;
  trace_gpu_arch: string;
  suggested_sim_config: string | null;
  real_gpu_config: string | null;
  sample_sim_configs: string[];
  sample_set: string | null;
  kernel_count: number;
  bytes: number;
  has_real_timing: boolean;
  has_step_timeline: boolean;
  /** Seeded from samples/ at startup: real, recorded results, with no instruction traces to simulate. */
  sample: boolean;
}

export interface TraceDetail extends TraceSummary {
  metadata: Record<string, unknown> | null;
  kernels: KernelLaunch[];
}

export interface RunEntry {
  gpu_config: string;
  count: number;
}

export interface RunCreate {
  trace_id: string;
  entries: RunEntry[];
}

export interface SimMetrics {
  gpu_tot_sim_cycle: number | null;
  gpu_tot_sim_insn: number | null;
  gpu_tot_ipc: number | null;
  sim_rate_ips: number | null;
}

export interface SimSummary {
  id: string;
  run_id: string;
  gpu_config: string;
  replica: number;
  state: SimState;
  error: string | null;
  launcher_ref: string | null;
  created_at: IsoDate;
  started_at: IsoDate | null;
  finished_at: IsoDate | null;
  last_heartbeat_at: IsoDate | null;
  exit_code: number | null;
  kernels_done: number;
  kernels_total: number;
  current_kernel: string | null;
  latest: SimMetrics;
  log_tail: string[];
}

export interface KernelStat {
  sim_id: string;
  kernel_index: number;
  kernel_name: string;
  kernel_launch_uid: number | null;
  gpu_sim_cycle: number | null;
  gpu_sim_insn: number | null;
  gpu_ipc: number | null;
  sim_time_ns: number | null;
  stats: Record<string, number | string>;
}

export interface RunSummary {
  id: string;
  trace_id: string;
  created_at: IsoDate;
  state: RunState;
  entries: RunEntry[];
  sim_counts: Partial<Record<SimState, number>>;
  /** Seeded from samples/ with its recorded results. */
  sample: boolean;
  /** Launched on a sample trace: replays its recorded Accel-Sim results instead of simulating. */
  replay: boolean;
}

export interface RunDetail extends RunSummary {
  trace: TraceSummary;
  sims: SimSummary[];
}

/** `simulated` is Alessio's simulated_timeline.json and `compare` his compare.json; the contract keeps them opaque. */
export interface StepTimelineUpload {
  simulated: Record<string, unknown>;
  compare: Record<string, unknown> | null;
}

export interface SimStepTimeline extends StepTimelineUpload {
  sim_id: string;
  gpu_config: string;
  replica: number;
}

export interface SimFile {
  name: string;
  size: number;
  content_type: string;
  created_at: IsoDate;
}

export interface RunFile extends SimFile {
  sim_id: string;
  gpu_config: string;
  replica: number;
}

export interface ComparisonRow {
  kernel_index: number;
  kernel_name: string;
  real_ns: number | null;
  sim_ns: Record<string, number | null>;
}

export interface Comparison {
  has_real_timing: boolean;
  configs: string[];
  rows: ComparisonRow[];
}

export interface SimResults {
  sim: SimSummary;
  trace_id: string;
  gpu_config: GpuConfig;
  kernels: KernelStat[];
  totals: SimMetrics;
}

export interface RunResults {
  run: RunSummary;
  trace: TraceDetail;
  sims: SimResults[];
  comparison: Comparison;
}

export type RunEvent =
  | { type: "snapshot"; run: RunDetail; kernels: KernelStat[] }
  | { type: "sim"; sim: SimSummary }
  | { type: "log"; sim_id: string; lines: string[] }
  | { type: "kernel"; stat: KernelStat };
