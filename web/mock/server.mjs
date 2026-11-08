// Throwaway dev backend for the web UI. Not part of the app.
// Serves fixtures shaped like web/src/api/types.ts and an SSE stream that advances sims.
// Usage: npm run mock   (listens on :8000, token "cascade-dev")
//        MOCK_SCENARIO=empty npm run mock   (no traces or runs; an upload starts after 30s)

import { readFileSync } from "node:fs";
import http from "node:http";
import { URL } from "node:url";
import { gzipSync } from "node:zlib";

const PORT = Number(process.env.PORT ?? 8000);
const TOKEN = process.env.CASCADE_TOKEN ?? "cascade-dev";
const BOOT = Date.now();
const SCENARIO = process.env.MOCK_SCENARIO ?? "default";

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = rng(42);
const hex = (n) => Array.from({ length: n }, () => Math.floor(rand() * 16).toString(16)).join("");
const iso = (ms) => new Date(ms).toISOString();
const MIN = 60_000;
const HOUR = 60 * MIN;

const CONFIGS = [
  { name: "SM90_H100", arch: "sm_90", core_clock_mhz: 1755 },
  { name: "SM90_H200", arch: "sm_90", core_clock_mhz: 1980 },
  { name: "SM90_GH200", arch: "sm_90", core_clock_mhz: 1980 },
  { name: "SM80_A100", arch: "sm_80", core_clock_mhz: 1410 },
  { name: "SM86_RTX3070", arch: "sm_86", core_clock_mhz: 1132 },
];
const IPC_SCALE = { SM90_H100: 1, SM90_H200: 1.08, SM90_GH200: 1.12, SM80_A100: 0.74, SM86_RTX3070: 0.41 };

const KERNEL_KINDS = [
  { kind: "norm", ipc: 38, bias: 1.34, name: "_ZN4vllm15rms_norm_kernelIN3c104HalfELi8ELi1EEEvPT_PKS3_S6_fii", grid: [16, 1, 1], block: [256, 1, 1], insn: 98_304 },
  { kind: "gemm", ipc: 112, bias: 0.9, name: "sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas", grid: [18, 1, 1], block: [384, 1, 1], insn: 2_359_296 },
  { kind: "rope", ipc: 54, bias: 1.08, name: "_ZN4vllm23rotary_embedding_kernelIN3c104HalfELb1EEEvPKlPT_S6_PKS4_iiiiii", grid: [16, 1, 1], block: [512, 1, 1], insn: 262_144 },
  { kind: "cache", ipc: 29, bias: 1.58, name: "_ZN4vllm30reshape_and_cache_flash_kernelIN3c104HalfES2_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S7_PT0_S9_PKlliiiiiPKfSD_", grid: [16, 1, 1], block: [768, 1, 1], insn: 147_456 },
  { kind: "attn", ipc: 86, bias: 0.8, name: "_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi64ELi64ELi128ELi4ELb0ELb0EN7cutlass6half_tELb0EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEv13Flash_fwd_params", grid: [1, 12, 16], block: [128, 1, 1], insn: 1_572_864 },
  { kind: "gemm_out", ipc: 104, bias: 0.96, name: "sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas", grid: [12, 1, 1], block: [384, 1, 1], insn: 786_432 },
  { kind: "add", ipc: 47, bias: 1.2, name: "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIN3c104HalfEEESt5arrayIPcLm3EEEEviT0_T1_", grid: [96, 1, 1], block: [128, 1, 1], insn: 196_608 },
];

const DECODE_KINDS = [
  ...KERNEL_KINDS,
  { kind: "silu", ipc: 44, bias: 1.12, name: "triton_poi_fused_mul_silu_0", grid: [112, 1, 1], block: [128, 1, 1], insn: 344_064 },
  { kind: "down", ipc: 98, bias: 0.86, name: "_ZN7cutlass7Kernel2I64cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_64x4_tn_align8EEvNT_6ParamsE", grid: [24, 4, 1], block: [128, 1, 1], insn: 1_179_648 },
  { kind: "argmax", ipc: 31, bias: 1.27, name: "void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue>(float*, float const*, int)", grid: [1, 1, 1], block: [1024, 1, 1], insn: 425_984 },
];
const LM_HEAD = { kind: "lm_head", ipc: 118, bias: 0.93, name: "sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas", grid: [132, 1, 1], block: [384, 1, 1], insn: 31_457_280 };

function buildKernels(count, seed, kinds = KERNEL_KINDS) {
  const r = rng(seed);
  const kernels = [];
  let id = 1;
  for (let i = 0; i < count; i++) {
    const k = kinds === DECODE_KINDS && i === count - 1 ? LM_HEAD : kinds[i % kinds.length];
    const scale = 0.8 + r() * 0.5;
    kernels.push({
      kernel_id: id,
      name: k.name,
      grid: k.grid,
      block: k.block,
      instructions: Math.round(k.insn * scale),
      _kind: k.kind,
      _ipc: k.ipc,
      _bias: k.bias,
    });
    id += r() < 0.18 ? 2 : 1;
  }
  return kernels;
}
const publicKernel = ({ _kind, _ipc, _bias, ...k }) => k;

function makeTrace({ id, state, error = null, createdAgo, model, layers, sourceGpu, arch = "sm_90", suggested = "SM90_H100", kernelCount, bytes, realTiming, stepTimeline = false, seed, kinds }) {
  const created = BOOT - createdAgo;
  const kernels = buildKernels(kernelCount, seed, kinds);
  const real = realTiming
    ? kernels.map((k) => {
        const cycles = k.instructions / k._ipc;
        const simNs = (cycles / 1755) * 1000;
        return { kernel_id: k.kernel_id, kernel_name: k.name, duration_ns: Math.round(simNs * k._bias * (0.92 + rand() * 0.16)), grid: k.grid, block: k.block };
      })
    : null;
  return {
    id,
    state,
    error,
    created_at: iso(created),
    updated_at: iso(created + 4 * MIN),
    model,
    layers,
    source_gpu: sourceGpu,
    trace_gpu_arch: arch,
    suggested_sim_config: suggested,
    kernel_count: state === "uploading" ? 0 : kernelCount,
    bytes,
    has_real_timing: Boolean(realTiming),
    has_step_timeline: stepTimeline,
    _kernels: kernels,
    _real: real,
    _meta: {
            status: state === "failed" ? "failed" : "ok",
            model,
            layers,
            gpu: sourceGpu,
            trace_gpu_arch: arch,
            suggested_sim_config: suggested,
            prompts: [
              "Explain the difference between a GPU warp and a CPU thread in two sentences.",
              "Write a haiku about memory bandwidth.",
              "Summarize the plot of Hamlet for a five year old.",
            ],
            versions: { vllm: "0.11.0", torch: "2.8.0+cu128", cuda: "12.8", driver: "570.124.06", accelsim_tracer: "v2.0.1" },
            accelsim_commit: "1c9cc831",
            max_tokens: 16,
            real_timing: realTiming ? "real_timing.json" : null,
            files: [{ key: "metadata.json", bytes: 2048 }],
          },
  };
}


const traces = [
  makeTrace({
    id: "20261107-174820__Qwen_Qwen2.5-7B-Instruct__model.layers.12.self_attn",
    state: "uploading",
    createdAgo: 0,
    model: "Qwen/Qwen2.5-7B-Instruct",
    layers: ["model.layers.12.self_attn"],
    sourceGpu: "NVIDIA H100 80GB HBM3",
    kernelCount: 42,
    bytes: 0,
    realTiming: false,
    seed: 5,
  }),
  makeTrace({
    id: "20261107-174102__mistralai_Mistral-7B-v0.3__model.layers.3.mlp",
    state: "processing",
    createdAgo: 6 * MIN,
    model: "mistralai/Mistral-7B-v0.3",
    layers: ["model.layers.3.mlp"],
    sourceGpu: "NVIDIA H100 80GB HBM3",
    kernelCount: 31,
    bytes: 1_204_000_000,
    realTiming: false,
    seed: 4,
  }),
  makeTrace({
    id: "20261107-161544__meta-llama_Llama-3.1-8B-Instruct__model.layers.0.mlp",
    state: "ready",
    createdAgo: 92 * MIN,
    model: "meta-llama/Llama-3.1-8B-Instruct",
    layers: ["model.layers.0.mlp", "model.layers.1.mlp"],
    sourceGpu: "NVIDIA H200",
    suggested: "SM90_H200",
    kernelCount: 18,
    bytes: 1_932_735_283,
    realTiming: false,
    seed: 3,
  }),
  makeTrace({
    id: "20261107-153012__facebook_opt-125m__model.decoder.layers.10.self_attn",
    state: "ready",
    createdAgo: 3 * HOUR,
    model: "facebook/opt-125m",
    layers: ["model.decoder.layers.10.self_attn"],
    sourceGpu: "NVIDIA H100 80GB HBM3",
    kernelCount: 21,
    bytes: 14_889_861,
    realTiming: true,
    stepTimeline: true,
    seed: 1,
  }),
  makeTrace({
    id: "20261107-120301__facebook_opt-1.3b__model.decoder.layers.0.self_attn",
    state: "failed",
    error: "post-traces-processing exited 1: kernel-12-ctx_0x5a3c1e0.trace.xz: unexpected end of input",
    createdAgo: 6 * HOUR,
    model: "facebook/opt-1.3b",
    layers: [],
    sourceGpu: "NVIDIA H100 80GB HBM3",
    kernelCount: 12,
    bytes: 88_120_311,
    realTiming: false,
    seed: 2,
  }),
  makeTrace({
    id: "20261107-110110__meta-llama_Llama-3.1-70B-Instruct__decode_step",
    state: "ready",
    createdAgo: 7 * HOUR,
    model: "meta-llama/Llama-3.1-70B-Instruct",
    layers: ["model.layers.*"],
    sourceGpu: "NVIDIA H100 80GB HBM3",
    kernelCount: 1000,
    bytes: 9_412_331_008,
    realTiming: true,
    seed: 9,
  }),
];
for (const t of traces) t._metadata = t.state === "uploading" ? null : t._meta;
const traceById = (id) => allTraces.find((t) => t.id === id);
const traceSummary = ({ _kernels, _real, _metadata, _meta, _processingAt, ...t }) => t;
const traceDetail = (t) => ({ ...traceSummary(t), metadata: t._metadata, kernels: t.state === "uploading" ? t._kernels.slice(0, t.kernel_count).map(publicKernel) : t._kernels.map(publicKernel) });

const SMALL_TRACE = makeTrace({
  id: "20261107-142215__Qwen_Qwen2.5-0.5B__model.layers.0.input_layernorm",
  state: "ready",
  createdAgo: 4 * HOUR,
  model: "Qwen/Qwen2.5-0.5B",
  layers: ["model.layers.0.input_layernorm"],
  sourceGpu: "NVIDIA H100 80GB HBM3",
  kernelCount: 6,
  bytes: 3_204_112,
  realTiming: true,
  seed: 6,
});
const DECODE_TRACE = makeTrace({
  id: "20261107-131907__meta-llama_Llama-3.1-8B-Instruct__decode_step",
  state: "ready",
  createdAgo: 5 * HOUR,
  model: "meta-llama/Llama-3.1-8B-Instruct",
  layers: ["model.layers.*"],
  sourceGpu: "NVIDIA H100 80GB HBM3",
  kernelCount: 242,
  bytes: 2_610_883_420,
  realTiming: true,
  seed: 8,
  kinds: DECODE_KINDS,
});
for (const t of [SMALL_TRACE, DECODE_TRACE]) {
  t._metadata = t._meta;
  traces.push(t);
}

const uploading = traces[0];
uploading._uploadStart = SCENARIO === "empty" ? BOOT + 30_000 : BOOT;
const allTraces = traces.splice(0, traces.length, ...(SCENARIO === "empty" ? [] : traces));
function advanceUpload(now) {
  if (SCENARIO === "empty" && traces.length === 0) {
    if (now < uploading._uploadStart) return;
    uploading.created_at = iso(now);
    traces.push(uploading);
  }
  const t = now - uploading._uploadStart;
  if (uploading.state === "uploading") {
    uploading.kernel_count = Math.min(uploading._kernels.length, Math.floor(t / 3000));
    uploading.bytes = Math.round(uploading.kernel_count * 41_300_000 + (t % 3000) * 4_000);
    uploading.updated_at = iso(now);
    if (uploading.kernel_count >= uploading._kernels.length) {
      uploading.state = "processing";
      uploading._processingAt = now;
    }
  } else if (uploading.state === "processing" && now - uploading._processingAt > 20_000) {
    uploading.state = "ready";
    uploading.updated_at = iso(now);
    uploading._metadata = uploading._meta;
  }
}

function kernelStat(sim, kernel, index) {
  const cfg = CONFIGS.find((c) => c.name === sim.gpu_config);
  const jitter = 1 + (rand() - 0.5) * 0.02;
  const ipc = kernel._ipc * IPC_SCALE[cfg.name] * jitter;
  const cycles = Math.round(kernel.instructions / ipc);
  return {
    sim_id: sim.id,
    kernel_index: index,
    kernel_name: kernel.name,
    kernel_launch_uid: kernel.kernel_id,
    gpu_sim_cycle: cycles,
    gpu_sim_insn: kernel.instructions,
    gpu_ipc: Number(ipc.toFixed(4)),
    sim_time_ns: Math.round((cycles / cfg.core_clock_mhz) * 1000),
    stats: { gpu_stall_dramfull: Math.round(rand() * 400), gpu_stall_icnt2sh: Math.round(rand() * 1200), L2_BW: `${(rand() * 400).toFixed(2)} GB/Sec` },
  };
}

function kernelLogLines(sim, kernel, stat, totals) {
  return [
    `-kernel name = ${kernel.name}`,
    `-kernel id = ${kernel.kernel_id}`,
    `-grid dim = (${kernel.grid.join(",")})`,
    `-block dim = (${kernel.block.join(",")})`,
    `kernel_name = ${kernel.name}`,
    `kernel_launch_uid = ${kernel.kernel_id}`,
    `gpu_sim_cycle = ${stat.gpu_sim_cycle}`,
    `gpu_sim_insn = ${stat.gpu_sim_insn}`,
    `gpu_ipc = ${stat.gpu_ipc.toFixed(4).padStart(12)}`,
    `gpu_tot_sim_cycle = ${totals.gpu_tot_sim_cycle}`,
    `gpu_tot_sim_insn = ${totals.gpu_tot_sim_insn}`,
    `gpu_tot_ipc = ${totals.gpu_tot_ipc.toFixed(4).padStart(12)}`,
    `gpu_stall_dramfull = ${stat.stats.gpu_stall_dramfull}`,
    `L2_BW  = ${stat.stats.L2_BW}`,
  ];
}

const runs = [];
const sims = new Map();
const subscribers = new Map();

function emit(runId, event) {
  for (const res of subscribers.get(runId) ?? []) res.write(`data: ${JSON.stringify(event)}\n\n`);
}

function newSim(run, gpu_config, replica, createdAt) {
  return {
    id: `sim-${hex(6)}`,
    run_id: run.id,
    gpu_config,
    replica,
    state: "pending",
    error: null,
    launcher_ref: null,
    created_at: iso(createdAt),
    started_at: null,
    finished_at: null,
    last_heartbeat_at: null,
    exit_code: null,
    kernels_done: 0,
    kernels_total: traceById(run.trace_id)._kernels.length,
    current_kernel: null,
    latest: { gpu_tot_sim_cycle: null, gpu_tot_sim_insn: null, gpu_tot_ipc: null, sim_rate_ips: null },
    log_tail: [],
    _log: [],
    _stats: [],
  };
}

function pushLog(sim, lines, live) {
  sim._log.push(...lines);
  sim.log_tail = sim._log.slice(-20);
  if (live) emit(sim.run_id, { type: "log", sim_id: sim.id, lines });
}

function completeKernel(sim, live, quiet = false) {
  const trace = traceById(sim._traceId);
  const kernel = trace._kernels[sim.kernels_done];
  const stat = kernelStat(sim, kernel, sim.kernels_done);
  sim._stats.push(stat);
  sim.kernels_done += 1;
  const cyc = (sim.latest.gpu_tot_sim_cycle ?? 0) + stat.gpu_sim_cycle;
  const insn = (sim.latest.gpu_tot_sim_insn ?? 0) + stat.gpu_sim_insn;
  sim.latest = { gpu_tot_sim_cycle: cyc, gpu_tot_sim_insn: insn, gpu_tot_ipc: insn / cyc, sim_rate_ips: Math.round(38_000 + rand() * 52_000) };
  if (!quiet) pushLog(sim, kernelLogLines(sim, kernel, stat, sim.latest), live);
  const next = trace._kernels[sim.kernels_done];
  sim.current_kernel = next ? next.name : null;
  if (next && !quiet) pushLog(sim, [`launching kernel name: ${next.name} uid: ${next.kernel_id}`, `Processing kernel ./traces/kernel-${next.kernel_id}.traceg`], live);
  if (live) emit(sim.run_id, { type: "kernel", stat });
}

function finish(sim, state, now, extra = {}) {
  Object.assign(sim, { state, finished_at: iso(now), current_kernel: null, ...extra });
}

function createRun({ trace_id, entries, createdAt, live, settle, pace = 1, launchSpread = 1600, trouble = {} }) {
  const run = { id: `run-${hex(6)}`, trace_id, created_at: iso(createdAt), entries };
  runs.unshift(run);
  for (const { gpu_config, count } of entries) {
    for (let r = 1; r <= count; r++) {
      const sim = newSim(run, gpu_config, r, createdAt);
      sim._traceId = trace_id;
      const trace = traceById(trace_id);
      const speed = (gpu_config === "SM90_H100" ? 1 : 0.85) * pace;
      sim._plan = {
        pendingMs: 600 + rand() * launchSpread,
        launchMs: 1800 + rand() * 1200,
        kernelMs: (() => {
          const replicaPace = 0.7 + rand() * 0.6;
          return trace._kernels.map((k) => (900 + (k.instructions / 2_400_000) * 2600) * speed * replicaPace * (0.9 + rand() * 0.2));
        })(),
      };
      sims.set(sim.id, sim);
      run._sims = [...(run._sims ?? []), sim.id];
      if (live) {
        sim._live = { startAt: createdAt, progress: 0, lastHb: 0, lastProgressLog: 0 };
        if (r === 2 && gpu_config === entries[0].gpu_config) sim._plan.lost = { atKernel: 4, ms: 9000 };
        const fail = trouble[`${gpu_config}#${r}`];
        if (fail) sim._plan.fail = fail;
      } else {
        settle(sim, createdAt);
      }
    }
  }
  return run;
}

function settleSucceeded(sim, createdAt, quiet = false) {
  const started = createdAt + 4_000;
  sim.state = "running";
  sim.started_at = iso(started);
  sim.launcher_ref = `sbx_${hex(10)}`;
  pushLog(sim, ["Accel-Sim [build 1c9cc831]", `GPGPU-Sim: Configuration options: ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/${sim.gpu_config}/gpgpusim.config`], false);
  while (sim.kernels_done < sim.kernels_total) completeKernel(sim, false, quiet);
  pushLog(sim, ["GPGPU-Sim: *** exit detected ***"], false);
  finish(sim, "succeeded", started + 11 * MIN, { exit_code: 0, last_heartbeat_at: iso(started + 11 * MIN) });
}

const OOM = {
  atKernel: 3,
  exit: 134,
  error: "simulator exited 134 (SIGABRT)",
  log: ["GPGPU-Sim PTX: ERROR ** out of memory while allocating 4194304 bytes for shader core L1D", "Aborted (core dumped)"],
};

// Real `cascade simulate` + `cascade compare` output (results/20261108-043849, H100 OPT-125m), with each
// step's kernel list cut to 3 so the mock stays light. Configs borrow it: H100 gets the default run,
// H200 the one with launch latency subtracted.
const RESULTS = new URL("../../results/20261108-043849/", import.meta.url);
function stepResult(dir) {
  const read = (name) => JSON.parse(readFileSync(new URL(`${dir}/${name}`, RESULTS), "utf8"));
  const simulated = read("simulated_timeline.json");
  return { simulated: { ...simulated, steps: simulated.steps.map((s) => ({ ...s, kernels: s.kernels.slice(0, 3) })) }, compare: read("compare.json") };
}
const STEP_RESULTS = { SM90_H100: stepResult("accelsim"), SM90_H200: stepResult("accelsim-nolat") };
const stepTimelines = new Map();
function attachStepTimelines(run) {
  const list = run._sims.map((id) => sims.get(id))
    .filter((s) => s.state === "succeeded" && STEP_RESULTS[s.gpu_config])
    .toSorted((a, b) => a.gpu_config.localeCompare(b.gpu_config) || a.replica - b.replica)
    .map((s) => ({ sim_id: s.id, gpu_config: s.gpu_config, replica: s.replica, ...STEP_RESULTS[s.gpu_config] }));
  stepTimelines.set(run.id, list);
}

// Alessio's real OPT-125m H100 Accel-Sim outputs; the text files are rebuilt from each sim's own stats.
const OUTPUTS = new URL("../../results/20261108-021007-opt125m-h100-accelsim/outputs/", import.meta.url);
const CONTENT_TYPES = { ".csv": "text/csv", ".txt": "text/plain", ".log": "text/plain", ".json": "application/json", ".gz": "application/gzip" };
const simFiles = new Map();
function attachFiles(run) {
  const real = (name) => readFileSync(new URL(name, OUTPUTS));
  for (const sim of run._sims.map((id) => sims.get(id))) {
    const stats = ["uid,name,gpu_sim_cycle,gpu_sim_insn,ipc", ...sim._stats.map((k) => [k.kernel_launch_uid, k.kernel_name, k.gpu_sim_cycle, k.gpu_sim_insn, k.gpu_ipc].join(","))];
    const files = {
      "accelsim.log.gz": gzipSync(sim._log.join("\n") + "\n"),
      "exit-status.txt": Buffer.from(`${sim.exit_code}\n`),
      "gpgpu_inst_stats.txt": real("gpgpu_inst_stats.txt"),
      "kernel-stats.csv": Buffer.from(stats.join("\n") + "\n"),
      "opt125m_H100_perfetto.json.gz": real("opt125m_H100_perfetto.json.gz"),
      "perf_counter.csv.gz": real("perf_counter.csv.gz"),
    };
    simFiles.set(sim.id, Object.entries(files).map(([name, bytes]) => ({ name, size: bytes.length, content_type: CONTENT_TYPES[name.slice(name.lastIndexOf("."))], created_at: sim.finished_at, _bytes: bytes })));
  }
}
const publicFile = ({ _bytes, ...f }) => f;

// cascade/perfetto.py's run_trace, with raw kernel names (the readable-name port lives in TypeScript and Python only).
function perfettoTrace(run) {
  const meta = (name, pid, value, tid) => ({ ph: "M", pid, name, args: { [name.endsWith("sort_index") ? "sort_index" : "name"]: value }, ...(tid === undefined ? {} : { tid }) });
  const label = (uid, name) => (uid == null ? name : `k${uid} ${name}`);
  const configs = [...new Set(run.entries.map((e) => e.gpu_config))];
  const events = [];
  configs.forEach((config, i) => {
    const pid = i + 1;
    events.push(meta("process_name", pid, `${config} · Accel-Sim simulated GPU`), meta("process_sort_index", pid, pid));
    for (const sim of runSims(run).filter((s) => s.gpu_config === config).toSorted((a, b) => a.replica - b.replica)) {
      const tid = sim.replica + 1;
      events.push(meta("thread_name", pid, `replica ${sim.replica} · ${sim.id}`, tid), meta("thread_sort_index", pid, tid, tid));
      let startNs = 0;
      let cycle = 0;
      for (const k of sim._stats.toSorted((a, b) => a.kernel_index - b.kernel_index)) {
        if (k.sim_time_ns == null) continue;
        const cycles = k.gpu_sim_cycle ?? 0;
        events.push({
          ph: "X", pid, tid, cat: "kernel", name: label(k.kernel_launch_uid, k.kernel_name), ts: startNs / 1000, dur: k.sim_time_ns / 1000,
          args: { kernel_index: k.kernel_index, uid: k.kernel_launch_uid, kernel_name: k.kernel_name, gpu_sim_cycle: k.gpu_sim_cycle, gpu_sim_insn: k.gpu_sim_insn, ipc: k.gpu_ipc, sim_time_ns: k.sim_time_ns, start_cycle: cycle, end_cycle: cycle + cycles },
        });
        startNs += k.sim_time_ns;
        cycle += cycles;
      }
    }
  });
  const trace = traceById(run.trace_id);
  if (trace._real) {
    const pid = configs.length + 1;
    events.push(meta("process_name", pid, trace.source_gpu ? `Real hardware · ${trace.source_gpu}` : "Real hardware"), meta("process_sort_index", pid, pid), meta("thread_name", pid, "measured kernels", 1));
    let startNs = 0;
    for (const row of trace._real) {
      events.push({ ph: "X", pid, tid: 1, cat: "kernel", name: label(row.kernel_id, row.kernel_name), ts: startNs / 1000, dur: row.duration_ns / 1000, args: row });
      startNs += row.duration_ns;
    }
  }
  return {
    traceEvents: events,
    displayTimeUnit: "ns",
    otherData: {
      run_id: run.id,
      trace_id: run.trace_id,
      time_axis: "simulated time, microseconds; each sim's kernels laid end to end",
      slice_duration: "sim_time_ns = gpu_sim_cycle / core clock (MHz) * 1000",
      kernel_scope: "sequential kernels, each including the configured launch latency; excludes host time and memcpy",
    },
  };
}

if (SCENARIO !== "empty") {
const filesRun = createRun({
  trace_id: traces[3].id,
  entries: [{ gpu_config: "SM90_H100", count: 2 }, { gpu_config: "SM90_H200", count: 1 }],
  createdAt: BOOT - 2 * HOUR,
  live: false,
  settle: (sim, at) => {
    settleSucceeded(sim, at);
    // Like the stand-in replay: its log has more kernels than the trace lists.
    if (sim.gpu_config === "SM90_H200") sim.kernels_done += 3;
  },
});
attachStepTimelines(filesRun);
attachFiles(filesRun);
createRun({
  trace_id: traces[2].id,
  entries: [{ gpu_config: "SM90_H100", count: 2 }, { gpu_config: "SM90_H200", count: 2 }],
  createdAt: BOOT - 70 * MIN,
  live: false,
  settle: (sim, at) => {
    if (sim.gpu_config === "SM90_H200" && sim.replica === 2) {
      sim.state = "running";
      sim.started_at = iso(at + 5000);
      completeKernel(sim, false);
      completeKernel(sim, false);
      pushLog(sim, ["GPGPU-Sim PTX: ERROR ** out of memory while allocating 4194304 bytes for shader core L1D", "Aborted (core dumped)"], false);
      finish(sim, "failed", at + 3 * MIN, { exit_code: 134, error: "simulator exited 134 (SIGABRT)" });
    } else settleSucceeded(sim, at);
  },
});
createRun({
  trace_id: traces[2].id,
  entries: [{ gpu_config: "SM90_H200", count: 4 }],
  createdAt: BOOT - 26 * HOUR,
  live: false,
  settle: (sim, at) => {
    if (sim.replica === 1) finish(sim, "failed", at + 2000, { error: "launcher: Sailbox.from_checkpoint failed: checkpoint ckpt_91ab not found" });
    else finish(sim, "cancelled", at + 9 * MIN);
  },
});
createRun({
  trace_id: traces[3].id,
  entries: [{ gpu_config: "SM90_GH200", count: 2 }, { gpu_config: "SM90_H100", count: 1 }],
  createdAt: BOOT - 40 * MIN,
  live: false,
  settle: settleSucceeded,
});
createRun({
  trace_id: traces[5].id,
  entries: [
    { gpu_config: "SM90_H100", count: 16 },
    { gpu_config: "SM90_H200", count: 16 },
    { gpu_config: "SM90_GH200", count: 16 },
  ],
  createdAt: BOOT - 3 * HOUR,
  live: false,
  settle: (sim, at) => settleSucceeded(sim, at, true),
});
createRun({
  trace_id: traces[3].id,
  entries: [{ gpu_config: "SM90_H100", count: 3 }, { gpu_config: "SM90_H200", count: 2 }],
  createdAt: BOOT,
  live: true,
});
createRun({
  trace_id: traces[3].id,
  entries: [
    { gpu_config: "SM90_H100", count: 16 },
    { gpu_config: "SM90_H200", count: 16 },
    { gpu_config: "SM90_GH200", count: 16 },
  ],
  createdAt: BOOT,
  live: true,
  pace: 1.5,
  launchSpread: 9000,
  trouble: { "SM90_H200#7": OOM, "SM90_GH200#12": { ...OOM, atKernel: 8 } },
});
createRun({
  trace_id: SMALL_TRACE.id,
  entries: [{ gpu_config: "SM90_H100", count: 2 }],
  createdAt: BOOT - 4 * HOUR + 10 * MIN,
  live: false,
  settle: settleSucceeded,
});
createRun({
  trace_id: traces[3].id,
  entries: [{ gpu_config: "SM90_H100", count: 2 }, { gpu_config: "SM90_H200", count: 1 }, { gpu_config: "SM90_GH200", count: 1 }],
  createdAt: BOOT - 2.5 * HOUR,
  live: false,
  settle: settleSucceeded,
});
createRun({
  trace_id: DECODE_TRACE.id,
  entries: [{ gpu_config: "SM90_H100", count: 2 }, { gpu_config: "SM90_H200", count: 2 }],
  createdAt: BOOT - 5 * HOUR + 20 * MIN,
  live: false,
  settle: settleSucceeded,
});
}

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);
function runState(states) {
  if (!states.length || states.some((s) => !TERMINAL.has(s))) return "active";
  if (states.every((s) => s === "succeeded")) return "succeeded";
  if (states.every((s) => s === "failed")) return "failed";
  if (states.includes("cancelled") && !states.includes("succeeded")) return "cancelled";
  return "partial";
}
const publicSim = ({ _log, _stats, _plan, _live, _traceId, ...s }) => ({ ...s, log_tail: s.log_tail.slice(-20) });
const runSims = (run) => run._sims.map((id) => sims.get(id));
function runSummary(run) {
  const ss = runSims(run);
  const counts = {};
  for (const s of ss) counts[s.state] = (counts[s.state] ?? 0) + 1;
  const { _sims, ...rest } = run;
  return { ...rest, state: runState(ss.map((s) => s.state)), sim_counts: counts };
}
const lightStat = ({ stats: _stats, ...k }) => ({ ...k, stats: {} });
const runDetail = (run) => ({ ...runSummary(run), trace: traceSummary(traceById(run.trace_id)), sims: runSims(run).map(publicSim) });

function tick() {
  const now = Date.now();
  advanceUpload(now);
  for (const sim of sims.values()) {
    const L = sim._live;
    if (!L || TERMINAL.has(sim.state)) continue;
    const t = now - L.startAt;
    const p = sim._plan;
    const prev = sim.state;
    let changed = false;
    if (t < p.pendingMs) continue;
    if (sim.state === "pending") {
      sim.state = "launching";
      sim.launcher_ref = `sbx_${hex(10)}`;
      changed = true;
    }
    if (sim.state === "launching" && t >= p.pendingMs + p.launchMs) {
      sim.state = "running";
      sim.started_at = iso(now);
      sim.current_kernel = traceById(sim._traceId)._kernels[0].name;
      pushLog(sim, ["Accel-Sim [build 1c9cc831]", `GPGPU-Sim: Configuration options: ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/${sim.gpu_config}/gpgpusim.config`, `Processing kernel ./traces/kernel-${traceById(sim._traceId)._kernels[0].kernel_id}.traceg`], true);
      changed = true;
    }
    if (sim.state === "running" && p.lost && sim.kernels_done === p.lost.atKernel && !L.lostUntil) {
      L.lostUntil = now + p.lost.ms;
      sim.state = "lost";
      changed = true;
    }
    if (sim.state === "lost" && now >= L.lostUntil) {
      sim.state = "running";
      changed = true;
    }
    if (sim.state === "running" && p.fail && sim.kernels_done >= p.fail.atKernel) {
      pushLog(sim, p.fail.log, true);
      finish(sim, "failed", now, { exit_code: p.fail.exit, error: p.fail.error });
      emit(sim.run_id, { type: "sim", sim: publicSim(sim) });
      continue;
    }
    if (sim.state === "running") {
      L.progress += 250;
      while (sim.kernels_done < sim.kernels_total && L.progress >= p.kernelMs[sim.kernels_done]) {
        L.progress -= p.kernelMs[sim.kernels_done];
        completeKernel(sim, true);
        changed = true;
      }
      if (now - L.lastProgressLog > 800 && sim.kernels_done < sim.kernels_total) {
        L.lastProgressLog = now;
        const cyc = (sim.latest.gpu_tot_sim_cycle ?? 0) + Math.round(L.progress * 4.1);
        pushLog(sim, [`GPGPU-Sim uArch: cycles simulated: ${cyc}  inst.: ${(sim.latest.gpu_tot_sim_insn ?? 0) + Math.round(L.progress * 260)} (ipc=${(sim.latest.gpu_tot_ipc ?? 0).toFixed(1)}) sim_rate=${sim.latest.sim_rate_ips ?? 0} (inst/sec) elapsed = 0:0:00:${String(Math.floor(t / 1000) % 60).padStart(2, "0")} / ${new Date(now).toString().slice(0, 24)}`], true);
      }
      if (sim.kernels_done >= sim.kernels_total) {
        pushLog(sim, ["GPGPU-Sim: *** exit detected ***"], true);
        finish(sim, "succeeded", now, { exit_code: 0 });
        changed = true;
      }
    }
    if ((sim.state === "running" || sim.state === "launching") && now - L.lastHb >= 1000) {
      L.lastHb = now;
      sim.last_heartbeat_at = iso(now);
      changed = true;
    }
    if (changed || prev !== sim.state) emit(sim.run_id, { type: "sim", sim: publicSim(sim) });
  }
}
setInterval(tick, 250);

function median(xs) {
  const s = xs.filter((x) => x != null).sort((a, b) => a - b);
  if (!s.length) return null;
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function comparison(run) {
  const trace = traceById(run.trace_id);
  const configs = run.entries.map((e) => e.gpu_config);
  const ss = runSims(run);
  const rows = trace._kernels.map((k, i) => {
    const sim_ns = {};
    for (const c of configs) sim_ns[c] = median(ss.filter((s) => s.gpu_config === c).map((s) => s._stats[i]?.sim_time_ns));
    return { kernel_index: i, kernel_name: k.name, real_ns: trace._real ? trace._real[i].duration_ns : null, sim_ns };
  });
  return { has_real_timing: trace.has_real_timing, configs, rows };
}
function simResults(sim) {
  const totals = sim.latest;
  return { sim: publicSim(sim), trace_id: sim._traceId, gpu_config: CONFIGS.find((c) => c.name === sim.gpu_config), kernels: sim._stats, totals };
}

function send(res, status, body, headers = {}) {
  const isText = typeof body === "string";
  res.writeHead(status, { "content-type": isText ? "text/plain; charset=utf-8" : "application/json", ...headers });
  res.end(isText ? body : JSON.stringify(body));
}
async function readJson(req) {
  let raw = "";
  for await (const chunk of req) raw += chunk;
  return raw ? JSON.parse(raw) : null;
}

const routes = [
  ["GET", /^\/api\/health$/, (_m, _req, res) => send(res, 200, { ok: true }), { open: true }],
  ["GET", /^\/api\/gpu-configs$/, (_m, _req, res) => send(res, 200, CONFIGS)],
  ["GET", /^\/api\/traces$/, (_m, _req, res) => send(res, 200, traces.map(traceSummary))],
  ["GET", /^\/api\/traces\/([^/]+)$/, ([id], _req, res) => {
    const t = traceById(id);
    return t ? send(res, 200, traceDetail(t)) : send(res, 404, { detail: "trace not found" });
  }],
  ["GET", /^\/api\/runs$/, (_m, _req, res) => send(res, 200, runs.toSorted((a, b) => b.created_at.localeCompare(a.created_at)).map(runSummary))],
  ["POST", /^\/api\/runs$/, async (_m, req, res) => {
    const body = await readJson(req);
    const trace = traceById(body?.trace_id);
    if (!trace) return send(res, 404, { detail: "trace not found" });
    if (trace.state !== "ready") return send(res, 409, { detail: `trace is ${trace.state}, not ready` });
    for (const e of body.entries ?? []) {
      const cfg = CONFIGS.find((c) => c.name === e.gpu_config);
      if (!cfg) return send(res, 422, { detail: `unknown gpu_config ${e.gpu_config}` });
      if (cfg.arch !== trace.trace_gpu_arch) return send(res, 422, { detail: `${cfg.name} is ${cfg.arch}; trace needs ${trace.trace_gpu_arch}` });
    }
    const run = createRun({ trace_id: trace.id, entries: body.entries, createdAt: Date.now(), live: true });
    return send(res, 201, runDetail(run));
  }],
  ["GET", /^\/api\/runs\/([^/]+)$/, ([id], _req, res) => {
    const run = runs.find((r) => r.id === id);
    return run ? send(res, 200, runDetail(run)) : send(res, 404, { detail: "run not found" });
  }],
  ["POST", /^\/api\/runs\/([^/]+)\/cancel$/, ([id], _req, res) => {
    const run = runs.find((r) => r.id === id);
    if (!run) return send(res, 404, { detail: "run not found" });
    for (const sim of runSims(run)) {
      if (TERMINAL.has(sim.state)) continue;
      finish(sim, "cancelled", Date.now());
      emit(run.id, { type: "sim", sim: publicSim(sim) });
    }
    return send(res, 200, runDetail(run));
  }],
  ["GET", /^\/api\/runs\/([^/]+)\/events$/, ([id], req, res) => {
    const run = runs.find((r) => r.id === id);
    if (!run) return send(res, 404, { detail: "run not found" });
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
    res.write(`data: ${JSON.stringify({ type: "snapshot", run: runDetail(run), kernels: runSims(run).flatMap((s) => s._stats.map(lightStat)) })}\n\n`);
    const set = subscribers.get(id) ?? new Set();
    set.add(res);
    subscribers.set(id, set);
    const ping = setInterval(() => res.write(": ping\n\n"), 15_000);
    req.on("close", () => {
      clearInterval(ping);
      set.delete(res);
    });
  }],
  ["GET", /^\/api\/runs\/([^/]+)\/comparison$/, ([id], _req, res) => {
    const run = runs.find((r) => r.id === id);
    return run ? send(res, 200, comparison(run)) : send(res, 404, { detail: "run not found" });
  }],
  ["GET", /^\/api\/runs\/([^/]+)\/kernels$/, ([id], _req, res) => {
    const run = runs.find((r) => r.id === id);
    if (!run) return send(res, 404, { detail: "run not found" });
    const ordered = runSims(run).toSorted((a, b) => a.gpu_config.localeCompare(b.gpu_config) || a.replica - b.replica);
    return send(res, 200, ordered.flatMap((s) => s._stats.map(lightStat)));
  }],
  ["GET", /^\/api\/runs\/([^/]+)\/step-timelines$/, ([id], _req, res) => {
    const run = runs.find((r) => r.id === id);
    return run ? send(res, 200, stepTimelines.get(id) ?? []) : send(res, 404, { detail: "run not found" });
  }],
  ["GET", /^\/api\/runs\/([^/]+)\/results$/, ([id], _req, res) => {
    const run = runs.find((r) => r.id === id);
    if (!run) return send(res, 404, { detail: "run not found" });
    return send(res, 200, { run: runSummary(run), trace: traceDetail(traceById(run.trace_id)), sims: runSims(run).map(simResults), comparison: comparison(run) });
  }],
  ["GET", /^\/api\/sims\/([^/]+)\/results$/, ([id], _req, res) => {
    const sim = sims.get(id);
    return sim ? send(res, 200, simResults(sim)) : send(res, 404, { detail: "sim not found" });
  }],
  ["GET", /^\/api\/runs\/([^/]+)\/files$/, ([id], _req, res) => {
    const run = runs.find((r) => r.id === id);
    if (!run) return send(res, 404, { detail: "run not found" });
    const ordered = runSims(run).toSorted((a, b) => a.gpu_config.localeCompare(b.gpu_config) || a.replica - b.replica || a.id.localeCompare(b.id));
    return send(res, 200, ordered.flatMap((s) => (simFiles.get(s.id) ?? []).map((f) => ({ ...publicFile(f), sim_id: s.id, gpu_config: s.gpu_config, replica: s.replica }))));
  }],
  ["GET", /^\/api\/runs\/([^/]+)\/perfetto$/, ([id], _req, res) => {
    const run = runs.find((r) => r.id === id);
    if (!run) return send(res, 404, { detail: "run not found" });
    return send(res, 200, perfettoTrace(run), { "content-disposition": `attachment; filename="${run.id}-perfetto.json"` });
  }],
  ["GET", /^\/api\/sims\/([^/]+)\/files$/, ([id], _req, res) => {
    if (!sims.has(id)) return send(res, 404, { detail: "sim not found" });
    return send(res, 200, (simFiles.get(id) ?? []).map(publicFile));
  }],
  ["GET", /^\/api\/sims\/([^/]+)\/files\/([^/]+)$/, ([id, name], _req, res) => {
    if (!sims.has(id)) return send(res, 404, { detail: "sim not found" });
    const file = simFiles.get(id)?.find((f) => f.name === name);
    if (!file) return send(res, 404, { detail: `sim ${id} has no file '${name}'` });
    res.writeHead(200, { "content-type": file.content_type, "content-length": file.size, "content-disposition": `attachment; filename="${name}"` });
    res.end(file._bytes);
  }],
  ["GET", /^\/api\/sims\/([^/]+)\/log$/, ([id], _req, res) => {
    const sim = sims.get(id);
    return sim ? send(res, 200, sim._log.join("\n") + "\n") : send(res, 404, { detail: "sim not found" });
  }],
];

http
  .createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    const route = routes.find(([method, re]) => method === req.method && re.test(url.pathname));
    if (!route) return send(res, 404, { detail: "not found" });
    const [, re, handler, opts] = route;
    const auth = req.headers.authorization === `Bearer ${TOKEN}` || url.searchParams.get("token") === TOKEN;
    if (!opts?.open && !auth) return send(res, 401, { detail: "invalid token" });
    await new Promise((r) => setTimeout(r, 80 + rand() * 120));
    const match = url.pathname.match(re).slice(1).map(decodeURIComponent);
    try {
      await handler(match, req, res);
    } catch (err) {
      send(res, 500, { detail: String(err) });
    }
  })
  .listen(PORT, () => {
    console.log(`cascade mock on http://localhost:${PORT}  token: ${TOKEN}`);
  });
