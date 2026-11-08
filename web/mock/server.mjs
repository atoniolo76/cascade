// Throwaway dev backend for the web UI. Not part of the app.
// Serves fixtures shaped like web/src/api/types.ts and an SSE stream that advances sims.
// Usage: npm run mock   (listens on :8000, token "cascade-dev")
//        MOCK_SCENARIO=empty npm run mock   (no traces or runs; an upload starts after 30s)

import http from "node:http";
import { URL } from "node:url";

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
  { kind: "norm", ipc: 38, name: "_ZN4vllm15rms_norm_kernelIN3c104HalfELi8ELi1EEEvPT_PKS3_S6_fii", grid: [16, 1, 1], block: [256, 1, 1], insn: 98_304 },
  { kind: "gemm", ipc: 112, name: "sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas", grid: [18, 1, 1], block: [384, 1, 1], insn: 2_359_296 },
  { kind: "rope", ipc: 54, name: "_ZN4vllm23rotary_embedding_kernelIN3c104HalfELb1EEEvPKlPT_S6_PKS4_iiiiii", grid: [16, 1, 1], block: [512, 1, 1], insn: 262_144 },
  { kind: "cache", ipc: 29, name: "_ZN4vllm30reshape_and_cache_flash_kernelIN3c104HalfES2_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S7_PT0_S9_PKlliiiiiPKfSD_", grid: [16, 1, 1], block: [768, 1, 1], insn: 147_456 },
  { kind: "attn", ipc: 86, name: "_ZN5flash22flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi64ELi64ELi128ELi4ELb0ELb0EN7cutlass6half_tELb0EELb0ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEv13Flash_fwd_params", grid: [1, 12, 16], block: [128, 1, 1], insn: 1_572_864 },
  { kind: "gemm_out", ipc: 104, name: "sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas", grid: [12, 1, 1], block: [384, 1, 1], insn: 786_432 },
  { kind: "add", ipc: 47, name: "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIN3c104HalfEEESt5arrayIPcLm3EEEEviT0_T1_", grid: [96, 1, 1], block: [128, 1, 1], insn: 196_608 },
];

function buildKernels(count, seed) {
  const r = rng(seed);
  const kernels = [];
  let id = 1;
  for (let i = 0; i < count; i++) {
    const k = KERNEL_KINDS[i % KERNEL_KINDS.length];
    const scale = 0.8 + r() * 0.5;
    kernels.push({
      kernel_id: id,
      name: k.name,
      grid: k.grid,
      block: k.block,
      instructions: Math.round(k.insn * scale),
      _kind: k.kind,
      _ipc: k.ipc,
    });
    id += r() < 0.18 ? 2 : 1;
  }
  return kernels;
}
const publicKernel = ({ _kind, _ipc, ...k }) => k;

function makeTrace({ id, state, error = null, createdAgo, model, layers, sourceGpu, arch = "sm_90", suggested = "SM90_H100", kernelCount, bytes, realTiming, seed }) {
  const created = BOOT - createdAgo;
  const kernels = buildKernels(kernelCount, seed);
  const real = realTiming
    ? kernels.map((k) => {
        const cycles = k.instructions / k._ipc;
        const simNs = (cycles / 1755) * 1000;
        return { kernel_id: k.kernel_id, kernel_name: k.name, duration_ns: Math.round(simNs * (0.72 + rand() * 0.5)), grid: k.grid, block: k.block };
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
];
for (const t of traces) t._metadata = t.state === "uploading" ? null : t._meta;
const traceById = (id) => allTraces.find((t) => t.id === id);
const traceSummary = ({ _kernels, _real, _metadata, _meta, _processingAt, ...t }) => t;
const traceDetail = (t) => ({ ...traceSummary(t), metadata: t._metadata, kernels: t.state === "uploading" ? t._kernels.slice(0, t.kernel_count).map(publicKernel) : t._kernels.map(publicKernel) });

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

function completeKernel(sim, live) {
  const trace = traceById(sim._traceId);
  const kernel = trace._kernels[sim.kernels_done];
  const stat = kernelStat(sim, kernel, sim.kernels_done + 1);
  sim._stats.push(stat);
  sim.kernels_done += 1;
  const cyc = (sim.latest.gpu_tot_sim_cycle ?? 0) + stat.gpu_sim_cycle;
  const insn = (sim.latest.gpu_tot_sim_insn ?? 0) + stat.gpu_sim_insn;
  sim.latest = { gpu_tot_sim_cycle: cyc, gpu_tot_sim_insn: insn, gpu_tot_ipc: insn / cyc, sim_rate_ips: Math.round(38_000 + rand() * 52_000) };
  pushLog(sim, kernelLogLines(sim, kernel, stat, sim.latest), live);
  const next = trace._kernels[sim.kernels_done];
  sim.current_kernel = next ? next.name : null;
  if (next) pushLog(sim, [`launching kernel name: ${next.name} uid: ${next.kernel_id}`, `Processing kernel ./traces/kernel-${next.kernel_id}.traceg`], live);
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

function settleSucceeded(sim, createdAt) {
  const started = createdAt + 4_000;
  sim.state = "running";
  sim.started_at = iso(started);
  sim.launcher_ref = `sbx_${hex(10)}`;
  pushLog(sim, ["Accel-Sim [build 1c9cc831]", `GPGPU-Sim: Configuration options: ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/${sim.gpu_config}/gpgpusim.config`], false);
  while (sim.kernels_done < sim.kernels_total) completeKernel(sim, false);
  pushLog(sim, ["GPGPU-Sim: *** exit detected ***"], false);
  finish(sim, "succeeded", started + 11 * MIN, { exit_code: 0, last_heartbeat_at: iso(started + 11 * MIN) });
}

const OOM = {
  atKernel: 3,
  exit: 134,
  error: "simulator exited 134 (SIGABRT)",
  log: ["GPGPU-Sim PTX: ERROR ** out of memory while allocating 4194304 bytes for shader core L1D", "Aborted (core dumped)"],
};

if (SCENARIO !== "empty") {
createRun({
  trace_id: traces[3].id,
  entries: [{ gpu_config: "SM90_H100", count: 2 }, { gpu_config: "SM90_H200", count: 1 }],
  createdAt: BOOT - 2 * HOUR,
  live: false,
  settle: (sim, at) => {
    settleSucceeded(sim, at);
    // Like the stand-in fake sim: its log has more kernels than the trace lists.
    if (sim.gpu_config === "SM90_H200") sim.kernels_done += 3;
  },
});
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
    for (const c of configs) sim_ns[c] = median(ss.filter((s) => s.gpu_config === c).map((s) => s._stats.find((st) => st.kernel_index === i + 1)?.sim_time_ns));
    return { kernel_index: i + 1, kernel_name: k.name, real_ns: trace._real ? trace._real[i].duration_ns : null, sim_ns };
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
  ["GET", /^\/api\/runs$/, (_m, _req, res) => send(res, 200, runs.map(runSummary))],
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
    res.write(`data: ${JSON.stringify({ type: "snapshot", run: runDetail(run), kernels: runSims(run).flatMap((s) => s._stats) })}\n\n`);
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
  ["GET", /^\/api\/runs\/([^/]+)\/results$/, ([id], _req, res) => {
    const run = runs.find((r) => r.id === id);
    if (!run) return send(res, 404, { detail: "run not found" });
    return send(res, 200, { run: runSummary(run), trace: traceDetail(traceById(run.trace_id)), sims: runSims(run).map(simResults), comparison: comparison(run) });
  }],
  ["GET", /^\/api\/sims\/([^/]+)\/results$/, ([id], _req, res) => {
    const sim = sims.get(id);
    return sim ? send(res, 200, simResults(sim)) : send(res, 404, { detail: "sim not found" });
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
