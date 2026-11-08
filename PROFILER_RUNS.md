# Profiler runs for sim-to-real calibration

## Python package (bring your own GPU container)

```
pip install "git+https://github.com/atoniolo76/cascade@profiler"
# Optional dependency sets: cascade-profiler[modal] / cascade-profiler[vllm]
```

`import cascade_profiler` requires neither Modal nor torch/vLLM. The package exports
`Workload`, `RunResult`, `profile`, `trace_plan`, and lazily loaded `image`,
`register`. The caller supplies the GPU/container, dependencies, output mount, and
volume commit. Runtime timing needs vLLM 0.21.0, torch, nvtx, nsys and nvidia-smi; optional
NVBit tracing also needs a built Accel-Sim tracer/spinlock tool/postprocessor at `accelsim_root`.
Baseline-only execution does not need nsys or Accel-Sim. All stages run sequentially inside
the same caller container, each workload in a fresh subprocess.

```python
from cascade_profiler import Workload, profile

# Inside your own Modal GPU function (with your chosen image and volume mounts):
result = profile(
    Workload("facebook/opt-125m", ["Hello, my name is"], max_tokens=32, repeats=4),
    out_dir="/outputs", trace_steps=[0, 1], trace_repeat=1,
    accelsim_root="/opt/accel-sim-framework",
)
print(result.run_dir, result.paths["corrected_timeline.json"], result.stage_wall_s)
# Commit your own output volume here.
```

`Workload.to_dict()` / `Workload.from_dict()` round-trip JSON-compatible configuration:
model, prompts, max_tokens (32), repeats (4), exec_mode (eager), max_model_len (2048),
llm_kwargs, sampling_kwargs, and optional `driver="your_module:your_function"`. model is required;
library code does not choose a model or layer path. Layer-scope tracing requires user-specified
paths (`--list-layers` helps discover them). Only compatibility CLI wrappers retain OPT defaults.
Sampling defaults are temperature=0, seed=0, ignore_eos=True, max_tokens=workload.max_tokens;
sampling_kwargs can override them. LLM extras can override non-required defaults such as
gpu_memory_utilization=0.8, enforce_eager, and compilation_config: **user kwargs win over exec_mode
defaults**. dtype, quantization, trust_remote_code, max_num_seqs and other JSON-compatible kwargs
pass through unchanged. run.json records per-phase effective kwargs and overridden execution defaults.
Conflicting truly required settings (async scheduling off, iteration/metrics instrumentation, CUDA
capture, layerwise NVTX's eager requirement) raise with an explanation before launching.
TP/PP/DP>1 and distributed executors are explicitly unsupported: engine-core multiprocessing being
off does not prevent vLLM from spawning GPU worker processes. In vLLM 0.21, parallel.py:831-876
selects mp/ray for multi-worker world sizes; StepRecorder's local NVBit enable/disable and synchronize
cannot control those workers. Single-device uni execution is required; register's gpu argument
is explicit and can name/type/count the caller's allocation, but multi-device workloads are rejected.
Custom sampling/drivers must preserve the kernel sequence across processes for later alignment.

A driver must be importable in both parent and subprocesses (not a closure passed as a callable
or a function in `__main__`); validate/import happens before launching. It receives
`driver(llm, repeat: int, workload: Workload)` once per repeat and returns a list of RequestOutput.
Without a driver, the runner uses `llm.generate(workload.prompts, SamplingParams(...))`.
Step recording, capture windows, GC, and metrics surround either driver. The runner supports
`python -m cascade_profiler.runner --workload-json /path/workload.json`; the old vllm_trace.py
CLI remains a compatibility shim. Workload JSON defaults to step scope with async scheduling off.

`profile()` orders requested timing phases baseline then profiled, irrespective of argument order;
`phases=("baseline",)` permits repeats=1, but profiled capture needs at least two repeats.
If trace_steps is not None it then runs spinlock detection twice, NVBit tracing, and postprocessing.
Trace indices are **within trace_repeat**, not global indices: `[0,1]` selects that repeat's first
two steps even if earlier repeats have different lengths. `[]` selects all steps in that repeat;
None disables tracing. Legacy `--trace-steps` retains its global-index meaning. Every raw step
now records `step_in_repeat`; traced/trace_steps.json includes repeat, step_in_repeat, global
`step`, traced boolean, and `[first_kernel_id,end_kernel_id)` ranges, including untraced steps.
Requested indices that are never reached fail explicitly. NVBit timings are never used for timing.
Existing NVBit CUDA-graph replay coverage limitations still apply: choose eager/compile when
complete instruction traces are needed. The pipeline never silently changes the workload mode.

The run directory is `<out_dir>/<timestamp-random>/` (or the explicit run_name). Existing run
directories are rejected rather than mixing old/new artifacts. Baseline/profiled artifact names
below are unchanged; traced/ holds the instruction traces, detection logs, and postprocessing
outputs. The root also contains workload.json and run.json (configuration, selected phases,
trace selection, stage_wall_s, package_version, git_sha when available, traced_timing_used=false,
and completion status). Failures retain partial logs and a failed manifest. `RunResult.paths`
maps artifact-relative names to Paths; corrected_timeline.json exists only with both timing phases.

Module map: workload.py (configuration/driver validation), runner.py (GPU subprocess), pipeline.py
(orchestration), nsys.py (CLI, scrubbing, SQLite/CSV helpers and existing alignment), clocks.py
(NVML/smi), timeline.py (pure reconciliation), tracing.py (parameterized NVBit commands), utils.py
(logging/naming/serialization), legacy.py (existing trace/nsys/microbench orchestration and storage).
align.py handles step-local trace alignment, and modal_support.py optional GPU image
factories/registration, without module-level Modal objects. CPU execution lives in
`cascade.sim.accelsim` in the separate simulator checkout, not in this package.
accelsim_trace.py remains the compatibility Modal app, with package sources added to its image;
all existing Modal commands below remain valid. Offline checks: `python3 -m unittest discover -s tests -v`.

## End-to-end tracing and CPU simulation

With Python **3.12** and Modal credentials, the bring-your-own-app example is:

```
modal run examples/modal_app.py
# GPU profiling only:
modal run accelsim_trace.py::e2e_main --run-name my-run --trace-steps plan --trace-repeat 1
# From the simulator checkout (same accelsim-traces volume, phases/<run>):
modal run scripts/simulate_accelsim.py --run-name my-run --gpu-config SM90_H100 --propagate interp
```

`register(app, volume, gpu=<required>, image=None, hf_cache=None, mount="/data",
timeout=14400)` returns only the `.profile` Modal function.
`.profile.remote(workload.to_dict(), run_name=None, trace_steps=[0,5], trace_repeat=1)` writes
`<mount>/<run_name>/`, commits the caller's volume (also on errors), and returns run_name and
stage_wall_s. The simulator package's `cascade.sim.modal_support.register` separately returns
a CPU function accepting run_name, gpu_config, propagate, dedup, and verify_dedup. Set its
mount to the volume root and `run_subdir="phases"` for the compatibility app's layout.
Defaults use 32 CPU simulation workers.
Functions use `serialized=True` because they are nested: client and container **Python minor
versions must match** (default images are 3.12). Import checks do not build/deploy images.
Use separate apps when registering profiling and simulation.

`image(base=None, accelsim_commit=..., nvbit=True, nsys=True, arch="all")` builds the profiling
image with package sources. A custom base must supply vLLM/torch/nvtx and be CUDA **12.x devel,
Ubuntu 22.04**, with NVIDIA's apt repository for cuda-nsight-systems-12-9. `nvbit=False` skips
tracer installation/build; `nsys=False` skips nsys. arch="all" supports multiple CUDA architectures
(slower image build); the old H100 app explicitly uses arch="sm_90". The simulator package's
`cascade.sim.modal_support.accelsim_image()` builds a separate CPU-only
simulator image with CUDA's toolkit, compiler/build dependencies, and the standalone CPU trace
postprocessor; it does not request a GPU or install vLLM. Default pins:

- Accel-Sim: `7ba54ded60c4ba0083410c37d22d719166acd6dc`.
- GPGPU-Sim dev: `4195dd4f229cbf03f4f38c49139ae94d01d23f03` (resolved with git ls-remote).
  setup_environment.sh honors `GPGPUSIM_BRANCH`; the CPU image sets it to this commit before
  setup clones/checks out GPGPU-Sim. Pass `gpgpusim_ref` to override deliberately.

### Alignment

When both timing phases and tracing exist, profile() writes `trace_alignment.json` after
postprocessing. Corrected steps now add `step_in_repeat`; older timelines are indexed by counting
steps within each repeat. Corrected kernel `name` is already nsys's **mangledName** when available
(`_nsys_kernels` selects it rather than shortName; demangledName is only an old-schema fallback),
so no extra name field is needed. Alignment also accepts explicit `mangled_name` from external
timelines. It never guesses equivalence between demangled and mangled names.

Tracer rows are limited to each selected repeat-local step's `[first_kernel_id,end_kernel_id)`;
only positive total_insts or existing raw/processed trace files count as traced. Ordered
(name,grid,block) SequenceMatcher anchors are followed by name-only matches within unmatched
ordered gaps, marked `dims_match=false`. UID collisions across contexts are rejected rather than
silently mapping the wrong kernel. Output contains provenance (including traced_timing_used=false),
per-step counts/exactness, UID/file → (repeat, baseline-global step, seq) mappings, and unmatched
counts/lists. Missing capture windows remain unmatched, not fabricated timing kernels.

### Simulation and artifacts

```
from cascade.sim.accelsim import simulate_accelsim  # in the simulator checkout
summary = simulate_accelsim("/data/my-run", jobs=32, kernel_timeout_s=600, propagate=True)
```

Each aligned UID runs as a **separate process** in a process pool (jobs defaults to os.cpu_count()).
The simulator uses gpu-simulator/bin/release/accel-sim.out, GPGPU-Sim's
configs/tested-cfgs/<gpu_config>/gpgpusim.config, and Accel-Sim's corresponding trace.config.
Config companions are copied to each working directory. A bash wrapper sources the runtime
setup **once before parallel work**, captures its environment, and passes library paths to workers;
this avoids setup scripts racing over shared symlinks. Setup/build/config failures are fatal;
individual simulator exits, missing traces, invalid stats, and timeouts are recorded, not fatal.
Missing kernelslist.g/.tracez triggers postprocessing first. Partial postprocessing failure is
recorded in provenance/summary; existing processed kernels still run and missing kernels fail individually.

`sim/kernel-<uid>/kernelslist.g` contains only that kernel filename, symlinked to its .tracez.
The parser accepts kernel-only lists (`trace_parser.cc:390-428`); MemcpyHtoD lines are omitted.
These are isolated **cold-cache** simulations, not warm-cache application replay: copies can warm
caches in a full replay, and that behavior is deliberately not reproduced here. Per-kernel logs
are `sim/kernel-<uid>.log`; `sim/kernels.csv` records UID/name/cycles/wall seconds/status/error and
the simulator's own launch UID (which need not equal the tracer UID). Parsing requires completed
single-kernel kernel_name/kernel_launch_uid/gpu_sim_cycle stats; cycle-limit truncation is rejected.

`sim_durations.json` has provenance and scoped `kernels` records:
`{repeat,step,seq,sim_cycles,clock_mhz,source,tracer_uid,name}` (propagated rows omit tracer_uid).
The simulator reader accepts extra fields. Clock MHz is the **first domain** of the actual config;
measured profiled SM clock is separately recorded, never silently substituted. Failed kernels have
no duration record so the simulator consumer can fall back to measured durations. `uids` selects
aligned IDs explicitly; each invocation replaces durations/CSV/summary (old per-UID logs may remain).

With propagate="identical" (or legacy True), only **untraced steps** get assignments with identical kernel (name,grid,block)
and step kind. Dimension-mismatched alignments do not seed propagation. Differing measured cycles
for the same key use their median and are listed in provenance. This is an extrapolation, not an
exact simulation of those steps. `sim/summary.json` reports coverage per traced step, success/failure
counts, summed per-kernel simulator wall seconds, mean/max, and overall elapsed wall seconds.

### Generic models, planning, deduplication and interpolation

For example, a second-architecture timing-only run needs no NVBit and never invokes simulation:

```
modal run accelsim_trace.py::e2e_main --model Qwen/Qwen2.5-0.5B --timing-only
```

The runner collects `meta.model_config` from the loaded model, not a preset: architecture,
num_layers, hidden_size, num_heads, num_kv_heads, head_dim, dtype, tensor_parallel_size, quantization.
vLLM paths checked: v1/engine/llm_engine.py:61-62 (`engine.model_config`), config/model.py:164-167
(`hf_config`, `hf_text_config`), and engine.vllm_config.parallel_config. HF text config supplies
GQA's num_key_value_heads and head_dim (or hidden_size/num_attention_heads). Missing fields remain
null; model-specific assumptions are not substituted. The standalone **OPT-only** kernel_microbench.py
is intentionally outside the installable package and is not a generic model benchmark.

simulate_accelsim defaults gpu_config=None: measured nvidia-smi names H100/H200/A100/RTX3070/RTX2060
map to existing tested configs; unknown names require an explicit gpu_config. Provenance records
the measured name, chosen config, and whether the selection was explicit or mapped. An explicit
override is the caller's responsibility (including precision, memory-system and clock validity).

`trace_plan(workload, context_lengths=(1,8,16,32), baseline_steps=..., iterations=..., repeat=1)`
returns (repeat,step_in_repeat) pairs: prefill plus available decode contexts. It requires baseline
inputs; corrected steps additionally expose nullable `context_length` derived from those inputs.
Multi-request prompt lengths require uniform per-request token counts; aggregate averages alone
are not treated as a known context length. The helper requires baseline
inputs because prompt strings are **not** token counts. ctx_tokens/ctx_reqs comes from baseline
iteration logs (raw scheduled-token fields are a fallback); decode k has context prompt_tokens+k.
Unavailable target lengths are skipped. Plans require one homogeneous, non-chunked prefill per
repeat; ambiguous/mixed/chunked schedules error rather than guessing. Use `profile(...,
trace_steps="plan")`, or `trace_plan=(1,8,16,32)`, to compute after baseline and before tracing.
The resolved selection is persisted in run.json. Planning does not require Modal.

`simulate_accelsim(..., dedup=True, verify_dedup=0)` groups globally across selected traced steps by
mangled name/grid/block plus cheap stable header fields. tracez has a null-terminated plain-text
header before its compressed instruction data (post-traces-processing.cpp:394-406); .trace/.trace.xz
are text headers. shmem, nregs, binary/tracer versions, cluster dimensions and any explicit shape/args
fields are included; kernel IDs, stream IDs and base addresses are excluded. Standard headers do
**not** expose complete kernel arguments, so this is an assumption about repeated instruction traces,
not proof of equivalence. Unknown/missing headers disable dedup for that occurrence.

One representative runs per group, with cycles fanned out. Each duration record's `sim` is `exact`
(own run) or `dedup:<uid>`. verify_dedup=N runs up to N additional group members and uses their own
cycles, reporting per-group and overall maximum absolute/relative differences. A zero reference with
nonzero difference is separately flagged. Nonzero verification differences do not silently disable
dedup: choose dedup=False for independent simulations. Summary/provenance include unique runs vs
traced occurrences, verification details, and counts for every provenance label.

propagate="none" is default; "identical" labels assignments `identical:<representative uid>` (median
for ambiguous same-key cycles, detailed in provenance). "interp" additionally detects varying cycles
by traced decode **seq/name/grid/block**, with no kernel-name allowlists. It linearly interpolates
between nearest traced context lengths, clamps outside their range, and labels `interp:<uidA>,<uidB>`
or `interp:<uid>,<uid>:clamped`. Targets with different dimensions, names, seq, or unavailable contexts
do not get interpolated. Unmatched untraced kernels stay absent and are counted; the simulator reader
falls back to measured durations. Use dedup=False or verification when probing context dependence:
unverified dedup can otherwise hide variation. These labels are extra fields accepted by the simulator.

All commands run from the repo root. Without `--s3-*` flags, `trace_vllm` writes the whole run
dir (raw traces, `kernelslist.g` + `.tracez` from post-traces-processing, `metadata.json`,
`trace_steps.json`, logs) to the `accelsim-traces` Modal volume at `/<run_name>/`, and
`nsys_main --trace-run <run_name>` reads it from there. The run name is printed at the end (and is
`<timestamp>-<random hex>__<model>__<scope>__<exec_mode>`; `modal volume ls accelsim-traces` lists them).
All automatic names include a six-hex-digit random suffix, including phase/microbench names
and nsys output folders (`nsys_<mode>_<timestamp>-<random hex>`). Explicit names are unchanged.
Add `--no-post-process` to skip post-processing, or `--s3-endpoint/--s3-bucket` to stream to S3
instead (no post-processing; then copy the run dir to the volume yourself before `nsys_main`).
Defaults are unchanged: `modal run accelsim_trace.py::main` still traces one layer in eager mode.

## 1. Whole-model trace of complete engine steps (eager)

Traces every kernel of every engine step (step 0 = prefill, 1..3 = decode for `--max-tokens 4`),
including input prep, logits and sampler kernels. Startup/profiling/warmup kernels (inside
`LLM()`) are never traced.

```
modal run accelsim_trace.py::main --trace-scope steps --exec-mode eager
# only some steps, e.g. prefill + first decode:  --trace-steps 0,1
```

```
modal run accelsim_trace.py::nsys_main --trace-run <run_name>
```

## 2. GPU-bound case (CUDA graphs)

NVBit can't trace CUDA-graph replays properly: kernels are only instrumented when they are
captured (vLLM captures inside `LLM()`, while tracing is off) and replayed kernels get no
tracer launch ids. So trace the same torch.compile'd (Triton) kernels with graphs off, and
time them with graphs on:

```
modal run accelsim_trace.py::main --trace-scope steps --exec-mode compile
modal run accelsim_trace.py::nsys_main --trace-run <compile_run_name>                     # -> nsys/
modal run accelsim_trace.py::nsys_main --trace-run <compile_run_name> --exec-mode graphs  # -> nsys_graphs/
```

Grid dims in `graphs` may differ from the trace (vLLM pads batches to captured sizes); kernels
are matched by name and differences are flagged in `launch_dims_match`.

## 3. Launch-overhead microbenchmark

N back-to-back launches of small opt-125m-shaped kernels (fill, copy, add, layer_norm, relu,
GEMMs, reshape_and_cache_flash), eager vs one CUDA-graph replay:

```
modal run accelsim_trace.py::microbench_main --n 200 --tokens 1,6
```

## Optional flags for `nsys_main`

- `--lock-clocks-mhz 1980` tries `nvidia-smi -lgc 1980,1980` (probably refused in Modal; the result is in `summary.json` under `clock_lock`).
- `--gpu-metrics` adds nsys GPU metrics sampling (may need privileges).
- `--out-name <dir>` sets the output folder name.

## vLLM-native baseline and steady-state profiling

```
modal run accelsim_trace.py::phases_main --max-tokens 32 --repeats 4
```

Phase A runs without a profiler or NVBit; Phase B runs sequentially on the same GPU under
light nsys capture. Both use identical prompts, greedy sampling and synchronous scheduling.
Repeat 0 is warmup; only repeats 1 onward are captured in B via cudaProfilerStart/Stop.
Layerwise NVTX is **off by default**: opt in with `--layerwise-nvtx` (B, eager mode only).
On the H100 test it doubled step time (~110% overhead versus ~25% without it).
Use `--phase baseline` or
`--phase profiled` to run one side (unavailable reconciliation columns are null).
NVBit runs are for instruction traces only, **never timing**.

Outputs live at `/phases/<run_name>/` on `accelsim-traces`:

| File | Contents |
|---|---|
| `baseline_iterations.jsonl`, `profiled_iterations.jsonl` | Parsed vLLM iteration counts/timing joined by global step index to repeat, warmup, wall timestamps/span and prefill/decode/mixed kind |
| `reconcile.json`, `reconcile.csv` | Non-warmup prefill/decode distributions (n/mean/p50/p95), B−A mean/p50 overhead in ns and %, iteration/step mismatch flags in JSON |
| `corrected_timeline.json` | Simulator input combining baseline wall time and request latency with matched profiled GPU/API intervals and kernel durations; only written when both phases run |
| `baseline/baseline.log`, `profiled/profiled.log` | Full subprocess logs |
| `<phase>/steps.json` | StepRecorder metadata for every repeat, including uncaptured warmup |
| `<phase>/metrics.json` | Per-request RequestStateStats plus derived latencies and final cumulative `llm.get_metrics()` snapshot (histogram count/sum/buckets included) |
| `<phase>/clocks.csv` | NVML samples for each phase; summary/error in reconcile.json |
| `profiled/profile.sqlite` | Scrubbed nsys export |
| `profiled/kernels.csv`, `events.csv`, `api_calls.csv`, `steps.csv` | Kernel, GPU activity, runtime API and per-step exports (timestamps in ns; activity/API exports use absolute nsys timeline) |
| `profiled/nvtx.csv` | All NVTX events, including layerwise ranges/shapes, with start/end/text/domain ID/pid/tid |

Derived request metrics use monotonic timestamps only: `queue_s = scheduled_ts - queued_ts`,
`ttft_s = first_token_ts - scheduled_ts`, `queued_ttft_s = first_token_ts - queued_ts`,
`decode_s = last_token_ts - first_token_ts`, and `mean_itl_s = decode_s / (num_generation_tokens - 1)`
(null for fewer than two generated tokens). Arrival time is wall-clock and is never subtracted
from monotonic timestamps. Metrics snapshots include warmup. Quantiles use linear interpolation.
Reconciliation keeps vLLM elapsed time in ms, but converts its overhead delta to ns.
**vLLM iteration `elapsed_ms` is not step time**: its timer starts after launch (roughly
0.2 ms in the real run). StepRecorder's synchronized wall span is the step-time ground truth;
elapsed_ms remains only a diagnostic and is never used in the corrected timeline.
GPU busy is the clipped union of kernel/memcpy/memset intervals; CUDA API time is the clipped
union of runtime calls on the NVTX step's thread. Other host time is NVTX span minus API time,
not pure CPU compute (it includes waits). Steps outside capture are excluded from nsys columns.
Iteration counts/indices are checked against StepRecorder; inspect mismatch flags before
trusting joins. Mixed steps are exported but excluded from the prefill/decode comparison.

### Corrected timeline schema

```
meta: run_name, model, prompt, max_tokens, repeats, exec_mode, gpu_name,
      sm_clock_mhz_median, mem_clock_mhz, vllm_version, layerwise_nvtx, clock
steps[]: step, repeat, warmup, kind, num_reqs, num_scheduled_tokens,
         baseline_wall_ns, profiled_wall_ns, gpu_busy_ns, cuda_api_ns, host_ns,
         baseline_wall_start_ns, baseline_wall_end_ns, baseline_gap_before_ns,
         profiled_wall_start_ns, profiled_wall_end_ns, profiled_gap_before_ns,
         kernels: [{seq, name, grid, block, duration_ns, start_offset_ns}]
requests[]: repeat, warmup, queued_ms, ttft_ms, itl_ms_mean, itl_ms_p50,
            e2e_ms, num_output_tokens, first_step_start_ns, last_step_end_ns,
            arrival_ns, queued_ns, first_token_ns, last_token_ns,
            pre_first_step_ns, post_last_step_ns,
            arrival_to_first_step_ns, last_step_end_to_return_ns
summary[kind]: n, baseline_wall_ns: {mean, p50, p95}, gpu_busy_ns: {mean, p50},
               host_ns: {mean, p50}, profiler_overhead_pct,
               baseline_gap_before_ns: {mean, p50, p95},
               profiled_gap_before_ns: {mean, p50, p95}
```

Steps come from the baseline records; match to B by `(repeat, step-in-repeat)`, where
step-in-repeat is the ordinal of the step records within each repeat, not a fixed token count.
`step` stays the baseline global index. Request/token counts sum baseline iteration context
and generation counts, falling back to step metadata when the iteration is missing.
`baseline_wall_ns` is the StepRecorder wall span. `profiled_wall_ns` is the matched nsys
NVTX step span; GPU/API intervals are clipped and unioned within that range.
`host_ns = baseline_wall_ns - gpu_busy_ns` estimates corrected non-GPU time, not pure CPU
compute; it is not clamped if noisy measurements make it negative. Uncaptured/unmatched
steps have null profiled/GPU/API/host values and no kernels (including repeat-0 warmup).
Kernels are selected by GPU start inside the matched NVTX range, sorted by GPU start, with
zero-based per-step `seq`; `name` is the mangled name and grid/block are `[x,y,z]` lists.
`start_offset_ns` is relative to the **profiled step's NVTX range start**, not baseline time.

Requests use baseline RequestStateStats: `queued_ms = (scheduled_ts - queued_ts) * 1000`,
`ttft_ms = (first_token_ts - queued_ts) * 1000` (the metrics JSON's `queued_ttft_s`, not
`ttft_s`), and `e2e_ms = (last_token_ts - queued_ts) * 1000`. Per-token timestamps are not
recorded: `itl_ms_mean = (last_token_ts - first_token_ts) * 1000 / (num_generation_tokens - 1)`
and `itl_ms_p50` is null. Mean ITL is also null for fewer than two tokens. Missing request
metrics produce null latency/token fields. Repeat 0 marks warmup for requests and steps.

Summary `n` counts all non-warmup baseline steps of that kind, as do baseline wall statistics;
GPU/host statistics use only matched captured steps. Overhead compares profiled and baseline
wall p50 over the same matched non-warmup steps: `(B_p50 - A_p50) / A_p50 * 100`.
Empty statistics are null. Clock metadata uses medians of all profiled-phase ClockSampler
samples (both SM and memory; null if sampling failed), and vLLM version comes from container
package metadata. Single-phase runs print a skip note and do not write this artifact.

### What's between steps

The new absolute `*_wall_start_ns` / `*_wall_end_ns` fields use **CLOCK_MONOTONIC**, despite
the historical `wall` name. StepRecorder retains its original `time.time_ns` fields and spans
unchanged and additionally records `monotonic_start_ns` / `monotonic_end_ns` in steps.json.
`meta.clock` records `step_clock="monotonic"`, baseline `wall_minus_monotonic_ns`, sampling
`how` and `sample_bracket_ns`, plus `profiled_wall_minus_monotonic_ns`. Each workload samples
wall time between two monotonic reads once and uses their midpoint for the offset. Raw clock
metadata is also in each phase's metrics.json. Request core timestamps are monotonic seconds
converted to ns; `arrival_ns = round(arrival_time * 1e9) - baseline offset` converts wall time.
Linux CLOCK_MONOTONIC is comparable across processes in one container (not across hosts).
vLLM defaults to a separate EngineCore process (`vllm/envs.py:1112-1114`), but these phase runs
force multiprocessing off (`cascade_profiler.nsys::_nsys_env`).

Gaps are this step's monotonic start minus the preceding step's end **within the same repeat**;
the first gap is null. The profiled timestamp/gap fields come from B's step records even when
outside capture, unlike the existing `profiled_wall_ns` NVTX-span field, which stays null for
uncaptured steps. A missing partner yields null new B fields. Summary gaps use non-warmup,
non-null gaps, attributed to the kind of the following step, independent of capture availability.
Legacy records can be converted if their run offset is provided; without either monotonic
timestamps or an offset, the new absolute times and gaps are null rather than mixed clocks.

Request step endpoints use the first/last baseline step of that repeat (a batch envelope,
not individual request membership). `pre_first_step_ns = first_step_start_ns - queued_ns` and
`post_last_step_ns = last_token_ns - last_step_end_ns` align with reference metrics:
`ttft_ms = (first_token_ns - queued_ns) / 1e6` and
`e2e_ms = (last_token_ns - queued_ns) / 1e6`. Post-step time may be negative because the last
token is stamped inside the final step; the sign is preserved. Pre-step time plus all baseline
step walls, intervening gaps, and post-step time reconstructs e2e (subject to timestamp rounding
and any wall-clock adjustment during steps). Pre-step time plus the first step wall approximates
TTFT; real TTFT can differ by the first token's in-step stamp offset. The offline TTFT fixture
places that timestamp exactly at first-step end and checks reconstruction within 0.1 ms.

The previous endpoints remain available as `arrival_to_first_step_ns = first_step_start_ns -
arrival_ns` and `last_step_end_to_return_ns = generate_return_monotonic_ns - last_step_end_ns`.
The latter uses the monotonic timestamp immediately after generate returns; for multiple
requests it is the batch-return upper bound, not each request's precise completion time.
Missing timestamps yield null endpoint fields. Existing reference latency metrics are unchanged.

Code-path findings in vLLM 0.21.0 (paths relative to the unpacked vLLM source):

- StepRecorder wraps **LLMEngine.step**, not just EngineCore.step. In-process get_output calls
  EngineCore.step_fn and post_step (`v1/engine/core_client.py:287-290`). Scheduler execution and
  `update_from_output` are **inside** our span (`v1/engine/core.py:406-435`).
- Output processing, incremental detokenization, output construction and stats logging are
  also **inside** (`v1/engine/llm_engine.py:295-323`, `v1/engine/output_processor.py:633-709`,
  `v1/engine/detokenizer.py:95-135`); they cannot be assigned again to the inter-step gap.
- Our pre-step device synchronize is **outside** the span; post-step synchronize and NVTX
  pushes/pops are **inside**. Record assembly plus writing the entire growing, indented step
  JSON list used to happen **after** the end timestamp on every step. That growing rewrite is
  now removed: records append to memory and the same JSON array is written once at finalization
  (`cascade_profiler.runner::StepRecorder.step/close/write`).
- `LLM._run_engine` does loop checks and output collection outside the wrapper, and tqdm setup,
  updates/refresh, closing and final sorting (`entrypoints/llm.py:1839-1892`). However `_add_request`
  sets FINAL_ONLY (`1826-1828`): progress updates are on completed outputs, **not each decode
  token** here, so they cannot alone explain every decode gap. Prompt rendering/tokenization and
  adding requests happen before the first step (`1592-1651`, `1789-1837`); some rendering precedes
  recorded arrival (`v1/engine/input_processor.py:269-281`).
- The default is two extra device synchronizations per step (64 for 32 steps), plus unchanged
  initialization sync. Synchronous non-speculative sampling already copies generated IDs to CPU
  and waits on its transfer event (`v1/worker/gpu_model_runner.py:3466-3472`, `7090-7103`), so
  additional device syncs need not wait for all forward work again, but may wait for other
  streams and have API overhead. Their real cost is **not measured by code inspection**.

`steps.json` now includes `pre_step_sync_ns` and `post_step_sync_ns` measured around our calls
(host duration including waits). To measure the intervention, use
`modal run accelsim_trace.py::phases_main --no-step-sync` (also available directly in
vllm_trace.py). The default remains synchronized. With sync disabled, asynchronous work may
cross range boundaries: treat GPU attribution and host subtraction with caution. No numeric
attribution of the reported 34–57 ms deficit is claimed without a fresh instrumented run;
optional gap-specific CUDA/NVTX attribution is not exported in this change.

### Growing gaps and GC diagnostics

The old `StepRecorder.step()` called `write()` after every appended record; `write()` did
`json.dumps(self.steps, indent=2)` and `Path.write_text` of the entire array. This was O(n)
serialization/I/O per step, O(N²) cumulative work across repeats, outside measured spans.
Both phases set the step-log path, so that path was reachable. It is a concrete mechanism for
growing gaps, but its share of the observed stall was not measured. Finalization now serializes
once, O(N) total, with amortized O(1) buffering per step. A try/finally flushes completed records
on Python exceptions; an idempotent atexit fallback handles normal interpreter exit. JSON-array
format is unchanged. Abrupt SIGKILL/container loss cannot flush memory. Final file writing is
after generate-return timestamp collection, not inside request envelopes.

**`--gc-between-repeats` defaults ON**, in both vllm_trace.py and phases_main/profile_phases.
It calls `gc.collect()` before repeats 1 onward, before generate and outside any step span;
first-step gaps remain null, so collection is not charged to the next repeat's decode gaps.
Use `--no-gc-between-repeats` for an A/B comparison. Step synchronization stays ON by default;
the reported sync/no-sync runs found no difference. Explicit collection may change GC timing
but does not disable automatic GC or remove live recorded steps.

Each raw step adds `gc_before` and `gc_after`, each containing `count` (`gc.get_count()`) and
`collections` (cumulative collection counts per generation from `gc.get_stats()`). GC callbacks
sample monotonic start/stop timestamps and maintain a cumulative pause counter without rescanning
history. `gc_pause_ns_in_step` is the counter delta across the span; `gc_pause_ns_in_gap_before`
is the delta since the preceding step end in the same repeat (null for the first step).
The corrected timeline passes these through as `baseline_gc_before`, `baseline_gc_after`,
`baseline_gc_pause_ns_in_step`, `baseline_gc_pause_ns_in_gap_before` and the corresponding
`profiled_*` fields. Legacy/missing records yield null. These are process-local diagnostics
(appropriate here because EngineCore multiprocessing is disabled).

Each phase's metrics.json adds `gc = {between_repeats, repeats, events}`. Events contain
`generation`, `start_ns`, `end_ns`, `duration_ns`. Each completed repeat has `repeat`,
`gc_collect_before_ns` (host time around explicit collection, zero when not requested) and
`total_gc_pause_ns` (callback pauses during explicit collection, generate, and metrics copying).
The corrected timeline adds top-level `gc_summary = {baseline: [...], profiled: [...]}` with
these per-repeat summaries, including warmup. The repeat total can exceed step/gap-attributed
pauses because it includes collection and work outside step envelopes. Diagnostics themselves
allocate small fixed-size snapshots; callback event and step buffers append without rewriting.

vLLM growth audit (0.21.0 source; not an exhaustive memory-leak audit):

- Completed frontend requests are removed from request state/ID maps
  (`v1/engine/output_processor.py:714-727`); scheduler requests are freed/deleted
  (`v1/core/sched/scheduler.py:1750-1771`), finished IDs reset on schedule (`954-956`), and
  per-client finished-ID maps clear after export (`1487-1499`). No unbounded finished-request
  list scanned across these generate calls was found.
- `_run_engine` creates a fresh local outputs list for each generate and sorts only that batch
  (`entrypoints/llm.py:1857-1892`). The request counter (`387,1829`) and iteration counter
  (`v1/engine/core.py:381-404`) increase, but are scalar counters, not growing history scans.
- IterationStats lists are new each frontend step (`v1/engine/llm_engine.py:299`,
  `v1/metrics/stats.py:325-340`). Logging counters reset per reporting interval
  (`v1/metrics/loggers.py:130-148,196-201`); cache metrics retain a bounded recent-request deque
  (`v1/metrics/stats.py:35-95`). Prometheus is installed once in the logger manager
  (`v1/metrics/loggers.py:1290-1330`), with fixed bucket histogram objects updated by observe
  (`1163-1215`), not a new histogram/raw observation list per generate. Counts and cache state
  persist, but this does not establish increasing loop cost for four identical single requests.
- EngineCore freezes initialization objects (`v1/engine/core.py:224-226`,
  `utils/gc_utils.py:106-118`), reducing static-heap scanning but leaving later allocations
  collectable. A gen-2 pass over new live/garbage objects remains possible. GC callbacks now
  distinguish such pauses from ordinary host time; there is no evidence yet that it caused
  the reported 700 ms stall.

Stall candidates: the removed full-file rewrite was definitely reachable in gaps. Automatic
GC is plausible but requires callback evidence. Logger backpressure/flush is reachable, but
iteration/stat logging happens **inside** LLMEngine.step; tqdm final-only updates occur at
request completion, not every decode gap. CUDA-profiler start is lazy at repeat 1 but occurs
before its first step, whose gap is null; model initialization is before all repeats. These
cannot directly explain a decode-gap p95 spike on their own. No exact cause of the 700 ms event
is claimed without the new diagnostics or an nsys/Python host trace from another run.

All nsys entrypoints strip MODAL and secret-named environment variables (except the two HF
token variables required by gated models), remove environment rows from sqlite and VACUUM.
**`.nsys-rep` is no longer uploaded by default** because it retains the unscrubbed environment.
The remote functions accept `keep_nsys_rep=True` to opt in with a warning; `phases_main` also
exposes `--keep-nsys-rep`. Treat retained reports as secret-bearing files.

```
modal volume get accelsim-traces phases/<run_name> ./phases
```

## Download existing trace runs

```
modal volume get accelsim-traces <run_name> ./<run_name>        # trace dir (incl. nsys*/ once profiled)
modal volume get accelsim-traces <run_name>/nsys ./nsys
modal volume get accelsim-traces <run_name>/nsys_graphs ./nsys_graphs
modal volume get accelsim-traces microbench/<name> ./microbench
```

## Outputs

Trace run: `trace_steps.json` lists, for each engine step: kind (prefill/decode),
scheduled tokens, wall-clock start/end, and the tracer kernel-id range `[first_kernel_id,
end_kernel_id)`. Async scheduling is turned off for new runs so that each engine step is
exactly one forward pass.

nsys run (`<run>/nsys*/`):

| File | Contents |
|---|---|
| `kernels.csv` | Per kernel: `trace_kernel_id`, `step`, `step_kind`, GPU start/end/duration, `api_name`, `api_start_ns`/`api_end_ns`, `launch_to_start_ns`, `correlation_id`, `pid`/`tid`, stream, graph node, dims |
| `events.csv` | Kernels, memcpys (bytes, copy/src/dst kind), memsets and syncs (type, event id), each with its API call and step |
| `api_calls.csv` | Every CUDA runtime API call, with its step |
| `steps.csv` | Per step: NVTX span, GPU busy/idle, kernel and memcpy counts, tracer kernel range, `matched_in_step`, `exact_step_match` |
| `clocks.csv` | NVML SM/mem clocks and throttle reasons every 50 ms (`wall_ns` and `nsys_ns`) |
| `clocks_before.txt`, `clocks_after.txt` | `nvidia-smi -q` clock snapshots |
| `summary.json` | Global and per-step alignment, clock stats during the steps, nsys flags used |

NVTX ranges in both runs: `vllm_init` (`LLM()` construction), `vllm_step_<i>`, plus a nested
`prefill_<i>` / `decode_<i>` range.
