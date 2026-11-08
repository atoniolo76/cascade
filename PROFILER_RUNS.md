# Profiler runs for sim-to-real calibration

All commands run from the repo root. Without `--s3-*` flags, `trace_vllm` writes the whole run
dir (raw traces, `kernelslist.g` + `.tracez` from post-traces-processing, `metadata.json`,
`trace_steps.json`, logs) to the `accelsim-traces` Modal volume at `/<run_name>/`, and
`nsys_main --trace-run <run_name>` reads it from there. The run name is printed at the end (and is
`<timestamp>__<model>__<scope>__<exec_mode>`; `modal volume ls accelsim-traces` lists them).
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
            pre_first_step_ns, post_last_step_ns
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
force multiprocessing off (`accelsim_trace.py::_nsys_env`).

Gaps are this step's monotonic start minus the preceding step's end **within the same repeat**;
the first gap is null. The profiled timestamp/gap fields come from B's step records even when
outside capture, unlike the existing `profiled_wall_ns` NVTX-span field, which stays null for
uncaptured steps. A missing partner yields null new B fields. Summary gaps use non-warmup,
non-null gaps, attributed to the kind of the following step, independent of capture availability.
Legacy records can be converted if their run offset is provided; without either monotonic
timestamps or an offset, the new absolute times and gaps are null rather than mixed clocks.

Request step endpoints use the first/last baseline step of that repeat (a batch envelope,
not individual request membership). `pre_first_step_ns = first_step_start_ns - arrival_ns`.
RequestStateStats and RequestOutput have no finish timestamp; each request's raw metrics row
now records `generate_return_monotonic_ns`, sampled immediately after that repeat's generate
returns. `post_last_step_ns` subtracts the final step end from this return timestamp; for multiple
requests this is the batch-return upper bound, not each request's precise completion time.
Existing `e2e_ms` remains last-token minus queued time and is deliberately unchanged.

Code-path findings in vLLM 0.21.0 (paths relative to the unpacked vLLM source):

- StepRecorder wraps **LLMEngine.step**, not just EngineCore.step. In-process get_output calls
  EngineCore.step_fn and post_step (`v1/engine/core_client.py:287-290`). Scheduler execution and
  `update_from_output` are **inside** our span (`v1/engine/core.py:406-435`).
- Output processing, incremental detokenization, output construction and stats logging are
  also **inside** (`v1/engine/llm_engine.py:295-323`, `v1/engine/output_processor.py:633-709`,
  `v1/engine/detokenizer.py:95-135`); they cannot be assigned again to the inter-step gap.
- Our pre-step device synchronize is **outside** the span; post-step synchronize and NVTX
  pushes/pops are **inside**. Record assembly plus writing the entire growing, indented step
  JSON list happen **after** the end timestamp on every step (`vllm_trace.py::StepRecorder.step/write`).
  This logging is reachable in both phases and adds gap/post-step overhead, not model time.
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
