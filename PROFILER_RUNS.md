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
Layerwise NVTX is enabled only in B and only in eager mode. Use `--phase baseline` or
`--phase profiled` to run one side (unavailable reconciliation columns are null).
NVBit runs are for instruction traces only, **never timing**.

Outputs live at `/phases/<run_name>/` on `accelsim-traces`:

| File | Contents |
|---|---|
| `baseline_iterations.jsonl`, `profiled_iterations.jsonl` | Parsed vLLM iteration counts/timing joined by global step index to repeat, warmup, wall timestamps/span and prefill/decode/mixed kind |
| `reconcile.json`, `reconcile.csv` | Non-warmup prefill/decode distributions (n/mean/p50/p95), B−A mean/p50 overhead in ns and %, iteration/step mismatch flags in JSON |
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
GPU busy is the clipped union of kernel/memcpy/memset intervals; CUDA API time is the clipped
union of runtime calls on the NVTX step's thread. Other host time is NVTX span minus API time,
not pure CPU compute (it includes waits). Steps outside capture are excluded from nsys columns.
Iteration counts/indices are checked against StepRecorder; inspect mismatch flags before
trusting joins. Mixed steps are exported but excluded from the prefill/decode comparison.

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
