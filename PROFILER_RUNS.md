# Profiler runs for sim-to-real calibration

All commands run from the repo root. `trace_vllm` streams traces to S3 (same `--s3-*` flags as
before); `profile_nsys` reads the trace run directory (`metadata.json`, `traces/stats_ctx_*`,
`trace_steps.json`) from the `accelsim-traces` Modal volume, as with the existing run.
Defaults are unchanged: `modal run accelsim_trace.py` still traces one layer in eager mode.

## 1. Whole-model trace of complete engine steps (eager)

Traces every kernel of every engine step (step 0 = prefill, 1..3 = decode for `--max-tokens 4`),
including input prep, logits and sampler kernels. Startup/profiling/warmup kernels (inside
`LLM()`) are never traced.

```
modal run accelsim_trace.py --trace-scope steps --exec-mode eager \
    --s3-endpoint <endpoint> --s3-bucket <bucket>
# only some steps, e.g. prefill + first decode:  --trace-steps 0,1
```

Then (after the run dir is on the volume):

```
modal run accelsim_trace.py::nsys_main --trace-run <run_name>
```

## 2. GPU-bound case (CUDA graphs)

NVBit can't trace CUDA-graph replays properly: kernels are only instrumented when they are
captured (vLLM captures inside `LLM()`, while tracing is off) and replayed kernels get no
tracer launch ids. So trace the same torch.compile'd (Triton) kernels with graphs off, and
time them with graphs on:

```
modal run accelsim_trace.py --trace-scope steps --exec-mode compile \
    --s3-endpoint <endpoint> --s3-bucket <bucket>
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

## Download

```
modal volume get accelsim-traces <run_name>/nsys ./nsys
modal volume get accelsim-traces <run_name>/nsys_graphs ./nsys_graphs
modal volume get accelsim-traces microbench/<name> ./microbench
```

## Outputs

Trace run (S3): `trace_steps.json` lists, for each engine step: kind (prefill/decode),
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
