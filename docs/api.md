# Cascade app API

Types live in [cascade/models.py](../cascade/models.py) and are mirrored in [web/src/api/types.ts](../web/src/api/types.ts). Decisions behind this contract are in [cascade-app-decisions.md](cascade-app-decisions.md).

All `/api` routes require `Authorization: Bearer $CASCADE_TOKEN`, except `GET /api/health`. EventSource cannot send headers, so the SSE route also accepts `?token=`.

## UI and reads

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/api/health` | `{"ok": true, "auth": bool, "workspace": str | null, "s3_endpoint": str | null}`. `s3_endpoint` is the S3 endpoint the profiler should upload to, when the app knows it. `auth` is false when the app runs with `CASCADE_NO_AUTH=1` (local dev only), and the UI then skips the token screen. |
| GET | `/api/gpu-configs` | `GpuConfig[]` |
| GET | `/api/traces` | `TraceSummary[]`, newest first |
| GET | `/api/traces/{id}` | `TraceDetail` |
| GET | `/api/runs` | `RunSummary[]`, newest first |
| POST | `/api/runs` | body `RunCreate`, returns `RunDetail` (201). 409 if the trace is not `ready`, or is a sample (it has no instruction traces). 422 if a config is unknown or its arch differs from the trace's `trace_gpu_arch`. |
| GET | `/api/runs/{id}` | `RunDetail` |
| POST | `/api/runs/{id}/cancel` | `RunDetail`. Moves every non-terminal sim to `cancelled` and calls `Launcher.terminate`. Idempotent. On a sample run, whose sim already `succeeded`, it changes nothing. |
| GET | `/api/runs/{id}/events` | SSE stream of `RunEvent` as plain `data:` lines. First event is a `snapshot` with the run and every `KernelStat` so far, each with an empty `stats` map to keep it small. Full stats are in the results downloads and in live `kernel` events. |
| GET | `/api/runs/{id}/comparison` | `Comparison` |
| GET | `/api/runs/{id}/kernels` | `KernelStat[]` for every sim in the run, each with an empty `stats` map, ordered by config, replica, then kernel. Feeds the timeline. |
| GET | `/api/runs/{id}/step-timelines` | `SimStepTimeline[]`, one per sim that posted step timing, ordered by config then replica. Empty until a sim posts one. |
| GET | `/api/runs/{id}/results` | `RunResults` (download) |
| GET | `/api/sims/{id}/results` | `SimResults` (download) |
| GET | `/api/sims/{id}/log` | full raw log, `text/plain` (download) |
| GET | `/api/sims/{id}/files` | `SimFile[]`, the sim's uploaded output files, by name |
| GET | `/api/sims/{id}/files/{name}` | the file as an attachment, with its `content_type`. A `.gz` file is `application/gzip` with no `Content-Encoding`, so it downloads compressed. 404 if the sim has no such file. |
| GET | `/api/runs/{id}/files` | `RunFile[]` (a `SimFile` plus `sim_id`, `gpu_config`, `replica`), ordered by config, replica, then name |
| GET | `/api/runs/{id}/perfetto` | Chrome trace JSON (download) built from the run's parsed kernel stats, for ui.perfetto.dev. One process per config (`<config> · Accel-Sim simulated GPU`), one thread per replica (`replica <n> · <sim id>`), and one `X` slice per kernel named `k<uid> <readable kernel name>`, laid end to end: `dur` is `sim_time_ns / 1000` µs. `args` hold `kernel_name` (as recorded), `kernel_index`, `uid`, `gpu_sim_cycle`, `gpu_sim_insn`, `ipc`, `sim_time_ns` and cumulative `start_cycle`/`end_cycle`. When the trace has real timing, a `Real hardware · <source_gpu>` process holds the measured kernels the same way. Kernels without `sim_time_ns` are left out. |

## OpenAI-compatible simulation

`GET /v1/models`, `POST /v1/completions` and `POST /v1/chat/completions` use the same
bearer `CASCADE_TOKEN` authentication as `/api` (or `CASCADE_NO_AUTH=1` for local dev).
These routes run **performance models, not model inference**. Timing estimates the
requested prompt/output shape from a cached base simulation. **Each `/v1` request
is simulated alone: concurrent HTTP calls do not batch together.** For continuous
batching, use `/api/scenarios` below.

`/v1/models` returns an OpenAI `list` of ready traces, with trace IDs as model IDs,
`owned_by: "cascade"` and `metadata.model` containing the original model name. A
request's `model` may be that trace ID or the model name; the latter selects the
newest ready trace for that model. Unknown/not-ready models return 404.

```python
import os
from openai import OpenAI

client = OpenAI(base_url="https://your-cascade-app/v1", api_key=os.environ["CASCADE_TOKEN"])
response = client.chat.completions.create(
    model="facebook/opt-125m",  # or an exact ready trace ID
    messages=[{"role": "user", "content": "Hello, my name is"}],
    max_completion_tokens=32,
    seed=42,
    extra_body={"gpu_config": "SM90_H100", "fresh": False},
)
print(response.choices[0].message.content)
print(response.model_extra["cascade"])
```

- `gpu_config` defaults to the trace's `suggested_sim_config`; specify it if the
  trace has no suggestion. Unknown configs or trace/config architecture mismatches
  return 422, with the same checks as `POST /api/runs`. Request sizing currently
  supports `SM90_H100`, `SM90_H200` and `SM80_A100`; other targets or missing model
  metadata return 422 rather than inventing hardware/performance specifications.
- `fresh: false` reuses the newest successful simulation for that exact trace and
  config **only if it posted step timing**. Otherwise it attaches to a matching
  pending/launching/running simulation, or creates a normal one-sim run through
  the configured launcher. Concurrent callers share the new run. All runs appear
  in the existing UI. `fresh: true` creates a new base run. Bundled sample traces
  have no instruction traces: they reuse recorded simulations only and reject
  `fresh: true` with 422. Built performance profiles are cached in memory by sim ID
  (up to 128 profiles); requested scenarios themselves are not cached or persisted.
- Non-streaming calls wait until the simulation is terminal. Set
  `CASCADE_OPENAI_WAIT_S` (default 3600 seconds) to change the wait budget. A timeout
  returns 504 with `error.run_id` and `error.sim_id`; it does not cancel the run.
  Poll `/api/runs/{run_id}`. Failed/cancelled simulations return 502 with their
  error; success without posted step timing also returns 502.
- Output count is `max_tokens` for completions, `max_completion_tokens` or
  `max_tokens` for chat, defaulting to the trace workload's `max_tokens`. Supplying
  conflicting chat counts returns 422. Output counts must be nonnegative, prompt
  counts positive, and their sum must fit the supplied run/meta `max_model_len`
  (2048 when absent); invalid shapes return 422 before launching a base run.
  A zero-output request is prefill-only, with null TTFT and ITL.
- Output IDs are sampled uniformly with replacement from the trace model's
  non-special vocabulary. `seed` makes sampling deterministic. The lightweight
  `tokenizers` library loads public HF `tokenizer.json`, cached under
  `$CASCADE_DATA_DIR/tokenizers/<sha256(model)>.json`. No model weights or remote
  code are loaded. Offline, unsupported or unavailable tokenizers fall back to
  seeded `tok<i>` placeholders; failed loads are remembered until server restart.
  `cascade.tokens_source` is `vocab_uniform` or `placeholder`.
- `usage.completion_tokens` counts sampled IDs (also available as
  `cascade.output_token_ids`). Decoding a random ID sequence and re-tokenizing it
  can change boundaries/counts; it is not a claim about re-encoded text length.
  `usage.prompt_tokens` is exactly the count used for timing. Chat applies the
  public `tokenizer_config.json` chat template when usable (sandboxed Jinja,
  generation markers supported; no duplicate special-token insertion). Otherwise
  it tokenizes newline-joined message content. Without a tokenizer, it estimates
  `max(1, ceil(UTF-8 bytes / 4))`. `cascade.prompt_tokens_source` distinguishes
  `tokenizer`, `tokenizer:chat_template`, `tokenizer:joined` and
  `estimated:utf8-bytes/4`. Configs are cached beside tokenizer JSON files as
  `<sha256(model)>-config.json`. `usage.total_tokens` is prompt plus output count.
- Standard choices contain sampled text, `index: 0`, and `finish_reason: "length"`.
  Exactly one choice is returned. Other generation options are ignored: this is
  not semantic text generation. Chat accepts text-only message content.
- `cascade` contains base identifiers `trace_id`, `gpu_config`, `run_id`, `sim_id`,
  `cached`; **`timing`** contains requested `prompt_tokens`, `output_tokens`,
  `ttft_ms`, `itl_ms_mean`, `itl_ms_p50`, `e2e_ms`, `provenance_counts`,
  `calibration_range` and `assumptions`. **`profiled`** separately contains the base
  workload's `simulated`/`real` steady-repeat metrics and `compare` summary or null.
  `profile_match` is true only when the known profiled prompt and output count
  match. `profile_match_details` and `note` explain the distinction. Chat prompt
  equality without a template is checked only for a single user message;
  with a template, the rendered prompt is compared. A match is not independent
  accuracy validation. `measured` means a matching simulated step, which may itself
  use deduplication/interpolation; `interpolated` is inside calibration and
  `extrapolated:roofline` is outside. Host timing always carries `assumed-host`.
- `stream: true` sends `text/event-stream`: periodic `: progress done/total state`
  comments while waiting, OpenAI content chunks in the final phase, a final chunk
  with usage and `cascade`, then `data: [DONE]`. This is not timing-paced inference.
  Errors after headers are sent appear as `data: {"error": ...}` before `[DONE]`;
  their HTTP status is necessarily 200. Pre-stream validation errors retain their
  4xx status. Every error uses `{"error":{"message":...,"type":...,"code":...}}`.

For step timing, upload `corrected_timeline.json` plus either the legacy
`kernel_mapping.json` or the profiler's `trace_alignment.json` in the trace root.
Alignment records map `tracer_uid` to `uid` and retain `name`, `repeat`, `step`, `seq`;
UID/name validation remains strict. When both exist the legacy mapping wins.
Late-arriving sidecars update `has_step_timeline` during ingest. Alignment alone
maps only traced kernels: it does not imply untraced-kernel propagation or full
coverage. The response's comparison/coverage remains that of the posted timeline.

## Transient request scenarios

`POST /api/scenarios` uses normal `/api` bearer auth and runs the discrete-event
single-GPU scheduler. It shares the exact cache/attach/launch/wait path and profile
cache with `/v1`. It does not persist a scenario or alter base profiled timelines.
Only a missing/fresh base simulation creates a normal UI run.

Provide exactly one of `trace_id` or `model` (newest ready trace), optional
`gpu_config` and `fresh`, and either explicit requests:

```json
{
  "trace_id": "sample__facebook_opt-125m__h100-e2e",
  "requests": [
    {"id": "a", "arrival_s": 0, "prompt_tokens": 512, "output_tokens": 128},
    {"id": "b", "arrival_s": 0.5, "prompt_tokens": 512, "output_tokens": 128}
  ],
  "scheduler": {"max_num_seqs": 256, "max_num_batched_tokens": 8192}
}
```

or shorthand (mutually exclusive with `requests`):

```json
{
  "trace_id": "sample__facebook_opt-125m__h100-e2e",
  "num_requests": 8,
  "prompt_tokens": 512,
  "output_tokens": 128,
  "arrival_rate_rps": 0
}
```

Shorthand arrivals are all zero for null/zero/omitted rate; otherwise `i / rate`
seconds, not a stochastic process. Omitted request IDs become `request-<index>`;
IDs must be unique. Arrival ties preserve input order. Arrival times are finite,
nonnegative and at most 1e9 seconds. Requests respect the profile's model limit.
Bodies reject unknown fields and invalid values with 422. Limits: 4096 requests,
1,000,000 total prompt/output tokens and 100,000 executed scheduler steps.
These are service safety caps, not vLLM limits.

Scheduler options (all defaults are explicitly labelled assumptions, not captured
vLLM resolved scheduler values): `max_num_seqs=256`, `max_num_batched_tokens=8192`,
`block_size=16`, `enable_chunked_prefill=true`, `gpu_memory_utilization` from supplied
run/profile or 0.8, and optional `num_kv_blocks` override. Estimated KV capacity uses
datasheet memory times utilization minus dense body plus tied embedding weights
and a fixed 1 GiB activation reserve, divided by block size times KV bytes/token.
Decode-first running work precedes FCFS admission. Memory pressure preempts the
most recently admitted running request by recomputation, preserving emitted tokens.
An impossible isolation capacity, or an unchunked prompt/recompute exceeding the
token budget, returns 422 rather than looping indefinitely.

The 200 response contains:
- Base `trace_id`, `gpu_config`, `run_id`, `sim_id`, `cached`.
- `requests`: ID/arrival, TTFT, mean/p50 ITL, E2E and queue time in ms,
  preemption count and exact emitted `output_tokens`. Latencies start at arrival;
  queue time ends at first GPU step start.
- `aggregate`: output tokens/s, p50/p99 TTFT/E2E, pooled p50 token interval, idle
  seconds. Throughput spans first arrival through last completion.
- `steps`: start/end/wall seconds, sequence/token budgets, prefill/decode/sampling
  counts, peak allocated KV blocks, request IDs/query/context vectors, GPU group
  durations in ns, and provenance counts (one GPU label and `assumed-host` per step).
- `steps_total`, `steps_returned`, `steps_downsampled`, `steps_sampling`: at most
  5000 uniformly indexed records, retaining first/last and original `step_index`.
  Metrics/provenance counts always use **all** executed steps.
- `scheduler` resolved values, `provenance_counts`, `calibration_range`, `assumptions`.

All multi-sequence/chunked predictions are labelled roofline extrapolations, not
measured batching performance. Pre-step envelopes apply once per busy period;
signed post-step envelopes determine completion while token emission uses actual
step ends, preserving the standalone model convention. Errors use normal `/api`
`detail`; timeouts include base run/sim IDs and never cancel the base run.

## Simulator → main

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| POST | `/api/sims/{id}/heartbeat` | `Heartbeat` | `HeartbeatAck`. A heartbeat for a terminal sim changes nothing. `cancel: true` tells the sim to stop. |
| POST | `/api/sims/{id}/logs` | `LogChunk` | 204. A repeated `seq` is ignored, so retries are safe. Main parses kernel stats from the lines. |
| PUT | `/api/sims/{id}/files/{name}` | raw file bytes | `SimFile`. Stores one output file, streamed to disk under `$CASCADE_DATA_DIR/files/<sim id>/`. `name` must match `^[A-Za-z0-9._-]{1,128}$` and not be `.` or `..` (422). 413 if the file is over 64 MiB (`FILE_MAX_BYTES`) or the sim's files would pass 256 MiB in all (`SIM_FILES_MAX_BYTES`). A re-put replaces the file, so retries are safe. 404 for an unknown sim. |
| POST | `/api/sims/{id}/step-timeline` | `StepTimelineUpload` | 204. A re-post replaces the earlier one. 404 for an unknown sim. 422 unless `simulated` is an object with a `steps` list and a `summary` object, and `compare` is an object or null. |

Simulator env: `CASCADE_SIM_ID`, `CASCADE_MAIN_URL`, `CASCADE_TOKEN`, `CASCADE_GPU_CONFIG`, `CASCADE_TRACE_DIR`, `CASCADE_ACCELSIM_DIR`.

`cascade worker` is the client of these routes that every sim runs. It sends a `running` heartbeat every 5 s, log chunks of up to 200 lines or 0.5 s with `seq` from 0, retries with backoff on 5xx and network errors, and stops when an ack says `cancel`. All log chunks are sent before its final heartbeat, which carries the exit code. It keeps the full log at `$TMPDIR/cascade-worker/<sim id>/accelsim.log`.

Output files: before the simulator starts, the worker records the run dir's listing (the copied configs). After it exits, succeeded or failed (not cancelled), the worker uploads every regular file directly in the run dir that the simulator created or changed, in name order, then its own log as `accelsim.log.gz`, all before the final heartbeat. Symlinks and subdirectories are skipped. A text file over 1 MiB is gzipped as it streams and stored as `<name>.gz`. A file over 64 MiB, a name the route would refuse, and anything once 256 MiB has been sent are skipped with one log line each. Uploads retry like the other calls; one that still fails is a log line and never changes the sim's state.

- `cascade worker` runs `$CASCADE_ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out -trace $CASCADE_TRACE_DIR/traces/kernelslist.g -config gpu-simulator/gpgpu-sim/configs/tested-cfgs/<cfg>/gpgpusim.config -config gpu-simulator/configs/tested-cfgs/<cfg>/trace.config`, both configs under `$CASCADE_ACCELSIM_DIR`, in that run dir with both config dirs' files copied in. A missing binary, config, or `kernelslist.g` is a `failed` heartbeat with exit code 2.
- `cascade worker --replay <log> [--delay S] [--replay-outputs DIR]` replays a recorded Accel-Sim log instead, pausing `S` seconds (default 0.05) every 200 lines. `--replay-outputs` copies DIR's files (except `run.log` and `setup.log`) into the run dir as if Accel-Sim wrote them, so they are uploaded like real output. The stand-in deploy uses `cascade worker --replay tests/fixtures/logs/opt125m_sm90_h100.log --replay-outputs results/20261108-021007-opt125m-h100-accelsim/outputs`.

Simulator image (the Dockerfile's `BASE_IMAGE`): the app expects the trace post-processor at `CASCADE_POSTPROCESS_CMD` (default `$CASCADE_ACCELSIM_DIR/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing`), Accel-Sim's `gpu-simulator/gpgpu-sim/configs/tested-cfgs/*/gpgpusim.config` under `CASCADE_ACCELSIM_DIR`, and an Accel-Sim build at `$CASCADE_ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out`. Set `CASCADE_SIM_CMD` to `/app/.venv/bin/cascade worker`, which main runs with the env above.

## Step timing

### cascade-samples

The built-in [OPT-125m H100 sample](../samples/opt125m-h100-e2e/README.md) belongs to
`cascade-samples`. It seeds automatically at startup, with recorded SM90_H100 and SM90_H200
simulations; no upload, instruction traces or simulator execution is needed. Sample traces
cannot launch new runs. The native A100 sample contains its own measured baseline and recorded
SM80_A100 simulation, not a simulation of the H100 instruction trace.

To rebuild bundles from complete profiler run directories (never commit `.tracez` files):

```sh
modal volume get accelsim-traces phases/e2e-plan-v1 /path/to/samples/e2e-plan-v1
uv run python scripts/build_sample.py /path/to/samples/e2e-plan-v1 \
  --config SM90_H200=/path/to/hw-sweep/SM90_H200
# Rebuild the native A100 bundle after its simulation outputs are present:
modal volume get accelsim-traces phases/e2e-a100-plan-v1 /path/to/samples/e2e-a100-plan-v1
uv run python scripts/build_sample.py /path/to/samples/e2e-a100-plan-v1 --sample opt125m-a100-e2e
```

`sample.json` supplies `real_gpu_config`, `sample_sim_configs`, `sample_set` and
`sample_pending_configs` to trace metadata. Architecture timing hides a
pending entry once a trace in the same set identifies that config as its `real_gpu_config`.
It includes the latest succeeded run for each other native trace in the set, with its measured
reference and simulated metrics. Signed E2E error is shown **only** for each trace's
`real_gpu_config` against that trace's measured baseline; counterfactual configs are metrics-only.
Without an explicit `real_gpu_config`, no error is inferred from the suggested launch config.

Bundles are seeded once per trace id; use a fresh `CASCADE_DATA_DIR` to inspect rebuilt bundles.
Live, non-sample traces may also specify `sample_sim_configs` to preselect compatible configs.

`TraceSummary.has_step_timeline` is true when the trace dir holds `corrected_timeline.json`
(the profiler's phases output) and either `kernel_mapping.json` (a list of
`{uid, name, repeat, step, seq}`) or `trace_alignment.json`. For alignment, the worker maps
simulator launch positions through `traces/kernelslist.g` to tracer UIDs, checks names, and
propagates durations with interpolation. Ingest rechecks the files on every scan, so the flag
turns true when they land after the trace is `ready`.
Sample traces already carry recorded step timelines and ingest skips their directories.

When a sim succeeds on such a trace, `cascade worker` imports the sim's kernel durations from its own log (`cascade import-sim-durations` with the config's core clock), runs `cascade simulate` with its defaults and `cascade compare` against the corrected timeline, and posts `{"simulated": simulated_timeline.json, "compare": compare.json}` before its final heartbeat. The shapes of both objects are Alessio's; see [corrected-timeline-simulation.md](corrected-timeline-simulation.md).

## Sim state machine

`SIM_TRANSITIONS` in models.py is the only source of allowed moves.

```
pending → launching → running → succeeded | failed
any non-terminal → cancelled            (user)
launching | running → lost              (no heartbeat within timeout)
lost → running | succeeded | failed     (heartbeat resumes)
pending | launching → failed            (launcher error)
```

## Trace state machine

```
uploading → processing → ready
uploading | processing → failed
```

`uploading` starts when the first file of `<prefix>/<run_name>/` appears. The trace leaves `uploading` when `metadata.json` exists and either lists `files` that are all present with matching sizes, or (older profiler) `traces/kernelslist_ctx_*` exists. A `metadata.json` with `status: "failed"` moves the trace to `failed`.

## Samples

On startup, `cascade serve` seeds each bundle under `samples/` into the workspace once, so a new workspace opens with a finished trace and run to explore while its first trace profiles (about 30 minutes). Set `CASCADE_SAMPLES=false` to skip it; a sample already seeded stays.

Today there is one, [samples/opt125m-h100-e2e](../samples/opt125m-h100-e2e/README.md): Alessio's real facebook/opt-125m run on an H100, simulated with Accel-Sim on SM90_H100. It appears as:

- The trace `sample__facebook_opt-125m__h100-e2e`, `ready`, with `sample: true`. Its trace dir holds `corrected_timeline.json`, `trace_alignment.json`, `trace_steps.json` and `traces/kernelslist.g`, but no `.tracez` instruction traces, so `POST /api/runs` on it is 409.
- The run `run-sample-opt125m-h100-e2e`, `succeeded`, with `sample: true` and one SM90_H100 sim. Its kernel stats are the 93 kernels Accel-Sim simulated, parsed from their logs, in launch order, with `kernel_launch_uid` set to the tracer's uid. It has the bundle's step timeline and five output files. The trace's real timing is those 93 kernels' nsys durations, in the same order.

Seeding is one SQLite transaction keyed on the trace id, so a restart never duplicates it.
