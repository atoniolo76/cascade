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
| POST | `/api/runs` | body `RunCreate`, returns `RunDetail` (201). 409 if the trace is not `ready`. 422 if a config is unknown or its arch differs from the trace's `trace_gpu_arch`. |
| GET | `/api/runs/{id}` | `RunDetail` |
| POST | `/api/runs/{id}/cancel` | `RunDetail`. Moves every non-terminal sim to `cancelled` and calls `Launcher.terminate`. Idempotent. |
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

The committed [manifest](../scripts/cascade-samples.json) describes real OPT-125m profiler
runs, not demo labels. H100 (`e2e-plan-v1`, `sm_90`) launches SM90_H100 and SM90_H200;
A100 (`e2e-a100-plan-v1`, `sm_80`) launches SM80_A100 and is **pending** while profiling finishes.
Trace binaries are external and must not be committed.

Fetch a run directory from the Modal volume (or use an existing local copy):

```sh
modal volume get accelsim-traces phases/e2e-plan-v1 /path/to/samples/e2e-plan-v1
# When A100 finishes:
modal volume get accelsim-traces phases/e2e-a100-plan-v1 /path/to/samples/e2e-a100-plan-v1
uv run python scripts/seed_samples.py --runs-dir /path/to/samples \
  --app http://127.0.0.1:8030 --s3 http://127.0.0.1:9030 --token "$CASCADE_TOKEN"
```

Alternatively, repeat `--source e2e-plan-v1=/local/run/dir` (or the A100 entry id) to override
individual locations. Missing entries print `SKIP`; a pending entry is automatically seeded
once its complete local directory exists. No manifest edit is needed for A100. Each invocation
uploads available entries and launches a new run per trace with one sim per listed config.
It waits for trace readiness, not for the launched simulations to finish.

The seed stages `traced/traces/` as `traces/`, excludes raw `.trace` / `.trace.xz` files, and
includes `corrected_timeline.json`, `trace_alignment.json`, `run.json`, and `workload.json`.
Uploads use `upload_trace.py`, with `metadata.json` last. Metadata includes `trace_gpu_arch`,
`suggested_sim_config`, `model`, `gpu`, `real_gpu_config`, `sample_sim_configs`, and
`sample_set: "cascade-samples"`. The last three fields are available on both trace summaries
and details; the full original metadata is also on details. Launch defaults select all compatible
sample configs, falling back to the suggested compatible config for non-sample traces.

The seed also writes `sample_pending_configs` in detail metadata. Architecture timing hides a
pending entry once a trace in the same set identifies that config as its `real_gpu_config`.
It includes the latest succeeded run for each other native trace in the set, with its measured
reference and simulated metrics. Signed E2E error is shown **only** for the current trace's
`real_gpu_config` on that trace; other traces and counterfactual configs are metrics-only.
Without an explicit `real_gpu_config`, no error is inferred from the suggested launch config.

For local development, build the web app (`cd web && npm ci && npm run build`), install `rclone`,
and start `cascade serve --port 8030 --s3-port 9030` with `CASCADE_DATA_DIR`, `CASCADE_TOKEN`,
`CASCADE_PUBLIC_URL=http://127.0.0.1:8030`, and `CASCADE_ACCELSIM_DIR` pointing at your setup.
Use a real simulator or set `CASCADE_SIM_CMD` to a replay worker command for UI testing;
replayed results are not new simulator executions.

`TraceSummary.has_step_timeline` is true when the trace dir holds `corrected_timeline.json`
(the profiler's phases output) and either `kernel_mapping.json` (a list of
`{uid, name, repeat, step, seq}`) or `trace_alignment.json`. For alignment, the worker maps
simulator launch positions through `traces/kernelslist.g` to tracer UIDs, checks names, and
propagates durations with interpolation. Ingest rechecks the files on every scan, so the flag
turns true when they land after the trace is `ready`.

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
