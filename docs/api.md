# Cascade app API

Types live in [cascade/models.py](../cascade/models.py) and are mirrored in [web/src/api/types.ts](../web/src/api/types.ts). Decisions behind this contract are in [cascade-app-decisions.md](cascade-app-decisions.md).

All `/api` routes require `Authorization: Bearer $CASCADE_TOKEN`, except `GET /api/health`. EventSource cannot send headers, so the SSE route also accepts `?token=`.

## UI and reads

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/api/health` | `{"ok": true, "auth": bool}`. `auth` is false when the app runs with `CASCADE_NO_AUTH=1` (local dev only), and the UI then skips the token screen. |
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

## Simulator → main

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| POST | `/api/sims/{id}/heartbeat` | `Heartbeat` | `HeartbeatAck`. A heartbeat for a terminal sim changes nothing. `cancel: true` tells the sim to stop. |
| POST | `/api/sims/{id}/logs` | `LogChunk` | 204. A repeated `seq` is ignored, so retries are safe. Main parses kernel stats from the lines. |
| POST | `/api/sims/{id}/step-timeline` | `StepTimelineUpload` | 204. A re-post replaces the earlier one. 404 for an unknown sim. 422 unless `simulated` is an object with a `steps` list and a `summary` object, and `compare` is an object or null. |

Simulator env: `CASCADE_SIM_ID`, `CASCADE_MAIN_URL`, `CASCADE_TOKEN`, `CASCADE_GPU_CONFIG`, `CASCADE_TRACE_DIR`, `CASCADE_ACCELSIM_DIR`.

`cascade worker` is the client of these routes that every sim runs. It sends a `running` heartbeat every 5 s, log chunks of up to 200 lines or 0.5 s with `seq` from 0, retries with backoff on 5xx and network errors, and stops when an ack says `cancel`. All log chunks are sent before its final heartbeat, which carries the exit code. It keeps the full log at `$TMPDIR/cascade-worker/<sim id>/accelsim.log`.

- `cascade worker` runs `$CASCADE_ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out -trace $CASCADE_TRACE_DIR/traces/kernelslist.g -config gpu-simulator/gpgpu-sim/configs/tested-cfgs/<cfg>/gpgpusim.config -config gpu-simulator/configs/tested-cfgs/<cfg>/trace.config`, both configs under `$CASCADE_ACCELSIM_DIR`, in that run dir with both config dirs' files copied in. A missing binary, config, or `kernelslist.g` is a `failed` heartbeat with exit code 2.
- `cascade worker --replay <log> [--delay S]` replays a recorded Accel-Sim log instead, pausing `S` seconds (default 0.05) every 200 lines. Dev and the stand-in deploy use `cascade worker --replay tests/fixtures/logs/babelstream_sm86_rtx3070.log`.

Simulator image (the Dockerfile's `BASE_IMAGE`): the app expects the trace post-processor at `CASCADE_POSTPROCESS_CMD` (default `$CASCADE_ACCELSIM_DIR/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing`), Accel-Sim's `gpu-simulator/gpgpu-sim/configs/tested-cfgs/*/gpgpusim.config` under `CASCADE_ACCELSIM_DIR`, and an Accel-Sim build at `$CASCADE_ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out`. Set `CASCADE_SIM_CMD` to `/app/.venv/bin/cascade worker`, which main runs with the env above.

## Step timing

`TraceSummary.has_step_timeline` is true when the trace dir holds both `corrected_timeline.json` (the profiler's phases output) and `kernel_mapping.json` (a list of `{uid, name, repeat, step, seq}`). Ingest rechecks it on every scan, so it turns true when the files land after the trace is `ready`.

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
