# Cascade

You are working with a Cascade app. Cascade takes a trace of a GPU workload, captured once on a real GPU, and simulates it with Accel-Sim on other GPU configs, many sims at once. You drive the app over its HTTP API. You can check traces, launch runs, watch them, pull results and comparisons, and explain the numbers to the user.

You cannot capture a trace. Tracing runs the workload on a real GPU under Cascade's Modal profiler, outside this API. When the user wants a new trace, tell them to run the profiler (the "Trace a workload" page of the Cascade docs) and then wait for the trace to appear in the app.

Words used below:

- A **trace** is one profiler run. Its `id` is the profiler's run name.
- A **GPU config** is one of Accel-Sim's tested configs, such as `SM90_H100`.
- A **run** is one trace plus a list of entries. Each entry names a GPU config and a `count`.
- A **sim** is one simulator working through the trace on one config. Sims of one entry differ only in `replica`, numbered from 0.

## Connect

The app's URL and token come from two environment variables.

- `CASCADE_URL` is the app's base URL with no trailing slash, such as `http://127.0.0.1:8000`.
- `CASCADE_TOKEN` is the bearer token the app was started with.

Check that both are set without printing them:

```bash
test -n "$CASCADE_URL" && test -n "$CASCADE_TOKEN" && echo set
```

If either is unset, ask the user to export both in the terminal they start you from, and to start you again. If the user gives you the URL or the token in the conversation instead, write them once to a file only they can read, and load that file at the start of every command. Variables you export in one command may be gone in the next. Writing the file is the one command that holds the token's value. When you finish the task, delete it with `rm ~/.cascade.env` and tell the user, unless they ask you to keep it.

```bash
(umask 077 && printf 'export CASCADE_URL=%s CASCADE_TOKEN=%s\n' '<url>' '<token>' > ~/.cascade.env)
. ~/.cascade.env && curl -sS --fail-with-body -H "Authorization: Bearer $CASCADE_TOKEN" "$CASCADE_URL/api/traces"
```

Send the token on every request as an `Authorization: Bearer` header. Write every request in this form, with the variables unexpanded, so the token never appears in a command or its output:

```bash
curl -sS --fail-with-body -H "Authorization: Bearer $CASCADE_TOKEN" "$CASCADE_URL/api/traces"
```

For a request with a JSON body, add `-X POST -H "Content-Type: application/json" -d '<json>'`. When the status is 400 or more, `--fail-with-body` makes curl exit with code 22, print `The requested URL returned error: <status>` and still print the JSON body. Parse responses with `jq` or `python3`, and aggregate large responses in the command instead of printing them whole.

Check the connection before anything else.

1. `GET /api/health` needs no token. Expect `"ok": true` and `"auth": true`. `workspace` names the workspace, or is `null`. Anything else means `CASCADE_URL` is wrong or the app is down. `"auth": false` means the app runs without a token check, which is fine.
2. `GET /api/traces` with the token. A `401` means the token is wrong. Stop and ask the user to check `CASCADE_TOKEN`.

The connection is good when both calls succeed.

## Procedure

Follow these steps to simulate a trace and report on it. When the user asks about an existing run, start at step 5 with its `id`. `GET /api/runs` lists runs newest first.

1. **Find the trace.** Call `GET /api/traces`. Keep the traces whose `state` is `ready`. Match the user's words against `id`, `model`, `layers` and `source_gpu`. If more than one trace matches, list each with its `id`, `model`, `layers`, `trace_gpu_arch`, `kernel_count` and `created_at`, and ask which one. If the trace the user means is `uploading` or `processing`, poll `GET /api/traces/{trace_id}` every 10 seconds until it is `ready`. If it is `failed`, report its `error` and stop. This step is done when you hold one `ready` trace and its `trace_gpu_arch`.
2. **Pick the configs.** Call `GET /api/gpu-configs`. Keep only the configs whose `arch` equals the trace's `trace_gpu_arch`. A trace can only run on configs of its own architecture. Use the configs the user named. If they named none, use the trace's `suggested_sim_config` when it is set, and otherwise every matching config. If a config the user named has another `arch`, say so and offer the matching ones. This step is done when every config you will use has the trace's `arch`.
3. **Size the run.** Each entry's `count`, from 1 to 64, is how many replicas run that config. Use 1 per config unless the user asks for more. Suggest 3 when the user wants to know whether the numbers are stable. Add up the counts. If the total is more than 4 sims, show the user the entries and the total and wait for them to agree. On Sail, every sim is its own forked sandbox, and each one costs money. This step is done when you have the entries and, above 4 sims, the user's yes.
4. **Launch the run.** Call `POST /api/runs` with `{"trace_id": "<trace id>", "entries": [{"gpu_config": "<name>", "count": 1}]}`. A `201` returns the run. Save its `id`. Launch once. If the call fails with a network error or a `5xx`, call `GET /api/runs` and look for a new run on this trace before you try again.
5. **Watch the run.** Poll `GET /api/runs/{run_id}` every 10 seconds until `state` is anything but `active`. Between polls, tell the user the counts in `sim_counts` and each sim's `kernels_done` out of `kernels_total`. If `kernels_done` passes `kernels_total`, give both numbers as they are, cap any percentage at 100, and treat it as a stand-in sign (see Rules). A `lost` sim can come back, so keep waiting. A sim that stays `lost` for 10 times the heartbeat timeout, 600 seconds by default, becomes `failed`. A real Accel-Sim run can take hours. If the run is still `active` after about 5 minutes, give the user the run `id` and the progress, and offer to check again later. `GET /api/runs/{run_id}/events` streams the same changes as server-sent events, but the stream never closes, so polling is simpler. To read the stream anyway, add `-N --max-time 30` to curl. This step is done when the run's `state` is `succeeded`, `partial`, `failed` or `cancelled`.
6. **Fetch the results.** Call `GET /api/runs/{run_id}/kernels` for every sim's per-kernel times and `GET /api/runs/{run_id}/comparison` for real against simulated time per kernel. A `KernelStat` names only its `sim_id`. Join it to the run's `sims` by `id` to get its `gpu_config`, `replica` and `state`. For each sim that did not succeed, read `error`, `exit_code` and `log_tail` from the run. If `log_tail` does not explain the failure, read the end of `GET /api/sims/{sim_id}/log`, then search the whole log with `grep -inE 'error|abort|assert|fault|exception'`. If nothing explains it, report that the log does not show a cause, and give the `exit_code`, the signal it implies (134 is SIGABRT, 137 is SIGKILL, 139 is SIGSEGV), and the last kernel the sim reached. Call `GET /api/runs/{run_id}/results` only when the user asks for the full Accel-Sim stats. It carries about 30 KB per kernel. This step is done when you hold the kernels, the comparison and the reason for every sim that did not succeed.
7. **Report.** Write the report as the next section describes.

## Report

Lead with one line that gives the run `id`, the trace `id`, the run `state`, and the sim counts. Then cover these points, in this order.

1. **Simulated time per config.** For each sim, add up `sim_time_ns` over its kernels from `GET /api/runs/{run_id}/kernels`. For each config, give the median of those totals over the replicas that succeeded, and the lowest and highest total when there is more than one replica. A sim that did not succeed has only the kernels it finished, so leave it out of the totals. If no replica of a config succeeded, say so. Name the fastest config and give every other config as a multiple of it. When configs are within 1% of each other, say they tie instead of naming a winner. When the numbers come from the stand-in, say the configs cannot be compared from this run. Also give `latest.gpu_tot_sim_cycle` and `latest.gpu_tot_ipc` per config, from the run's sims. This computes the totals:

```bash
curl -sS --fail-with-body -H "Authorization: Bearer $CASCADE_TOKEN" "$CASCADE_URL/api/runs/<run id>" > /tmp/cascade-run.json
curl -sS --fail-with-body -H "Authorization: Bearer $CASCADE_TOKEN" "$CASCADE_URL/api/runs/<run id>/kernels" > /tmp/cascade-kernels.json
python3 -c '
import json, statistics
run = json.load(open("/tmp/cascade-run.json"))
sims = {s["id"]: s for s in run["sims"]}
totals = {}
for k in json.load(open("/tmp/cascade-kernels.json")):
    totals[k["sim_id"]] = totals.get(k["sim_id"], 0) + (k["sim_time_ns"] or 0)
configs = {}
for sim, total in totals.items():
    if sims[sim]["state"] == "succeeded":
        configs.setdefault(sims[sim]["gpu_config"], []).append(total)
for name, values in sorted(configs.items()):
    print(name, "median", statistics.median(values), "lowest", min(values), "highest", max(values), "replicas", len(values))
'
```
2. **Accuracy against real hardware.** When the comparison has `has_real_timing` true, compute the mean absolute percentage error (MAPE) for each config. Use the rows where both `real_ns` and `sim_ns` for that config are set. MAPE is the mean of `abs(sim_ns - real_ns) / real_ns * 100`. Say how many rows went into it. A row's `sim_ns` is the median over every replica that finished that kernel, including replicas that failed later. When `has_real_timing` is false, say that the trace has no real timing yet, so the report covers simulated time only.
3. **Worst kernels.** Name up to 5 kernels. With real timing, pick those with the largest absolute percentage error. Without it, pick those with the largest share of simulated time, and say how much their time differs between configs. Name each kernel in readable form. Demangle `kernel_name` with `c++filt -n -p <kernel_name>` when `c++filt` is installed, so `_Z11copy_kernelIdEvPKT_PS0_` becomes `copy_kernel<double>`. The `-n` matters on macOS, where `c++filt` without it returns the name unchanged. Without `c++filt`, give the identifier inside the mangled name, such as `copy_kernel`. Give the `kernel_index` too.
4. **Failed, lost and cancelled sims.** List every sim whose `state` is not `succeeded`, with its `id`, `gpu_config`, `replica`, `state`, `error`, `exit_code` and the lines of `log_tail` that show the cause.

Give times in the largest unit that keeps the number above 1, such as 5.2 µs or 3.1 ms, and call them simulated. Take every number from an API response. When a value is `null`, say that it is missing. If the user asks what a number means, explain it from the field descriptions under "Types" below.

## Rules

- Keep the token secret. Refer to it only as `$CASCADE_TOKEN`, and never print, echo or repeat its value.
- Launch a run only when the user asked for one. Above 4 sims in total, get the user's yes to the entries first.
- Cancel a run only when the user asks. `POST /api/runs/{run_id}/cancel` stops every sim that has not finished. It is safe to repeat.
- Use `GET /api/runs/{run_id}/kernels` for per-kernel numbers. Use `GET /api/runs/{run_id}/results` only for the full stats maps.
- Check whether the numbers come from Cascade's stand-in simulator, unless the user has said the app runs a real Accel-Sim image. A default deploy runs the stand-in, `cascade worker --replay`, which replays one recorded Accel-Sim run of `facebook/opt-125m` on SM90_H100 (20 vLLM kernels), whatever the trace. A local dev server may replay a BabelStream log instead, whose kernels are named init, copy, mul, add, triad and dot. The signs are the same kernels for every trace (`nvjet_sm90_hsh_64x8_64x16_4x1_v_bz_bias_TNT`, `vllm::reshape_and_cache_flash_kernel`, `cutlass::device_kernel` and `at::native::vectorized_elementwise_kernel` on a default deploy), OPT-125m kernels for a trace of another model, and `kernels_done` different from `kernels_total`. When you see these signs, say at the top of the report that the numbers come from the stand-in and do not describe the trace.
- Never call `POST /api/sims/{sim_id}/heartbeat`, `POST /api/sims/{sim_id}/logs` or `POST /api/sims/{sim_id}/step-timeline`. Those routes are for simulators, and a heartbeat changes a sim's state.
- Lists come back whole, with no pagination.

## Errors

Every error is JSON with a `detail` field.

- **401.** The token is missing or wrong. Stop and ask the user to check `CASCADE_TOKEN`. Do not try other tokens.
- **404.** When `detail` names a trace, run or sim, that id does not exist. List again with `GET /api/traces` or `GET /api/runs` to find the right one. When `detail` is just `Not Found`, the path itself is wrong. Check it against the routes below.
- **409.** `POST /api/runs` named a trace that is not `ready`. `detail` gives its state. If the state is `uploading` or `processing`, wait until the trace is `ready` and launch again. If it is `failed`, report the trace's `error`.
- **422.** When `detail` is a sentence, a `gpu_config` is unknown or has an `arch` other than the trace's `trace_gpu_arch`. Call `GET /api/gpu-configs` and use matching configs. When `detail` is a list, the body is invalid: `entries` is empty, or a `count` is outside 1 to 64. Each item's `loc` and `msg` say what to fix.
- **5xx or a network error.** Retry a `GET` up to 3 times, a few seconds apart. Before you retry `POST /api/runs`, check `GET /api/runs` for the run, so you never launch it twice.

## API

Every route takes the bearer token except `GET /api/health`. Bodies are JSON, and timestamps are ISO 8601 in UTC.

| method | path | what it does |
|---|---|---|
| GET | `/api/health` | Liveness check. Needs no token. |
| GET | `/api/traces` | Every trace, newest first. |
| GET | `/api/traces/{trace_id}` | One trace with its metadata and kernels. |
| GET | `/api/gpu-configs` | The GPU configs a run can use. |
| POST | `/api/runs` | Launch sims for a ready trace. |
| GET | `/api/runs` | Every run, newest first. |
| GET | `/api/runs/{run_id}` | One run with its trace and every sim. |
| POST | `/api/runs/{run_id}/cancel` | Cancel every unfinished sim. |
| GET | `/api/runs/{run_id}/comparison` | Per-kernel real time against simulated time per config. |
| GET | `/api/runs/{run_id}/kernels` | Every sim's per-kernel stats, without the full stats maps. |
| GET | `/api/runs/{run_id}/step-timelines` | Each sim's predicted step timeline and its comparison with the real run. |
| GET | `/api/runs/{run_id}/results` | One JSON file with the run, trace, every sim's stats and the comparison. |
| GET | `/api/sims/{sim_id}/results` | One sim's config, per-kernel stats and totals. |
| GET | `/api/sims/{sim_id}/log` | The sim's full raw output. |
| GET | `/api/runs/{run_id}/events` | Server-sent events for one run. |

### GET /api/health

Check health. Returns `{"ok": true, "auth": true, "workspace": ...}` while the app process is serving requests. `auth` is `false` only when the app runs with `CASCADE_NO_AUTH=1`, a local-dev mode where no token is needed and the web UI skips its token screen. `workspace` is `CASCADE_WORKSPACE`, the name the web UI shows in its rail, or `null`. Health does not check the database, the S3 endpoint or the launcher.

- Auth: No token needed.
- Returns: `200` with `{"ok": true, "auth": true, "workspace": "<name or null>"}`.

### GET /api/traces

List traces. Lists every trace the app has registered, in every state, newest first by `created_at`. A trace appears as soon as its first file lands in the bucket, so this list also shows uploads in progress.

- Returns: `200` with a list of `TraceSummary`.

### GET /api/traces/{trace_id}

Get a trace. Returns one trace with the parsed `metadata.json` and every kernel launch read from the trace's `stats_ctx_*` files. `kernels` stays empty until the trace is `ready`.

- Path parameter `trace_id` (str): The trace's `id`, which is the profiler's run name.
- Returns: `200` with a `TraceDetail`.
- `404`: No trace has this id.

### GET /api/gpu-configs

List GPU configs. Lists the GPU configs the app found at startup under `$CASCADE_ACCELSIM_DIR/gpu-simulator/gpgpu-sim/configs/tested-cfgs/*/gpgpusim.config`. The app derives `arch` from the directory name (`SM90_H100` is `sm_90`, `SM7_QV100` is `sm_70`) and reads `core_clock_mhz` from the first value of `-gpgpu_clock_domains`. A directory whose name or clock it can't read is skipped.

- Returns: `200` with a list of `GpuConfig`, sorted by name.

### POST /api/runs

Create a run. Creates a run and starts launching its sims in the background. Each entry fans out to `count` sims, numbered by `replica` from 0. The response comes back before any sim has launched, so every sim is `pending` or `launching`. Follow progress with `GET /api/runs/{run_id}/events` or by polling `GET /api/runs/{run_id}`.

- Body: `RunCreate`.
- Returns: `201` with a `RunDetail`.
- `404`: `trace_id` names no trace.
- `409`: The trace is not `ready`.
- `422`: An entry names an unknown `gpu_config`, or a config whose `arch` differs from the trace's `trace_gpu_arch`.
- `422`: The body fails validation: `entries` is empty, or a `count` is outside 1 to 64.

### GET /api/runs

List runs. Lists every run, newest first, with per-state sim counts. Use `GET /api/runs/{run_id}` for the sims themselves.

- Returns: `200` with a list of `RunSummary`.

### GET /api/runs/{run_id}

Get a run. Returns the run, its trace, and every sim with its state, progress, latest metrics and last 20 log lines.

- Path parameter `run_id` (str): The run's `id`, such as `run-9f56e171af08`.
- Returns: `200` with a `RunDetail`.
- `404`: No run has this id.

### POST /api/runs/{run_id}/cancel

Cancel a run. Moves every sim that has not finished to `cancelled` and asks the launcher to stop it. Sims that already finished keep their state. The next heartbeat from a cancelled sim gets `cancel: true`. Calling it again changes nothing, so it is safe to retry.

- Path parameter `run_id` (str): The run's `id`, such as `run-9f56e171af08`.
- Returns: `200` with the `RunDetail` after cancelling.
- `404`: No run has this id.

### GET /api/runs/{run_id}/comparison

Compare real and simulated time. Lines up kernels by order. Row `i` holds the `i`th kernel the sims finished and the `i`th row of the trace's real timing, sorted by `kernel_id`. For each config, `sim_ns` is the median `sim_time_ns` over that config's replicas, or `null` if no replica has finished that kernel. `real_ns` is `null` until the trace has real timing, which the profiler does not capture yet.

- Path parameter `run_id` (str): The run's `id`, such as `run-9f56e171af08`.
- Returns: `200` with a `Comparison`.
- `404`: No run has this id.

### GET /api/runs/{run_id}/kernels

List a run's kernel timings. Returns a `KernelStat` for every kernel every sim in the run has finished, ordered by config, replica, then kernel. Each `stats` map is empty to keep the response small; the full maps are in the results downloads. Laying each sim's kernels end to end by `sim_time_ns` gives its timeline, which is what the app's timeline and Compare views draw.

- Path parameter `run_id` (str): The run's `id`, such as `run-9f56e171af08`.
- Returns: `200` with a list of `KernelStat`.
- `404`: No run has this id.

### GET /api/runs/{run_id}/step-timelines

List a run's step timing. Returns a `SimStepTimeline` for every sim in the run that has sent step timing, ordered by config, then replica. A sim sends one after it succeeds on a trace whose `has_step_timeline` is true, so the list is empty until then, and stays empty for other traces. `simulated` is a full `simulated_timeline.json`, with every kernel of every step, so for a real model it can be several megabytes per sim. Per-kind step means are in `simulated.summary.per_kind`, coverage is in `simulated.summary.simulation.coverage`, and per-step and per-request errors against the real run are in `compare`.

- Path parameter `run_id` (str): The run's `id`, such as `run-9f56e171af08`.
- Returns: `200` with a list of `SimStepTimeline`.
- `404`: No run has this id.

### GET /api/runs/{run_id}/results

Download run results. Returns everything about a run as one JSON download, with `Content-Disposition: attachment; filename="<run_id>-results.json"`. Each sim's `kernels` carry the full `stats` map, every `key = value` line Accel-Sim printed for that kernel, which is about 30 KB per kernel.

- Path parameter `run_id` (str): The run's `id`, such as `run-9f56e171af08`.
- Returns: `200` with a `RunResults`.
- `404`: No run has this id.

### GET /api/sims/{sim_id}/results

Download sim results. Returns one sim's results as a JSON download named `<sim_id>-results.json`. `totals` is the sim's latest `SimMetrics`.

- Path parameter `sim_id` (str): The sim's `id`, such as `sim-6b4c91e71437`. Find it in `RunDetail.sims`.
- Returns: `200` with a `SimResults`.
- `404`: No sim has this id.

### GET /api/sims/{sim_id}/log

Download a sim's log. Streams every line the simulator sent, in `seq` order, as `text/plain` named `<sim_id>.log`.

- Path parameter `sim_id` (str): The sim's `id`, such as `sim-6b4c91e71437`. Find it in `RunDetail.sims`.
- Returns: `200` with the log as plain text.
- `404`: No sim has this id.

### GET /api/runs/{run_id}/events

Stream run events. Opens a `text/event-stream`. The first event is always a `snapshot`. After that the stream sends `sim`, `log` and `kernel` events as they happen, and a `: keepalive` comment after 15 seconds without one. The stream does not close when the run finishes.

- Auth: Bearer token, or the token as the `token` query parameter.
- Path parameter `run_id` (str): The run's `id`, such as `run-9f56e171af08`.
- Query parameter `token` (str, optional): Your `CASCADE_TOKEN`. Use it when you can't set headers, as with `EventSource`. If present, it is checked instead of the header.
- Returns: `200` with a `text/event-stream` of `RunEvent`, one JSON object per `data:` line.
- `404`: No run has this id.

## States

### Trace states

A trace's `state`. Runs can only launch on a `ready` trace.

- `uploading`: Files are arriving. `kernel_count` and `bytes` grow on each scan, every 2 seconds. Next: `processing` or `failed`.
- `processing`: The upload is complete. The app is running the trace post-processor, unless the upload already has `traces/kernelslist.g`. Next: `ready` or `failed`.
- `ready`: Post-processing produced `traces/kernelslist.g`. Runs can launch. Terminal.
- `failed`: `metadata.json` has `status: "failed"`, or post-processing failed. `error` says which. Terminal.

### Sim states

A sim's `state`. The app ignores any move not listed here.

- `pending`: Created with the run. Waiting for the launcher. Next: `launching`, `failed` or `cancelled`.
- `launching`: The launcher is starting the sim's sandbox or process. Next: `running`, `lost`, `succeeded`, `failed` or `cancelled`.
- `running`: The sim sent a `running` heartbeat. Next: `lost`, `succeeded`, `failed` or `cancelled`.
- `lost`: No heartbeat for `CASCADE_HEARTBEAT_TIMEOUT_S`. A later heartbeat brings it back. Next: `running`, `succeeded`, `failed` or `cancelled`.
- `succeeded`: The sim sent a `succeeded` heartbeat. Terminal.
- `failed`: The sim sent a `failed` heartbeat, the launcher failed, or it stayed `lost` for 10 times the timeout. Terminal.
- `cancelled`: The run was cancelled before this sim finished. Terminal.

### Run states

`RunState` is not stored. `run_state()` in `cascade/models.py` derives it from the run's sim states each time a run is read.

- `active`: At least one sim has not finished.
- `succeeded`: Every sim succeeded.
- `partial`: Every sim finished, at least one succeeded, and at least one did not.
- `failed`: Every sim failed.
- `cancelled`: Every sim finished, at least one was cancelled, and none succeeded.

## Types

These are the request and response bodies, from `cascade/models.py`. A field written `X | None` can be `null`. `TraceState`, `SimState` and `RunState` are under States.

### GpuConfig

- `name` (`str`): Accel-Sim tested-cfgs directory name, e.g. "SM90_H100".
- `arch` (`str`): For example, "sm_90".
- `core_clock_mhz` (`float`): First value of -gpgpu_clock_domains in gpgpusim.config.

### KernelLaunch

One kernel launch from the trace's stats_ctx_* file (original hardware).

- `kernel_id` (`int`): Matches kernel-<N> in the trace filename.
- `name` (`str`): Mangled kernel name.
- `grid` (`tuple[int, int, int]`): Grid size, x, y and z.
- `block` (`tuple[int, int, int]`): Block size, x, y and z.
- `instructions` (`int | None`): Instructions the tracer counted. `null` if the kernel's trace was cut short.

### TraceSummary

- `id` (`str`): The profiler's run_name.
- `state` (`TraceState`): Where the trace is in its lifecycle.
- `error` (`str | None`): Why the trace failed. `null` unless `state` is `failed`.
- `created_at` (`datetime`): When the first file of the trace landed.
- `updated_at` (`datetime`): When the app last changed this trace.
- `model` (`str | None`): `metadata.json` `model`, such as `facebook/opt-125m`.
- `layers` (`list[str]`): `metadata.json` `layers`. Empty if missing or not a list.
- `source_gpu` (`str | None`): metadata.json "gpu".
- `trace_gpu_arch` (`str`): metadata.json "trace_gpu_arch", default "sm_90".
- `suggested_sim_config` (`str | None`): `metadata.json` `suggested_sim_config`. The UI preselects it on the New run screen.
- `kernel_count` (`int`): Traced kernels seen so far.
- `bytes` (`int`): Bytes on disk under the trace dir.
- `has_real_timing` (`bool`): True when the trace has real per-kernel timings (`real_timing.json`). The profiler does not write them yet.
- `has_step_timeline` (`bool`): corrected_timeline.json + kernel_mapping.json are in the trace dir.

### TraceDetail

Every field of `TraceSummary`, plus these.

- `metadata` (`dict | None`): The trace's `metadata.json`, parsed. `null` until it arrives.
- `kernels` (`list[KernelLaunch]`): Kernel launches from the trace's `stats_ctx_*` files, by `kernel_id`. Filled in when the trace becomes `ready`.

### RunEntry

- `gpu_config` (`str`): A `GpuConfig.name` whose `arch` matches the trace's `trace_gpu_arch`.
- `count` (`int`): How many sims run this config. Allowed: at least 1, at most 64.

### RunCreate

- `trace_id` (`str`): A trace in state `ready`.
- `entries` (`list[RunEntry]`): One entry per config. Each fans out to `count` sims. Allowed: at least 1 item.

### RunSummary

- `id` (`str`): Generated, `run-` and 12 hex digits.
- `trace_id` (`str`): The trace the run simulates.
- `created_at` (`datetime`): When the run was created.
- `state` (`RunState`): Derived from the sims. See `RunState`.
- `entries` (`list[RunEntry]`): The entries the run was created with.
- `sim_counts` (`dict[SimState, int]`): How many sims are in each state. States with no sims are left out.

### RunDetail

Every field of `RunSummary`, plus these.

- `trace` (`TraceSummary`): The run's trace.
- `sims` (`list[SimSummary]`): Every sim, sorted by `gpu_config` and then `replica`.

### SimSummary

- `id` (`str`): Generated, `sim-` and 12 hex digits. The simulator receives it as `CASCADE_SIM_ID`.
- `run_id` (`str`): The run this sim belongs to.
- `gpu_config` (`str`): The config this sim simulates. The simulator receives it as `CASCADE_GPU_CONFIG`.
- `replica` (`int`): 0-based index within its {gpu_config, count} entry.
- `state` (`SimState`): Where the sim is in its lifecycle.
- `error` (`str | None`): Why the sim failed or went `lost`, from a heartbeat or from the app. A `running` heartbeat clears it.
- `launcher_ref` (`str | None`): Sailbox id or local pid.
- `created_at` (`datetime`): When the run was created.
- `started_at` (`datetime | None`): The first `running` heartbeat.
- `finished_at` (`datetime | None`): When the sim reached a terminal state.
- `last_heartbeat_at` (`datetime | None`): The last heartbeat. `null` before the first one.
- `exit_code` (`int | None`): From the final heartbeat, when the simulator sent one.
- `kernels_done` (`int`): Kernel stats blocks parsed from the log so far.
- `kernels_total` (`int`): The trace's kernel count when the run was created.
- `current_kernel` (`str | None`): The kernel named by the last `launching kernel name:` line in the log.
- `latest` (`SimMetrics`): The most recent totals parsed from the log.
- `log_tail` (`list[str]`): Last 20 lines.

### SimMetrics

- `gpu_tot_sim_cycle` (`float | None`): Simulated cycles so far, over every kernel.
- `gpu_tot_sim_insn` (`float | None`): Simulated instructions so far, over every kernel.
- `gpu_tot_ipc` (`float | None`): Instructions per cycle so far, over every kernel.
- `sim_rate_ips` (`float | None`): gpgpu_simulation_rate, instructions/sec.

### KernelStat

One per-kernel stats block parsed from Accel-Sim output.

- `sim_id` (`str`): The sim that ran this kernel.
- `kernel_index` (`int`): 0-based order of completion within the sim.
- `kernel_name` (`str`): The block's `kernel_name`.
- `kernel_launch_uid` (`int | None`): The block's `kernel_launch_uid`.
- `gpu_sim_cycle` (`int | None`): Cycles this kernel took on the simulated GPU.
- `gpu_sim_insn` (`int | None`): Instructions this kernel ran.
- `gpu_ipc` (`float | None`): This kernel's instructions per cycle.
- `sim_time_ns` (`float | None`): gpu_sim_cycle / core_clock_mhz * 1000.
- `stats` (`dict[str, float | str]`): Every "key = value" line in the block.

### ComparisonRow

- `kernel_index` (`int`): Row position. Kernels are matched by order.
- `kernel_name` (`str`): From real timing if present, otherwise from the first sim that finished this kernel.
- `real_ns` (`float | None`): `duration_ns` from real timing. `null` without real timing.
- `sim_ns` (`dict[str, float | None]`): gpu_config -> median sim_time_ns over replicas.

### Comparison

- `has_real_timing` (`bool`): False until the trace has real timing.
- `configs` (`list[str]`): The run's configs, in entry order, without repeats.
- `rows` (`list[ComparisonRow]`): One row per kernel position.

### SimResults

- `sim` (`SimSummary`): The sim as `SimSummary`.
- `trace_id` (`str`): The trace the sim ran.
- `gpu_config` (`GpuConfig`): The full config. `arch` is `unknown` and the clock is 0 if the config disappeared since the run started.
- `kernels` (`list[KernelStat]`): Every kernel the sim finished, with the full `stats` map.
- `totals` (`SimMetrics`): Same as `sim.latest`.

### RunResults

- `run` (`RunSummary`): The run without its trace and sims, which are below.
- `trace` (`TraceDetail`): The full trace.
- `sims` (`list[SimResults]`): One entry per sim.
- `comparison` (`Comparison`): Same as `GET /api/runs/{run_id}/comparison`.

### SimStepTimeline

- `sim_id` (`str`): The sim that sent it.
- `gpu_config` (`str`): The sim's config.
- `replica` (`int`): The sim's replica.
- `simulated` (`dict`): As sent in `StepTimelineUpload.simulated`.
- `compare` (`dict | None`): As sent in `StepTimelineUpload.compare`.

### RunEvent

`RunEvent` is `SnapshotEvent | SimEvent | LogEvent | KernelEvent`. One event on the run event stream. The `type` field says which model it is.

### SnapshotEvent

- `type` (`Literal["snapshot"]`): Always `snapshot`.
- `run` (`RunDetail`): The run as it is at connect time.
- `kernels` (`list[KernelStat]`): Every sim's stats with `stats` emptied; full stats are ~30 KB each.

### SimEvent

- `type` (`Literal["sim"]`): Always `sim`.
- `sim` (`SimSummary`): The sim after a change: state, heartbeat, progress or metrics.

### LogEvent

- `type` (`Literal["log"]`): Always `log`.
- `sim_id` (`str`): The sim the lines came from.
- `lines` (`list[str]`): The new chunk, as the simulator sent it.

### KernelEvent

- `type` (`Literal["kernel"]`): Always `kernel`.
- `stat` (`KernelStat`): A kernel that just finished, with the full `stats` map.
