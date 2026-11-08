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

1. `GET /api/health` needs no token. Expect `{"ok": true, "auth": true}`. Anything else means `CASCADE_URL` is wrong or the app is down. `"auth": false` means the app runs without a token check, which is fine.
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

When the user wants the simulator's own files, list them with `GET /api/runs/{run_id}/files`, download one with `GET /api/sims/{sim_id}/files/{name}`, and fetch a Perfetto trace of the run with `GET /api/runs/{run_id}/perfetto`.

Give times in the largest unit that keeps the number above 1, such as 5.2 µs or 3.1 ms, and call them simulated. Take every number from an API response. When a value is `null`, say that it is missing. If the user asks what a number means, explain it from the field descriptions under "Types" below.

## Rules

- Keep the token secret. Refer to it only as `$CASCADE_TOKEN`, and never print, echo or repeat its value.
- Launch a run only when the user asked for one. Above 4 sims in total, get the user's yes to the entries first.
- Cancel a run only when the user asks. `POST /api/runs/{run_id}/cancel` stops every sim that has not finished. It is safe to repeat.
- Use `GET /api/runs/{run_id}/kernels` for per-kernel numbers. Use `GET /api/runs/{run_id}/results` only for the full stats maps.
- Check whether the numbers come from Cascade's stand-in simulator, unless the user has said the app runs a real Accel-Sim image. A default deploy runs the stand-in, `cascade worker --replay`, which replays one recorded Accel-Sim run of `facebook/opt-125m` on SM90_H100 (20 vLLM kernels), whatever the trace. A local dev server may replay a BabelStream log instead, whose kernels are named init, copy, mul, add, triad and dot. The signs are the same kernels for every trace ({{stand-in-kernels}} on a default deploy), OPT-125m kernels for a trace of another model, and `kernels_done` different from `kernels_total`. When you see these signs, say at the top of the report that the numbers come from the stand-in and do not describe the trace.
{{simulator-routes}}
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

{{route-index}}

{{routes}}

## States

{{states}}

## Types

These are the request and response bodies, from `cascade/models.py`. A field written `X | None` can be `null`. `TraceState`, `SimState` and `RunState` are under States.

{{types}}
