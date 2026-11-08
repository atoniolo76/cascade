"""Every /api route cascade/api.py serves. build.py fails if this list and the app disagree."""

# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cascade.models import FILE_MAX_BYTES, SIM_FILES_MAX_BYTES


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    where: Literal["path", "query"]
    doc: str
    required: bool = True


@dataclass(frozen=True)
class Endpoint:
    key: str
    page: str
    method: Literal["GET", "POST", "PUT"]
    path: str
    title: str
    summary: str
    doc: str
    returns: str
    example: str
    auth: Literal["bearer", "none", "bearer-or-query"] = "bearer"
    kind: Literal["json", "sse", "download"] = "json"
    params: tuple[Param, ...] = ()
    body: str | None = None
    raw_body: bool = False
    nested: tuple[str, ...] = ()
    errors: tuple[tuple[int, str], ...] = ()
    more_examples: tuple[tuple[str, str], ...] = ()
    extra: str | None = None


UNAUTHORIZED = (
    401,
    "The `Authorization` header is missing, is not `Bearer <token>`, or the token is wrong.",
)
RUN_ID = Param("run_id", "str", "path", "The run's `id`, such as `run-9f56e171af08`.")
SIM_ID = Param(
    "sim_id",
    "str",
    "path",
    "The sim's `id`, such as `sim-6b4c91e71437`. Find it in `RunDetail.sims`.",
)
FILE_NAME = Param(
    "name",
    "str",
    "path",
    "The file's name, such as `kernel-stats.csv`. Letters, digits, `.`, `_` and `-`, 1 to 128 characters, and not `.` or `..`.",
)
TRACE_ID = Param("trace_id", "str", "path", "The trace's `id`, which is the profiler's run name.")

ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint(
        key="health",
        page="api",
        method="GET",
        path="/api/health",
        title="Check health",
        summary="Liveness check. Needs no token.",
        doc='Returns `{"ok": true, "auth": true, "workspace": ...}` while the app process is serving requests. `auth` is `false` only when the app runs with `CASCADE_NO_AUTH=1`, a local-dev mode where no token is needed and the web UI skips its token screen. `workspace` is `CASCADE_WORKSPACE`, the name the web UI shows in its rail, or `null`. Health does not check the database, the S3 endpoint or the launcher.',
        returns='`200` with `{"ok": true, "auth": true, "workspace": "<name or null>"}`.',
        example="health",
        auth="none",
    ),
    Endpoint(
        key="list-traces",
        page="api-traces",
        method="GET",
        path="/api/traces",
        title="List traces",
        summary="Every trace, newest first.",
        doc="Lists every trace the app has registered, in every state, newest first by `created_at`. A trace appears as soon as its first file lands in the bucket, so this list also shows uploads in progress.",
        returns="`200` with a list of `TraceSummary`.",
        example="traces-list",
        errors=(UNAUTHORIZED,),
    ),
    Endpoint(
        key="get-trace",
        page="api-traces",
        method="GET",
        path="/api/traces/{trace_id}",
        title="Get a trace",
        summary="One trace with its metadata and kernels.",
        doc="Returns one trace with the parsed `metadata.json` and every kernel launch read from the trace's `stats_ctx_*` files. `kernels` stays empty until the trace is `ready`.",
        returns="`200` with a `TraceDetail`.",
        example="trace-detail",
        params=(TRACE_ID,),
        errors=(UNAUTHORIZED, (404, "No trace has this id.")),
        more_examples=(("404 response", "error-404-trace"),),
    ),
    Endpoint(
        key="list-gpu-configs",
        page="api-runs",
        method="GET",
        path="/api/gpu-configs",
        title="List GPU configs",
        summary="The GPU configs a run can use.",
        doc="Lists the GPU configs the app found at startup under `$CASCADE_ACCELSIM_DIR/gpu-simulator/gpgpu-sim/configs/tested-cfgs/*/gpgpusim.config`. The app derives `arch` from the directory name (`SM90_H100` is `sm_90`, `SM7_QV100` is `sm_70`) and reads `core_clock_mhz` from the first value of `-gpgpu_clock_domains`. A directory whose name or clock it can't read is skipped.",
        returns="`200` with a list of `GpuConfig`, sorted by name.",
        example="gpu-configs",
        errors=(UNAUTHORIZED,),
    ),
    Endpoint(
        key="create-run",
        page="api-runs",
        method="POST",
        path="/api/runs",
        title="Create a run",
        summary="Launch sims for a ready trace.",
        doc="Creates a run and starts launching its sims in the background. Each entry fans out to `count` sims, numbered by `replica` from 0. The response comes back before any sim has launched, so every sim is `pending` or `launching`. Follow progress with [Stream run events](api-events.html#run-events) or by polling [Get a run](#get-run).",
        returns="`201` with a `RunDetail`.",
        example="run-create",
        body="RunCreate",
        nested=("RunEntry",),
        errors=(
            UNAUTHORIZED,
            (404, "`trace_id` names no trace."),
            (409, "The trace is not `ready`."),
            (
                422,
                "An entry names an unknown `gpu_config`, or a config whose `arch` differs from the trace's `trace_gpu_arch`.",
            ),
            (
                422,
                "The body fails validation: `entries` is empty, or a `count` is outside 1 to 64.",
            ),
        ),
        more_examples=(
            ("409 response", "error-409"),
            ("422 response, arch mismatch", "error-422-arch"),
            ("422 response, validation", "error-422-validation"),
        ),
    ),
    Endpoint(
        key="list-runs",
        page="api-runs",
        method="GET",
        path="/api/runs",
        title="List runs",
        summary="Every run, newest first.",
        doc="Lists every run, newest first, with per-state sim counts. Use [Get a run](#get-run) for the sims themselves.",
        returns="`200` with a list of `RunSummary`.",
        example="runs-list",
        errors=(UNAUTHORIZED,),
    ),
    Endpoint(
        key="get-run",
        page="api-runs",
        method="GET",
        path="/api/runs/{run_id}",
        title="Get a run",
        summary="One run with its trace and every sim.",
        doc="Returns the run, its trace, and every sim with its state, progress, latest metrics and last 20 log lines.",
        returns="`200` with a `RunDetail`.",
        example="run-detail",
        params=(RUN_ID,),
        errors=(UNAUTHORIZED, (404, "No run has this id.")),
        more_examples=(("404 response", "error-404-run"),),
    ),
    Endpoint(
        key="cancel-run",
        page="api-runs",
        method="POST",
        path="/api/runs/{run_id}/cancel",
        title="Cancel a run",
        summary="Cancel every unfinished sim.",
        doc="Moves every sim that has not finished to `cancelled` and asks the launcher to stop it. Sims that already finished keep their state. The next heartbeat from a cancelled sim gets `cancel: true`. Calling it again changes nothing, so it is safe to retry.",
        returns="`200` with the `RunDetail` after cancelling.",
        example="run-cancel",
        params=(RUN_ID,),
        errors=(UNAUTHORIZED, (404, "No run has this id.")),
    ),
    Endpoint(
        key="get-comparison",
        page="api-runs",
        method="GET",
        path="/api/runs/{run_id}/comparison",
        title="Compare real and simulated time",
        summary="Per-kernel real time against simulated time per config.",
        doc="Lines up kernels by order. Row `i` holds the `i`th kernel the sims finished and the `i`th row of the trace's real timing, sorted by `kernel_id`. For each config, `sim_ns` is the median `sim_time_ns` over that config's replicas, or `null` if no replica has finished that kernel. `real_ns` is `null` until the trace has real timing, which the profiler does not capture yet.",
        returns="`200` with a `Comparison`.",
        example="comparison",
        params=(RUN_ID,),
        errors=(UNAUTHORIZED, (404, "No run has this id.")),
    ),
    Endpoint(
        key="get-run-kernels",
        page="api-runs",
        method="GET",
        path="/api/runs/{run_id}/kernels",
        title="List a run's kernel timings",
        summary="Every sim's per-kernel stats, without the full stats maps.",
        doc="Returns a `KernelStat` for every kernel every sim in the run has finished, ordered by config, replica, then kernel. Each `stats` map is empty to keep the response small; the full maps are in the results downloads. Laying each sim's kernels end to end by `sim_time_ns` gives its timeline, which is what the app's timeline and Compare views draw.",
        returns="`200` with a list of `KernelStat`.",
        example="run-kernels",
        params=(RUN_ID,),
        errors=(UNAUTHORIZED, (404, "No run has this id.")),
    ),
    Endpoint(
        key="get-step-timelines",
        page="api-runs",
        method="GET",
        path="/api/runs/{run_id}/step-timelines",
        title="List a run's step timing",
        summary="Each sim's predicted step timeline and its comparison with the real run.",
        doc="Returns a `SimStepTimeline` for every sim in the run that has sent step timing, ordered by config, then replica. A sim sends one after it succeeds on a trace whose `has_step_timeline` is true, so the list is empty until then, and stays empty for other traces. `simulated` is a full `simulated_timeline.json`, with every kernel of every step, so for a real model it can be several megabytes per sim. Per-kind step means are in `simulated.summary.per_kind`, coverage is in `simulated.summary.simulation.coverage`, and per-step and per-request errors against the real run are in `compare`.",
        returns="`200` with a list of `SimStepTimeline`.",
        example="run-step-timelines",
        params=(RUN_ID,),
        errors=(UNAUTHORIZED, (404, "No run has this id.")),
    ),
    Endpoint(
        key="get-run-results",
        page="api-runs",
        method="GET",
        path="/api/runs/{run_id}/results",
        title="Download run results",
        summary="One JSON file with the run, trace, every sim's stats and the comparison.",
        doc='Returns everything about a run as one JSON download, with `Content-Disposition: attachment; filename="<run_id>-results.json"`. Each sim\'s `kernels` carry the full `stats` map, every `key = value` line Accel-Sim printed for that kernel, which is about 30 KB per kernel.',
        returns="`200` with a `RunResults`.",
        example="run-results",
        kind="download",
        params=(RUN_ID,),
        errors=(UNAUTHORIZED, (404, "No run has this id.")),
    ),
    Endpoint(
        key="get-sim-results",
        page="api-runs",
        method="GET",
        path="/api/sims/{sim_id}/results",
        title="Download sim results",
        summary="One sim's config, per-kernel stats and totals.",
        doc="Returns one sim's results as a JSON download named `<sim_id>-results.json`. `totals` is the sim's latest `SimMetrics`.",
        returns="`200` with a `SimResults`.",
        example="sim-results",
        kind="download",
        params=(SIM_ID,),
        errors=(UNAUTHORIZED, (404, "No sim has this id.")),
    ),
    Endpoint(
        key="get-sim-log",
        page="api-runs",
        method="GET",
        path="/api/sims/{sim_id}/log",
        title="Download a sim's log",
        summary="The sim's full raw output.",
        doc="Streams every line the simulator sent, in `seq` order, as `text/plain` named `<sim_id>.log`.",
        returns="`200` with the log as plain text.",
        example="sim-log",
        kind="download",
        params=(SIM_ID,),
        errors=(UNAUTHORIZED, (404, "No sim has this id.")),
    ),
    Endpoint(
        key="list-run-files",
        page="api-runs",
        method="GET",
        path="/api/runs/{run_id}/files",
        title="List a run's output files",
        summary="Every output file the run's sims uploaded.",
        doc="Lists the files every sim in the run uploaded, ordered by config, replica, then name. Each `RunFile` names its sim, so one call covers the run. `cascade worker` uploads a sim's files just before its final heartbeat, so the list grows as sims finish. Fetch one with [Download an output file](#get-sim-file).",
        returns="`200` with a list of `RunFile`.",
        example="run-files",
        params=(RUN_ID,),
        errors=(UNAUTHORIZED, (404, "No run has this id.")),
    ),
    Endpoint(
        key="get-run-perfetto",
        page="api-runs",
        method="GET",
        path="/api/runs/{run_id}/perfetto",
        title="Download a Perfetto trace",
        summary="The run's simulated kernels as a Chrome trace JSON for ui.perfetto.dev.",
        doc='Builds a Chrome trace JSON from the run\'s parsed kernel stats and returns it with `Content-Disposition: attachment; filename="<run_id>-perfetto.json"`. Each config is a process, each replica is a thread in it, and each finished kernel is one slice. The app builds it from the stats it parsed, so every run has one, whether or not the simulator wrote its own. [Reading results](reading-results.html#the-perfetto-trace) describes the layout.',
        returns="`200` with a Chrome trace JSON object: `traceEvents`, `displayTimeUnit` and `otherData`.",
        example="run-perfetto",
        kind="download",
        params=(RUN_ID,),
        errors=(UNAUTHORIZED, (404, "No run has this id.")),
    ),
    Endpoint(
        key="list-sim-files",
        page="api-runs",
        method="GET",
        path="/api/sims/{sim_id}/files",
        title="List a sim's output files",
        summary="The output files one sim uploaded.",
        doc="Lists one sim's uploaded files, sorted by name. The list is empty until the simulator uploads something.",
        returns="`200` with a list of `SimFile`.",
        example="sim-files",
        params=(SIM_ID,),
        errors=(UNAUTHORIZED, (404, "No sim has this id.")),
    ),
    Endpoint(
        key="get-sim-file",
        page="api-runs",
        method="GET",
        path="/api/sims/{sim_id}/files/{name}",
        title="Download an output file",
        summary="One output file, as the sim uploaded it.",
        doc="Returns the file as a download named `name`, with the file's `content_type`. A `.gz` file comes back as `application/gzip` with no `Content-Encoding`, so it stays compressed.",
        returns="`200` with the file's bytes.",
        example="sim-file",
        kind="download",
        params=(SIM_ID, FILE_NAME),
        errors=(
            UNAUTHORIZED,
            (404, "No sim has this id, or the sim has no file with this name."),
        ),
        more_examples=(("404 response, no such file", "error-404-file"),),
    ),
    Endpoint(
        key="run-events",
        page="api-events",
        method="GET",
        path="/api/runs/{run_id}/events",
        title="Stream run events",
        summary="Server-sent events for one run.",
        doc="Opens a `text/event-stream`. The first event is always a `snapshot`. After that the stream sends `sim`, `log` and `kernel` events as they happen, and a `: keepalive` comment after 15 seconds without one. The stream does not close when the run finishes.",
        returns="`200` with a `text/event-stream` of `RunEvent`, one JSON object per `data:` line.",
        example="events",
        auth="bearer-or-query",
        kind="sse",
        params=(
            RUN_ID,
            Param(
                "token",
                "str",
                "query",
                "Your `CASCADE_TOKEN`. Use it when you can't set headers, as with `EventSource`. If present, it is checked instead of the header.",
                required=False,
            ),
        ),
        errors=(UNAUTHORIZED, (404, "No run has this id.")),
        extra="",
    ),
    Endpoint(
        key="heartbeat",
        page="api-simulator",
        method="POST",
        path="/api/sims/{sim_id}/heartbeat",
        title="Send a heartbeat",
        summary="Report a sim's state. The reply says whether to stop.",
        doc="Records that the sim is alive and, if `state` differs, moves the sim to it when `SIM_TRANSITIONS` allows the move. A `running` heartbeat brings a `lost` sim back. A heartbeat for a sim that already finished changes nothing and returns its final state. Send one every few seconds. A sim with no heartbeat for `CASCADE_HEARTBEAT_TIMEOUT_S` (60 by default) becomes `lost`.",
        returns="`200` with a `HeartbeatAck`. If `cancel` is `true`, stop simulating and exit.",
        example="heartbeat-finished",
        params=(SIM_ID,),
        body="Heartbeat",
        errors=(
            UNAUTHORIZED,
            (404, "No sim has this id."),
            (422, "`state` is not `running`, `succeeded` or `failed`."),
        ),
        more_examples=(
            ("Response to a cancelled sim", "heartbeat-cancelled"),
            ("404 response", "error-404-sim"),
        ),
    ),
    Endpoint(
        key="logs",
        page="api-simulator",
        method="POST",
        path="/api/sims/{sim_id}/logs",
        title="Send log lines",
        summary="Append a chunk of simulator output.",
        doc="Appends `lines` to the sim's log and parses any finished Accel-Sim kernel blocks out of them. Number chunks from 0 and add one each time. The app stores the first chunk it gets for each `seq` and ignores repeats, so a retry after a timeout is safe. Send lines without their trailing newline.",
        returns="`204` with no body, whether the chunk was new or a repeat.",
        example="logs-post",
        params=(SIM_ID,),
        body="LogChunk",
        errors=(
            UNAUTHORIZED,
            (404, "No sim has this id."),
            (422, "`seq` is negative or `lines` is not a list of strings."),
        ),
    ),
    Endpoint(
        key="post-step-timeline",
        page="api-simulator",
        method="POST",
        path="/api/sims/{sim_id}/step-timeline",
        title="Send step timing",
        summary="Store a sim's predicted step timeline and its comparison.",
        doc="Stores the output of `cascade simulate` and `cascade compare` for this sim. `cascade worker` sends it after a successful sim on a trace with step timing, before its final heartbeat. Sending again replaces the earlier upload, so a retry is safe. The app checks only that `simulated` has a `steps` list and a `summary` object and stores both objects as they are.",
        returns="`204` with no body.",
        example="step-timeline-post",
        params=(SIM_ID,),
        body="StepTimelineUpload",
        errors=(
            UNAUTHORIZED,
            (404, "No sim has this id."),
            (
                422,
                "`simulated` lacks a `steps` list or a `summary` object, or `compare` is neither an object nor `null`.",
            ),
        ),
    ),
    Endpoint(
        key="put-sim-file",
        page="api-simulator",
        method="PUT",
        path="/api/sims/{sim_id}/files/{name}",
        title="Upload an output file",
        summary="Store one of the simulator's output files.",
        doc="Stores the request body as the sim's file `name`, in `$CASCADE_DATA_DIR/files/<sim_id>/`. Putting the same name again replaces the file, so a retry is safe. `cascade worker` uploads every file the simulator wrote, then its own log as `accelsim.log.gz`, before its final heartbeat.",
        returns="`200` with the stored `SimFile`.",
        example="file-put",
        params=(SIM_ID, FILE_NAME),
        raw_body=True,
        errors=(
            UNAUTHORIZED,
            (404, "No sim has this id."),
            (413, f"The file is over {FILE_MAX_BYTES >> 20} MiB."),
            (
                413,
                f"The sim's files would pass {SIM_FILES_MAX_BYTES >> 20} MiB in all. The file a put replaces doesn't count.",
            ),
            (
                422,
                "`name` has a character other than letters, digits, `.`, `_` and `-`, is over 128 characters, or is `.` or `..`.",
            ),
        ),
        more_examples=(
            ("413 response, file too big", "error-413-file"),
            ("422 response, bad name", "error-422-file-name"),
        ),
    ),
)
