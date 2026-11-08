"""Prose for the Types page that cascade/models.py has no comment for.

build.py checks these against the models: every model sits in one group, every enum value has a
meaning, and a note can't name a field that is gone or that already has a comment in the source.
"""

# ruff: noqa: E501

from __future__ import annotations

MODEL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("States", ("TraceState", "SimState", "RunState")),
    ("Traces", ("GpuConfig", "KernelLaunch", "TraceSummary", "TraceDetail")),
    (
        "Runs and sims",
        ("RunEntry", "RunCreate", "RunSummary", "RunDetail", "SimSummary", "SimMetrics"),
    ),
    (
        "Results",
        (
            "KernelStat",
            "ComparisonRow",
            "Comparison",
            "SimResults",
            "RunResults",
            "SimStepTimeline",
            "SimFile",
            "RunFile",
        ),
    ),
    ("Simulator calls", ("Heartbeat", "HeartbeatAck", "LogChunk", "StepTimelineUpload")),
    ("Events", ("RunEvent", "SnapshotEvent", "SimEvent", "LogEvent", "KernelEvent")),
)

ENUM_NOTES: dict[str, dict[str, str]] = {
    "TraceState": {
        "uploading": "Files are arriving. `kernel_count` and `bytes` grow on each scan, every 2 seconds.",
        "processing": "The upload is complete. The app is running the trace post-processor, unless the upload already has `traces/kernelslist.g`.",
        "ready": "Post-processing produced `traces/kernelslist.g`. Runs can launch.",
        "failed": '`metadata.json` has `status: "failed"`, or post-processing failed. `error` says which.',
    },
    "SimState": {
        "pending": "Created with the run. Waiting for the launcher.",
        "launching": "The launcher is starting the sim's sandbox or process.",
        "running": "The sim sent a `running` heartbeat.",
        "lost": "No heartbeat for `CASCADE_HEARTBEAT_TIMEOUT_S`. A later heartbeat brings it back.",
        "succeeded": "The sim sent a `succeeded` heartbeat. Terminal.",
        "failed": "The sim sent a `failed` heartbeat, the launcher failed, or it stayed `lost` for 10 times the timeout. Terminal.",
        "cancelled": "The run was cancelled before this sim finished. Terminal.",
    },
    "RunState": {
        "active": "At least one sim has not finished.",
        "succeeded": "Every sim succeeded.",
        "partial": "Every sim finished, at least one succeeded, and at least one did not.",
        "failed": "Every sim failed.",
        "cancelled": "Every sim finished, at least one was cancelled, and none succeeded.",
    },
}

RUN_STATE_CASES: tuple[tuple[str, ...], ...] = (
    ("running", "succeeded"),
    ("succeeded", "succeeded"),
    ("succeeded", "failed"),
    ("failed", "failed"),
    ("cancelled", "failed"),
    ("cancelled", "succeeded"),
    ("cancelled", "cancelled"),
)

FIELD_NOTES: dict[str, str] = {
    "RunState": "`RunState` is not stored. `run_state()` in `cascade/models.py` derives it from the run's sim states each time a run is read.",
    "RunEvent": "One event on the [run event stream](api-events.html). The `type` field says which model it is.",
    "TraceSummary.state": "Where the trace is in [its lifecycle](concepts.html#trace-states).",
    "TraceSummary.error": "Why the trace failed. `null` unless `state` is `failed`.",
    "TraceSummary.created_at": "When the first file of the trace landed.",
    "TraceSummary.updated_at": "When the app last changed this trace.",
    "TraceSummary.model": "`metadata.json` `model`, such as `facebook/opt-125m`.",
    "TraceSummary.layers": "`metadata.json` `layers`. Empty if missing or not a list.",
    "TraceSummary.suggested_sim_config": "`metadata.json` `suggested_sim_config`. The UI preselects it on the New run screen.",
    "TraceSummary.has_real_timing": "True when the trace has real per-kernel timings (`real_timing.json`). The profiler does not write them yet.",
    "TraceSummary.sample": "True for the sample trace that `cascade serve` seeds from `samples/` (see [Introduction](index.html)), unless `CASCADE_SAMPLES` is false. It is `ready` but has no instruction traces, so it can't start a run.",
    "RunSummary.sample": "True for the sample run seeded with the sample trace. Its results were recorded when the run was made, not in this workspace, and cancelling it changes nothing.",
    "TraceDetail.metadata": "The trace's `metadata.json`, parsed. `null` until it arrives.",
    "TraceDetail.kernels": "Kernel launches from the trace's `stats_ctx_*` files, by `kernel_id`. Filled in when the trace becomes `ready`.",
    "RunEntry.gpu_config": "A `GpuConfig.name` whose `arch` matches the trace's `trace_gpu_arch`.",
    "RunEntry.count": "How many sims run this config.",
    "RunCreate.trace_id": "A trace in state `ready`.",
    "RunCreate.entries": "One entry per config. Each fans out to `count` sims.",
    "RunSummary.id": "Generated, `run-` and 12 hex digits.",
    "RunSummary.state": "Derived from the sims. See `RunState`.",
    "RunSummary.entries": "The entries the run was created with.",
    "RunSummary.sim_counts": "How many sims are in each state. States with no sims are left out.",
    "RunDetail.trace": "The run's trace.",
    "RunDetail.sims": "Every sim, sorted by `gpu_config` and then `replica`.",
    "SimSummary.id": "Generated, `sim-` and 12 hex digits. The simulator receives it as `CASCADE_SIM_ID`.",
    "SimSummary.gpu_config": "The config this sim simulates. The simulator receives it as `CASCADE_GPU_CONFIG`.",
    "SimSummary.error": "Why the sim failed or went `lost`, from a heartbeat or from the app. A `running` heartbeat clears it.",
    "SimSummary.started_at": "The first `running` heartbeat.",
    "SimSummary.finished_at": "When the sim reached a terminal state.",
    "SimSummary.exit_code": "From the final heartbeat, when the simulator sent one.",
    "SimSummary.kernels_done": "Kernel stats blocks parsed from the log so far.",
    "SimSummary.kernels_total": "The trace's kernel count when the run was created.",
    "SimSummary.current_kernel": "The kernel named by the last `launching kernel name:` line in the log.",
    "SimSummary.latest": "The most recent totals parsed from the log.",
    "KernelLaunch.name": "Mangled kernel name.",
    "KernelLaunch.grid": "Grid size, x, y and z.",
    "KernelLaunch.block": "Block size, x, y and z.",
    "KernelLaunch.instructions": "Instructions the tracer counted. `null` if the kernel's trace was cut short.",
    "SimMetrics.gpu_tot_sim_cycle": "Simulated cycles so far, over every kernel.",
    "SimMetrics.gpu_tot_sim_insn": "Simulated instructions so far, over every kernel.",
    "SimMetrics.gpu_tot_ipc": "Instructions per cycle so far, over every kernel.",
    "SimSummary.run_id": "The run this sim belongs to.",
    "SimSummary.state": "Where the sim is in [its lifecycle](concepts.html#sim-states).",
    "SimSummary.created_at": "When the run was created.",
    "SimSummary.last_heartbeat_at": "The last heartbeat. `null` before the first one.",
    "RunSummary.trace_id": "The trace the run simulates.",
    "RunSummary.created_at": "When the run was created.",
    "KernelStat.sim_id": "The sim that ran this kernel.",
    "Comparison.rows": "One row per kernel position.",
    "SimResults.sim": "The sim as `SimSummary`.",
    "SimResults.trace_id": "The trace the sim ran.",
    "KernelStat.kernel_name": "The block's `kernel_name`.",
    "KernelStat.kernel_launch_uid": "The block's `kernel_launch_uid`.",
    "KernelStat.gpu_sim_cycle": "Cycles this kernel took on the simulated GPU.",
    "KernelStat.gpu_sim_insn": "Instructions this kernel ran.",
    "KernelStat.gpu_ipc": "This kernel's instructions per cycle.",
    "ComparisonRow.kernel_index": "Row position. Kernels are matched by order.",
    "ComparisonRow.kernel_name": "From real timing if present, otherwise from the first sim that finished this kernel.",
    "ComparisonRow.real_ns": "`duration_ns` from real timing. `null` without real timing.",
    "Comparison.has_real_timing": "False until the trace has real timing.",
    "Comparison.configs": "The run's configs, in entry order, without repeats.",
    "SimResults.gpu_config": "The full config. `arch` is `unknown` and the clock is 0 if the config disappeared since the run started.",
    "SimResults.kernels": "Every kernel the sim finished, with the full `stats` map.",
    "SimResults.totals": "Same as `sim.latest`.",
    "RunResults.run": "The run without its trace and sims, which are below.",
    "RunResults.trace": "The full trace.",
    "RunResults.sims": "One entry per sim.",
    "RunResults.comparison": "Same as [Compare real and simulated time](api-runs.html#get-comparison).",
    "Heartbeat.state": "What the simulator reports. Send `running` while working and `succeeded` or `failed` once at the end.",
    "Heartbeat.exit_code": "Your process's exit code. Send it with the final heartbeat.",
    "Heartbeat.error": "A short reason, with a `failed` heartbeat.",
    "HeartbeatAck.state": "The sim's state after this heartbeat.",
    "LogChunk.lines": "Lines of simulator stdout, without trailing newlines.",
    "SimStepTimeline.sim_id": "The sim that sent it.",
    "SimStepTimeline.gpu_config": "The sim's config.",
    "SimStepTimeline.replica": "The sim's replica.",
    "SimStepTimeline.simulated": "As sent in `StepTimelineUpload.simulated`.",
    "SimStepTimeline.compare": "As sent in `StepTimelineUpload.compare`.",
    "SimFile.content_type": "From the name: `text/csv` for `.csv`, `text/plain` for `.txt` and `.log`, `application/json` for `.json`, `application/gzip` for `.gz`, and otherwise a guess, falling back to `application/octet-stream`.",
    "RunFile.sim_id": "The sim that uploaded the file.",
    "RunFile.gpu_config": "The sim's config.",
    "RunFile.replica": "The sim's replica.",
    "SnapshotEvent.run": "The run as it is at connect time.",
    "SimEvent.sim": "The sim after a change: state, heartbeat, progress or metrics.",
    "LogEvent.sim_id": "The sim the lines came from.",
    "LogEvent.lines": "The new chunk, as the simulator sent it.",
    "KernelEvent.stat": "A kernel that just finished, with the full `stats` map.",
    "SnapshotEvent.type": "Always `snapshot`.",
    "SimEvent.type": "Always `sim`.",
    "LogEvent.type": "Always `log`.",
    "KernelEvent.type": "Always `kernel`.",
}

# Env vars read by cascade/settings.py: (default to show instead of the parsed one, description).
SETTINGS_NOTES: dict[str, tuple[str | None, str]] = {
    "CASCADE_TOKEN": (
        "required",
        "The shared bearer token. The UI and every sim send it. The app refuses to start without it unless `CASCADE_NO_AUTH` is set.",
    ),
    "CASCADE_NO_AUTH": (
        "false",
        "Local development only. Set to `1` to turn off the token check: the API accepts any request, `/api/health` reports `auth: false`, and the web UI skips its token screen. Never set it on a deployment someone else can reach.",
    ),
    "CASCADE_ACCELSIM_DIR": (
        None,
        "Accel-Sim checkout. The app reads GPU configs from `gpu-simulator/gpgpu-sim/configs/tested-cfgs/*/gpgpusim.config` under it.",
    ),
    "CASCADE_DATA_DIR": (
        None,
        "Where the app keeps `cascade.db`, the S3 bucket directory `s3/`, sims' output files in `files/`, and scratch files in `work/`.",
    ),
    "CASCADE_PUBLIC_URL": (
        None,
        "The URL sims use to reach the app. Each sim gets it as `CASCADE_MAIN_URL`.",
    ),
    "CASCADE_S3_BUCKET": (
        None,
        "The bucket the profiler uploads to. It is a directory under `$CASCADE_DATA_DIR/s3/`.",
    ),
    "CASCADE_S3_PREFIX": (
        None,
        "The key prefix traces live under. Each trace is `<prefix>/<run_name>/`.",
    ),
    "CASCADE_SIM_CMD": (
        None,
        "The command each sim runs. On Sail it runs through a shell. Locally it is split into arguments and run without one. Without it, every sim fails at launch.",
    ),
    "CASCADE_POSTPROCESS_CMD": (
        "`$CASCADE_ACCELSIM_DIR/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing`",
        "Accel-Sim's trace post-processor. The app runs it once per trace as `<cmd> <trace>/traces -j <cpus>`.",
    ),
    "CASCADE_HEARTBEAT_TIMEOUT_S": (
        None,
        "Seconds without a heartbeat before a sim becomes `lost`. After 10 times this, a `lost` sim becomes `failed`.",
    ),
    "CASCADE_OPENAI_WAIT_S": (
        None,
        "Seconds `/v1` and `/api/scenarios` wait for base-simulation step timing. A timeout returns 504 with the run/sim IDs; it does not cancel the simulation.",
    ),
    "CASCADE_WORKSPACE": (
        None,
        "The workspace's name. `/api/health` returns it and the web UI shows it in the rail. `cascade deploy` and the Get started page set it.",
    ),
    "CASCADE_S3_PUBLIC_URL": (
        None,
        "The S3 endpoint the profiler uploads to. `/api/health` returns it as `s3_endpoint` and the UI fills it into the send-a-trace command. `cascade deploy` sets it; a local `cascade serve` derives it from `CASCADE_PUBLIC_URL`'s host and `--s3-port`.",
    ),
    "CASCADE_SAMPLES": (
        None,
        "Whether `cascade serve` seeds the bundled sample trace and run (`samples/`) into the workspace on startup. Set `false` to start empty. A sample already seeded stays.",
    ),
    "CASCADE_RUN_S3": (
        None,
        "Whether `cascade serve` starts `rclone serve s3` next to the app. Set `false` to run the S3 endpoint yourself.",
    ),
}
