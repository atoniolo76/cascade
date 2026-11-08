"""API contract between main, simulators, and the UI. Mirrored by web/src/api/types.ts."""

from __future__ import annotations

import mimetypes
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class TraceState(StrEnum):
    uploading = "uploading"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class SimState(StrEnum):
    pending = "pending"
    launching = "launching"
    running = "running"
    lost = "lost"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL_SIM_STATES = frozenset({SimState.succeeded, SimState.failed, SimState.cancelled})

# Allowed sim transitions. Main owns pending/launching/cancelled/lost; heartbeats own the rest.
SIM_TRANSITIONS: dict[SimState, frozenset[SimState]] = {
    SimState.pending: frozenset({SimState.launching, SimState.cancelled, SimState.failed}),
    SimState.launching: frozenset(
        {SimState.running, SimState.succeeded, SimState.failed, SimState.cancelled, SimState.lost}
    ),
    SimState.running: frozenset(
        {SimState.succeeded, SimState.failed, SimState.cancelled, SimState.lost}
    ),
    SimState.lost: frozenset(
        {SimState.running, SimState.succeeded, SimState.failed, SimState.cancelled}
    ),
    SimState.succeeded: frozenset(),
    SimState.failed: frozenset(),
    SimState.cancelled: frozenset(),
}

RunState = Literal["active", "succeeded", "partial", "failed", "cancelled"]


def run_state(sim_states: list[SimState]) -> RunState:
    if not sim_states or any(s not in TERMINAL_SIM_STATES for s in sim_states):
        return "active"
    if all(s == SimState.succeeded for s in sim_states):
        return "succeeded"
    if all(s == SimState.failed for s in sim_states):
        return "failed"
    if any(s == SimState.cancelled for s in sim_states) and not any(
        s == SimState.succeeded for s in sim_states
    ):
        return "cancelled"
    return "partial"


class GpuConfig(BaseModel):
    name: str  # Accel-Sim tested-cfgs directory name, e.g. "SM90_H100"
    arch: str  # e.g. "sm_90"
    core_clock_mhz: float  # first value of -gpgpu_clock_domains in gpgpusim.config


class KernelLaunch(BaseModel):
    """One kernel launch from the trace's stats_ctx_* file (original hardware)."""

    kernel_id: int  # matches kernel-<N> in the trace filename
    name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    instructions: int | None


class TraceSummary(BaseModel):
    id: str  # the profiler's run_name
    state: TraceState
    error: str | None
    created_at: datetime
    updated_at: datetime
    model: str | None
    layers: list[str]
    source_gpu: str | None  # metadata.json "gpu"
    trace_gpu_arch: str  # metadata.json "trace_gpu_arch", default "sm_90"
    suggested_sim_config: str | None
    real_gpu_config: str | None = None
    sample_sim_configs: list[str] = Field(default_factory=list)
    sample_set: str | None = None
    kernel_count: int  # traced kernels seen so far
    bytes: int  # bytes on disk under the trace dir
    has_real_timing: bool
    has_step_timeline: bool  # corrected_timeline.json + kernel_mapping.json are in the trace dir
    sample: bool


class TraceDetail(TraceSummary):
    metadata: dict | None
    kernels: list[KernelLaunch]


class RunEntry(BaseModel):
    gpu_config: str
    count: Annotated[int, Field(ge=1, le=64)]


class RunCreate(BaseModel):
    trace_id: str
    entries: Annotated[list[RunEntry], Field(min_length=1)]


class SimMetrics(BaseModel):
    gpu_tot_sim_cycle: float | None = None
    gpu_tot_sim_insn: float | None = None
    gpu_tot_ipc: float | None = None
    sim_rate_ips: float | None = None  # gpgpu_simulation_rate, instructions/sec


class SimSummary(BaseModel):
    id: str
    run_id: str
    gpu_config: str
    replica: int  # 0-based index within its {gpu_config, count} entry
    state: SimState
    error: str | None
    launcher_ref: str | None  # sailbox id or local pid
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    last_heartbeat_at: datetime | None
    exit_code: int | None
    kernels_done: int
    kernels_total: int
    current_kernel: str | None
    latest: SimMetrics
    log_tail: list[str]  # last 20 lines


class KernelStat(BaseModel):
    """One per-kernel stats block parsed from Accel-Sim output."""

    sim_id: str
    kernel_index: int  # 0-based order of completion within the sim
    kernel_name: str
    kernel_launch_uid: int | None
    gpu_sim_cycle: int | None
    gpu_sim_insn: int | None
    gpu_ipc: float | None
    sim_time_ns: float | None  # gpu_sim_cycle / core_clock_mhz * 1000
    stats: dict[str, float | str]  # every "key = value" line in the block


class RunSummary(BaseModel):
    id: str
    trace_id: str
    created_at: datetime
    state: RunState
    entries: list[RunEntry]
    sim_counts: dict[SimState, int]
    sample: bool


class RunDetail(RunSummary):
    trace: TraceSummary
    sims: list[SimSummary]


class Heartbeat(BaseModel):
    state: Literal["running", "succeeded", "failed"]
    exit_code: int | None = None
    error: str | None = None


class HeartbeatAck(BaseModel):
    state: SimState
    cancel: bool  # true when main cancelled this sim; the simulator must stop


class LogChunk(BaseModel):
    seq: Annotated[int, Field(ge=0)]  # per-sim, increasing; a repeated seq is ignored
    lines: list[str]


class StepTimelineUpload(BaseModel):
    """What a sim posts after step timing: `cascade simulate` and `cascade compare` output."""

    simulated: dict  # `cascade simulate` output (simulated_timeline.json), with steps and summary
    compare: dict | None  # `cascade compare` output (compare.json) against the real timeline

    @field_validator("simulated")
    @classmethod
    def _timeline_shape(cls, value: dict) -> dict:
        if not isinstance(value.get("steps"), list) or not isinstance(value.get("summary"), dict):
            raise ValueError("simulated needs a steps list and a summary object")
        return value


class SimStepTimeline(BaseModel):
    sim_id: str
    gpu_config: str
    replica: int
    simulated: dict
    compare: dict | None


FILE_NAME_PATTERN = r"^(?!\.\.?$)[A-Za-z0-9._-]{1,128}$"
FILE_MAX_BYTES = 64 << 20  # per output file; a bigger PUT is 413
SIM_FILES_MAX_BYTES = 256 << 20  # all of one sim's output files together
CONTENT_TYPES = {
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".json": "application/json",
    ".gz": "application/gzip",
}


def content_type(name: str) -> str:
    known = CONTENT_TYPES.get(Path(name).suffix.lower())
    return known or mimetypes.guess_type(name, strict=False)[0] or "application/octet-stream"


class SimFile(BaseModel):
    """An output file a sim uploaded with PUT /api/sims/{id}/files/{name}."""

    name: str  # matches FILE_NAME_PATTERN; Accel-Sim's name, plus .gz when the worker gzipped it
    size: int  # bytes as stored
    content_type: str
    created_at: datetime  # when this version was stored; a re-put replaces the file


class RunFile(SimFile):
    sim_id: str
    gpu_config: str
    replica: int


class ComparisonRow(BaseModel):
    kernel_index: int
    kernel_name: str
    real_ns: float | None
    sim_ns: dict[str, float | None]  # gpu_config -> median sim_time_ns over replicas


class Comparison(BaseModel):
    has_real_timing: bool
    configs: list[str]
    rows: list[ComparisonRow]


class SimResults(BaseModel):
    sim: SimSummary
    trace_id: str
    gpu_config: GpuConfig
    kernels: list[KernelStat]
    totals: SimMetrics


class RunResults(BaseModel):
    run: RunSummary
    trace: TraceDetail
    sims: list[SimResults]
    comparison: Comparison


class SnapshotEvent(BaseModel):
    type: Literal["snapshot"] = "snapshot"
    run: RunDetail
    kernels: list[KernelStat]  # every sim's stats with `stats` emptied; full stats are ~30 KB each


class SimEvent(BaseModel):
    type: Literal["sim"] = "sim"
    sim: SimSummary


class LogEvent(BaseModel):
    type: Literal["log"] = "log"
    sim_id: str
    lines: list[str]


class KernelEvent(BaseModel):
    type: Literal["kernel"] = "kernel"
    stat: KernelStat


RunEvent = Annotated[SnapshotEvent | SimEvent | LogEvent | KernelEvent, Field(discriminator="type")]
