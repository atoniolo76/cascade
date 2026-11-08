"""A run as a Chrome trace JSON for ui.perfetto.dev, built from its parsed kernel stats.

Follows Alessio's Accel-Sim export (results/20261108-021007-opt125m-h100-accelsim/outputs/
opt125m_H100_perfetto.json.gz): `X` slices named `k<uid> <kernel>`, ts and dur in µs, and
cumulative start/end cycles in args. Kernels run one after another, so each sim's slices are laid
end to end in simulated time.
"""

from __future__ import annotations

from collections import defaultdict

from cascade.kernel_names import kernel_display_name
from cascade.models import KernelStat, RunDetail


def _meta(name: str, pid: int, value: str | int, tid: int | None = None) -> dict:
    key = "sort_index" if name.endswith("sort_index") else "name"
    event = {"ph": "M", "pid": pid, "name": name, "args": {key: value}}
    if tid is not None:
        event["tid"] = tid
    return event


def _label(uid: int | None, name: str) -> str:
    readable = kernel_display_name(name)
    return readable if uid is None else f"k{uid} {readable}"


def _sim_slices(pid: int, tid: int, kernels: list[KernelStat]) -> list[dict]:
    events = []
    start_ns, cycle = 0.0, 0
    for k in sorted(kernels, key=lambda k: k.kernel_index):
        if k.sim_time_ns is None:
            continue
        cycles = k.gpu_sim_cycle or 0
        events.append(
            {
                "ph": "X",
                "pid": pid,
                "tid": tid,
                "cat": "kernel",
                "name": _label(k.kernel_launch_uid, k.kernel_name),
                "ts": start_ns / 1000,
                "dur": k.sim_time_ns / 1000,
                "args": {
                    "kernel_index": k.kernel_index,
                    "uid": k.kernel_launch_uid,
                    "kernel_name": k.kernel_name,
                    "gpu_sim_cycle": k.gpu_sim_cycle,
                    "gpu_sim_insn": k.gpu_sim_insn,
                    "ipc": k.gpu_ipc,
                    "sim_time_ns": k.sim_time_ns,
                    "start_cycle": cycle,
                    "end_cycle": cycle + cycles,
                },
            }
        )
        start_ns += k.sim_time_ns
        cycle += cycles
    return events


def _real_slices(pid: int, rows: list[dict]) -> list[dict]:
    events = []
    start_ns = 0.0
    for row in rows:
        duration = row.get("duration_ns")
        if duration is None:
            continue
        events.append(
            {
                "ph": "X",
                "pid": pid,
                "tid": 1,
                "cat": "kernel",
                "name": _label(row.get("kernel_id"), str(row.get("kernel_name") or "")),
                "ts": start_ns / 1000,
                "dur": float(duration) / 1000,
                "args": {
                    k: row[k]
                    for k in ("kernel_id", "kernel_name", "duration_ns", "grid", "block")
                    if k in row
                },
            }
        )
        start_ns += float(duration)
    return events


def run_trace(run: RunDetail, kernels: list[KernelStat], real: list[dict] | None) -> dict:
    """One process per GPU config with one thread per replica, plus a real-hardware process
    when the trace has real timing."""
    by_sim: dict[str, list[KernelStat]] = defaultdict(list)
    for k in kernels:
        by_sim[k.sim_id].append(k)
    configs = list(dict.fromkeys(e.gpu_config for e in run.entries))
    events: list[dict] = []
    for pid, config in enumerate(configs, 1):
        events += [
            _meta("process_name", pid, f"{config} · Accel-Sim simulated GPU"),
            _meta("process_sort_index", pid, pid),
        ]
        for sim in sorted((s for s in run.sims if s.gpu_config == config), key=lambda s: s.replica):
            tid = sim.replica + 1
            events += [
                _meta("thread_name", pid, f"replica {sim.replica} · {sim.id}", tid),
                _meta("thread_sort_index", pid, tid, tid),
                *_sim_slices(pid, tid, by_sim[sim.id]),
            ]
    if real:
        pid = len(configs) + 1
        gpu = run.trace.source_gpu
        events += [
            _meta("process_name", pid, f"Real hardware · {gpu}" if gpu else "Real hardware"),
            _meta("process_sort_index", pid, pid),
            _meta("thread_name", pid, "measured kernels", 1),
            *_real_slices(pid, real),
        ]
    return {
        "traceEvents": events,
        "displayTimeUnit": "ns",
        "otherData": {
            "run_id": run.id,
            "trace_id": run.trace_id,
            "time_axis": "simulated time, microseconds; each sim's kernels laid end to end",
            "slice_duration": "sim_time_ns = gpu_sim_cycle / core clock (MHz) * 1000",
            "kernel_scope": "sequential kernels, each including the configured launch latency;"
            " excludes host time and memcpy",
        },
    }
