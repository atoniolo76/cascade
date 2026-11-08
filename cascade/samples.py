"""Seeds each bundle under samples/ (a real profiling and Accel-Sim run) into the workspace, so it
opens with a finished trace and run to explore. A bundle is a trimmed run directory in the layout
of samples/opt125m-h100-e2e/README.md plus its sample.json; scripts/build_sample.py makes one.

Seeding is one store transaction keyed on the trace id, so it runs at most once per workspace and
an interrupted start redoes it from scratch.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cascade.models import (
    KernelLaunch,
    KernelStat,
    SimFile,
    SimMetrics,
    SimState,
    SimSummary,
    StepTimelineUpload,
    TraceDetail,
    TraceState,
    content_type,
)
from cascade.settings import Settings
from cascade.stats import LogParser
from cascade.store import SampleRows, Store

log = logging.getLogger(__name__)

SAMPLES_ROOT = Path(__file__).resolve().parent.parent / "samples"
TRACE_FILES = (
    "corrected_timeline.json",
    "trace_alignment.json",
    "traced/trace_steps.json",
    "traced/traces/kernelslist.g",
)


class Bundle:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = json.loads((root / "sample.json").read_text())

    def read(self, rel: str) -> bytes:
        """A bundled file's original bytes, whether it is stored plain or as `<rel>.gz`."""
        path = self.root / rel
        if path.is_file():
            return path.read_bytes()
        return gzip.decompress(path.with_name(f"{path.name}.gz").read_bytes())

    def read_json(self, rel: str):
        return json.loads(self.read(rel))


@dataclass(frozen=True)
class Sample:
    rows: SampleRows
    trace_files: dict[str, bytes]  # path under the trace dir -> bytes
    output_files: dict[str, bytes]  # output file name -> bytes


def _utc(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, UTC)


def _simulated(bundle: Bundle) -> list[dict]:
    """kernels.csv rows Accel-Sim ran itself (not copied from a dedup representative), by uid."""
    rows = csv.DictReader(io.StringIO(bundle.read("sim/kernels.csv").decode()))
    return sorted((r for r in rows if r["sim"] == "exact"), key=lambda r: int(r["uid"]))


def _stats(bundle: Bundle, sim_id: str, simulated: list[dict]) -> list[KernelStat]:
    """One stat per simulated kernel in launch order, parsed from that kernel's own Accel-Sim log.
    Each kernel ran alone, so its log's launch uid is 1; the tracer's uid replaces it."""
    clock = bundle.manifest["core_clock_mhz"]
    stats = []
    for index, row in enumerate(simulated):
        parser = LogParser(sim_id, clock)
        lines = bundle.read(f"sim/kernel-{row['uid']}.log").decode().splitlines()
        parsed = parser.feed(lines) + parser.finish()
        if (
            len(parsed) != 1
            or parsed[0].gpu_sim_cycle != int(row["cycles"])
            or parsed[0].kernel_name != row["name"]
        ):
            raise ValueError(f"kernel-{row['uid']}.log does not match kernels.csv")
        stats.append(
            parsed[0].model_copy(
                update={"kernel_index": index, "kernel_launch_uid": int(row["uid"])}
            )
        )
    return stats


def _real_timing(alignment: dict, corrected: dict, simulated: list[dict]) -> list[dict]:
    """The nsys duration of each simulated kernel, in the same order as its stats, found through
    trace_alignment.json's (repeat, step, seq) for its tracer uid."""
    steps = {(s["repeat"], s["step"]): s for s in corrected["steps"]}
    by_uid = {k["tracer_uid"]: k for k in alignment["kernels"]}
    rows = []
    for row in simulated:
        k = by_uid[int(row["uid"])]
        timed = next(t for t in steps[(k["repeat"], k["step"])]["kernels"] if t["seq"] == k["seq"])
        if timed["name"] != k["name"] or k["name"] != row["name"]:
            raise ValueError(f"kernel {row['uid']}: names differ across the alignment")
        rows.append(
            {
                "kernel_id": k["tracer_uid"],
                "kernel_name": k["name"],
                "duration_ns": timed["duration_ns"],
                "grid": k["grid"],
                "block": k["block"],
            }
        )
    return rows


def load(bundle: Bundle) -> Sample:
    m = bundle.manifest
    run = bundle.read_json("run.json")
    workload = bundle.read_json("workload.json")
    corrected = bundle.read_json("corrected_timeline.json")
    alignment = bundle.read_json("trace_alignment.json")
    trace_steps = bundle.read_json("traced/trace_steps.json")
    summary = bundle.read_json("sim/summary.json")
    simulated = _simulated(bundle)

    traced_at = _utc(min(s["wall_start_ns"] for s in trace_steps))
    started = datetime.fromisoformat(m["sim_started_at"])
    finished = started + timedelta(seconds=summary["wall_s"])
    trace_files = {rel.removeprefix("traced/"): bundle.read(rel) for rel in TRACE_FILES}
    kernelslist = trace_files["traces/kernelslist.g"].decode().splitlines()
    metadata = {
        "status": run["status"],
        "run_name": run["run_name"],
        "model": workload["model"],
        "gpu": corrected["meta"]["gpu_name"],
        "trace_gpu_arch": m["trace_gpu_arch"],
        "suggested_sim_config": m["gpu_config"],
        "layers": [
            f"step {s['step_in_repeat']} ({s['kind']})"
            for s in trace_steps
            if s["traced"] and s["repeat"] == run["trace_repeat"]
        ],
        "prompts": workload["prompts"],
        "max_tokens": workload["max_tokens"],
        "repeats": workload["repeats"],
        "exec_mode": workload["exec_mode"],
        "vllm_version": corrected["meta"]["vllm_version"],
        "trace_repeat": run["trace_repeat"],
        "trace_steps": run["trace_steps"],
        "stage_wall_s": run["stage_wall_s"],
        "profiler_version": run["package_version"],
    }
    trace = TraceDetail(
        id=m["trace_id"],
        state=TraceState.ready,
        error=None,
        created_at=traced_at,
        updated_at=traced_at,
        model=metadata["model"],
        layers=metadata["layers"],
        source_gpu=metadata["gpu"],
        trace_gpu_arch=m["trace_gpu_arch"],
        suggested_sim_config=m["gpu_config"],
        kernel_count=sum(1 for line in kernelslist if line.startswith("kernel")),
        bytes=sum(len(b) for b in trace_files.values()),
        has_real_timing=True,
        has_step_timeline=True,
        sample=True,
        metadata=metadata,
        kernels=[
            KernelLaunch(
                kernel_id=k["tracer_uid"],
                name=k["name"],
                grid=tuple(k["grid"]),
                block=tuple(k["block"]),
                instructions=None,
            )
            for k in sorted(alignment["kernels"], key=lambda k: k["tracer_uid"])
        ],
    )
    stats = _stats(bundle, m["sim_id"], simulated)
    insns = [s.gpu_sim_insn for s in stats]
    log_lines = bundle.read("sim/setup.log").decode().splitlines()
    sim = SimSummary(
        id=m["sim_id"],
        run_id=m["run_id"],
        gpu_config=m["gpu_config"],
        replica=0,
        state=SimState.succeeded,
        error=None,
        launcher_ref=None,
        created_at=started,
        started_at=started,
        finished_at=finished,
        last_heartbeat_at=None,
        exit_code=None,
        kernels_done=len(stats),
        kernels_total=len(stats),
        current_kernel=None,
        latest=SimMetrics(
            gpu_tot_sim_cycle=sum(s.gpu_sim_cycle or 0 for s in stats),
            gpu_tot_sim_insn=None if None in insns else sum(insns),
        ),
        log_tail=log_lines[-20:],
    )
    outputs = {rel.rsplit("/", 1)[-1]: bundle.read(rel) for rel in m["output_files"]}
    files = [
        SimFile(name=name, size=len(data), content_type=content_type(name), created_at=finished)
        for name, data in outputs.items()
    ]
    rows = SampleRows(
        trace=trace,
        real_timing=_real_timing(alignment, corrected, simulated),
        run_created_at=started,
        sim=sim,
        log=log_lines,
        stats=stats,
        step_timeline=StepTimelineUpload(
            simulated=bundle.read_json("sim/simulated_timeline.json"),
            compare=bundle.read_json("sim/compare.json"),
        ),
        files=files,
    )
    return Sample(rows=rows, trace_files=trace_files, output_files=outputs)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_bytes(data)
    part.replace(path)


def seed(settings: Settings, store: Store, root: Path) -> bool:
    """True when this call added the sample. Files go to disk before the rows that point at
    them, so a crash in between leaves no rows and the next start rewrites the files."""
    bundle = Bundle(root)
    if store.get_trace_summary(bundle.manifest["trace_id"]) is not None:
        return False
    sample = load(bundle)
    trace_dir = settings.trace_dir(sample.rows.trace.id)
    for rel, data in sample.trace_files.items():
        _write(trace_dir / rel, data)
    for name, data in sample.output_files.items():
        _write(settings.files_root / sample.rows.sim.id / name, data)
    return store.add_sample(sample.rows)


def seed_samples(settings: Settings, store: Store, root: Path = SAMPLES_ROOT) -> None:
    for manifest in sorted(root.glob("*/sample.json")):
        if seed(settings, store, manifest.parent):
            log.info("seeded sample %s", manifest.parent.name)
