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
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cascade.ingest import parse_stats
from cascade.models import (
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
from cascade.store import SampleRows, SampleSimRows, Store

log = logging.getLogger(__name__)

SAMPLES_ROOT = Path(__file__).resolve().parent.parent / "samples"
TRACE_FILES = (
    "corrected_timeline.json",
    "trace_alignment.json",
    "traced/trace_steps.json",
    "traced/traces/kernelslist.g",
)
REPLAY_DIR = "replay"
REPLAY_DELAY_S = 0.02
_UID_RE = re.compile(r"^(launching kernel name: \S+ uid: |kernel_launch_uid = )\d+")


class Bundle:
    def __init__(self, root: Path, config: dict | None = None):
        self.root = root
        self.manifest = json.loads((root / "sample.json").read_text())
        self.manifest.update(config or {})

    def read(self, rel: str) -> bytes:
        """A bundled file's original bytes, whether it is stored plain or as `<rel>.gz`."""
        if rel.startswith("sim/"):
            rel = f"{self.manifest.get('sim_dir', 'sim')}/{rel.removeprefix('sim/')}"
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
    output_files: dict[str, dict[str, bytes]]  # sim id -> output file name -> bytes


def _utc(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, UTC)


def _simulated(bundle: Bundle) -> list[dict]:
    """kernels.csv rows Accel-Sim ran itself (not copied from a dedup representative), by uid."""
    rows = csv.DictReader(io.StringIO(bundle.read("sim/kernels.csv").decode()))
    return sorted((r for r in rows if r["sim"] == "exact"), key=lambda r: int(r["uid"]))


def _stats(
    bundle: Bundle, sim_id: str, simulated: list[dict], indices: dict[int, int]
) -> list[KernelStat]:
    """One stat per simulated kernel in launch order, parsed from that kernel's own Accel-Sim log.
    Each kernel ran alone, so its log's launch uid is 1; the tracer's uid replaces it."""
    clock = bundle.manifest["core_clock_mhz"]
    stats = []
    for row in simulated:
        uid = int(row["uid"])
        if bundle.manifest.get("stats_source") == "kernels.csv":
            stats.append(
                KernelStat(
                    sim_id=sim_id,
                    kernel_index=indices[uid],
                    kernel_name=row["name"],
                    kernel_launch_uid=uid,
                    gpu_sim_cycle=int(row["cycles"]),
                    gpu_sim_insn=None,
                    gpu_ipc=None,
                    stats={},
                    sim_time_ns=int(row["cycles"]) / clock * 1000,
                )
            )
            continue
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
            parsed[0].model_copy(update={"kernel_index": indices[uid], "kernel_launch_uid": uid})
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
    bundles = [Bundle(bundle.root, c) for c in m.get("configs", [{}])]
    if not bundles:
        raise ValueError("sample needs at least one simulation config")
    simulated = sorted(
        {int(r["uid"]): r for b in bundles for r in _simulated(b)}.values(),
        key=lambda r: int(r["uid"]),
    )
    indices = {int(r["uid"]): i for i, r in enumerate(simulated)}

    traced_at = _utc(min(s["wall_start_ns"] for s in trace_steps))
    stats_file = m.get("stats_file", "traced/traces/stats_ctx_0x54c35c0")
    trace_files = {
        rel.removeprefix("traced/"): bundle.read(rel) for rel in (*TRACE_FILES, stats_file)
    }
    kernelslist = [
        line
        for line in trace_files["traces/kernelslist.g"].decode().splitlines()
        if line.startswith("kernel")
    ]
    traced = {int(line.split("-")[1]) for line in kernelslist}
    launches = parse_stats(trace_files[stats_file.removeprefix("traced/")].decode())
    metadata = {
        "status": run["status"],
        "run_name": run["run_name"],
        "model": workload["model"],
        "gpu": corrected["meta"]["gpu_name"],
        "trace_gpu_arch": m["trace_gpu_arch"],
        "suggested_sim_config": m["gpu_config"],
        "real_gpu_config": m.get("real_gpu_config", m["gpu_config"]),
        "sample_sim_configs": [b.manifest["gpu_config"] for b in bundles],
        "sample_set": m.get("sample_set"),
        "sample_pending_configs": m.get("sample_pending_configs", []),
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
        real_gpu_config=metadata["real_gpu_config"],
        sample_sim_configs=metadata["sample_sim_configs"],
        sample_set=metadata["sample_set"],
        kernel_count=len(kernelslist),
        bytes=sum(len(b) for b in trace_files.values()),
        has_real_timing=True,
        has_step_timeline=True,
        sample=True,
        metadata=metadata,
        kernels=sorted((k for k in launches if k.kernel_id in traced), key=lambda k: k.kernel_id),
    )
    results = [_load_sim(b, traced_at, indices) for b in bundles]
    rows = SampleRows(
        trace=trace,
        real_timing=_real_timing(alignment, corrected, simulated),
        run_created_at=min(r.sim.created_at for r, _ in results),
        sims=[r for r, _ in results],
    )
    return Sample(
        rows=rows,
        trace_files=trace_files,
        output_files={r.sim.id: outputs for r, outputs in results},
    )


def _load_sim(
    bundle: Bundle, traced_at: datetime, indices: dict[int, int]
) -> tuple[SampleSimRows, dict[str, bytes]]:
    m = bundle.manifest
    summary = bundle.read_json("sim/summary.json")
    if any(summary.get("counts", {}).get(key, 0) for key in ("failed", "timeout")):
        raise ValueError(f"{m['gpu_config']}: sample simulation has failed or timed-out kernels")
    started = datetime.fromisoformat(m["sim_started_at"]) if m.get("sim_started_at") else None
    finished = started + timedelta(seconds=summary["wall_s"]) if started else None
    stats = _stats(bundle, m["sim_id"], _simulated(bundle), indices)
    insns = [s.gpu_sim_insn for s in stats]
    log_lines = (
        []
        if m.get("stats_source") == "kernels.csv"
        else bundle.read("sim/setup.log").decode().splitlines()
    )
    sim = SimSummary(
        id=m["sim_id"],
        run_id=m["run_id"],
        gpu_config=m["gpu_config"],
        replica=0,
        state=SimState.succeeded,
        error=None,
        launcher_ref=None,
        created_at=started or traced_at,
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
        SimFile(
            name=name,
            size=len(data),
            content_type=content_type(name),
            created_at=finished or traced_at,
        )
        for name, data in outputs.items()
    ]
    rows = SampleSimRows(
        sim=sim,
        log=log_lines,
        stats=stats,
        step_timeline=StepTimelineUpload(
            simulated=bundle.read_json("sim/simulated_timeline.json"),
            compare=bundle.read_json("sim/compare.json"),
        ),
        files=files,
    )
    return rows, outputs


def _replay_log(bundle: Bundle, uids: list[int]) -> tuple[str, int]:
    """One Accel-Sim log for the reference kernels, in launch order. A kernel's own log (it ran
    alone, so its launch uid is 1) gets the tracer uid; without a log, its kernels.csv cycles
    become a minimal stats block."""
    rows = csv.DictReader(io.StringIO(bundle.read("sim/kernels.csv").decode()))
    by_uid = {int(r["uid"]): r for r in rows}
    sim_dir = bundle.root / bundle.manifest.get("sim_dir", "sim")
    lines = []
    for uid in uids:
        row = by_uid[uid]
        if any((sim_dir / f"kernel-{uid}.log{ext}").is_file() for ext in ("", ".gz")):
            log_lines = bundle.read(f"sim/kernel-{uid}.log").decode().splitlines()
            lines += [_UID_RE.sub(rf"\g<1>{uid}", line) for line in log_lines]
            continue
        lines += [
            f"launching kernel name: {row['name']} uid: {uid} cuda_stream_id: 0",
            f"kernel_name = {row['name']} ",
            f"kernel_launch_uid = {uid} ",
            f"gpu_sim_cycle = {row['cycles']}",
        ]
    lines.append("GPGPU-Sim: *** exit detected ***")
    return "\n".join(lines) + "\n", len(uids)


def write_replays(bundle: Bundle, trace_dir: Path) -> None:
    """Writes replay/<config>/ for each recorded config that lacks one: accelsim.log, the
    config's output files under outputs/, then replay.json last, so a partial write redoes."""
    bundles = [Bundle(bundle.root, c) for c in bundle.manifest.get("configs", [{}])]
    uids = sorted({int(r["uid"]) for b in bundles for r in _simulated(b)})
    for b in bundles:
        out = trace_dir / REPLAY_DIR / b.manifest["gpu_config"]
        timing = out / "outputs" / "step-timeline.json"
        if not timing.is_file():
            _write(
                timing,
                json.dumps(
                    {
                        "simulated": b.read_json("sim/simulated_timeline.json"),
                        "compare": b.read_json("sim/compare.json"),
                    }
                ).encode(),
            )
        if (out / "replay.json").is_file():
            continue
        text, kernels = _replay_log(b, uids)
        _write(out / "accelsim.log", text.encode())
        for rel in b.manifest["output_files"]:
            _write(out / "outputs" / rel.rsplit("/", 1)[-1], b.read(rel))
        _write(out / "replay.json", json.dumps({"kernels": kernels}).encode())


def recorded_configs(trace_dir: Path) -> dict[str, int]:
    """The configs a sample trace can replay, with how many kernels each replays."""
    return {
        p.parent.name: json.loads(p.read_text())["kernels"]
        for p in sorted((trace_dir / REPLAY_DIR).glob("*/replay.json"))
    }


def replay_command(trace_dir: Path, gpu_config: str) -> str:
    out = trace_dir / REPLAY_DIR / gpu_config
    return shlex.join(
        [
            sys.executable,
            "-m",
            "cascade.cli",
            "worker",
            "--replay",
            str(out / "accelsim.log"),
            "--replay-outputs",
            str(out / "outputs"),
            "--delay",
            str(REPLAY_DELAY_S),
        ]
    )


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_bytes(data)
    part.replace(path)


def seed(settings: Settings, store: Store, root: Path) -> bool:
    """True when this call added the sample. Files go to disk before the rows that point at
    them, so a crash in between leaves no rows and the next start rewrites the files."""
    bundle = Bundle(root)
    if bundle.manifest.get("status") == "pending":
        log.info("skipping pending sample %s", root.name)
        return False
    trace_dir = settings.trace_dir(bundle.manifest["trace_id"])
    write_replays(bundle, trace_dir)
    if store.get_trace_summary(bundle.manifest["trace_id"]) is not None:
        return False
    sample = load(bundle)
    for rel, data in sample.trace_files.items():
        _write(trace_dir / rel, data)
    for sim_id, outputs in sample.output_files.items():
        for name, data in outputs.items():
            _write(settings.files_root / sim_id / name, data)
    return store.add_sample(sample.rows)


def seed_samples(settings: Settings, store: Store, root: Path = SAMPLES_ROOT) -> None:
    for manifest in sorted(root.glob("*/sample.json")):
        if seed(settings, store, manifest.parent):
            log.info("seeded sample %s", manifest.parent.name)
