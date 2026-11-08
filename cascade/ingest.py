"""Turns the plain files rclone writes under <bucket>/<prefix>/<run_name>/ into traces.

Drives the trace state machine in docs/api.md. Every step reads the tree and the DB afresh, so a
restart at any point converges to the same end state.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from cascade.models import KernelLaunch, TraceState
from cascade.settings import Settings
from cascade.store import Store

log = logging.getLogger(__name__)

POLL_S = 2.0
STEP_TIMELINE_FILES = ("corrected_timeline.json", "kernel_mapping.json")
RAW_KERNEL_RE = re.compile(r"^kernel-(\d+)(?:-ctx_0x[0-9a-f]+)?\.trace(?:\.xz)?$")
STATS_LINE_RE = re.compile(
    r"^kernel-(\d+)[^,]*,\s*(.+?),"
    r"\s*(\d+),\s*(\d+),\s*(\d+),\s*\d+,"
    r"\s*(\d+),\s*(\d+),\s*(\d+),\s*\d+,"
    r"\s*(?:(\d+)\s*,\s*\d+)?\s*$"
)


def scan_tree(run_dir: Path) -> dict[str, int]:
    """relative path -> size, skipping rclone's in-flight `.rclone_temp_put_*` objects and
    other dotfiles."""
    files = {}
    for path in run_dir.rglob("*"):
        rel = path.relative_to(run_dir)
        if any(part.startswith(".") for part in rel.parts) or not path.is_file():
            continue
        files[rel.as_posix()] = path.stat().st_size
    return files


def raw_kernel_count(files: dict[str, int]) -> int:
    return sum(
        1
        for rel in files
        if rel.startswith("traces/") and RAW_KERNEL_RE.match(rel.removeprefix("traces/"))
    )


def read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def upload_complete(
    files: dict[str, int], metadata: dict, key_prefix: str, previous: dict[str, int] | None
) -> bool:
    """docs/api.md done signal. `files` keys may be relative to the run dir or full object keys.

    Without `files`, the older profiler uploads kernelslist_ctx_* before stats_ctx_*, so also
    require a matching stats file and a tree unchanged since the previous poll.
    """
    listed = metadata.get("files")
    if isinstance(listed, list) and listed:
        for entry in listed:
            if not isinstance(entry, dict) or "key" not in entry:
                return False
            key = str(entry["key"]).removeprefix(key_prefix)
            if files.get(key) != entry.get("bytes"):
                return False
        return True
    lists = [r for r in files if r.startswith("traces/kernelslist_ctx_")]
    if not lists:
        return False
    if any(r.replace("kernelslist_ctx_", "stats_ctx_") not in files for r in lists):
        return False
    return files == previous


def parse_stats(text: str) -> list[KernelLaunch]:
    """Rows of the tracer's stats_ctx_* (or older stats.csv): see enter/leave_kernel_launch in
    util/tracer_nvbit/tracer_tool/tracer_tool.cu at v2.0.1. A kernel whose leave callback never ran
    has no instruction counts."""
    kernels = []
    for line in text.splitlines():
        m = STATS_LINE_RE.match(line.strip())
        if not m:
            continue
        kid, name, gx, gy, gz, bx, by, bz, insts = m.groups()
        kernels.append(
            KernelLaunch(
                kernel_id=int(kid),
                name=name.strip(),
                grid=(int(gx), int(gy), int(gz)),
                block=(int(bx), int(by), int(bz)),
                instructions=int(insts) if insts is not None else None,
            )
        )
    return kernels


def load_kernels(traces_dir: Path) -> list[KernelLaunch]:
    kernels: list[KernelLaunch] = []
    for path in sorted(traces_dir.glob("stats_ctx_*")) + sorted(traces_dir.glob("stats.csv")):
        kernels += parse_stats(path.read_text(errors="replace"))
    return sorted(kernels, key=lambda k: k.kernel_id)


def load_real_timing(run_dir: Path, metadata: dict) -> list[dict] | None:
    rel = metadata.get("real_timing") or "real_timing.json"
    rows = read_json(run_dir / str(rel))
    if not isinstance(rows, list) or not all(
        isinstance(r, dict) and isinstance(r.get("kernel_id"), int) for r in rows
    ):
        if rows is not None:
            log.warning("ignoring malformed real timing in %s", run_dir / str(rel))
        return None
    return sorted(rows, key=lambda r: r["kernel_id"])


def has_step_timeline(run_dir: Path) -> bool:
    """The profiler's phases output and its kernel mapping, which `cascade worker` needs for
    step timing. Either can land after the trace is ready."""
    return all((run_dir / name).is_file() for name in STEP_TIMELINE_FILES)


def kernelslist_kernel_count(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.startswith("kernel"))


class Interrupted(Exception):
    """Main stopped mid post-processing; the trace stays `processing` so a restart reruns it."""


class Ingestor:
    def __init__(self, settings: Settings, store: Store):
        self.settings = settings
        self.store = store
        self._previous: dict[str, dict[str, int]] = {}
        self._inflight: dict[str, Future] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="postprocess")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._child: subprocess.Popen | None = None

    @property
    def key_prefix_base(self) -> str:
        return f"{self.settings.s3_prefix}/" if self.settings.s3_prefix else ""

    def start(self) -> None:
        self.settings.traces_root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="ingest", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        # A post-processor left running would race the rerun that restart starts.
        if self._child and self._child.poll() is None:
            self._child.kill()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def drain(self) -> None:
        for fut in list(self._inflight.values()):
            fut.result()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                log.exception("ingest poll failed")
            self._stop.wait(POLL_S)

    def poll_once(self) -> None:
        root = self.settings.traces_root
        if not root.is_dir():
            return
        states = self.store.trace_states()
        flags = self.store.step_timeline_flags()
        samples = self.store.sample_trace_ids()
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            trace_id = run_dir.name
            if trace_id.startswith(".") or trace_id in samples:
                continue
            state = states.get(trace_id)
            if state == TraceState.processing:
                self._submit(trace_id)
            elif state not in (TraceState.ready, TraceState.failed):
                self._advance_upload(trace_id, run_dir, registered=state is not None)
            present = has_step_timeline(run_dir)
            if present != flags.get(trace_id, False):
                self.store.update_trace(trace_id, has_step_timeline=present)

    def _advance_upload(self, trace_id: str, run_dir: Path, registered: bool) -> None:
        files = scan_tree(run_dir)
        if not files:
            return
        kernels, nbytes = raw_kernel_count(files), sum(files.values())
        if not registered:
            self.store.register_trace(trace_id, kernels, nbytes)
        meta = read_json(run_dir / "metadata.json") if "metadata.json" in files else None
        meta = meta if isinstance(meta, dict) else None
        previous = self._previous.get(trace_id)
        self._previous[trace_id] = files
        if meta is not None and meta.get("status") == "failed":
            self.store.update_trace(
                trace_id,
                state=TraceState.failed,
                error="profiler reported status 'failed' in metadata.json",
                metadata=meta,
                kernel_count=kernels,
                nbytes=nbytes,
            )
            return
        prefix = f"{self.key_prefix_base}{trace_id}/"
        if meta is not None and upload_complete(files, meta, prefix, previous):
            self.store.update_trace(
                trace_id,
                state=TraceState.processing,
                metadata=meta,
                kernel_count=kernels,
                nbytes=nbytes,
            )
            self._previous.pop(trace_id, None)
            self._submit(trace_id)
        else:
            self.store.update_trace(trace_id, metadata=meta, kernel_count=kernels, nbytes=nbytes)

    def _submit(self, trace_id: str) -> None:
        fut = self._inflight.get(trace_id)
        if fut is None or fut.done():
            self._inflight[trace_id] = self._pool.submit(self._process_safely, trace_id)

    def _process_safely(self, trace_id: str) -> None:
        try:
            self._process(trace_id)
        except Interrupted:
            log.info("post-processing %s interrupted; it reruns on the next start", trace_id)
        except Exception as e:
            log.exception("processing trace %s failed", trace_id)
            self.store.update_trace(trace_id, state=TraceState.failed, error=f"processing: {e}")

    def _process(self, trace_id: str) -> None:
        trace = self.store.get_trace_summary(trace_id)
        if trace is None or trace.state != TraceState.processing:
            return
        run_dir = self.settings.trace_dir(trace_id)
        traces = run_dir / "traces"
        work = self.settings.work_dir / "postprocess"
        work.mkdir(parents=True, exist_ok=True)
        started, done = work / f"{trace_id}.started", work / f"{trace_id}.done"
        kernelslist = traces / "kernelslist.g"
        # post-traces-processing creates kernelslist.g before converting, so a kernelslist.g
        # left by our own interrupted run is not proof of a finished one.
        uploaded_processed = kernelslist.is_file() and not started.exists()
        if not uploaded_processed and not done.exists():
            error = self._postprocess(trace_id, traces, started, done, work)
            if error:
                self.store.update_trace(trace_id, state=TraceState.failed, error=error)
                return
        if not kernelslist.is_file():
            self.store.update_trace(
                trace_id, state=TraceState.failed, error="post-processing produced no kernelslist.g"
            )
            return
        meta = read_json(run_dir / "metadata.json")
        meta = meta if isinstance(meta, dict) else {}
        kernels = load_kernels(traces)
        self.store.update_trace(
            trace_id,
            state=TraceState.ready,
            kernels=kernels,
            kernel_count=kernelslist_kernel_count(kernelslist) or len(kernels),
            nbytes=sum(scan_tree(run_dir).values()),
            real_timing=load_real_timing(run_dir, meta),
        )

    def _postprocess(
        self, trace_id: str, traces: Path, started: Path, done: Path, work: Path
    ) -> str | None:
        if not traces.is_dir() or not any(traces.glob("kernelslist*")):
            return "no traces/kernelslist* to post-process"
        argv = shlex.split(self.settings.postprocess_cmd)
        exe = shutil.which(argv[0]) if argv else None
        if exe is None:
            return (
                f"post-processor not found: {self.settings.postprocess_cmd!r}"
                " (set CASCADE_POSTPROCESS_CMD or CASCADE_ACCELSIM_DIR)"
            )
        started.touch()
        log_path = work / f"{trace_id}.log"
        cmd = [exe, *argv[1:], str(traces), "-j", str(os.cpu_count() or 8)]
        log.info("post-processing %s: %s", trace_id, shlex.join(cmd))
        with log_path.open("w") as out:
            proc = self._child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
            proc.wait()
        if self._stop.is_set():
            raise Interrupted(trace_id)
        if proc.returncode != 0:
            tail = log_path.read_text(errors="replace").splitlines()[-5:]
            return f"post-processing exited {proc.returncode}: " + " | ".join(tail)
        done.touch()
        return None
