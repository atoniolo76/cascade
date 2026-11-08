"""HTTP API from docs/api.md, plus the background loops that keep main's state honest."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import statistics
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from cascade.gpu_configs import GpuConfigRegistry
from cascade.ingest import Ingestor
from cascade.launcher import Launcher, make_launcher
from cascade.models import (
    FILE_MAX_BYTES,
    FILE_NAME_PATTERN,
    SIM_FILES_MAX_BYTES,
    TERMINAL_SIM_STATES,
    Comparison,
    ComparisonRow,
    GpuConfig,
    Heartbeat,
    HeartbeatAck,
    KernelEvent,
    KernelStat,
    LogChunk,
    LogEvent,
    RunCreate,
    RunDetail,
    RunFile,
    RunResults,
    RunSummary,
    SimEvent,
    SimFile,
    SimResults,
    SimState,
    SimStepTimeline,
    SimSummary,
    SnapshotEvent,
    StepTimelineUpload,
    TraceDetail,
    TraceState,
    TraceSummary,
    content_type,
)
from cascade.openai_api import install_openai_routes
from cascade.perfetto import run_trace
from cascade.samples import seed_samples
from cascade.settings import Settings
from cascade.stats import LogParser
from cascade.store import IllegalTransition, Store, now

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
SWEEP_S = 5.0
SSE_KEEPALIVE_S = 15.0
SSE_QUEUE_MAX = 2000
LOST_GIVE_UP_FACTOR = 10


class Bus:
    """In-process fan-out of run events to SSE subscribers. publish() is thread-safe."""

    def __init__(self) -> None:
        self._subs: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, run_id: str, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(SSE_QUEUE_MAX)
        with self._lock:
            self._subs.setdefault(run_id, set()).add((loop, q))
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subs.get(run_id, set())
            subs.difference_update({s for s in subs if s[1] is q})
            if not subs:
                self._subs.pop(run_id, None)

    def publish(self, run_id: str, event: BaseModel) -> None:
        with self._lock:
            subs = list(self._subs.get(run_id, ()))
        for loop, q in subs:
            loop.call_soon_threadsafe(_offer, q, event)


def _offer(q: asyncio.Queue, event: BaseModel) -> None:
    # A subscriber this far behind gets disconnected; its reconnect starts from a snapshot.
    if q.full():
        while not q.empty():
            q.get_nowait()
        q.put_nowait(None)
    else:
        q.put_nowait(event)


class Cascade:
    """Main-mode services shared by the routes and background loops."""

    def __init__(self, settings: Settings, launcher: Launcher | None = None):
        self.settings = settings
        self.store = Store(settings.db_path)
        self.configs = GpuConfigRegistry(settings.accelsim_dir)
        self.launcher = launcher or make_launcher(settings)
        self.ingestor = Ingestor(settings, self.store)
        self.bus = Bus()
        self._parsers: dict[str, LogParser] = {}
        # Held across every store write that publishes an event, and across subscribe+snapshot,
        # so subscribers see events in commit order and a snapshot never overlaps queued events.
        self._lock = threading.RLock()
        self._launch_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="launch")
        self._stop = threading.Event()
        self._sweeper: threading.Thread | None = None

    def start(self) -> None:
        if self.settings.samples:
            seed_samples(self.settings, self.store)
        self.ingestor.start()
        self._sweeper = threading.Thread(target=self._sweep_loop, name="sweeper", daemon=True)
        self._sweeper.start()
        for run_id in self.store.run_ids_with_pending_sims():
            self.launch_run(run_id)

    def stop(self) -> None:
        self._stop.set()
        self.ingestor.stop()
        if self._sweeper:
            self._sweeper.join(timeout=10)
        self._launch_pool.shutdown(wait=True, cancel_futures=True)
        self.store.close()

    def subscribe(
        self, run_id: str, loop: asyncio.AbstractEventLoop
    ) -> tuple[asyncio.Queue, SnapshotEvent] | None:
        with self._lock:
            run = self.store.get_run(run_id)
            if run is None:
                return None
            snapshot = SnapshotEvent(run=run, kernels=self.light_kernels(run_id))
            return self.bus.subscribe(run_id, loop), snapshot

    def light_kernels(self, run_id: str) -> list[KernelStat]:
        return [k.model_copy(update={"stats": {}}) for k in self.store.kernel_stats_for_run(run_id)]

    def publish_sim(self, sim: SimSummary) -> None:
        self.bus.publish(sim.run_id, SimEvent(sim=sim))

    def launch_run(self, run_id: str) -> None:
        self._launch_pool.submit(self._launch_run, run_id)

    def _launch_run(self, run_id: str) -> None:
        pending = [s for s in self.store.sims_for_run(run_id) if s.state == SimState.pending]
        try:
            self.launcher.prepare(run_id)
        except Exception as e:
            log.exception("preparing run %s failed", run_id)
            for sim in pending:
                self._fail(sim.id, f"launcher prepare: {e}")
            return
        trace_dir = self.settings.trace_dir(self.store.run_trace_id(run_id))
        for sim in pending:
            self._launch_pool.submit(self._launch_sim, sim.id, trace_dir)

    def _launch_sim(self, sim_id: str, trace_dir: Path) -> None:
        with self._lock:
            try:
                sim = self.store.transition_sim(sim_id, SimState.launching)
            except IllegalTransition:
                return
            self.publish_sim(sim)
        try:
            ref = self.launcher.launch(sim, trace_dir)
        except Exception as e:
            log.exception("launching %s failed", sim_id)
            self._fail(sim_id, f"launch: {e}")
            return
        with self._lock:
            sim = self.store.set_launcher_ref(sim_id, ref)
            self.publish_sim(sim)
        if sim.state in TERMINAL_SIM_STATES:
            # Cancelled or finished while launching: the earlier terminate had no ref to stop.
            self._terminate_later(sim)

    def _fail(self, sim_id: str, error: str) -> None:
        with self._lock:
            try:
                self.publish_sim(self.store.transition_sim(sim_id, SimState.failed, error=error))
            except IllegalTransition:
                pass

    def _terminate_later(self, sim: SimSummary) -> None:
        def run() -> None:
            try:
                self.launcher.terminate(sim)
            except Exception:
                log.exception("terminating %s failed", sim.id)

        self._launch_pool.submit(run)

    def _end(self, sims: list[SimSummary]) -> None:
        for sim in sims:
            self._parsers.pop(sim.id, None)
            self.publish_sim(sim)

    def cancel_run(self, run_id: str) -> None:
        with self._lock:
            cancelled = []
            for sim in self.store.sims_for_run(run_id):
                try:
                    cancelled.append(self.store.transition_sim(sim.id, SimState.cancelled))
                except IllegalTransition:
                    continue
            self._end(cancelled)
        for sim in cancelled:
            self._terminate_later(sim)

    def heartbeat(self, sim_id: str, hb: Heartbeat) -> HeartbeatAck:
        with self._lock:
            sim, changed = self.store.apply_heartbeat(sim_id, hb)
            finished = changed and sim.state in TERMINAL_SIM_STATES
            if finished:
                sim = self._finish_parsing(sim)
            if changed:
                self.publish_sim(sim)
        if finished:
            self._terminate_later(sim)
        return HeartbeatAck(state=sim.state, cancel=sim.state == SimState.cancelled)

    def _sweep_loop(self) -> None:
        while not self._stop.wait(SWEEP_S):
            try:
                self.sweep()
            except Exception:
                log.exception("sweep failed")

    def sweep(self) -> None:
        timeout = self.settings.heartbeat_timeout_s
        with self._lock:
            for sim in self.store.mark_stale_lost(now() - timedelta(seconds=timeout)):
                self.publish_sim(sim)
            given_up = self.store.fail_lost(
                now() - timedelta(seconds=timeout * LOST_GIVE_UP_FACTOR),
                error=f"no heartbeat for {timeout * LOST_GIVE_UP_FACTOR:.0f}s",
            )
            self._end(given_up)
        for sim in given_up:
            self._terminate_later(sim)

    def _parser(self, sim: SimSummary) -> LogParser:
        """Parsers live in memory; after a restart one is rebuilt by replaying the stored log."""
        parser = self._parsers.get(sim.id)
        if parser is None:
            cfg = self.configs.get(sim.gpu_config)
            parser = LogParser(sim.id, cfg.core_clock_mhz if cfg else None)
            replayed = []
            for lines in self.store.log_chunks(sim.id):
                replayed += parser.feed(lines)
            self.store.record_progress(sim.id, replayed, parser.current_kernel, parser.latest)
            self._parsers[sim.id] = parser
        return parser

    def add_logs(self, sim: SimSummary, chunk: LogChunk) -> None:
        with self._lock:
            parser = self._parser(sim)
            if not self.store.add_log_chunk(sim.id, chunk.seq, chunk.lines):
                return
            stats = parser.feed(chunk.lines)
            sim, added = self.store.record_progress(
                sim.id, stats, parser.current_kernel, parser.latest
            )
            self.bus.publish(sim.run_id, LogEvent(sim_id=sim.id, lines=chunk.lines))
            for stat in added:
                self.bus.publish(sim.run_id, KernelEvent(stat=stat))
            self.publish_sim(sim)

    def _finish_parsing(self, sim: SimSummary) -> SimSummary:
        """Closes a kernel block left open when the log ended, then forgets the parser."""
        parser = self._parser(sim)
        sim, added = self.store.record_progress(
            sim.id, parser.finish(), parser.current_kernel, parser.latest
        )
        self._parsers.pop(sim.id, None)
        for stat in added:
            self.bus.publish(sim.run_id, KernelEvent(stat=stat))
        return sim

    def comparison(self, run: RunDetail) -> Comparison:
        configs = list(dict.fromkeys(e.gpu_config for e in run.entries))
        real = self.store.real_timing(run.trace_id) or []
        by_config: dict[str, dict[int, list[float]]] = {c: {} for c in configs}
        names: dict[int, str] = {}
        for sim in run.sims:
            for stat in self.store.kernel_stats(sim.id):
                names.setdefault(stat.kernel_index, stat.kernel_name)
                if stat.sim_time_ns is not None:
                    by_config[sim.gpu_config].setdefault(stat.kernel_index, []).append(
                        stat.sim_time_ns
                    )
        n = max([len(real), *(i + 1 for i in names)], default=0)
        rows = []
        for i in range(n):
            r = real[i] if i < len(real) else {}
            duration = r.get("duration_ns")
            rows.append(
                ComparisonRow(
                    kernel_index=i,
                    kernel_name=str(r.get("kernel_name") or names.get(i, "")),
                    real_ns=float(duration) if duration is not None else None,
                    sim_ns={
                        c: statistics.median(v) if (v := by_config[c].get(i)) else None
                        for c in configs
                    },
                )
            )
        return Comparison(has_real_timing=bool(real), configs=configs, rows=rows)

    def sim_results(self, sim: SimSummary, trace_id: str) -> SimResults:
        cfg = self.configs.get(sim.gpu_config) or GpuConfig(
            name=sim.gpu_config, arch="unknown", core_clock_mhz=0
        )
        return SimResults(
            sim=sim,
            trace_id=trace_id,
            gpu_config=cfg,
            kernels=self.store.kernel_stats(sim.id),
            totals=sim.latest,
        )


def _attachment(model: BaseModel, filename: str) -> Response:
    return Response(
        model.model_dump_json(),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def create_app(settings: Settings | None = None, launcher: Launcher | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    cascade = Cascade(settings, launcher)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cascade.start()
        try:
            yield
        finally:
            cascade.stop()

    app = FastAPI(title="Cascade", lifespan=lifespan)
    app.state.cascade = cascade
    store = cascade.store

    def check(token: str | None) -> None:
        if settings.no_auth:
            return
        if token is None or not secrets.compare_digest(token, settings.token):
            raise HTTPException(401, "missing or wrong token", {"WWW-Authenticate": "Bearer"})

    def bearer(request: Request) -> None:
        header = request.headers.get("authorization", "")
        check(header[7:] if header.lower().startswith("bearer ") else None)

    def bearer_or_query(request: Request, token: str | None = None) -> None:
        if token is not None:
            check(token)
        else:
            bearer(request)

    auth = [Depends(bearer)]
    install_openai_routes(app, cascade, bearer)

    def get_run_or_404(run_id: str) -> RunDetail:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        return run

    def get_sim_or_404(sim_id: str) -> SimSummary:
        sim = store.get_sim(sim_id)
        if sim is None:
            raise HTTPException(404, f"sim {sim_id} not found")
        return sim

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "auth": not settings.no_auth,
            "workspace": settings.workspace,
            "s3_endpoint": settings.s3_public_url,
        }

    @app.get("/api/gpu-configs", dependencies=auth)
    def gpu_configs() -> list[GpuConfig]:
        return cascade.configs.all()

    @app.get("/api/traces", dependencies=auth)
    def list_traces() -> list[TraceSummary]:
        return store.list_traces()

    @app.get("/api/traces/{trace_id}", dependencies=auth)
    def get_trace(trace_id: str) -> TraceDetail:
        trace = store.get_trace(trace_id)
        if trace is None:
            raise HTTPException(404, f"trace {trace_id} not found")
        return trace

    @app.get("/api/runs", dependencies=auth)
    def list_runs() -> list[RunSummary]:
        return store.list_runs()

    @app.post("/api/runs", dependencies=auth, status_code=201)
    def create_run(body: RunCreate) -> RunDetail:
        trace = store.get_trace_summary(body.trace_id)
        if trace is None:
            raise HTTPException(404, f"trace {body.trace_id} not found")
        if trace.state != TraceState.ready:
            raise HTTPException(409, f"trace {trace.id} is {trace.state}, not ready")
        if trace.sample:
            raise HTTPException(
                409, f"trace {trace.id} is a sample without its instruction traces; it can't run"
            )
        for entry in body.entries:
            cfg = cascade.configs.get(entry.gpu_config)
            if cfg is None:
                raise HTTPException(422, f"unknown gpu_config {entry.gpu_config!r}")
            if cfg.arch != trace.trace_gpu_arch:
                raise HTTPException(
                    422,
                    f"{cfg.name} is {cfg.arch} but trace {trace.id} is {trace.trace_gpu_arch}",
                )
        run_id = store.create_run(trace.id, body.entries, trace.kernel_count)
        cascade.launch_run(run_id)
        return get_run_or_404(run_id)

    @app.get("/api/runs/{run_id}", dependencies=auth)
    def get_run(run_id: str) -> RunDetail:
        return get_run_or_404(run_id)

    @app.post("/api/runs/{run_id}/cancel", dependencies=auth)
    def cancel_run(run_id: str) -> RunDetail:
        get_run_or_404(run_id)
        cascade.cancel_run(run_id)
        return get_run_or_404(run_id)

    @app.get("/api/runs/{run_id}/events", dependencies=[Depends(bearer_or_query)])
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        subscribed = await run_in_threadpool(cascade.subscribe, run_id, asyncio.get_running_loop())
        if subscribed is None:
            raise HTTPException(404, f"run {run_id} not found")
        q, snapshot = subscribed

        async def stream():
            try:
                yield f"data: {snapshot.model_dump_json()}\n\n"
                while not await request.is_disconnected():
                    try:
                        event = await asyncio.wait_for(q.get(), SSE_KEEPALIVE_S)
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    if event is None:
                        return
                    yield f"data: {event.model_dump_json()}\n\n"
            finally:
                cascade.bus.unsubscribe(run_id, q)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{run_id}/comparison", dependencies=auth)
    def comparison(run_id: str) -> Comparison:
        return cascade.comparison(get_run_or_404(run_id))

    @app.get("/api/runs/{run_id}/kernels", dependencies=auth)
    def run_kernels(run_id: str) -> list[KernelStat]:
        get_run_or_404(run_id)
        return cascade.light_kernels(run_id)

    @app.get("/api/runs/{run_id}/step-timelines", dependencies=auth)
    def run_step_timelines(run_id: str) -> list[SimStepTimeline]:
        get_run_or_404(run_id)
        return store.step_timelines_for_run(run_id)

    @app.get("/api/runs/{run_id}/results", dependencies=auth)
    def run_results(run_id: str) -> Response:
        run = get_run_or_404(run_id)
        trace = store.get_trace(run.trace_id)
        results = RunResults(
            run=RunSummary(**run.model_dump(exclude={"trace", "sims"})),
            trace=trace,
            sims=[cascade.sim_results(s, run.trace_id) for s in run.sims],
            comparison=cascade.comparison(run),
        )
        return _attachment(results, f"{run_id}-results.json")

    @app.get("/api/sims/{sim_id}/results", dependencies=auth)
    def sim_results(sim_id: str) -> Response:
        sim = get_sim_or_404(sim_id)
        trace_id = store.run_trace_id(sim.run_id)
        return _attachment(cascade.sim_results(sim, trace_id), f"{sim_id}-results.json")

    @app.get("/api/sims/{sim_id}/log", dependencies=auth)
    def sim_log(sim_id: str) -> StreamingResponse:
        get_sim_or_404(sim_id)

        def lines() -> Iterator[str]:
            for chunk in store.log_chunks(sim_id):
                yield "".join(f"{line}\n" for line in chunk)

        return StreamingResponse(
            lines(),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{sim_id}.log"'},
        )

    @app.post("/api/sims/{sim_id}/heartbeat", dependencies=auth)
    def heartbeat(sim_id: str, body: Heartbeat) -> HeartbeatAck:
        get_sim_or_404(sim_id)
        return cascade.heartbeat(sim_id, body)

    @app.post("/api/sims/{sim_id}/logs", dependencies=auth, status_code=204)
    def logs(sim_id: str, body: LogChunk) -> Response:
        cascade.add_logs(get_sim_or_404(sim_id), body)
        return Response(status_code=204)

    @app.post("/api/sims/{sim_id}/step-timeline", dependencies=auth, status_code=204)
    def step_timeline(sim_id: str, body: StepTimelineUpload) -> Response:
        get_sim_or_404(sim_id)
        store.set_step_timeline(sim_id, body)
        return Response(status_code=204)

    files_lock = threading.Lock()

    def file_path(sim_id: str, name: str) -> Path:
        return settings.files_root / sim_id / name

    def too_big(size: int, others: int) -> HTTPException | None:
        if size > FILE_MAX_BYTES:
            return HTTPException(413, f"file is over the {FILE_MAX_BYTES >> 20} MiB per-file limit")
        if others + size > SIM_FILES_MAX_BYTES:
            return HTTPException(
                413, f"sim's files would pass the {SIM_FILES_MAX_BYTES >> 20} MiB total limit"
            )
        return None

    def commit_file(sim_id: str, name: str, part: Path, size: int) -> SimFile:
        with files_lock:
            if error := too_big(size, store.files_bytes(sim_id, name)):
                raise error
            dest = file_path(sim_id, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            part.replace(dest)
            return store.put_file(sim_id, name, size, content_type(name))

    @app.put("/api/sims/{sim_id}/files/{name}", dependencies=auth)
    async def put_file(sim_id: str, name: str, request: Request) -> SimFile:
        if not re.fullmatch(FILE_NAME_PATTERN, name):
            raise HTTPException(422, f"file name {name!r} must match {FILE_NAME_PATTERN}")
        await run_in_threadpool(get_sim_or_404, sim_id)
        others = await run_in_threadpool(store.files_bytes, sim_id, name)
        part = settings.files_root / ".parts" / uuid.uuid4().hex
        part.parent.mkdir(parents=True, exist_ok=True)
        size, error = 0, None
        try:
            with part.open("wb") as f:
                # Reads the whole body even past a limit, so the client gets the 413 instead of
                # a reset connection mid-upload.
                async for chunk in request.stream():
                    size += len(chunk)
                    error = error or too_big(size, others)
                    if error is None:
                        f.write(chunk)
            if error is not None:
                raise error
            return await run_in_threadpool(commit_file, sim_id, name, part, size)
        finally:
            part.unlink(missing_ok=True)

    @app.get("/api/sims/{sim_id}/files", dependencies=auth)
    def sim_files(sim_id: str) -> list[SimFile]:
        get_sim_or_404(sim_id)
        return store.sim_files(sim_id)

    @app.get("/api/sims/{sim_id}/files/{name}", dependencies=auth)
    def get_file(sim_id: str, name: str) -> FileResponse:
        get_sim_or_404(sim_id)
        file = store.get_file(sim_id, name)
        if file is None:
            raise HTTPException(404, f"sim {sim_id} has no file {name!r}")
        return FileResponse(file_path(sim_id, name), media_type=file.content_type, filename=name)

    @app.get("/api/runs/{run_id}/files", dependencies=auth)
    def run_files(run_id: str) -> list[RunFile]:
        get_run_or_404(run_id)
        return store.run_files(run_id)

    @app.get("/api/runs/{run_id}/perfetto", dependencies=auth)
    def run_perfetto(run_id: str) -> Response:
        run = get_run_or_404(run_id)
        trace = run_trace(run, cascade.light_kernels(run_id), store.real_timing(run.trace_id))
        return Response(
            json.dumps(trace, allow_nan=False),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{run_id}-perfetto.json"'},
        )

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def api_not_found(path: str) -> JSONResponse:
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    if (STATIC_DIR / "index.html").is_file():
        root = STATIC_DIR.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = (root / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(root):
                return FileResponse(candidate)
            return FileResponse(root / "index.html")

    return app
