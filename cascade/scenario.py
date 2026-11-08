from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from cascade.ingest import read_json
from cascade.models import TERMINAL_SIM_STATES, RunEntry, SimState, TraceState
from cascade.samples import recorded_configs
from cascade.sim.request_model import GPU_SPECS, PerfProfile, build_profile, model_limit

if TYPE_CHECKING:
    from cascade.api import Cascade


def error(status: int, message: str, code: str, **extra) -> HTTPException:
    return HTTPException(
        status,
        {
            "message": message,
            "type": "invalid_request_error" if status < 500 else "server_error",
            "code": code,
            **extra,
        },
    )


@dataclass(frozen=True)
class BaseScenario:
    trace_id: str
    gpu_config: str
    run_id: str
    sim_id: str
    cached: bool
    real: dict
    simulated: dict
    compare: dict | None
    profile: PerfProfile

    def identifiers(self):
        return {
            k: getattr(self, k) for k in ("trace_id", "gpu_config", "run_id", "sim_id", "cached")
        }


class Scenarios:
    def __init__(self, cascade: Cascade):
        self.cascade = cascade
        self.store = cascade.store
        self._lock = threading.Lock()
        self._profile_lock = threading.Lock()
        self._profiles: dict[str, PerfProfile] = {}

    def trace(self, model: str):
        ready = [t for t in self.store.list_traces() if t.state == TraceState.ready]
        trace = next((t for t in ready if t.id == model), None)
        trace = trace or next((t for t in ready if t.model == model), None)
        if trace is None:
            raise error(404, f"No ready trace for model {model!r}", "model_not_found")
        return trace

    def metadata(self, trace):
        root = self.cascade.settings.trace_dir(trace.id)

        def load(name):
            value = read_json(root / name)
            return value if isinstance(value, dict) else {}

        real = load("corrected_timeline.json")
        if not isinstance(real.get("meta"), dict) or not isinstance(
            real["meta"].get("model_config"), dict
        ):
            raise error(
                422, "Trace lacks model_config required for request sizing", "unsupported_profile"
            )
        real.setdefault("meta", {}).update(run=load("run.json"), workload=load("workload.json"))
        return real

    def validate_shape(self, real, prompt, output):
        try:
            limit = model_limit(real)
        except ValueError as exc:
            raise error(422, str(exc), "unsupported_profile") from exc
        if prompt < 1 or output < 0 or prompt + output > limit:
            raise error(
                422, f"request exceeds model limits (max_model_len={limit})", "invalid_request"
            )

    def timeline(self, sim):
        return next(
            (t for t in self.store.step_timelines_for_run(sim.run_id) if t.sim_id == sim.id), None
        )

    def select(self, trace, gpu_config: str | None, fresh: bool):
        config = gpu_config or trace.suggested_sim_config
        cfg = self.cascade.configs.get(config) if config else None
        if cfg is None:
            raise error(422, f"unknown gpu_config {config!r}", "invalid_gpu_config")
        if cfg.arch != trace.trace_gpu_arch:
            raise error(
                422,
                f"{cfg.name} is {cfg.arch} but trace {trace.id} is {trace.trace_gpu_arch}",
                "architecture_mismatch",
            )
        if cfg.name not in GPU_SPECS:
            raise error(422, f"request sizing is unsupported for {cfg.name}", "unsupported_profile")
        with self._lock:
            if not fresh:
                sims = self.store.sims_for_trace_config(trace.id, cfg.name)
                for sim in sims:
                    if sim.state == SimState.succeeded and self.timeline(sim) is not None:
                        return sim, True
                for sim in sims:
                    if sim.state in (SimState.pending, SimState.launching, SimState.running):
                        return sim, False
            kernels_total = trace.kernel_count
            if trace.sample:
                recorded = recorded_configs(self.cascade.settings.trace_dir(trace.id))
                if cfg.name not in recorded:
                    raise error(
                        422,
                        f"Sample trace {trace.id} has no recorded results for {cfg.name!r}",
                        "sample_replay_unavailable",
                    )
                kernels_total = recorded[cfg.name]
            run_id = self.store.create_run(
                trace.id,
                [RunEntry(gpu_config=cfg.name, count=1)],
                {cfg.name: kernels_total},
                replay=trace.sample,
            )
            self.cascade.launch_run(run_id)
            return self.store.sims_for_run(run_id)[0], False

    def base(self, trace, sim, cached, timing):
        real = self.metadata(trace)
        with self._profile_lock:
            if sim.id not in self._profiles:
                try:
                    profile = build_profile(real, timing.simulated, sim.gpu_config)
                except (ValueError, KeyError, TypeError) as exc:
                    raise error(
                        422,
                        f"Cannot build request performance profile: {exc}",
                        "unsupported_profile",
                    ) from exc
                if len(self._profiles) >= 128:
                    self._profiles.pop(next(iter(self._profiles)))
                self._profiles[sim.id] = profile
            profile = self._profiles[sim.id]
        return BaseScenario(
            trace.id,
            sim.gpu_config,
            sim.run_id,
            sim.id,
            cached,
            real,
            timing.simulated,
            timing.compare,
            profile,
        )

    async def events(self, trace, sim, cached, request: Request, *, poll_s=0.25, progress_s=5.0):
        deadline = time.monotonic() + self.cascade.settings.openai_wait_s
        last_progress = float("-inf")
        while True:
            current = await run_in_threadpool(self.store.get_sim, sim.id)
            if current.state in TERMINAL_SIM_STATES:
                if current.state != SimState.succeeded:
                    raise error(
                        502,
                        current.error or f"sim {current.state}",
                        "simulation_failed",
                        run_id=sim.run_id,
                        sim_id=sim.id,
                    )
                timing = await run_in_threadpool(self.timeline, current)
                if timing is None:
                    raise error(
                        502,
                        "Sim succeeded without a step timeline",
                        "missing_timeline",
                        run_id=sim.run_id,
                        sim_id=sim.id,
                    )
                yield await run_in_threadpool(self.base, trace, current, cached, timing)
                return
            if time.monotonic() >= deadline:
                raise error(
                    504,
                    f"Timed out waiting for run {sim.run_id}",
                    "simulation_timeout",
                    run_id=sim.run_id,
                    sim_id=sim.id,
                )
            if await request.is_disconnected():
                raise error(499, "client disconnected", "disconnected")
            if time.monotonic() - last_progress >= progress_s:
                yield (
                    f": progress {current.kernels_done}/{current.kernels_total} {current.state}\n\n"
                )
                last_progress = time.monotonic()
            await asyncio.sleep(poll_s)

    async def resolve(self, trace, gpu_config, fresh, request):
        sim, cached = await run_in_threadpool(self.select, trace, gpu_config, fresh)
        async for event in self.events(trace, sim, cached, request):
            if isinstance(event, BaseScenario):
                return event
