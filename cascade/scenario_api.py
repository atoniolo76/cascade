from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from math import isfinite
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from cascade.scenario import Scenarios, error
from cascade.sim.scheduler import SchedulerConfig, SchedulerRequest, simulate_scheduler

MAX_REQUESTS = 4096
MAX_WORK_TOKENS = 1_000_000
MAX_STEPS = 100_000
MAX_STEP_RECORDS = 5000
Positive = Annotated[int, Field(strict=True, ge=1, le=1_000_000)]
Output = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
Arrival = Annotated[float, Field(strict=True, ge=0, le=1e9, allow_inf_nan=False)]


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScenarioRequest(Options):
    id: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    arrival_s: Arrival = 0
    prompt_tokens: Positive
    output_tokens: Output


class SchedulerOptions(Options):
    max_num_seqs: Annotated[int, Field(strict=True, ge=1, le=MAX_REQUESTS)] = 256
    max_num_batched_tokens: Positive = 8192
    block_size: Positive = 16
    enable_chunked_prefill: StrictBool = True
    gpu_memory_utilization: Annotated[float, Field(strict=True, gt=0, le=1)] | None = None
    num_kv_blocks: Positive | None = None


class Scenario(Options):
    trace_id: Annotated[str, Field(min_length=1)] | None = None
    model: Annotated[str, Field(min_length=1)] | None = None
    gpu_config: str | None = None
    fresh: StrictBool = False
    requests: (
        Annotated[list[ScenarioRequest], Field(min_length=1, max_length=MAX_REQUESTS)] | None
    ) = None
    num_requests: Annotated[int, Field(strict=True, ge=1, le=MAX_REQUESTS)] | None = None
    prompt_tokens: Positive | None = None
    output_tokens: Output | None = None
    arrival_rate_rps: (
        Annotated[float, Field(strict=True, ge=0, le=1e9, allow_inf_nan=False)] | None
    ) = None
    scheduler: SchedulerOptions = Field(default_factory=SchedulerOptions)

    @model_validator(mode="after")
    def validate_scenario(self):
        if (self.trace_id is None) == (self.model is None):
            raise ValueError("provide exactly one of trace_id or model")
        shorthand = (
            self.num_requests,
            self.prompt_tokens,
            self.output_tokens,
            self.arrival_rate_rps,
        )
        if self.requests is not None:
            if any(value is not None for value in shorthand):
                raise ValueError("requests and shorthand fields are mutually exclusive")
        elif any(value is None for value in shorthand[:3]):
            raise ValueError("shorthand requires num_requests, prompt_tokens and output_tokens")
        expanded = self.expanded()
        if any(not isfinite(r.arrival_s) or r.arrival_s > 1e9 for r in expanded):
            raise ValueError("arrival spacing exceeds supported time range")
        if not self.scheduler.enable_chunked_prefill and any(
            r.prompt_tokens > self.scheduler.max_num_batched_tokens for r in expanded
        ):
            raise ValueError("prompt exceeds token budget with chunked prefill disabled")
        if len({r.id for r in expanded}) != len(expanded):
            raise ValueError("request IDs must be unique, including generated IDs")
        if sum(r.prompt_tokens + r.output_tokens for r in expanded) > MAX_WORK_TOKENS:
            raise ValueError(f"scenario exceeds {MAX_WORK_TOKENS} total prompt/output tokens")
        return self

    def expanded(self):
        if self.requests is not None:
            return [
                SchedulerRequest(
                    r.id or f"request-{i}", r.arrival_s, r.prompt_tokens, r.output_tokens
                )
                for i, r in enumerate(self.requests)
            ]
        return [
            SchedulerRequest(
                f"request-{i}",
                i / self.arrival_rate_rps if self.arrival_rate_rps else 0,
                self.prompt_tokens,
                self.output_tokens,
            )
            for i in range(self.num_requests)
        ]


def evaluate(base, requests, options):
    try:
        result = simulate_scheduler(
            base.profile, requests, SchedulerConfig(**options.model_dump()), max_steps=MAX_STEPS
        )
    except ValueError as exc:
        raise error(422, str(exc), "invalid_request") from exc
    total = len(result.steps)
    indices = (
        list(range(total))
        if total <= MAX_STEP_RECORDS
        else [i * (total - 1) // (MAX_STEP_RECORDS - 1) for i in range(MAX_STEP_RECORDS)]
    )
    counts = Counter()
    for step in result.steps:
        counts.update(step.provenance_counts)
    metrics = []
    for row in result.requests:
        record = asdict(row)
        record["output_tokens"] = len(record.pop("token_timestamps_s"))
        metrics.append(record)
    return {
        **base.identifiers(),
        "requests": metrics,
        "aggregate": {
            k: getattr(result, k)
            for k in (
                "throughput_output_tokens_s",
                "p50_ttft_ms",
                "p99_ttft_ms",
                "p50_itl_ms",
                "p50_e2e_ms",
                "p99_e2e_ms",
                "idle_time_s",
            )
        },
        "steps": [{"step_index": i, **asdict(result.steps[i])} for i in indices],
        "steps_total": total,
        "steps_returned": len(indices),
        "steps_downsampled": len(indices) < total,
        "steps_sampling": "uniform-index; first and last retained",
        "provenance_counts": dict(counts),
        "calibration_range": base.profile.calibration_range,
        "assumptions": result.assumptions,
        "scheduler": asdict(result.config),
    }


def install_scenario_routes(app, service: Scenarios, bearer):
    router = APIRouter(dependencies=[Depends(bearer)])

    @router.post("/api/scenarios")
    async def scenarios(body: Scenario, request: Request):
        trace = await run_in_threadpool(service.trace, body.trace_id or body.model)
        real = await run_in_threadpool(service.metadata, trace)
        requests = body.expanded()
        for row in requests:
            service.validate_shape(real, row.prompt_tokens, row.output_tokens)
        base = await service.resolve(trace, body.gpu_config, body.fresh, request)
        return await run_in_threadpool(evaluate, base, requests, body.scheduler)

    app.include_router(router)
