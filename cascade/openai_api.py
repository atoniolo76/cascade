from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, StrictBool

from cascade.ingest import read_json
from cascade.models import TERMINAL_SIM_STATES, RunEntry, SimState, TraceState
from cascade.openai_tokens import Tokenizers, sample_tokens

if TYPE_CHECKING:
    from cascade.api import Cascade

POLL_S = 0.25
PROGRESS_S = 5.0
NOTE = (
    "Timings simulate the profiled workload, not the requested prompt or output size. "
    "Content is uniformly sampled vocabulary tokens, not model inference. "
    "Token counts count sampled IDs; decoded text can re-tokenize differently."
)


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


class OpenAIRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def wrapped(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                return JSONResponse(
                    {
                        "error": {
                            "message": str(exc),
                            "type": "invalid_request_error",
                            "code": "invalid_request",
                        }
                    },
                    status_code=422,
                )
            except HTTPException as exc:
                detail = (
                    exc.detail
                    if isinstance(exc.detail, dict)
                    else {
                        "message": str(exc.detail),
                        "type": "invalid_request_error",
                        "code": "authentication_error"
                        if exc.status_code == 401
                        else "invalid_request",
                    }
                )
                return JSONResponse(
                    {"error": detail}, status_code=exc.status_code, headers=exc.headers
                )

        return wrapped


TokenCount = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]


class Completion(BaseModel):
    model: Annotated[str, Field(min_length=1)]
    prompt: str = ""
    max_tokens: TokenCount | None = None
    max_completion_tokens: TokenCount | None = None
    gpu_config: str | None = None
    fresh: StrictBool = False
    stream: StrictBool = False
    seed: Annotated[int, Field(strict=True)] | None = None


class Message(BaseModel):
    role: str
    content: str


class ChatCompletion(Completion):
    messages: Annotated[list[Message], Field(min_length=1)]


class OpenAI:
    def __init__(self, cascade: Cascade):
        self.cascade = cascade
        self.store = cascade.store
        self.tokenizers = Tokenizers(cascade.settings.data_dir)
        self._lock = threading.Lock()

    def trace(self, model: str):
        ready = [t for t in self.store.list_traces() if t.state == TraceState.ready]
        trace = next((t for t in ready if t.id == model), None)
        trace = trace or next((t for t in ready if t.model == model), None)
        if trace is None:
            raise error(404, f"No ready trace for model {model!r}", "model_not_found")
        return trace

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
        with self._lock:
            if not fresh:
                sims = self.store.sims_for_trace_config(trace.id, cfg.name)
                for sim in sims:
                    if sim.state == SimState.succeeded and self.timeline(sim) is not None:
                        return sim, True
                for sim in sims:
                    if sim.state in (SimState.pending, SimState.launching, SimState.running):
                        return sim, False
            run_id = self.store.create_run(
                trace.id, [RunEntry(gpu_config=cfg.name, count=1)], {cfg.name: trace.kernel_count}
            )
            self.cascade.launch_run(run_id)
            return self.store.sims_for_run(run_id)[0], False

    def inputs(self, body: Completion, chat: bool):
        trace = self.trace(body.model)
        root = self.cascade.settings.trace_dir(trace.id)
        real = read_json(root / "corrected_timeline.json")
        real = real if isinstance(real, dict) else {}
        workload = read_json(root / "workload.json")
        workload = workload if isinstance(workload, dict) else {}
        meta = real.get("meta", {})
        profiled_max = workload.get("max_tokens", meta.get("max_tokens"))
        requested = body.max_tokens
        if chat and body.max_completion_tokens is not None:
            if requested is not None and requested != body.max_completion_tokens:
                raise error(422, "max_tokens and max_completion_tokens disagree", "invalid_request")
            requested = body.max_completion_tokens
        count = requested if requested is not None else profiled_max
        if type(count) is not int or not 0 <= count <= 1_000_000:
            raise error(
                422, "Provide max_tokens; trace has no supported default", "invalid_request"
            )
        prompts = workload.get("prompts")
        profiled_prompt = (
            prompts[0] if isinstance(prompts, list) and len(prompts) == 1 else (meta.get("prompt"))
        )
        prompt = body.prompt
        match_prompt = prompt
        if chat:
            prompt = "\n".join(m.content for m in body.messages)
            match_prompt = (
                body.messages[0].content
                if len(body.messages) == 1 and body.messages[0].role == "user"
                else None
            )
        return (
            trace,
            real,
            count,
            prompt,
            {
                "prompt_matches": match_prompt is not None and match_prompt == profiled_prompt,
                "max_tokens_matches": count == profiled_max,
                "profiled_max_tokens": profiled_max,
            },
        )

    def result(self, body, chat, trace, real, count, prompt, match, sim, cached, timing):
        sampled = sample_tokens(self.tokenizers.get(trace.model), count, body.seed, prompt)
        fields = ("repeat", "ttft_ms", "itl_ms_mean", "e2e_ms")

        def metrics(data):
            return [
                {k: row.get(k) for k in fields}
                for row in data.get("requests", [])
                if not row.get("warmup")
            ]

        usage = {
            "prompt_tokens": sampled["prompt_tokens"],
            "completion_tokens": count,
            "total_tokens": (
                sampled["prompt_tokens"] + count if sampled["prompt_tokens"] is not None else None
            ),
        }
        payload = {
            "id": f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}",
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    **(
                        {"message": {"role": "assistant", "content": sampled["text"]}}
                        if chat
                        else {"text": sampled["text"], "logprobs": None}
                    ),
                }
            ],
            "usage": usage,
            "cascade": {
                "trace_id": trace.id,
                "gpu_config": sim.gpu_config,
                "run_id": sim.run_id,
                "sim_id": sim.id,
                "cached": cached,
                "tokens_source": sampled["source"],
                "output_token_ids": sampled["ids"],
                "simulated": metrics(timing.simulated),
                "real": metrics(real),
                "compare": timing.compare.get("summary") if timing.compare else None,
                "profile_match": match["prompt_matches"] and match["max_tokens_matches"],
                "profile_match_details": match,
                "note": NOTE,
            },
        }
        return payload, sampled["pieces"]

    async def complete(self, body: Completion, request: Request, chat: bool):
        inputs = await run_in_threadpool(self.inputs, body, chat)
        trace = inputs[0]
        sim, cached = await run_in_threadpool(self.select, trace, body.gpu_config, body.fresh)
        deadline = time.monotonic() + self.cascade.settings.openai_wait_s

        async def wait():
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
                    result = await run_in_threadpool(
                        self.result, body, chat, *inputs, current, cached, timing
                    )
                    yield result
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
                    return
                if time.monotonic() - last_progress >= PROGRESS_S:
                    yield (
                        f": progress {current.kernels_done}/{current.kernels_total} "
                        f"{current.state}\n\n"
                    )
                    last_progress = time.monotonic()
                await asyncio.sleep(POLL_S)

        if not body.stream:
            async for event in wait():
                if not isinstance(event, str):
                    return JSONResponse(event[0])
            return JSONResponse(
                {
                    "error": {
                        "message": "client disconnected",
                        "type": "server_error",
                        "code": "disconnected",
                    }
                },
                499,
            )

        async def stream():
            try:
                async for event in wait():
                    if isinstance(event, str):
                        yield event
                        continue
                    payload, pieces = event
                    base = {k: payload[k] for k in ("id", "object", "created", "model")}
                    if chat:
                        base["object"] = "chat.completion.chunk"
                    for piece in pieces:
                        choice = {
                            "index": 0,
                            "finish_reason": None,
                            **(
                                {"delta": {"role": "assistant", "content": piece}}
                                if chat
                                else {"text": piece, "logprobs": None}
                            ),
                        }
                        yield f"data: {json.dumps({**base, 'choices': [choice]})}\n\n"
                    choice = {
                        "index": 0,
                        "finish_reason": "length",
                        **({"delta": {}} if chat else {"text": "", "logprobs": None}),
                    }
                    final = {
                        **base,
                        "choices": [choice],
                        "usage": payload["usage"],
                        "cascade": payload["cascade"],
                    }
                    yield f"data: {json.dumps(final)}\n\n"
            except HTTPException as exc:
                yield f"data: {json.dumps({'error': exc.detail})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def install_openai_routes(app: FastAPI, cascade: Cascade, bearer: Callable) -> None:
    service = OpenAI(cascade)
    router = APIRouter(prefix="/v1", route_class=OpenAIRoute, dependencies=[Depends(bearer)])

    @router.get("/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": trace.id,
                    "object": "model",
                    "created": int(trace.created_at.timestamp()),
                    "owned_by": "cascade",
                    "metadata": {"model": trace.model},
                }
                for trace in service.store.list_traces()
                if trace.state == TraceState.ready
            ],
        }

    @router.post("/completions")
    async def completions(body: Completion, request: Request):
        return await service.complete(body, request, False)

    @router.post("/chat/completions")
    async def chat_completions(body: ChatCompletion, request: Request):
        return await service.complete(body, request, True)

    app.include_router(router)
