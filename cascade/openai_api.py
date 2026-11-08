from __future__ import annotations

import json
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

from cascade.models import TraceState
from cascade.openai_tokens import Tokenizers, sample_tokens
from cascade.scenario import Scenarios, error
from cascade.sim.request_model import simulate_request

if TYPE_CHECKING:
    from cascade.api import Cascade

POLL_S = 0.25
PROGRESS_S = 5.0
NOTE = (
    "Timing estimates the requested shape; profiled workload metrics are separate. "
    "Each /v1 request is simulated alone, without cross-request batching. "
    "Content is uniformly sampled vocabulary tokens, not model inference. "
    "Token counts count sampled IDs; decoded text can re-tokenize differently."
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
    def __init__(self, cascade: Cascade, scenarios: Scenarios):
        self.cascade = cascade
        self.store = cascade.store
        self.tokenizers = Tokenizers(cascade.settings.data_dir)
        self.scenarios = scenarios

    def inputs(self, body: Completion, chat: bool):
        trace = self.scenarios.trace(body.model)
        real = self.scenarios.metadata(trace)
        workload = real["meta"]["workload"]
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
        prompt, prompt_tokens, source, loaded = self.tokenizers.prepare_prompt(
            trace.model,
            prompt,
            [m.model_dump() for m in body.messages] if chat else None,
        )
        self.scenarios.validate_shape(real, prompt_tokens, count, prompt_tokens_source=source)
        if source == "tokenizer:chat_template":
            match_prompt = prompt
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
            prompt_tokens,
            source,
            loaded,
        )

    def result(
        self, body, chat, trace, real, count, prompt, match, prompt_tokens, source, loaded, base
    ):
        sampled = sample_tokens(loaded, count, body.seed, prompt)
        try:
            timing = simulate_request(base.profile, prompt_tokens, count)
        except ValueError as exc:
            raise error(422, str(exc), "invalid_request") from exc
        fields = ("repeat", "ttft_ms", "itl_ms_mean", "e2e_ms")

        def metrics(data):
            return [
                {k: row.get(k) for k in fields}
                for row in data.get("requests", [])
                if not row.get("warmup")
            ]

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": count,
            "total_tokens": prompt_tokens + count,
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
                **base.identifiers(),
                "tokens_source": sampled["source"],
                "prompt_tokens_source": source,
                "output_token_ids": sampled["ids"],
                "timing": {
                    k: getattr(timing, k)
                    for k in (
                        "prompt_tokens",
                        "output_tokens",
                        "ttft_ms",
                        "itl_ms_mean",
                        "itl_ms_p50",
                        "e2e_ms",
                        "provenance_counts",
                        "calibration_range",
                        "assumptions",
                    )
                },
                "profiled": {
                    "simulated": metrics(base.simulated),
                    "real": metrics(real),
                    "compare": base.compare.get("summary") if base.compare else None,
                },
                "profile_match": match["prompt_matches"] and match["max_tokens_matches"],
                "profile_match_details": match,
                "note": NOTE,
            },
        }
        return payload, sampled["pieces"]

    async def complete(self, body: Completion, request: Request, chat: bool):
        inputs = await run_in_threadpool(self.inputs, body, chat)
        trace = inputs[0]
        sim, cached = await run_in_threadpool(
            self.scenarios.select, trace, body.gpu_config, body.fresh
        )

        async def wait():
            async for event in self.scenarios.events(
                trace,
                sim,
                cached,
                request,
                poll_s=POLL_S,
                progress_s=PROGRESS_S,
            ):
                yield (
                    event
                    if isinstance(event, str)
                    else await run_in_threadpool(
                        self.result,
                        body,
                        chat,
                        *inputs,
                        event,
                    )
                )

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


def install_openai_routes(
    app: FastAPI, cascade: Cascade, bearer: Callable, scenarios: Scenarios
) -> None:
    service = OpenAI(cascade, scenarios)
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
