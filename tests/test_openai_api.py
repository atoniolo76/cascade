import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from cascade.api import create_app
from cascade.models import RunEntry, SimState, StepTimelineUpload, TraceState
from cascade.openai_tokens import Tokenizers
from tests.helpers import AUTH, make_settings, wait_for, write_processed_trace
from tests.test_api import RecordingLauncher
from tests.test_openai_tokens import vocabulary

PROMPT = "Hello, my name is"
BODY = {"model": "facebook/opt-125m", "prompt": PROMPT}
TIMELINE = {
    "meta": {
        "prompt": PROMPT,
        "max_tokens": 4,
        "model_config": {
            "architecture": "OPTForCausalLM",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "dtype": "float16",
        },
    },
    "steps": [
        {
            "step": i,
            "repeat": 1,
            "warmup": False,
            "kind": "prefill" if i == 0 else "decode",
            "num_scheduled_tokens": 6 if i == 0 else 1,
            "context_length": 6 + i,
            "baseline_wall_ns": 1_000_000,
            "gpu_busy_ns": 200_000,
            "baseline_gap_before_ns": 0,
            "kernels": [{"name": "gemm", "duration_ns": 200_000, "grid": [1, 1, 1]}],
        }
        for i in range(4)
    ],
    "summary": {},
    "requests": [
        {"repeat": 0, "warmup": True, "e2e_ms": 999},
        {
            "repeat": 1,
            "warmup": False,
            "ttft_ms": 1,
            "itl_ms_mean": 2,
            "e2e_ms": 7,
            "num_output_tokens": 4,
        },
    ],
}


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(Tokenizers, "get", lambda self, model: None)
    monkeypatch.setattr(Tokenizers, "_chat_config", lambda self, model: {})
    settings = make_settings(tmp_path)
    root = write_processed_trace(settings, "trace-opt", arch="sm_90")
    (root / "corrected_timeline.json").write_text(json.dumps(TIMELINE))
    (root / "workload.json").write_text(json.dumps({"prompts": [PROMPT], "max_tokens": 4}))
    launcher = RecordingLauncher()
    app = create_app(settings, launcher)
    hub = app.state.cascade
    hub.store.register_trace("trace-opt", 6, 0)
    hub.store.update_trace("trace-opt", state=TraceState.processing)
    hub.store.update_trace(
        "trace-opt",
        state=TraceState.ready,
        metadata={
            "model": "facebook/opt-125m",
            "trace_gpu_arch": "sm_90",
            "suggested_sim_config": "SM90_H100",
        },
    )
    with TestClient(app, headers=AUTH) as client:
        yield client, hub, launcher


def pending(hub):
    run_id = hub.store.create_run("trace-opt", [RunEntry(gpu_config="SM90_H100", count=1)], 6)
    return hub.store.sims_for_run(run_id)[0]


def succeed(hub, sim_id, timeline=True):
    state = hub.store.get_sim(sim_id).state
    if state == SimState.pending:
        hub.store.transition_sim(sim_id, SimState.launching)
        state = SimState.launching
    if state == SimState.launching:
        hub.store.transition_sim(sim_id, SimState.running)
    if timeline:
        hub.store.set_step_timeline(
            sim_id,
            StepTimelineUpload(
                simulated=TIMELINE,
                compare={"summary": {"wall_ns": {"count": 1}}},
            ),
        )
    hub.store.transition_sim(sim_id, SimState.succeeded)


def cached(hub):
    sim = pending(hub)
    succeed(hub, sim.id)
    return sim


def test_models_only_lists_ready_traces_and_empty_list(endpoint):
    client, hub, _ = endpoint
    hub.store.register_trace("not-ready", 0, 0)
    data = client.get("/v1/models").json()
    assert data["object"] == "list"
    assert [(m["id"], m["owned_by"], m["metadata"]["model"]) for m in data["data"]] == [
        ("trace-opt", "cascade", "facebook/opt-125m")
    ]


def test_cache_hit_trace_id_and_model_alias_default_count(endpoint):
    client, hub, launcher = endpoint
    sim = cached(hub)
    body = client.post("/v1/completions", json=BODY).json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] == "length"
    assert len(body["choices"][0]["text"].split()) == 4
    assert body["usage"] == {"completion_tokens": 4, "prompt_tokens": 5, "total_tokens": 9}
    extra = body["cascade"]
    assert extra["cached"] and extra["run_id"] == sim.run_id and extra["sim_id"] == sim.id
    assert (
        extra["profile_match"]
        and len(extra["profiled"]["real"]) == len(extra["profiled"]["simulated"]) == 1
    )
    assert extra["profiled"]["compare"] == {"wall_ns": {"count": 1}}
    assert not launcher.launched
    alias = client.post("/v1/completions", json={**BODY, "model": "trace-opt"}).json()
    assert alias["cascade"]["sim_id"] == sim.id


def test_model_alias_selects_newest_ready_trace(endpoint):
    client, hub, _ = endpoint
    root = hub.settings.trace_dir("newer")
    root.mkdir()
    (root / "workload.json").write_text(json.dumps({"max_tokens": 2}))
    (root / "corrected_timeline.json").write_text(json.dumps(TIMELINE))
    hub.store.register_trace("newer", 6, 0)
    hub.store.update_trace("newer", state=TraceState.processing)
    hub.store.update_trace(
        "newer",
        state=TraceState.ready,
        metadata={
            "model": "facebook/opt-125m",
            "trace_gpu_arch": "sm_90",
            "suggested_sim_config": "SM90_H100",
        },
    )
    run = hub.store.create_run("newer", [RunEntry(gpu_config="SM90_H100", count=1)], 6)
    sim = hub.store.sims_for_run(run)[0]
    succeed(hub, sim.id)
    result = client.post("/v1/completions", json=BODY).json()
    assert result["cascade"]["trace_id"] == "newer"
    assert result["usage"]["completion_tokens"] == 2


@pytest.mark.parametrize("fresh,old_timeline", [(False, None), (True, True), (False, False)])
def test_miss_and_fresh_launch_normal_run(endpoint, fresh, old_timeline):
    client, hub, launcher = endpoint
    if old_timeline is not None:
        succeed(hub, pending(hub).id, timeline=old_timeline)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(client.post, "/v1/completions", json={**BODY, "fresh": fresh})
        sim_id = wait_for(lambda: launcher.launched and launcher.launched[-1])
        succeed(hub, sim_id)
        response = future.result(timeout=5)
    assert response.status_code == 200, response.text
    assert response.json()["cascade"]["sim_id"] == sim_id
    assert response.json()["cascade"]["cached"] is False
    assert len(launcher.launched) == 1


def test_attach_to_inflight_including_concurrent_callers(endpoint):
    client, hub, launcher = endpoint
    sim = pending(hub)
    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(client.post, "/v1/completions", json=BODY) for _ in range(2)]
        time.sleep(0.05)
        succeed(hub, sim.id)
        results = [f.result(timeout=5).json() for f in futures]
    assert {r["cascade"]["sim_id"] for r in results} == {sim.id}
    assert not launcher.launched and len(hub.store.list_runs()) == 1


def test_concurrent_misses_only_launch_once(endpoint):
    client, hub, launcher = endpoint
    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(client.post, "/v1/completions", json=BODY) for _ in range(2)]
        sim_id = wait_for(lambda: launcher.launched and launcher.launched[-1])
        succeed(hub, sim_id)
        results = [f.result(timeout=5).json() for f in futures]
    assert {r["cascade"]["sim_id"] for r in results} == {sim_id}
    assert len(hub.store.list_runs()) == 1


@pytest.mark.parametrize("chat", [False, True])
def test_exact_token_count_seed_and_profile_mismatch(endpoint, chat):
    client, hub, _ = endpoint
    cached(hub)
    body = {**BODY, "max_tokens": 13, "seed": 123}
    path = "/v1/completions"
    if chat:
        path = "/v1/chat/completions"
        body = {
            "model": "trace-opt",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_completion_tokens": 13,
            "seed": 123,
        }
    first, second = [client.post(path, json=body).json() for _ in range(2)]
    text = first["choices"][0]["message"]["content"] if chat else first["choices"][0]["text"]
    assert len(re.findall(r"tok\d+", text)) == 13
    assert first["usage"]["completion_tokens"] == 13
    assert first["cascade"]["output_token_ids"] == second["cascade"]["output_token_ids"]
    assert first["choices"] == second["choices"]
    assert first["cascade"]["tokens_source"] == "placeholder"
    assert first["cascade"]["profile_match"] is False
    assert first["cascade"]["profiled"]["simulated"][0]["e2e_ms"] == 7
    assert "profiled workload" in first["cascade"]["note"]


@pytest.mark.parametrize("chat", [False, True])
def test_stream_progress_content_final_metrics_and_done(endpoint, monkeypatch, chat):
    client, hub, _ = endpoint
    monkeypatch.setattr("cascade.openai_api.PROGRESS_S", 0.01)
    monkeypatch.setattr("cascade.openai_api.POLL_S", 0.01)
    sim = pending(hub)
    thread = threading.Thread(target=lambda: (time.sleep(0.1), succeed(hub, sim.id)))
    body = {**BODY, "stream": True, "max_tokens": 5, "seed": 12}
    path = "/v1/completions"
    if chat:
        path = "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": PROMPT}]
    thread.start()
    try:
        with client.stream("POST", path, json=body) as response:
            lines = list(response.iter_lines())
    finally:
        thread.join()
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert any(line.startswith(": progress 0/6 ") for line in lines)
    assert lines[-2] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert all(
        chunk["object"] == ("chat.completion.chunk" if chat else "text_completion")
        for chunk in chunks
    )
    text = "".join(
        chunk["choices"][0].get("delta", {}).get("content", "")
        if chat
        else chunk["choices"][0]["text"]
        for chunk in chunks
    )
    assert len(text.split()) == 5
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 5
    assert chunks[-1]["cascade"]["sim_id"] == sim.id


def test_empty_completion_and_chat_default(endpoint):
    client, hub, _ = endpoint
    cached(hub)
    result = client.post("/v1/completions", json={**BODY, "max_tokens": 0}).json()
    assert result["choices"][0]["text"] == "" and result["usage"]["completion_tokens"] == 0
    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "trace-opt",
            "messages": [{"role": "user", "content": PROMPT}],
        },
    ).json()
    assert chat["usage"]["completion_tokens"] == 4 and chat["cascade"]["profile_match"]


def test_vocabulary_content_and_tokenized_prompt_usage(endpoint, monkeypatch):
    client, hub, _ = endpoint
    cached(hub)
    monkeypatch.setattr(Tokenizers, "get", lambda self, model: (vocabulary(), (1, 2)))
    response = client.post(
        "/v1/completions",
        json={
            **BODY,
            "prompt": "hello world",
            "max_tokens": 10,
            "seed": 42,
        },
    ).json()
    assert response["usage"] == {"prompt_tokens": 2, "completion_tokens": 10, "total_tokens": 12}
    assert response["cascade"]["tokens_source"] == "vocab_uniform"
    assert len(response["cascade"]["output_token_ids"]) == 10
    assert set(response["cascade"]["output_token_ids"]) == {1, 2}
    assert response["cascade"]["profile_match"] is False


def test_conflicting_chat_token_counts_are_rejected(endpoint):
    client, _, _ = endpoint
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "trace-opt",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 3,
            "max_completion_tokens": 4,
        },
    )
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "body,status,code",
    [
        ({"model": "missing"}, 404, "model_not_found"),
        ({**BODY, "gpu_config": "unknown"}, 422, "invalid_gpu_config"),
        ({**BODY, "gpu_config": "SM86_RTX3070"}, 422, "architecture_mismatch"),
        ({**BODY, "max_tokens": -1}, 422, "invalid_request"),
        ({**BODY, "max_tokens": True}, 422, "invalid_request"),
        ({**BODY, "fresh": "yes"}, 422, "invalid_request"),
    ],
)
def test_openai_error_shapes(endpoint, body, status, code):
    client, _, launcher = endpoint
    result = client.post("/v1/completions", json=body)
    assert result.status_code == status, result.text
    assert result.json()["error"]["code"] == code
    assert set(result.json()["error"]) >= {"message", "type", "code"}
    assert not launcher.launched


def test_auth_and_no_auth(tmp_path):
    app = create_app(make_settings(tmp_path), RecordingLauncher())
    with TestClient(app) as client:
        for path in ("/v1/models", "/v1/completions", "/v1/chat/completions"):
            response = client.get(path) if path.endswith("models") else client.post(path, json=BODY)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "authentication_error"
    app = create_app(make_settings(tmp_path, CASCADE_NO_AUTH="1"), RecordingLauncher())
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 200
        assert client.get("/v1/models").json()["data"] == []


@pytest.mark.parametrize("stream", [False, True])
def test_timeout_returns_pollable_run_and_does_not_cancel(endpoint, stream):
    client, hub, _ = endpoint
    hub.settings = replace(hub.settings, openai_wait_s=0.01)
    sim = pending(hub)
    response = client.post("/v1/completions", json={**BODY, "stream": stream})
    if stream:
        assert response.status_code == 200 and "data: [DONE]" in response.text
        data = next(
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        )
    else:
        assert response.status_code == 504
        data = response.json()
    assert data["error"]["code"] == "simulation_timeout"
    assert data["error"]["run_id"] == sim.run_id
    assert hub.store.get_sim(sim.id).state == SimState.pending


def test_failed_sim_and_missing_step_timeline(endpoint):
    client, hub, _ = endpoint
    sim = pending(hub)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(client.post, "/v1/completions", json=BODY)
        time.sleep(0.05)
        hub.store.transition_sim(sim.id, SimState.failed, error="bad simulator")
        response = future.result(timeout=5)
    assert response.status_code == 502 and "bad simulator" in response.text
    sim = pending(hub)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(client.post, "/v1/completions", json=BODY)
        time.sleep(0.05)
        succeed(hub, sim.id, timeline=False)
        response = future.result(timeout=5)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "missing_timeline"
