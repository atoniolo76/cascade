import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import cascade.scenario as scenario_module
from cascade.api import create_app
from cascade.openai_tokens import Tokenizers
from cascade.sim.request_model import simulate_request
from tests.helpers import AUTH, make_settings, wait_for
from tests.test_api import RecordingLauncher
from tests.test_openai_api import BODY, succeed
from tests.test_openai_api import endpoint as endpoint
from tests.test_openai_tokens import vocabulary

TRACE = "sample__facebook_opt-125m__h100-e2e"
SCENARIO = {"trace_id": TRACE, "num_requests": 2, "prompt_tokens": 6, "output_tokens": 4}


@pytest.fixture(scope="module")
def sample_api(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Tokenizers, "get", lambda self, model: None)
        patch.setattr(Tokenizers, "_chat_config", lambda self, model: {})
        settings = make_settings(tmp_path_factory.mktemp("scenarios"), CASCADE_SAMPLES="true")
        launcher = RecordingLauncher()
        app = create_app(settings, launcher)
        with TestClient(app, headers=AUTH) as client:
            yield client, app, launcher


def test_seeded_scenario_is_transient_and_cache_first(sample_api):
    client, _, launcher = sample_api
    runs = client.get("/api/runs").json()
    response = client.post("/api/scenarios", json=SCENARIO)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["cached"] and data["gpu_config"] == "SM90_H100"
    assert len(data["requests"]) == 2 and all(r["output_tokens"] == 4 for r in data["requests"])
    assert data["steps"][0]["num_seqs"] == 2
    assert data["provenance_counts"]["assumed-host"] == data["steps_total"] == 4
    assert data["calibration_range"]["prefill_tokens"] == [6, 6]
    assert client.get("/api/runs").json() == runs
    assert not launcher.launched


@pytest.mark.parametrize("rate", [None, 0, 2])
def test_shorthand_matches_explicit_requests(sample_api, rate):
    client, _, _ = sample_api
    shorthand = {**SCENARIO, "arrival_rate_rps": rate}
    explicit = {
        "trace_id": TRACE,
        "requests": [
            {
                "id": f"request-{i}",
                "arrival_s": i / rate if rate else 0,
                "prompt_tokens": 6,
                "output_tokens": 4,
            }
            for i in range(2)
        ],
    }
    assert (
        client.post("/api/scenarios", json=shorthand).json()
        == client.post("/api/scenarios", json=explicit).json()
    )


def test_profile_cache_shared_across_api_and_openai(sample_api, monkeypatch):
    client, app, _ = sample_api
    original = scenario_module.build_profile
    calls = []

    def build(*args):
        calls.append(args[2])
        return original(*args)

    monkeypatch.setattr(scenario_module, "build_profile", build)
    app.state.scenarios._profiles.clear()
    first = client.post("/api/scenarios", json=SCENARIO).json()
    second = client.post("/api/scenarios", json={**SCENARIO, "output_tokens": 10}).json()
    completion = client.post(
        "/v1/completions", json={"model": TRACE, "prompt": "hello", "max_tokens": 3}
    ).json()
    assert first["sim_id"] == second["sim_id"] == completion["cascade"]["sim_id"]
    assert calls == ["SM90_H100"]
    h200 = client.post("/api/scenarios", json={**SCENARIO, "gpu_config": "SM90_H200"}).json()
    assert h200["sim_id"] != first["sim_id"]
    assert calls == ["SM90_H100", "SM90_H200"]


def test_model_alias_and_native_a100(sample_api):
    client, app, _ = sample_api
    newest = next(
        t for t in app.state.cascade.store.list_traces() if t.model == "facebook/opt-125m"
    )
    result = client.post(
        "/api/scenarios",
        json={
            k: v for k, v in {**SCENARIO, "model": "facebook/opt-125m"}.items() if k != "trace_id"
        },
    )
    assert result.status_code == 200, result.text
    assert result.json()["trace_id"] == newest.id
    a100 = client.post(
        "/api/scenarios", json={**SCENARIO, "trace_id": "sample__facebook_opt-125m__a100-e2e"}
    )
    assert a100.status_code == 200 and a100.json()["gpu_config"] == "SM80_A100"


def test_downsampling_does_not_change_totals(sample_api):
    client, _, _ = sample_api
    result = client.post(
        "/api/scenarios",
        json={
            **SCENARIO,
            "num_requests": 6,
            "output_tokens": 1000,
            "scheduler": {"max_num_seqs": 1},
        },
    )
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["steps_total"] == 6000 and data["steps_returned"] == 5000
    assert data["steps_downsampled"]
    assert data["steps"][0]["step_index"] == 0 and data["steps"][-1]["step_index"] == 5999
    assert data["provenance_counts"]["assumed-host"] == 6000
    assert all(r["output_tokens"] == 1000 for r in data["requests"])


@pytest.mark.parametrize(
    "changes",
    [
        {"num_requests": 4097},
        {"num_requests": True},
        {"prompt_tokens": "6"},
        {"prompt_tokens": 2048, "output_tokens": 1},
        {"output_tokens": -1},
        {"scheduler": {"max_num_seqs": 0}},
        {"scheduler": {"gpu_memory_utilization": 1.1}},
        {"scheduler": {"typo": 1}},
        {"model": "facebook/opt-125m"},
        {"requests": []},
        {"requests": [{"prompt_tokens": 6, "output_tokens": 1}]},
        {"num_requests": 4096, "prompt_tokens": 1024},
        {"arrival_rate_rps": -1},
        {"gpu_config": "unknown"},
        {"gpu_config": "SM86_RTX3070"},
        {"fresh": True},
    ],
)
def test_validation_errors_do_not_launch(sample_api, changes):
    client, _, launcher = sample_api
    response = client.post("/api/scenarios", json={**SCENARIO, **changes})
    assert response.status_code == 422, response.text
    assert not launcher.launched


def test_explicit_request_validation_and_caps(sample_api, monkeypatch):
    client, _, _ = sample_api
    request = {"prompt_tokens": 6, "output_tokens": 4}
    for rows in ([{**request, "arrival_s": -1}], [{**request, "id": "r"}] * 2, [request] * 4097):
        assert (
            client.post("/api/scenarios", json={"trace_id": TRACE, "requests": rows}).status_code
            == 422
        )
    monkeypatch.setattr("cascade.scenario_api.MAX_STEPS", 1)
    response = client.post("/api/scenarios", json=SCENARIO)
    assert response.status_code == 422 and "step limit=1" in response.text


def test_auth(sample_api):
    client, _, _ = sample_api
    assert (
        client.post(
            "/api/scenarios", json=SCENARIO, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )


def test_scenario_and_v1_attach_to_one_new_base_sim(endpoint):
    client, hub, launcher = endpoint
    with ThreadPoolExecutor() as pool:
        scenario = pool.submit(
            client.post, "/api/scenarios", json={**SCENARIO, "trace_id": "trace-opt"}
        )
        completion = pool.submit(client.post, "/v1/completions", json=BODY)
        sim_id = wait_for(lambda: launcher.launched and launcher.launched[-1])
        succeed(hub, sim_id)
        a, b = scenario.result(timeout=5), completion.result(timeout=5)
    assert a.status_code == b.status_code == 200, (a.text, b.text)
    assert a.json()["sim_id"] == b.json()["cascade"]["sim_id"] == sim_id
    assert len(launcher.launched) == 1


def test_scenario_timeout_keeps_base_run(endpoint):
    client, hub, _ = endpoint
    hub.settings = replace(hub.settings, openai_wait_s=0.01)
    response = client.post("/api/scenarios", json={**SCENARIO, "trace_id": "trace-opt"})
    assert response.status_code == 504
    detail = response.json()["detail"]
    assert detail["code"] == "simulation_timeout"
    assert hub.store.get_sim(detail["sim_id"]).run_id == detail["run_id"]


def test_openai_requested_timing_matches_model(sample_api, monkeypatch):
    client, app, _ = sample_api
    monkeypatch.setattr(Tokenizers, "get", lambda self, model: (vocabulary(), (1, 2)))
    outputs = []
    for prompt, count in (("hello", 4), ("hello", 16), ("hello " * 1800, 16)):
        response = client.post(
            "/v1/completions", json={"model": TRACE, "prompt": prompt, "max_tokens": count}
        )
        assert response.status_code == 200, response.text
        data = response.json()
        profile = app.state.scenarios._profiles[data["cascade"]["sim_id"]]
        expected = simulate_request(profile, data["usage"]["prompt_tokens"], count)
        assert data["cascade"]["timing"]["e2e_ms"] == expected.e2e_ms
        assert data["cascade"]["timing"]["ttft_ms"] == expected.ttft_ms
        assert data["cascade"]["prompt_tokens_source"] == "tokenizer"
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + count
        assert data["cascade"]["profiled"]["simulated"]
        outputs.append(data)
    assert outputs[1]["cascade"]["timing"]["e2e_ms"] > outputs[0]["cascade"]["timing"]["e2e_ms"]
    assert outputs[2]["cascade"]["timing"]["ttft_ms"] > outputs[1]["cascade"]["timing"]["ttft_ms"]
    assert outputs[2]["cascade"]["timing"]["provenance_counts"]["extrapolated:roofline"] == 16
    assert len({data["cascade"]["sim_id"] for data in outputs}) == 1


def test_chat_template_count_and_streamed_timing(sample_api, monkeypatch):
    client, app, _ = sample_api
    monkeypatch.setattr(Tokenizers, "get", lambda self, model: (vocabulary(), (1, 2)))
    monkeypatch.setattr(
        Tokenizers,
        "_chat_config",
        lambda self, model: {
            "chat_template": "{% for m in messages %}{{ m.content }} {% endfor %}"
            "{% generation %}hello{% endgeneration %}",
        },
    )
    body = {
        "model": TRACE,
        "messages": [{"role": "user", "content": "hello world"}],
        "max_tokens": 4,
    }
    data = client.post("/v1/chat/completions", json=body).json()
    assert data["usage"]["prompt_tokens"] == 3
    assert data["cascade"]["prompt_tokens_source"] == "tokenizer:chat_template"
    stream = client.post("/v1/chat/completions", json={**body, "stream": True})
    chunks = [
        json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: {")
    ]
    assert chunks[-1]["cascade"]["timing"] == data["cascade"]["timing"]
    assert chunks[-1]["usage"] == data["usage"]
    assert stream.text.endswith("data: [DONE]\n\n")


def test_openai_fallback_and_model_limit(sample_api):
    client, _, launcher = sample_api
    response = client.post(
        "/v1/completions", json={"model": TRACE, "prompt": "hello world", "max_tokens": 4}
    ).json()
    assert response["cascade"]["prompt_tokens_source"] == "estimated:utf8-bytes/4"
    assert response["usage"]["prompt_tokens"] == 3
    invalid = client.post(
        "/v1/completions", json={"model": TRACE, "prompt": "x" * 8192, "max_tokens": 1}
    )
    assert invalid.status_code == 422
    assert not launcher.launched
