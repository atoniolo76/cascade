import json

from cascade.sim.importer import aligned_durations, read_stats
from tests.conftest import BABELSTREAM_LOG, FIXTURES
from tests.helpers import CASCADE, running_app, wait_for, write_processed_trace

TIMELINE = FIXTURES / "timeline"


def test_local_replay_worker_accepts_alignment_and_serves_openai(tmp_path, monkeypatch):
    monkeypatch.setattr("cascade.openai_tokens.Tokenizers.get", lambda self, model: None)
    cmd = f"{CASCADE} worker --replay {BABELSTREAM_LOG} --delay 0 --heartbeat-s 0.3"
    with running_app(tmp_path, CASCADE_SIM_CMD=cmd) as (client, _, settings):
        root = write_processed_trace(settings, "trace-rtx", arch="sm_90")
        real = json.loads((TIMELINE / "corrected_timeline.json").read_text())
        real["meta"]["model_config"] = {
            "architecture": "OPTForCausalLM",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "dtype": "float16",
        }
        for step in real["steps"]:
            step["context_length"] = 6 if step["kind"] == "prefill" else 7
        (root / "corrected_timeline.json").write_text(json.dumps(real))
        mapping = json.loads((TIMELINE / "babelstream_kernel_mapping.json").read_text())
        alignment = {
            "kernels": [
                {
                    "tracer_uid": row["uid"],
                    "dims_match": True,
                    **{k: row[k] for k in ("name", "repeat", "step", "seq")},
                }
                for row in mapping
            ],
            "provenance": {},
            "steps": [{"repeat": 0, "step": step} for step in (0, 1)],
            "unmatched": {},
        }
        (root / "trace_alignment.json").write_text(json.dumps(alignment))
        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(model="facebook/opt-125m", suggested_sim_config="SM90_H100")
        metadata_path.write_text(json.dumps(metadata))
        wait_for(
            lambda: (
                (t := client.get("/api/traces/trace-rtx").json()).get("state") == "ready"
                and t.get("has_step_timeline")
            )
        )
        body = {"model": "trace-rtx", "prompt": "Hello, my name is", "max_tokens": 7, "seed": 9}
        response = client.post("/v1/completions", json=body)
        assert response.status_code == 200, response.text
        data = response.json()
        assert not data["cascade"]["cached"] and data["cascade"]["gpu_config"] == "SM90_H100"
        assert len(data["choices"][0]["text"].split()) == 7
        run_id = data["cascade"]["run_id"]
        (timing,) = client.get(f"/api/runs/{run_id}/step-timelines").json()
        assert timing["simulated"]["summary"]["simulation"]["coverage"]["simulated_kernels"] == 6
        second = client.post("/v1/completions", json=body).json()
        assert second["cascade"]["cached"] and second["cascade"]["run_id"] == run_id
        fresh = client.post("/v1/completions", json={**body, "fresh": True}).json()
        assert not fresh["cascade"]["cached"] and fresh["cascade"]["run_id"] != run_id
        assert len(client.get("/api/runs").json()) == 2
        assert aligned_durations(
            read_stats(BABELSTREAM_LOG),
            json.loads((root / "corrected_timeline.json").read_text()),
            alignment,
            root / "traces/kernelslist.g",
            1132,
        )["kernels"]
