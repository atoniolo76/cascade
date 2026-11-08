import json
import shutil

from cascade.sim.importer import import_durations
from tests.conftest import BABELSTREAM_LOG, FIXTURES
from tests.helpers import CASCADE, running_app, wait_for, write_processed_trace

TIMELINE = FIXTURES / "timeline"


def test_local_replay_worker_accepts_alignment_and_serves_openai(tmp_path, monkeypatch):
    monkeypatch.setattr("cascade.openai_tokens.Tokenizers.get", lambda self, model: None)
    cmd = f"{CASCADE} worker --replay {BABELSTREAM_LOG} --delay 0 --heartbeat-s 0.3"
    with running_app(tmp_path, CASCADE_SIM_CMD=cmd) as (client, _, settings):
        root = write_processed_trace(settings, "trace-rtx", arch="sm_86")
        shutil.copy(TIMELINE / "corrected_timeline.json", root / "corrected_timeline.json")
        mapping = json.loads((TIMELINE / "babelstream_kernel_mapping.json").read_text())
        alignment = {
            "kernels": [
                {"tracer_uid": row["uid"], **{k: row[k] for k in ("name", "repeat", "step", "seq")}}
                for row in mapping
            ],
            "provenance": {},
            "steps": [],
            "unmatched": {},
        }
        (root / "trace_alignment.json").write_text(json.dumps(alignment))
        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(model="facebook/opt-125m", suggested_sim_config="SM86_RTX3070")
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
        assert not data["cascade"]["cached"] and data["cascade"]["gpu_config"] == "SM86_RTX3070"
        assert len(data["choices"][0]["text"].split()) == 7
        run_id = data["cascade"]["run_id"]
        (timing,) = client.get(f"/api/runs/{run_id}/step-timelines").json()
        assert timing["simulated"]["summary"]["simulation"]["coverage"]["simulated_kernels"] == 6
        second = client.post("/v1/completions", json=body).json()
        assert second["cascade"]["cached"] and second["cascade"]["run_id"] == run_id
        fresh = client.post("/v1/completions", json={**body, "fresh": True}).json()
        assert not fresh["cascade"]["cached"] and fresh["cascade"]["run_id"] != run_id
        assert len(client.get("/api/runs").json()) == 2
        assert import_durations(BABELSTREAM_LOG, root / "trace_alignment.json", 1132)["kernels"]
