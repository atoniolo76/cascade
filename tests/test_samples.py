"""The bundled sample (samples/opt125m-h100-e2e) that every workspace starts with."""

import csv
import gzip
import io
import json

import pytest
from fastapi.testclient import TestClient

from cascade.api import create_app
from cascade.samples import SAMPLES_ROOT
from tests.helpers import AUTH, make_settings

BUNDLE = SAMPLES_ROOT / "opt125m-h100-e2e"
TRACE_ID = "sample__facebook_opt-125m__h100-e2e"
RUN_ID = "run-sample-opt125m-h100-e2e"


def bundled(rel: str) -> bytes:
    path = BUNDLE / rel
    return (
        path.read_bytes() if path.exists() else gzip.decompress((BUNDLE / f"{rel}.gz").read_bytes())
    )


def simulated_rows() -> list[dict]:
    rows = csv.DictReader(io.StringIO(bundled("sim/kernels.csv").decode()))
    return sorted((r for r in rows if r["sim"] == "exact"), key=lambda r: int(r["uid"]))


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    settings = make_settings(tmp_path_factory.mktemp("sample"), CASCADE_SAMPLES="true")
    with TestClient(create_app(settings)) as client:
        client.headers.update(AUTH)
        yield client, settings


def test_starting_twice_leaves_one_sample_trace_and_run(tmp_path):
    settings = make_settings(tmp_path, CASCADE_SAMPLES="true")
    for _ in range(2):
        with TestClient(create_app(settings)) as client:
            traces = client.get("/api/traces", headers=AUTH).json()
            runs = client.get("/api/runs", headers=AUTH).json()
            kernels = client.get(f"/api/runs/{RUN_ID}/kernels", headers=AUTH).json()
    assert [(t["id"], t["sample"]) for t in traces] == [(TRACE_ID, True)]
    assert [(r["id"], r["sample"], r["state"]) for r in runs] == [(RUN_ID, True, "succeeded")]
    assert len(kernels) == len(simulated_rows())


def test_samples_off_seeds_nothing(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as client:
        assert client.get("/api/traces", headers=AUTH).json() == []
        assert client.get("/api/runs", headers=AUTH).json() == []
    assert not (tmp_path / "data" / "files").exists()


def test_sample_trace_is_ready_and_describes_the_real_trace(app):
    client, _ = app
    trace = client.get(f"/api/traces/{TRACE_ID}").json()
    kernelslist = bundled("traced/traces/kernelslist.g").decode().splitlines()
    assert {k: trace[k] for k in ("state", "model", "source_gpu", "trace_gpu_arch")} == {
        "state": "ready",
        "model": "facebook/opt-125m",
        "source_gpu": "NVIDIA H100 80GB HBM3",
        "trace_gpu_arch": "sm_90",
    }
    assert trace["layers"] == [
        "step 0 (prefill)",
        "step 2 (decode)",
        "step 10 (decode)",
        "step 26 (decode)",
    ]
    assert trace["kernel_count"] == len(kernelslist) == 712
    assert trace["has_step_timeline"] and trace["has_real_timing"] and trace["sample"]
    assert [k["kernel_id"] for k in trace["kernels"]] == [
        int(line.split("-")[1]) for line in kernelslist
    ]


def test_sample_kernel_stats_are_kernels_csv_simulated_rows_in_launch_order(app):
    client, _ = app
    kernels = client.get(f"/api/runs/{RUN_ID}/kernels").json()
    rows = simulated_rows()
    assert len(rows) == 93
    assert [(k["kernel_index"], k["kernel_launch_uid"], k["kernel_name"]) for k in kernels] == [
        (i, int(r["uid"]), r["name"]) for i, r in enumerate(rows)
    ]
    assert [k["gpu_sim_cycle"] for k in kernels] == [int(r["cycles"]) for r in rows]
    assert [k["sim_time_ns"] for k in kernels] == [int(r["cycles"]) / 1980 * 1000 for r in rows]
    first = kernels[0]
    assert (first["gpu_sim_insn"], first["gpu_ipc"]) == (2816, 0.5145)


def test_sample_step_timeline_carries_compare_json(app):
    client, _ = app
    [timeline] = client.get(f"/api/runs/{RUN_ID}/step-timelines").json()
    compare = json.loads(bundled("sim/compare.json"))
    assert timeline["gpu_config"] == "SM90_H100"
    assert timeline["compare"]["requests"] == compare["requests"]
    assert [r["e2e_ms"]["error_pct"] for r in timeline["compare"]["requests"]] == [
        r["e2e_ms"]["error_pct"] for r in compare["requests"]
    ]
    assert timeline["simulated"] == json.loads(bundled("sim/simulated_timeline.json"))


def test_sample_files_are_listed_and_download_byte_identical(app):
    client, _ = app
    files = client.get(f"/api/runs/{RUN_ID}/files").json()
    expected = {
        "compare.json": "sim/compare.json",
        "kernel-6207.log": "sim/kernel-6207.log",
        "kernels.csv": "sim/kernels.csv",
        "setup.log": "sim/setup.log",
        "summary.json": "sim/summary.json",
    }
    assert [f["name"] for f in files] == sorted(expected)
    for f in files:
        got = client.get(f"/api/sims/{f['sim_id']}/files/{f['name']}")
        assert got.status_code == 200
        assert got.content == bundled(expected[f["name"]])
        assert f["size"] == len(got.content)


def test_sample_comparison_pairs_each_kernel_with_its_real_duration(app):
    client, _ = app
    comparison = client.get(f"/api/runs/{RUN_ID}/comparison").json()
    rows = simulated_rows()
    assert comparison["has_real_timing"] is True
    assert [r["kernel_name"] for r in comparison["rows"]] == [r["name"] for r in rows]
    assert comparison["rows"][0]["real_ns"] == 928
    assert all(r["real_ns"] is not None and r["sim_ns"]["SM90_H100"] for r in comparison["rows"])
    perfetto = client.get(f"/api/runs/{RUN_ID}/perfetto").json()
    slices = [e for e in perfetto["traceEvents"] if e["ph"] == "X"]
    assert len(slices) == 2 * len(rows)


def test_cancelling_the_sample_run_changes_nothing(app):
    client, _ = app
    before = client.get(f"/api/runs/{RUN_ID}").json()
    after = client.post(f"/api/runs/{RUN_ID}/cancel")
    assert after.status_code == 200
    assert after.json() == before
    assert after.json()["state"] == "succeeded"


def test_sample_trace_refuses_new_runs(app):
    client, _ = app
    body = {"trace_id": TRACE_ID, "entries": [{"gpu_config": "SM90_H100", "count": 1}]}
    resp = client.post("/api/runs", json=body)
    assert resp.status_code == 409
    assert "sample" in resp.json()["detail"]
    assert len(client.get("/api/runs").json()) == 1


def test_ingest_leaves_the_sample_trace_alone(app):
    client, _ = app
    client.app.state.cascade.ingestor.poll_once()
    trace = client.get(f"/api/traces/{TRACE_ID}").json()
    assert (trace["state"], trace["has_step_timeline"]) == ("ready", True)
