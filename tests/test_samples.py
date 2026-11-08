"""The bundled sample (samples/opt125m-h100-e2e) that every workspace starts with."""

import csv
import gzip
import io
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from cascade.api import create_app
from cascade.samples import SAMPLES_ROOT, Bundle, _load_sim, load, seed
from cascade.store import Store
from tests.helpers import AUTH, make_settings

BUNDLE = SAMPLES_ROOT / "opt125m-h100-e2e"
TRACE_ID = "sample__facebook_opt-125m__h100-e2e"
RUN_ID = "run-sample-opt125m-h100-e2e"
SIM_ID = "sim-sample-opt125m-h100-e2e"
H200_ID = f"{SIM_ID}-sm90_h200"


def available_bundles():
    return [
        m
        for p in SAMPLES_ROOT.glob("*/sample.json")
        if (m := json.loads(p.read_text())).get("status") != "pending"
    ]


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
    assert {(t["id"], t["sample"]) for t in traces} == {
        (m["trace_id"], True) for m in available_bundles()
    }
    assert {(r["id"], r["sample"], r["state"]) for r in runs} == {
        (m["run_id"], True, "succeeded") for m in available_bundles()
    }
    assert len(kernels) == len(simulated_rows()) + 31


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
    assert trace["real_gpu_config"] == "SM90_H100"
    assert trace["sample_sim_configs"] == ["SM90_H100", "SM90_H200"]
    assert trace["sample_set"] == "cascade-samples"
    assert trace["metadata"]["sample_pending_configs"] == ["SM80_A100"]
    assert [k["kernel_id"] for k in trace["kernels"]] == [
        int(line.split("-")[1]) for line in kernelslist
    ]
    assert trace["kernels"][0] == {
        "kernel_id": 6207,
        "name": simulated_rows()[0]["name"],
        "grid": [1, 1, 1],
        "block": [128, 1, 1],
        "instructions": 92,
    }


def test_sample_kernel_stats_are_kernels_csv_simulated_rows_in_launch_order(app):
    client, _ = app
    kernels = [k for k in client.get(f"/api/runs/{RUN_ID}/kernels").json() if k["sim_id"] == SIM_ID]
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
    timelines = client.get(f"/api/runs/{RUN_ID}/step-timelines").json()
    assert {t["gpu_config"] for t in timelines} == {"SM90_H100", "SM90_H200"}
    timeline = next(t for t in timelines if t["gpu_config"] == "SM90_H100")
    compare = json.loads(bundled("sim/compare.json"))
    assert timeline["gpu_config"] == "SM90_H100"
    assert timeline["compare"]["requests"] == compare["requests"]
    assert [r["e2e_ms"]["error_pct"] for r in timeline["compare"]["requests"]] == [
        r["e2e_ms"]["error_pct"] for r in compare["requests"]
    ]
    assert timeline["simulated"] == json.loads(bundled("sim/simulated_timeline.json"))


def test_sample_files_are_listed_and_download_byte_identical(app):
    client, _ = app
    files = [f for f in client.get(f"/api/runs/{RUN_ID}/files").json() if f["sim_id"] == SIM_ID]
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
    assert len(slices) == 2 * len(rows) + 31


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
    assert len(client.get("/api/runs").json()) == len(available_bundles())


def test_h200_uses_recorded_sweep_cycles_without_inventing_log_counters(app):
    client, _ = app
    rows = list(csv.DictReader(io.StringIO(bundled("sim/SM90_H200/kernels.csv").decode())))
    exact = {int(r["uid"]): r for r in rows if r["sim"] == "exact"}
    kernels = [
        k for k in client.get(f"/api/runs/{RUN_ID}/kernels").json() if k["sim_id"] == H200_ID
    ]
    reference = {int(r["uid"]): i for i, r in enumerate(simulated_rows())}
    assert len(kernels) == len(exact) == 31
    for k in kernels:
        uid = k["kernel_launch_uid"]
        assert k["kernel_index"] == reference[uid]
        assert k["kernel_name"] == exact[uid]["name"]
        assert k["gpu_sim_cycle"] == int(exact[uid]["cycles"])
        assert k["gpu_sim_insn"] is None and k["gpu_ipc"] is None
    timeline = next(
        t
        for t in client.get(f"/api/runs/{RUN_ID}/step-timelines").json()
        if t["gpu_config"] == "SM90_H200"
    )
    assert timeline["compare"] == json.loads(bundled("sim/SM90_H200/compare.json"))
    assert timeline["simulated"] == json.loads(bundled("sim/SM90_H200/simulated_timeline.json"))
    means = [r["e2e_ms"]["simulated"] for r in timeline["compare"]["requests"]]
    assert sum(means) / len(means) == pytest.approx(262.161664, abs=0.000001)
    run = client.get(f"/api/runs/{RUN_ID}").json()
    assert [s["gpu_config"] for s in run["sims"]] == ["SM90_H100", "SM90_H200"]
    assert next(s for s in run["sims"] if s["id"] == H200_ID)["started_at"] is None


def test_pending_bundle_is_skipped_without_reading_nonexistent_payload(tmp_path):
    root = tmp_path / "pending"
    root.mkdir()
    (root / "sample.json").write_text(json.dumps({"status": "pending"}))
    settings = make_settings(tmp_path)
    store = Store(settings.db_path)
    try:
        assert seed(settings, store, root) is False
        assert store.sample_trace_ids() == set()
    finally:
        store.close()


def test_multiple_sample_sims_are_inserted_atomically(tmp_path):
    rows = load(Bundle(BUNDLE)).rows
    duplicate = replace(
        rows.sims[1], sim=rows.sims[1].sim.model_copy(update={"id": rows.sims[0].sim.id})
    )
    broken = replace(rows, sims=[rows.sims[0], duplicate])
    store = Store(make_settings(tmp_path).db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store.add_sample(broken)
        assert store.get_trace_summary(TRACE_ID) is None
        assert store.get_run(RUN_ID) is None
    finally:
        store.close()


def test_sample_rejects_empty_configs():
    bundle = Bundle(BUNDLE)
    bundle.manifest["configs"] = []
    with pytest.raises(ValueError, match="at least one simulation"):
        load(bundle)


@pytest.mark.parametrize("kind", ["failed", "timeout"])
def test_failed_simulation_cannot_be_seeded_as_a_successful_sample(tmp_path, kind):
    (tmp_path / "sample.json").write_text(json.dumps({"gpu_config": "SM80_A100"}))
    (tmp_path / "sim").mkdir()
    (tmp_path / "sim/summary.json").write_text(json.dumps({"counts": {kind: 1}}))
    with pytest.raises(ValueError, match="failed or timed-out kernels"):
        _load_sim(Bundle(tmp_path), datetime.now(UTC), {})


def test_native_a100_has_its_own_baseline_and_real_simulator_results(app):
    client, _ = app
    root = SAMPLES_ROOT / "opt125m-a100-e2e"
    manifest = json.loads((root / "sample.json").read_text())
    trace = client.get(f"/api/traces/{manifest['trace_id']}").json()
    assert (trace["real_gpu_config"], trace["trace_gpu_arch"]) == ("SM80_A100", "sm_80")
    assert trace["sample_sim_configs"] == ["SM80_A100"]
    assert trace["sample_set"] == "cascade-samples"
    assert trace["kernel_count"] == 624
    run = client.get(f"/api/runs/{manifest['run_id']}").json()
    assert run["sample"] and run["state"] == "succeeded"
    assert len(run["sims"]) == 1 and run["sims"][0]["kernels_done"] == 33
    [timeline] = client.get(f"/api/runs/{manifest['run_id']}/step-timelines").json()
    assert timeline["gpu_config"] == "SM80_A100"
    assert timeline["compare"] == json.loads(
        gzip.decompress((root / "sim/compare.json.gz").read_bytes())
    )
    requests = timeline["compare"]["requests"]
    for metric, real, simulated in [
        ("e2e_ms", 222.08311833333502, 229.58006488662718),
        ("ttft_ms", 7.817559666662532, 8.28083233333842),
        ("itl_ms_mean", 6.911792215053951, 7.144044437199725),
    ]:
        assert sum(r[metric]["real"] for r in requests) / len(requests) == pytest.approx(real)
        assert sum(r[metric]["simulated"] for r in requests) / len(requests) == pytest.approx(
            simulated
        )


def test_ingest_leaves_the_sample_trace_alone(app):
    client, _ = app
    client.app.state.cascade.ingestor.poll_once()
    trace = client.get(f"/api/traces/{TRACE_ID}").json()
    assert (trace["state"], trace["has_step_timeline"]) == ("ready", True)
