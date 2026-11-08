import threading
import time

import pytest
from fastapi.testclient import TestClient

from cascade.api import create_app
from cascade.models import RunEntry, SimState, TraceState
from cascade.store import Store
from tests.conftest import BABELSTREAM_LOG
from tests.helpers import AUTH, make_settings, wait_for, write_processed_trace

BABELSTREAM_CYCLES = [6174, 5923, 6216, 6312, 6379, 45709]


class RecordingLauncher:
    def __init__(self):
        self.launched: list[str] = []
        self.terminated: list[str] = []
        self.prepared: list[str] = []
        self.lock = threading.Lock()

    def prepare(self, run_id):
        self.prepared.append(run_id)

    def launch(self, sim, trace_dir):
        assert (trace_dir / "traces" / "kernelslist.g").is_file()
        with self.lock:
            self.launched.append(sim.id)
        return f"fake-{sim.id}"

    def terminate(self, sim):
        # Like the real launchers, nothing can be stopped without a launcher_ref.
        if sim.launcher_ref:
            with self.lock:
                self.terminated.append(sim.id)


@pytest.fixture
def setup(tmp_path):
    settings = make_settings(tmp_path)
    write_processed_trace(settings, "trace-h100", arch="sm_90")
    launcher = RecordingLauncher()
    with TestClient(create_app(settings, launcher)) as client:
        wait_for(
            lambda: (
                client.get("/api/traces/trace-h100", headers=AUTH).json().get("state") == "ready"
            )
        )
        yield client, launcher, settings


def test_auth(setup):
    client, _, _ = setup
    assert client.get("/api/health").json() == {"ok": True}
    assert client.get("/api/traces").status_code == 401
    assert client.get("/api/traces", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/runs/x/events?token=nope").status_code == 401
    assert client.get("/api/nope", headers=AUTH).status_code == 404


def test_run_validation(setup, tmp_path):
    client, _, settings = setup
    post = lambda body: client.post("/api/runs", json=body, headers=AUTH)  # noqa: E731
    unknown = post(
        {"trace_id": "trace-h100", "entries": [{"gpu_config": "SM2_GTX480", "count": 1}]}
    )
    assert unknown.status_code == 422
    wrong_arch = post(
        {"trace_id": "trace-h100", "entries": [{"gpu_config": "SM80_A100", "count": 1}]}
    )
    assert wrong_arch.status_code == 422 and "sm_80" in wrong_arch.json()["detail"]
    assert post({"trace_id": "trace-h100", "entries": []}).status_code == 422
    assert (
        post(
            {"trace_id": "missing", "entries": [{"gpu_config": "SM90_H100", "count": 1}]}
        ).status_code
        == 404
    )
    (settings.trace_dir("half") / "traces").mkdir(parents=True)
    (settings.trace_dir("half") / "traces" / "kernel-1-ctx_0x1.trace.xz").write_bytes(b"x")
    wait_for(lambda: client.get("/api/traces/half", headers=AUTH).status_code == 200)
    busy = post({"trace_id": "half", "entries": [{"gpu_config": "SM90_H100", "count": 1}]})
    assert busy.status_code == 409
    assert client.get("/api/runs", headers=AUTH).json() == []


def test_run_lifecycle(setup):
    client, launcher, _ = setup
    created = client.post(
        "/api/runs",
        json={
            "trace_id": "trace-h100",
            "entries": [
                {"gpu_config": "SM90_H100", "count": 2},
                {"gpu_config": "SM90_H200", "count": 1},
            ],
        },
        headers=AUTH,
    )
    assert created.status_code == 201
    run = created.json()
    run_id = run["id"]
    assert run["trace"]["id"] == "trace-h100" and len(run["sims"]) == 3
    assert {s["kernels_total"] for s in run["sims"]} == {6}
    wait_for(lambda: len(launcher.launched) == 3)
    get_run = lambda: client.get(f"/api/runs/{run_id}", headers=AUTH).json()  # noqa: E731
    wait_for(lambda: all(s["launcher_ref"] for s in get_run()["sims"]))
    sims = {(s["gpu_config"], s["replica"]): s["id"] for s in get_run()["sims"]}
    assert all(s["state"] == "launching" for s in get_run()["sims"])

    lines = BABELSTREAM_LOG.read_text().splitlines()
    chunks = [lines[i : i + 500] for i in range(0, len(lines), 500)]
    for key in [("SM90_H100", 0), ("SM90_H100", 1)]:
        sim_id = sims[key]
        ack = client.post(f"/api/sims/{sim_id}/heartbeat", json={"state": "running"}, headers=AUTH)
        assert ack.json() == {"state": "running", "cancel": False}
        for seq, chunk in enumerate(chunks):
            r = client.post(
                f"/api/sims/{sim_id}/logs", json={"seq": seq, "lines": chunk}, headers=AUTH
            )
            assert r.status_code == 204
            if seq == 3:
                retry = client.post(
                    f"/api/sims/{sim_id}/logs", json={"seq": 2, "lines": chunks[2]}, headers=AUTH
                )
                assert retry.status_code == 204
        live = client.get(f"/api/runs/{run_id}", headers=AUTH).json()
        sim = next(s for s in live["sims"] if s["id"] == sim_id)
        assert sim["kernels_done"] == 6
        assert sim["current_kernel"] == "_Z10dot_kernelIdEvPKT_S2_PS0_i"
        assert sim["latest"]["gpu_tot_sim_cycle"] == 76713
        assert sim["log_tail"] == lines[-20:]
        done = client.post(
            f"/api/sims/{sim_id}/heartbeat",
            json={"state": "succeeded", "exit_code": 0},
            headers=AUTH,
        )
        assert done.json() == {"state": "succeeded", "cancel": False}

    h100 = sims[("SM90_H100", 0)]
    results = client.get(f"/api/sims/{h100}/results", headers=AUTH)
    assert "attachment" in results.headers["content-disposition"]
    body = results.json()
    assert [k["gpu_sim_cycle"] for k in body["kernels"]] == BABELSTREAM_CYCLES
    assert body["kernels"][0]["sim_time_ns"] == pytest.approx(6174 / 1980 * 1000)
    assert body["gpu_config"] == {"name": "SM90_H100", "arch": "sm_90", "core_clock_mhz": 1980.0}
    assert body["sim"]["exit_code"] == 0
    log = client.get(f"/api/sims/{h100}/log", headers=AUTH).text
    assert log == "\n".join(lines) + "\n"

    comparison = client.get(f"/api/runs/{run_id}/comparison", headers=AUTH).json()
    assert comparison["has_real_timing"] and comparison["configs"] == ["SM90_H100", "SM90_H200"]
    assert [r["real_ns"] for r in comparison["rows"]] == [1000.0 * k for k in range(1, 7)]
    assert [r["kernel_name"] for r in comparison["rows"]] == [f"k{k}" for k in range(1, 7)]
    assert [r["sim_ns"]["SM90_H100"] for r in comparison["rows"]] == pytest.approx(
        [c / 1980 * 1000 for c in BABELSTREAM_CYCLES]
    )
    assert all(r["sim_ns"]["SM90_H200"] is None for r in comparison["rows"])

    h200 = sims[("SM90_H200", 0)]
    cancelled = client.post(f"/api/runs/{run_id}/cancel", headers=AUTH).json()
    states = {s["id"]: s["state"] for s in cancelled["sims"]}
    assert states == {
        sims[("SM90_H100", 0)]: "succeeded",
        sims[("SM90_H100", 1)]: "succeeded",
        h200: "cancelled",
    }
    assert cancelled["state"] == "partial"
    assert (
        client.post(f"/api/runs/{run_id}/cancel", headers=AUTH).json()["sims"] == cancelled["sims"]
    )
    ack = client.post(f"/api/sims/{h200}/heartbeat", json={"state": "running"}, headers=AUTH).json()
    assert ack == {"state": "cancelled", "cancel": True}
    wait_for(lambda: set(launcher.terminated) == set(sims.values()))

    run_results = client.get(f"/api/runs/{run_id}/results", headers=AUTH).json()
    assert len(run_results["sims"]) == 3 and run_results["comparison"] == comparison
    assert [k["kernel_id"] for k in run_results["trace"]["kernels"]] == []
    summary = client.get("/api/runs", headers=AUTH).json()
    assert summary[0]["sim_counts"] == {"succeeded": 2, "cancelled": 1}


def test_stale_sim_goes_lost_then_revives_or_is_given_up(setup):
    client, launcher, _ = setup
    run = client.post(
        "/api/runs",
        json={"trace_id": "trace-h100", "entries": [{"gpu_config": "SM90_H100", "count": 2}]},
        headers=AUTH,
    ).json()
    revived, abandoned = (s["id"] for s in run["sims"])
    cascade = client.app.state.cascade
    wait_for(lambda: all(cascade.store.get_sim(i).launcher_ref for i in (revived, abandoned)))
    object.__setattr__(cascade.settings, "heartbeat_timeout_s", 0.1)
    time.sleep(0.3)
    cascade.sweep()
    assert {cascade.store.get_sim(i).state for i in (revived, abandoned)} == {SimState.lost}
    assert cascade.store.get_sim(revived).error == "no heartbeat"

    ack = client.post(f"/api/sims/{revived}/heartbeat", json={"state": "running"}, headers=AUTH)
    assert ack.json() == {"state": "running", "cancel": False}
    assert cascade.store.get_sim(revived).error is None

    time.sleep(1.0)
    client.post(f"/api/sims/{revived}/heartbeat", json={"state": "running"}, headers=AUTH)
    cascade.sweep()
    assert cascade.store.get_sim(revived).state == SimState.running
    gone = cascade.store.get_sim(abandoned)
    assert (gone.state, gone.error) == (SimState.failed, "no heartbeat for 1s")
    wait_for(lambda: abandoned in launcher.terminated)
    assert revived not in launcher.terminated


def test_cancel_during_launch_terminates_once_the_ref_exists(setup):
    client, launcher, _ = setup
    gate = threading.Event()
    real_launch = launcher.launch

    def slow_launch(sim, trace_dir):
        gate.wait(5)
        return real_launch(sim, trace_dir)

    launcher.launch = slow_launch
    run = client.post(
        "/api/runs",
        json={"trace_id": "trace-h100", "entries": [{"gpu_config": "SM90_H100", "count": 1}]},
        headers=AUTH,
    ).json()
    sim_id = run["sims"][0]["id"]
    cascade = client.app.state.cascade
    wait_for(lambda: cascade.store.get_sim(sim_id).state == SimState.launching)
    client.post(f"/api/runs/{run['id']}/cancel", headers=AUTH)
    gate.set()
    wait_for(lambda: cascade.store.get_sim(sim_id).launcher_ref == f"fake-{sim_id}")
    wait_for(lambda: sim_id in launcher.terminated)
    assert cascade.store.get_sim(sim_id).state == SimState.cancelled


def test_launch_error_fails_sim(tmp_path):
    settings = make_settings(tmp_path)
    write_processed_trace(settings, "t", arch="sm_90")
    with TestClient(create_app(settings)) as client:
        wait_for(lambda: client.get("/api/traces/t", headers=AUTH).json().get("state") == "ready")
        run = client.post(
            "/api/runs",
            json={"trace_id": "t", "entries": [{"gpu_config": "SM90_H100", "count": 1}]},
            headers=AUTH,
        ).json()
        sim = wait_for(
            lambda: (
                (s := client.get(f"/api/runs/{run['id']}", headers=AUTH).json()["sims"][0])["state"]
                == "failed"
                and s
            )
        )
        assert "CASCADE_SIM_CMD is not set" in sim["error"]


def test_restart_launches_sims_left_pending(tmp_path):
    settings = make_settings(tmp_path)
    write_processed_trace(settings, "t", arch="sm_90")
    store = Store(settings.db_path)
    store.register_trace("t", 6, 1)
    store.update_trace("t", state=TraceState.processing)
    store.update_trace("t", state=TraceState.ready)
    run_id = store.create_run("t", [RunEntry(gpu_config="SM90_H100", count=2)], 6)
    store.close()
    launcher = RecordingLauncher()
    with TestClient(create_app(settings, launcher)) as client:
        wait_for(lambda: len(launcher.launched) == 2)
        sims = client.get(f"/api/runs/{run_id}", headers=AUTH).json()["sims"]
        wait_for(
            lambda: all(
                s["state"] == "launching"
                for s in client.get(f"/api/runs/{run_id}", headers=AUTH).json()["sims"]
            )
        )
        assert sorted(launcher.launched) == sorted(s["id"] for s in sims)
        assert launcher.prepared == [run_id]


def test_serves_built_ui_with_spa_fallback(tmp_path, monkeypatch):
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<div id=root></div>")
    (static / "assets" / "app.js").write_text("console.log(1)")
    monkeypatch.setattr("cascade.api.STATIC_DIR", static)
    with TestClient(create_app(make_settings(tmp_path), RecordingLauncher())) as client:
        assert client.get("/").text == "<div id=root></div>"
        assert client.get("/runs/run-123").text == "<div id=root></div>"
        assert client.get("/assets/app.js").text == "console.log(1)"
        assert client.get("/../../etc/passwd").text == "<div id=root></div>"
        assert client.get("/api/unknown", headers=AUTH).status_code == 404
