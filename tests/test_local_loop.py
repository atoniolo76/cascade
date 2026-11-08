"""Full loop over real HTTP: uvicorn, LocalLauncher, and `cascade worker --replay` as each sim."""

import json
import os
import shutil
import threading
from pathlib import Path

import httpx
import pytest

from tests.conftest import BABELSTREAM_LOG
from tests.helpers import (
    CASCADE,
    TOKEN,
    finished_run,
    launch,
    running_app,
    wait_for,
    write_processed_trace,
)

TIMELINE = Path(__file__).parent / "fixtures" / "timeline"


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    cmd = f"{CASCADE} worker --replay {BABELSTREAM_LOG} --delay 0.02 --heartbeat-s 0.3"
    with running_app(tmp_path, CASCADE_SIM_CMD=cmd) as (client, url, settings):
        write_processed_trace(settings, "trace-rtx", arch="sm_86")
        wait_for(lambda: client.get("/api/traces/trace-rtx").json().get("state") == "ready")
        yield client, url, settings


def collect_events(url: str, run_id: str, events: list, stop: threading.Event) -> None:
    with httpx.stream("GET", f"{url}/api/runs/{run_id}/events?token={TOKEN}", timeout=30) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
            if stop.is_set():
                return


def run_detail(client, run_id):
    return client.get(f"/api/runs/{run_id}").json()


def test_replay_worker_runs_to_success_and_streams_events(server, tmp_path):
    client, url, _ = server
    run = client.post(
        "/api/runs",
        json={"trace_id": "trace-rtx", "entries": [{"gpu_config": "SM86_RTX3070", "count": 2}]},
    ).json()
    events: list = []
    stop = threading.Event()
    sse = threading.Thread(target=collect_events, args=(url, run["id"], events, stop), daemon=True)
    sse.start()
    done = wait_for(
        lambda: (d := run_detail(client, run["id"]))["state"] == "succeeded" and d, timeout=60
    )
    assert all(s["kernels_done"] == 6 and s["exit_code"] == 0 for s in done["sims"])
    assert all(s["launcher_ref"].isdigit() for s in done["sims"])
    for sim in done["sims"]:
        kept = tmp_path / "tmp" / "cascade-worker" / sim["id"] / "accelsim.log"
        assert kept.read_text() == client.get(f"/api/sims/{sim['id']}/log").text
        assert kept.read_text().startswith(BABELSTREAM_LOG.read_text())
    comparison = client.get(f"/api/runs/{run['id']}/comparison").json()
    assert comparison["rows"][5]["sim_ns"]["SM86_RTX3070"] == pytest.approx(45709 / 1132 * 1000)

    wait_for(lambda: any(e["type"] == "sim" and e["sim"]["state"] == "succeeded" for e in events))
    stop.set()
    assert events[0]["type"] == "snapshot" and events[0]["run"]["id"] == run["id"]
    types = {e["type"] for e in events}
    assert {"snapshot", "sim", "log", "kernel"} <= types
    kernel_events = [e["stat"] for e in events if e["type"] == "kernel"]
    expected = sorted((s["id"], i) for s in done["sims"] for i in range(6))
    assert sorted((k["sim_id"], k["kernel_index"]) for k in kernel_events) == expected

    with httpx.stream("GET", f"{url}/api/runs/{run['id']}/events?token={TOKEN}") as r:
        first = json.loads(next(line for line in r.iter_lines() if line.startswith("data: "))[6:])
    assert first["type"] == "snapshot"
    assert sorted((k["sim_id"], k["kernel_index"]) for k in first["kernels"]) == expected
    assert first["kernels"][5]["gpu_sim_cycle"] == 45709
    assert all(k["stats"] == {} for k in first["kernels"])


def test_cancel_stops_the_replay_worker(server):
    client, _, _ = server
    run = client.post(
        "/api/runs",
        json={"trace_id": "trace-rtx", "entries": [{"gpu_config": "SM86_RTX3070", "count": 1}]},
    ).json()
    sim = wait_for(
        lambda: (s := run_detail(client, run["id"])["sims"][0])["state"] == "running" and s
    )
    pid = int(sim["launcher_ref"])
    cancelled = client.post(f"/api/runs/{run['id']}/cancel").json()
    assert cancelled["state"] == "cancelled"
    assert cancelled["sims"][0]["state"] == "cancelled"

    def gone():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False

    wait_for(gone, timeout=15)
    assert run_detail(client, run["id"])["sims"][0]["kernels_done"] < 6


def add_step_files(settings, trace_id: str, mapping: Path) -> None:
    run_dir = settings.trace_dir(trace_id)
    shutil.copy(TIMELINE / "corrected_timeline.json", run_dir / "corrected_timeline.json")
    shutil.copy(mapping, run_dir / "kernel_mapping.json")


def test_step_timing_is_posted_after_a_successful_sim(server):
    client, _, settings = server
    assert client.get("/api/traces/trace-rtx").json()["has_step_timeline"] is False
    add_step_files(settings, "trace-rtx", TIMELINE / "babelstream_kernel_mapping.json")
    wait_for(lambda: client.get("/api/traces/trace-rtx").json()["has_step_timeline"])

    run = finished_run(client, launch(client, "trace-rtx", "SM86_RTX3070", 2)["id"])
    assert run["state"] == "succeeded"
    timelines = client.get(f"/api/runs/{run['id']}/step-timelines").json()
    assert [(t["gpu_config"], t["replica"]) for t in timelines] == [
        ("SM86_RTX3070", 0),
        ("SM86_RTX3070", 1),
    ]
    simulated, report = timelines[0]["simulated"], timelines[0]["compare"]
    coverage = simulated["summary"]["simulation"]["coverage"]
    assert (coverage["simulated_kernels"], coverage["kernels"]) == (6, 683)
    assert set(simulated["summary"]["per_kind"]) == {"prefill", "decode"}
    assert [s["kind"] for s in report["steps"]] == ["prefill", "decode", "decode", "decode"]
    prefill = next(s for s in simulated["steps"] if not s["warmup"] and s["step"] == 0)
    triad = next(k for k in prefill["kernels"] if k["seq"] == 660)
    assert triad["duration_ns"] == pytest.approx(6379 / 1132 * 1000)
    log = client.get(f"/api/sims/{timelines[0]['sim_id']}/log").text
    assert "step timing posted, 6/683 kernels simulated" in log


def test_step_timing_failure_leaves_the_sim_succeeded(server):
    client, _, settings = server
    add_step_files(settings, "trace-rtx", TIMELINE / "mapping.json")
    wait_for(lambda: client.get("/api/traces/trace-rtx").json()["has_step_timeline"])
    run = finished_run(client, launch(client, "trace-rtx", "SM86_RTX3070")["id"])
    assert run["state"] == "succeeded" and run["sims"][0]["exit_code"] == 0
    assert client.get(f"/api/runs/{run['id']}/step-timelines").json() == []
    log = client.get(f"/api/sims/{run['sims'][0]['id']}/log").text
    assert "step timing failed: ValueError: kernel UID/name mapping mismatch" in log


def test_no_step_files_means_one_log_line_and_no_upload(server):
    client, _, _ = server
    run = finished_run(client, launch(client, "trace-rtx", "SM86_RTX3070")["id"])
    assert run["state"] == "succeeded"
    assert client.get(f"/api/runs/{run['id']}/step-timelines").json() == []
    log = client.get(f"/api/sims/{run['sims'][0]['id']}/log").text
    assert [line for line in log.splitlines() if "step timing" in line] == [
        "cascade worker: no corrected_timeline.json + kernel_mapping.json; no step timing"
    ]
