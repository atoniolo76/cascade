"""Full loop over real HTTP: uvicorn, LocalLauncher, and `cascade worker --replay` as each sim."""

import csv
import gzip
import json
import os
import shutil
import threading
from pathlib import Path

import httpx
import pytest

from tests.conftest import BABELSTREAM_LOG, OPT_H100_LOG, OPT_H100_STATS
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


def test_replay_worker_succeeds_when_the_app_runs_without_a_token(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    cmd = f"{CASCADE} worker --replay {BABELSTREAM_LOG} --delay 0.02 --heartbeat-s 0.3"
    with running_app(tmp_path, CASCADE_SIM_CMD=cmd, CASCADE_NO_AUTH="1", CASCADE_TOKEN="") as (
        client,
        _,
        settings,
    ):
        write_processed_trace(settings, "trace-rtx", arch="sm_86")
        wait_for(lambda: client.get("/api/traces/trace-rtx").json().get("state") == "ready")
        run = launch(client, "trace-rtx", "SM86_RTX3070")
        done = wait_for(
            lambda: (d := run_detail(client, run["id"]))["state"] != "active" and d, timeout=60
        )
        assert done["state"] == "succeeded", done["sims"][0]["error"]


def test_real_h100_replay_covers_every_mapped_kernel_in_step_timing(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    cmd = f"{CASCADE} worker --replay {OPT_H100_LOG} --delay 0 --heartbeat-s 0.3"
    with running_app(tmp_path, CASCADE_SIM_CMD=cmd) as (client, _, settings):
        write_processed_trace(settings, "trace-opt", arch="sm_90", kernels=20)
        add_step_files(settings, "trace-opt", TIMELINE / "mapping.json")
        wait_for(lambda: client.get("/api/traces/trace-opt").json().get("has_step_timeline"))
        run = finished_run(client, launch(client, "trace-opt", "SM90_H100")["id"], timeout=120)
        assert run["state"] == "succeeded"
        assert run["sims"][0]["kernels_done"] == 20
        (timeline,) = client.get(f"/api/runs/{run['id']}/step-timelines").json()
        coverage = timeline["simulated"]["summary"]["simulation"]["coverage"]
        assert coverage["simulated_kernels"] == 20


OPT_OUTPUTS = OPT_H100_STATS.parent


@pytest.fixture
def opt_run(tmp_path, monkeypatch):
    """Two SM90_H100 replicas replaying Alessio's OPT-125m run and its output files, plus a 2 MiB
    CSV so one output crosses the gzip threshold."""
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    outputs = tmp_path / "outputs"
    shutil.copytree(OPT_OUTPUTS, outputs)
    (outputs / "l2_samples.csv").write_text("".join(f"{i},{i * 7}\n" for i in range(250_000)))
    cmd = (
        f"{CASCADE} worker --replay {OPT_H100_LOG} --replay-outputs {outputs}"
        " --delay 0 --heartbeat-s 0.3"
    )
    with running_app(tmp_path, CASCADE_SIM_CMD=cmd) as (client, _, settings):
        write_processed_trace(settings, "trace-opt", arch="sm_90", kernels=20)
        wait_for(lambda: client.get("/api/traces/trace-opt").json().get("state") == "ready")
        run = finished_run(client, launch(client, "trace-opt", "SM90_H100", 2)["id"], timeout=120)
        assert run["state"] == "succeeded"
        yield client, run, outputs


def test_replay_outputs_are_uploaded_and_download_byte_for_byte(opt_run):
    client, run, outputs = opt_run
    files = client.get(f"/api/runs/{run['id']}/files").json()
    expected = [
        "accelsim.log.gz",
        "exit-status.txt",
        "gpgpu_inst_stats.txt",
        "kernel-stats.csv",
        "l2_samples.csv.gz",
        "opt125m_H100_perfetto.json.gz",
        "perf_counter.csv.gz",
    ]
    sims = [s["id"] for s in run["sims"]]
    assert [(f["sim_id"], f["replica"], f["name"]) for f in files] == [
        (sim, replica, name) for replica, sim in enumerate(sims) for name in expected
    ]
    for f in files:
        got = client.get(f"/api/sims/{f['sim_id']}/files/{f['name']}")
        assert len(got.content) == f["size"]
        if f["name"] == "accelsim.log.gz":
            assert gzip.decompress(got.content).startswith(OPT_H100_LOG.read_bytes())
        elif f["name"] == "l2_samples.csv.gz":
            assert gzip.decompress(got.content) == (outputs / "l2_samples.csv").read_bytes()
        else:
            assert got.content == (outputs / f["name"]).read_bytes(), f["name"]
    gz = next(f for f in files if f["name"] == "perf_counter.csv.gz")
    assert gz["content_type"] == "application/gzip"


def test_perfetto_has_one_slice_per_kernel_laid_end_to_end_in_simulated_time(opt_run):
    client, run, _ = opt_run
    resp = client.get(f"/api/runs/{run['id']}/perfetto")
    assert resp.headers["content-type"] == "application/json"
    trace = json.loads(resp.text)
    events = trace["traceEvents"]
    names = {
        (e["pid"], e.get("tid")): e["args"]["name"]
        for e in events
        if e["ph"] == "M" and e["name"] in ("process_name", "thread_name")
    }
    sims = run["sims"]
    assert names == {
        (1, None): "SM90_H100 · Accel-Sim simulated GPU",
        (1, 1): f"replica 0 · {sims[0]['id']}",
        (1, 2): f"replica 1 · {sims[1]['id']}",
        (2, None): "Real hardware · NVIDIA H100 80GB HBM3",
        (2, 1): "measured kernels",
    }
    with OPT_H100_STATS.open() as f:
        rows = list(csv.DictReader(f))
    expected_us = [int(r["gpu_sim_cycle"]) / 1980 for r in rows]
    for tid in (1, 2):
        slices = [e for e in events if e["ph"] == "X" and (e["pid"], e["tid"]) == (1, tid)]
        assert len(slices) == len(rows) == 20
        assert [s["dur"] for s in slices] == pytest.approx(expected_us)
        starts = [s["ts"] for s in slices]
        ends = [s["ts"] + s["dur"] for s in slices]
        assert starts[0] == 0 and starts[1:] == pytest.approx(ends[:-1])
        assert [s["args"]["uid"] for s in slices] == [int(r["uid"]) for r in rows]
        assert [s["args"]["kernel_name"] for s in slices] == [r["name"] for r in rows]
        assert [s["args"]["kernel_index"] for s in slices] == list(range(20))
        assert [s["args"]["gpu_sim_insn"] for s in slices] == [int(r["gpu_sim_insn"]) for r in rows]
        assert slices[1]["name"] == "k2 vllm::reshape_and_cache_flash_kernel"
        assert slices[-1]["args"]["end_cycle"] == 169_686
    real = [e for e in events if e["ph"] == "X" and e["pid"] == 2]
    assert [r["dur"] for r in real] == [k for k in range(1, 21)]
