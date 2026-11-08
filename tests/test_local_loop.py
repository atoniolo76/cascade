"""Full loop over real HTTP: uvicorn, LocalLauncher, and scripts/fake_sim.py as the simulator."""

import json
import os
import socket
import sys
import threading
from pathlib import Path

import httpx
import pytest
import uvicorn

from cascade.api import create_app
from tests.conftest import BABELSTREAM_LOG
from tests.helpers import AUTH, TOKEN, make_settings, wait_for, write_processed_trace

FAKE_SIM = Path(__file__).parent.parent / "scripts" / "fake_sim.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    settings = make_settings(
        tmp_path,
        CASCADE_PUBLIC_URL=url,
        CASCADE_SIM_CMD=f"{sys.executable} {FAKE_SIM} {BABELSTREAM_LOG} --delay 0.02 "
        "--heartbeat-s 0.3",
    )
    write_processed_trace(settings, "trace-rtx", arch="sm_86")
    server = uvicorn.Server(uvicorn.Config(create_app(settings), port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_for(lambda: server.started)
    with httpx.Client(base_url=url, headers=AUTH, timeout=10) as client:
        wait_for(lambda: client.get("/api/traces/trace-rtx").json().get("state") == "ready")
        yield client, url
    server.should_exit = True
    thread.join(timeout=10)


def collect_events(url: str, run_id: str, events: list, stop: threading.Event) -> None:
    with httpx.stream("GET", f"{url}/api/runs/{run_id}/events?token={TOKEN}", timeout=30) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
            if stop.is_set():
                return


def run_detail(client, run_id):
    return client.get(f"/api/runs/{run_id}").json()


def test_fake_sim_runs_to_success_and_streams_events(server):
    client, url = server
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
    comparison = client.get(f"/api/runs/{run['id']}/comparison").json()
    assert comparison["rows"][5]["sim_ns"]["SM86_RTX3070"] == pytest.approx(45709 / 1132 * 1000)

    wait_for(lambda: any(e["type"] == "sim" and e["sim"]["state"] == "succeeded" for e in events))
    stop.set()
    assert events[0]["type"] == "snapshot" and events[0]["run"]["id"] == run["id"]
    types = {e["type"] for e in events}
    assert {"snapshot", "sim", "log", "kernel"} <= types
    kernel_events = [e["stat"] for e in events if e["type"] == "kernel"]
    assert sorted((k["sim_id"], k["kernel_index"]) for k in kernel_events) == sorted(
        (s["id"], i) for s in done["sims"] for i in range(6)
    )


def test_cancel_stops_the_fake_sim(server):
    client, _ = server
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
