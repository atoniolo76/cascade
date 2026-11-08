"""`cascade worker` in real mode, with a stub standing in for Accel-Sim's accel-sim.out."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from cascade.launcher import LocalLauncher
from tests.conftest import ACCELSIM_DIR, BABELSTREAM_LOG
from tests.helpers import CASCADE, finished_run, launch, running_app, wait_for
from tests.helpers import write_processed_trace as write_trace

CONFIG = "SM86_RTX3070"

STUB = """\
#!{python}
import json, os, sys, time
from pathlib import Path
here = Path(__file__).parent
Path(here, "calls.json").write_text(json.dumps(
    {{"argv": sys.argv, "cwd": os.getcwd(), "files": sorted(os.listdir()), "pid": os.getpid()}}
))
mode = (here / "mode").read_text() if (here / "mode").exists() else "ok"
if mode == "hang":
    print("GPGPU-Sim: started", flush=True)
    time.sleep(600)
sys.stdout.write(Path({log!r}).read_text())
sys.exit(3 if mode == "crash" else 0)
"""


@pytest.fixture
def accelsim(tmp_path):
    """Real tested-cfgs plus an aux file and the stub binary, laid out like an Accel-Sim build."""
    root = tmp_path / "accel-sim"
    shutil.copytree(ACCELSIM_DIR / "gpu-simulator", root / "gpu-simulator")
    cfg_dir = root / "gpu-simulator/gpgpu-sim/configs/tested-cfgs" / CONFIG
    (cfg_dir / "config_ampere_islip.icnt").write_text("aux\n")
    binary = root / "gpu-simulator/bin/release/accel-sim.out"
    binary.parent.mkdir(parents=True)
    binary.write_text(STUB.format(python=sys.executable, log=str(BABELSTREAM_LOG)))
    binary.chmod(0o755)
    return root


@pytest.fixture
def app(tmp_path, accelsim, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()

    def start(launcher=None):
        return running_app(
            tmp_path,
            launcher,
            CASCADE_ACCELSIM_DIR=str(accelsim),
            CASCADE_SIM_CMD=f"{CASCADE} worker --heartbeat-s 0.3",
        )

    return start


def ready_trace(client, settings, trace_id="trace-rtx"):
    write_trace(settings, trace_id, arch="sm_86")
    wait_for(lambda: client.get(f"/api/traces/{trace_id}").json().get("state") == "ready")


def stub_calls(accelsim):
    return json.loads((accelsim / "gpu-simulator/bin/release/calls.json").read_text())


def test_real_mode_runs_accel_sim_on_the_trace_in_a_fresh_run_dir(app, accelsim, tmp_path):
    with app() as (client, _, settings):
        ready_trace(client, settings)
        run = finished_run(client, launch(client, "trace-rtx", CONFIG)["id"])
    sim = run["sims"][0]
    assert sim["state"] == "succeeded" and sim["exit_code"] == 0 and sim["kernels_done"] == 6
    calls = stub_calls(accelsim)
    trace_dir = settings.trace_dir("trace-rtx")
    assert calls["argv"] == [
        str(accelsim / "gpu-simulator/bin/release/accel-sim.out"),
        "-trace",
        str(trace_dir / "traces/kernelslist.g"),
        "-config",
        str(accelsim / "gpu-simulator/gpgpu-sim/configs/tested-cfgs" / CONFIG / "gpgpusim.config"),
        "-config",
        str(accelsim / "gpu-simulator/configs/tested-cfgs" / CONFIG / "trace.config"),
    ]
    run_dir = tmp_path / "tmp" / "cascade-worker" / sim["id"]
    assert os.path.samefile(calls["cwd"], run_dir)
    assert calls["files"] == [
        "accelsim.log",
        "config_ampere_islip.icnt",
        "gpgpusim.config",
        "trace.config",
    ]
    assert (run_dir / "accelsim.log").read_text().startswith(BABELSTREAM_LOG.read_text())


def test_nonzero_exit_fails_the_sim_with_its_code(app, accelsim):
    (accelsim / "gpu-simulator/bin/release/mode").write_text("crash")
    with app() as (client, _, settings):
        ready_trace(client, settings)
        run = finished_run(client, launch(client, "trace-rtx", CONFIG)["id"])
    sim = run["sims"][0]
    assert (sim["state"], sim["exit_code"], sim["error"]) == ("failed", 3, "accel-sim exited 3")
    assert sim["kernels_done"] == 6


def test_missing_binary_fails_with_a_clear_error(app, accelsim):
    binary = accelsim / "gpu-simulator/bin/release/accel-sim.out"
    binary.unlink()
    with app() as (client, _, settings):
        ready_trace(client, settings)
        run = finished_run(client, launch(client, "trace-rtx", CONFIG)["id"])
    sim = run["sims"][0]
    assert (sim["state"], sim["exit_code"]) == ("failed", 2)
    assert sim["error"] == f"Accel-Sim binary not found at {binary} (check CASCADE_ACCELSIM_DIR)"


def test_missing_trace_config_fails_with_a_clear_error(app, accelsim):
    trace_config = accelsim / "gpu-simulator/configs/tested-cfgs" / CONFIG / "trace.config"
    trace_config.unlink()
    with app() as (client, _, settings):
        ready_trace(client, settings)
        run = finished_run(client, launch(client, "trace-rtx", CONFIG)["id"])
    assert run["sims"][0]["error"] == (
        f"{CONFIG} trace.config not found at {trace_config} (check CASCADE_ACCELSIM_DIR)"
    )


class KeepRunning(LocalLauncher):
    """Never stops a sim itself, so only the worker's cancel ack can."""

    def terminate(self, sim):
        pass


def test_cancel_ack_stops_accel_sim_and_the_worker(app, accelsim):
    (accelsim / "gpu-simulator/bin/release/mode").write_text("hang")
    launcher: list[KeepRunning] = []

    def keep(settings):
        launcher.append(KeepRunning(settings))
        return launcher[0]

    with app(keep) as (client, _, settings):
        ready_trace(client, settings)
        run = launch(client, "trace-rtx", CONFIG)
        wait_for(
            lambda: client.get(f"/api/runs/{run['id']}").json()["sims"][0]["state"] == "running"
        )
        wait_for(lambda: (accelsim / "gpu-simulator/bin/release/calls.json").exists())
        child = stub_calls(accelsim)["pid"]
        client.post(f"/api/runs/{run['id']}/cancel")
        worker = next(iter(launcher[0]._procs.values()))
        assert worker.wait(timeout=15) == 0

    def gone():
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return True
        return False

    wait_for(gone, timeout=15)


def test_worker_without_sim_env_says_what_is_missing(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
    done = subprocess.run([CASCADE, "worker"], env=env, capture_output=True, text=True)
    assert done.returncode == 2
    assert (
        "missing sim env CASCADE_SIM_ID, CASCADE_MAIN_URL, CASCADE_GPU_CONFIG, CASCADE_TRACE_DIR"
        in done.stderr
    )
