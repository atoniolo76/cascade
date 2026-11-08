"""Deploy Cascade on Sail, simulate native A100 traces, collect results and tear down."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import sail

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "sail-partial"
SOURCE = Path("/home/ubuntu/runs/phases/e2e-a100-plan-v1")
APP = "cascade-a100-plan-v1"
NAME = APP + "-main"
ACCELSIM = "/opt/accel-sim-framework"
CONFIG = "SM80_A100"
TRACE_ID = SOURCE.name
spec = importlib.util.spec_from_file_location("hw_sweep", HERE.parent / "hw-sweep/drive_sail.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def main():
    OUT.mkdir(exist_ok=True)
    state = {
        "app_name": APP,
        "created_sailboxes": [],
        "terminated_sailboxes": [],
        "deleted_checkpoints": [],
    }
    box = run = None
    try:
        deployment = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/deploy_sail.py"),
                "--app",
                APP,
                "--name",
                NAME,
                "--base-image",
                "nvidia/cuda:12.9.0-devel-ubuntu22.04",
                "--accelsim-dir",
                ACCELSIM,
                "--disk-gib",
                "64",
                "--sim-cmd",
                "/app/.venv/bin/python /app/scripts/a100_worker.py",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        print(deployment.stdout, flush=True)
        box = sail.Sailbox.list(name=NAME)[0]
        state.update(main_sailbox_id=box.sailbox_id, app_id=box.app_id)
        state["created_sailboxes"].append(box.sailbox_id)
        (OUT / "sail-resources.json").write_text(json.dumps(state, indent=2))
        print("Building pinned Accel-Sim", flush=True)
        box.fs.write("/data/a100-build.sh", sweep.BUILD)
        sweep.command(box, ["bash", "-c", "bash /data/a100-build.sh > /data/a100-build.log 2>&1"])
        print("Accel-Sim build complete", flush=True)
        box.fs.upload_file(HERE / "sail_worker.py", "/app/scripts/a100_worker.py")
        box.fs.write(
            "/app/scripts/a100-settings.json",
            json.dumps({"s3_url": box.listener(9000).endpoint.url.rstrip("/")}),
        )
        remote = f"/data/s3/cascade/accelsim-traces/{TRACE_ID}"
        box.fs.upload_dir(SOURCE / "traced/traces", remote + "/traces")
        for name in ("corrected_timeline.json", "trace_alignment.json"):
            box.fs.upload_file(SOURCE / name, remote + "/" + name)
        box.fs.write(
            remote + "/metadata.json",
            json.dumps(
                {
                    "status": "completed",
                    "gpu": "NVIDIA A100",
                    "trace_gpu_arch": "sm_80",
                    "files": [
                        {"key": "traces/" + p.name, "bytes": p.stat().st_size}
                        for p in (SOURCE / "traced/traces").iterdir()
                        if p.is_file()
                    ],
                }
            ),
        )
        url = box.listener(8000).endpoint.url.rstrip("/")
        state["app_url"] = url
        sweep.command(
            box, ["bash", "-c", "pkill -f '^/app/.venv/bin/python.*cascade serve' || true"]
        )
        box.exec(
            "cd /app && exec cascade serve >> /data/cascade-serve.log 2>&1",
            background=True,
            env={
                "CASCADE_TOKEN": os.environ["CASCADE_TOKEN"],
                "CASCADE_LAUNCHER": "sail",
                "CASCADE_PUBLIC_URL": url,
                "CASCADE_DATA_DIR": "/data",
                "CASCADE_ACCELSIM_DIR": ACCELSIM,
                "SAIL_API_KEY": os.environ["SAIL_API_KEY"],
                "CASCADE_SIM_CMD": "/app/.venv/bin/python /app/scripts/a100_worker.py",
            },
        )
        box.wait_for_listener(8000, timeout=120)
        with httpx.Client(
            base_url=url,
            timeout=120,
            headers={
                "Authorization": "Bearer " + os.environ["CASCADE_TOKEN"],
            },
        ) as client:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                try:
                    configs = client.get("/api/gpu-configs")
                    trace = client.get("/api/traces/" + TRACE_ID)
                    if (
                        configs.status_code == 200
                        and any(c["name"] == CONFIG for c in configs.json())
                        and trace.status_code == 200
                        and trace.json()["state"] == "ready"
                    ):
                        break
                except httpx.TransportError:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError("Main did not register native A100 config/trace")
            response = client.post(
                "/api/runs",
                json={
                    "trace_id": TRACE_ID,
                    "entries": [{"gpu_config": CONFIG, "count": 1}],
                },
            )
            response.raise_for_status()
            run = response.json()
            state["run_id"] = run["id"]
            print("Launched Cascade Sail run", run["id"], flush=True)
            deadline = time.monotonic() + 7200
            while time.monotonic() < deadline:
                response = client.get("/api/runs/" + run["id"])
                response.raise_for_status()
                run = response.json()
                print("Run state", run["state"], run["sim_counts"], flush=True)
                if run["state"] in ("succeeded", "failed", "cancelled", "partial"):
                    break
                time.sleep(30)
            else:
                raise RuntimeError("Simulation exceeded two-hour deadline")
            (OUT / "sail-run.json").write_text(json.dumps(run, indent=2))
            for filename in ("sim_durations.json", "kernels.csv", "summary.json", "log_tails.json"):
                box.fs.download_file(
                    "/data/s3/cascade/accelsim-traces/a100-results/" + filename, OUT / filename
                )
            for relative in (
                "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config",
                "gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config",
            ):
                box.fs.download_file(ACCELSIM + "/" + relative, OUT / Path(relative).name)
            print("A100 artifacts collected", flush=True)
    except subprocess.CalledProcessError as error:
        print(error.stdout[-10000:], error.stderr[-10000:], flush=True)
        raise
    finally:
        if box:
            for remote, name in (
                ("/data/a100-build.log", "sail-build.log"),
                ("/data/cascade-serve.log", "sail-serve.log"),
            ):
                try:
                    box.fs.download_file(remote, OUT / name)
                except Exception as error:
                    print("Log collection:", type(error).__name__, flush=True)
        for resource in sail.Sailbox.list():
            if resource.app_name == APP:
                if resource.sailbox_id not in state["created_sailboxes"]:
                    state["created_sailboxes"].append(resource.sailbox_id)
                resource.terminate()
                state["terminated_sailboxes"].append(resource.sailbox_id)
        if run:
            for checkpoint in sail.SailboxCheckpoint.list(name="cascade-" + run["id"]):
                checkpoint.delete()
                state["deleted_checkpoints"].append(checkpoint.checkpoint_id)
        (OUT / "sail-resources.json").write_text(json.dumps(state, indent=2))
        print("Sail cleanup:", state, flush=True)


if __name__ == "__main__":
    main()
