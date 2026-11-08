"""Deploy the repository app, ingest existing traces, launch the sweep, collect and clean up."""

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
SOURCE = Path("/home/ubuntu/runs/phases/e2e-plan-v1")
NAME = "cascade-hw-sweep-main"
APP = "cascade-hw-sweep"
ACCELSIM = "/opt/accel-sim-framework"

BUILD = r"""
set -e
export DEBIAN_FRONTEND=noninteractive CUDA_INSTALL_PATH=/usr/local/cuda
export GPGPUSIM_BRANCH=4195dd4f229cbf03f4f38c49139ae94d01d23f03
export CPLUS_INCLUDE_PATH=/opt/accel-sim-framework/util/tracer_nvbit/tracer_tool
apt-get update
apt-get install -y git wget build-essential xutils-dev bison flex zlib1g-dev \
    libglu1-mesa-dev libssl-dev libxml2-dev libboost-all-dev libzstd-dev \
    python3-dev python3-setuptools python3-pip bc cmake
/usr/local/bin/uv pip install --python /app/.venv/bin/python pyyaml plotly psutil
git clone https://github.com/accel-sim/accel-sim-framework.git /opt/accel-sim-framework
git -C /opt/accel-sim-framework checkout 7ba54ded60c4ba0083410c37d22d719166acd6dc
sed -i 's|-lzstd -pthread $(BUILD_DIR)/[*].o|-pthread $(BUILD_DIR)/*.o -lzstd|' \
    /opt/accel-sim-framework/gpu-simulator/Makefile
source /opt/accel-sim-framework/gpu-simulator/setup_environment.sh release
make -j8 -C /opt/accel-sim-framework/gpu-simulator
make -C /opt/accel-sim-framework/util/tracer_nvbit/tracer_tool/traces-processing
test -x /opt/accel-sim-framework/gpu-simulator/bin/release/accel-sim.out
"""


def command(box, args, timeout=3600):
    result = box.exec(args, timeout=timeout).wait()
    if result.exit_code:
        print(result.stdout[-12000:], result.stderr[-12000:], flush=True)
        raise RuntimeError(f"remote command exited {result.exit_code}")
    return result.stdout


def main():
    state = {
        "app_name": APP,
        "created_sailboxes": [],
        "terminated_sailboxes": [],
        "deleted_checkpoints": [],
    }
    run = None
    box = None
    try:
        deployed = subprocess.run(
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
                "/app/.venv/bin/python /app/scripts/hw_sweep_worker.py",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        print(deployed.stdout, flush=True)
        box = sail.Sailbox.list(name=NAME)[0]
        state["main_sailbox_id"] = box.sailbox_id
        state["app_id"] = box.app_id
        state["created_sailboxes"].append(box.sailbox_id)
        (HERE / "sail-resources.json").write_text(json.dumps(state, indent=2))
        print("Building pinned Accel-Sim in main sailbox", box.sailbox_id, flush=True)
        command(box, ["bash", "-c", "ls /data && ls /app && mkdir -p /data/hw-sweep"])
        box.fs.write("/data/hw-sweep/build.sh", BUILD)
        command(box, ["bash", "-c", "bash /data/hw-sweep/build.sh > /data/hw-sweep/build.log 2>&1"])
        print("Accel-Sim build complete", flush=True)
        box.fs.download_file("/data/hw-sweep/build.log", HERE / "sail-build.log")
        box.fs.upload_file(HERE / "sail_worker.py", "/app/scripts/hw_sweep_worker.py")
        box.fs.write(
            "/app/scripts/hw-sweep-settings.json",
            json.dumps({"s3_url": box.listener(9000).endpoint.url.rstrip("/")}),
        )
        trace_id = "e2e-plan-v1"
        remote = f"/data/s3/cascade/accelsim-traces/{trace_id}"
        print("Uploading existing traces", flush=True)
        box.fs.upload_dir(SOURCE / "traced/traces", remote + "/traces")
        for name in ("corrected_timeline.json", "trace_alignment.json"):
            box.fs.upload_file(SOURCE / name, remote + "/" + name)
        files = [
            {"key": "traces/" + p.name, "bytes": p.stat().st_size}
            for p in (SOURCE / "traced/traces").iterdir()
            if p.is_file()
        ]
        meta = {
            "status": "completed",
            "gpu": "NVIDIA H100",
            "trace_gpu_arch": "sm_90",
            "files": files,
        }
        box.fs.write(remote + "/metadata.json", json.dumps(meta))
        url = box.listener(8000).endpoint.url.rstrip("/")
        state["app_url"] = url
        with httpx.Client(
            base_url=url,
            headers={"Authorization": "Bearer " + os.environ["CASCADE_TOKEN"]},
            timeout=120,
        ) as client:
            deadline = time.monotonic() + 180
            while True:
                response = client.get("/api/traces/" + trace_id)
                if response.status_code == 200 and response.json()["state"] == "ready":
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("trace ingest not ready: " + response.text)
                time.sleep(2)
            # Main registered configs before the Accel-Sim build; restart before submitting.
            command(box, ["bash", "-c", "pkill -f '^/app/.venv/bin/python.*cascade serve' || true"])
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
                    "CASCADE_SIM_CMD": "/app/.venv/bin/python /app/scripts/hw_sweep_worker.py",
                },
            )
            box.wait_for_listener(8000, timeout=120)
            deadline = time.monotonic() + 120
            while True:
                try:
                    response = client.get("/api/gpu-configs")
                    if response.status_code == 200 and any(
                        cfg["name"] == "SM90_H100" for cfg in response.json()
                    ):
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("restarted main did not load the built GPU configs")
                time.sleep(1)
            rejected = []
            response = client.get("/api/gpu-configs")
            response.raise_for_status()
            for config in response.json():
                if config["arch"] == "sm_90":
                    continue
                response = client.post(
                    "/api/runs",
                    json={
                        "trace_id": trace_id,
                        "entries": [{"gpu_config": config["name"], "count": 1}],
                    },
                )
                rejected.append(
                    {
                        "config": config,
                        "http_status": response.status_code,
                        "response": response.json(),
                    }
                )
                if response.status_code != 422:
                    raise RuntimeError("expected architecture rejection: " + response.text)
            (HERE / "sail-incompatible-configs.json").write_text(json.dumps(rejected, indent=2))
            response = client.post(
                "/api/runs",
                json={
                    "trace_id": trace_id,
                    "entries": [
                        {"gpu_config": cfg, "count": 1} for cfg in ("SM90_H100", "SM90_H200")
                    ],
                },
            )
            response.raise_for_status()
            run = response.json()
            state["run_id"] = run["id"]
            print("Launched Cascade Sail run", run["id"], flush=True)
            deadline = time.monotonic() + 3600
            while True:
                response = client.get("/api/runs/" + run["id"])
                response.raise_for_status()
                run = response.json()
                print("Run state", run["state"], run["sim_counts"], flush=True)
                if run["state"] in ("succeeded", "failed", "cancelled", "partial"):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("run exceeded one-hour deadline")
                time.sleep(30)
            (HERE / "sail-run.json").write_text(json.dumps(run, indent=2))
            for sim in run["sims"]:
                if not sim["launcher_ref"]:
                    continue
                cfg = sim["gpu_config"]
                dest = HERE / cfg
                dest.mkdir(exist_ok=True)
                for filename in (
                    "sim_durations.json",
                    "kernels.csv",
                    "summary.json",
                    "log_tails.json",
                ):
                    box.fs.download_file(
                        f"/data/s3/cascade/accelsim-traces/hw-sweep-results/{cfg}/{filename}",
                        dest / filename,
                    )
                print(cfg, "collected", flush=True)
    except subprocess.CalledProcessError as error:
        print(error.stdout[-10000:], error.stderr[-10000:], flush=True)
        raise
    finally:
        if box:
            for remote_name, local in (
                ("/data/cascade-serve.log", "sail-serve.log"),
                ("/data/hw-sweep/build.log", "sail-build.log"),
            ):
                try:
                    box.fs.download_file(remote_name, HERE / local)
                except Exception as error:
                    print("log collection:", type(error).__name__, flush=True)
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
        (HERE / "sail-resources.json").write_text(json.dumps(state, indent=2))
        print("Sail resources cleaned:", state, flush=True)


if __name__ == "__main__":
    main()
