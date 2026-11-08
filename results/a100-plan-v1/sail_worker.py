"""Native A100 Accel-Sim worker; collect artifacts before Cascade terminates it."""

import json
import os
import sys
import threading
from pathlib import Path

import httpx
from fake_sim import Main, heartbeat_loop

from cascade.sim.accelsim import simulate_accelsim


def main():
    env = os.environ
    api = Main(env["CASCADE_MAIN_URL"], env["CASCADE_SIM_ID"], env["CASCADE_TOKEN"])
    api.heartbeat("running")
    stop = threading.Event()
    beat = threading.Thread(target=heartbeat_loop, args=(api, 5, stop), daemon=True)
    beat.start()
    try:
        source = Path(env["CASCADE_TRACE_DIR"])
        work = Path("/data/a100-simulation")
        work.mkdir(exist_ok=False)
        for name in ("corrected_timeline.json", "trace_alignment.json"):
            (work / name).symlink_to(source / name)
        (work / "traced").mkdir()
        (work / "traced/traces").symlink_to(source / "traces", target_is_directory=True)
        summary = simulate_accelsim(
            work,
            gpu_config=env["CASCADE_GPU_CONFIG"],
            jobs=8,
            kernel_timeout_s=900,
            dedup=True,
            verify_dedup=2,
            propagate="interp",
        )
        print("Simulation wall seconds:", summary["wall_s"], flush=True)
        print("Simulation counts:", summary["counts"], flush=True)
        settings = json.loads(Path("/app/scripts/a100-settings.json").read_text())
        prefix = settings["s3_url"] + "/cascade/accelsim-traces/a100-results"
        tails = {}
        for seq, log in enumerate(sorted((work / "sim").glob("kernel-*.log"))):
            lines = log.read_text(errors="replace").splitlines()
            tails[log.name] = lines[-40:]
            api.post("logs", {"seq": seq, "lines": lines})
        (work / "log_tails.json").write_text(json.dumps(tails, indent=2))
        for path in (
            work / "sim_durations.json",
            work / "sim/summary.json",
            work / "sim/kernels.csv",
            work / "log_tails.json",
        ):
            response = httpx.put(f"{prefix}/{path.name}", content=path.read_bytes(), timeout=120)
            response.raise_for_status()
        failed = summary["counts"]["failed"] + summary["counts"]["timeout"]
        api.heartbeat("failed" if failed else "succeeded", exit_code=int(bool(failed)))
        return int(bool(failed))
    except Exception as error:
        api.heartbeat("failed", exit_code=1, error=str(error))
        raise
    finally:
        stop.set()
        beat.join()


if __name__ == "__main__":
    sys.exit(main())
