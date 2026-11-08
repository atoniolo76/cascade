"""Simulate legacy instruction traces without inventing a timing alignment."""

import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cascade.sim.accelsim import (
    _commit,
    _runtime_environment,
    _simulate_kernel,
    core_clock,
    dedup_groups,
    fan_out,
    trace_header,
)
from cascade.sim.accelsim_contract import _write_csv


def _dims(header, field, path):
    value = re.fullmatch(r"\((\d+),\s*(\d+),\s*(\d+)\)", header.get(field, ""))
    if value is None:
        raise ValueError(f"missing/invalid {field} in {path}")
    return [int(n) for n in value.groups()]


def simulate_traces(
    run_dir,
    *,
    gpu_config,
    accelsim_root="/opt/accel-sim-framework",
    jobs=32,
    kernel_timeout_s=None,
    dedup=True,
    verify_dedup=2,
):
    started = time.monotonic()
    if not gpu_config or Path(gpu_config).name != gpu_config or gpu_config in {".", ".."}:
        raise ValueError("gpu_config must be a config directory name")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if isinstance(verify_dedup, bool) or not isinstance(verify_dedup, int) or verify_dedup < 0:
        raise ValueError("verify_dedup must be a nonnegative integer")
    if not isinstance(dedup, bool):
        raise ValueError("dedup must be boolean")
    if kernel_timeout_s is not None and (
        not math.isfinite(kernel_timeout_s) or kernel_timeout_s <= 0
    ):
        raise ValueError("kernel_timeout_s must be positive and finite")
    root, run = Path(accelsim_root).resolve(), Path(run_dir).resolve()
    traces = run / "traces"
    selected, files = [], {}
    for filename in (traces / "kernelslist.g").read_text().splitlines():
        match = re.fullmatch(r"kernel-(\d+)-ctx_.*\.tracez", filename.strip())
        if not match:
            continue
        uid = int(match[1])
        if uid in files:
            raise ValueError(f"duplicate tracer UID {uid}")
        path = traces / filename.strip()
        header = trace_header(path)

        if not header.get("kernel name"):
            raise ValueError(f"missing kernel name in {path}")
        selected.append(
            {
                "tracer_uid": uid,
                "name": header["kernel name"],
                "grid": _dims(header, "grid dim", path),
                "block": _dims(header, "block dim", path),
            }
        )
        files[uid] = path
    cfgdir = root / "gpu-simulator/gpgpu-sim/configs/tested-cfgs" / gpu_config
    tracecfg = root / "gpu-simulator/configs/tested-cfgs" / gpu_config / "trace.config"
    binary = root / "gpu-simulator/bin/release/accel-sim.out"
    for required in (
        cfgdir / "gpgpusim.config",
        tracecfg,
        binary,
        root / "gpu-simulator/setup_environment.sh",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    clock = core_clock(cfgdir / "gpgpusim.config")
    sim = run / "sim"
    sim.mkdir(exist_ok=True)
    groups = dedup_groups(selected, files, dedup)
    run_uids = {uid for group in groups for uid in group["members"][: 1 + verify_dedup]}
    env = _runtime_environment(root, sim / "setup.log") if selected else {}
    tasks = [
        {
            "uid": k["tracer_uid"],
            "name": k["name"],
            "env": env,
            "binary": str(binary),
            "directory": str(sim / f"kernel-{k['tracer_uid']}"),
            "cfgdir": str(cfgdir),
            "tracecfg": str(tracecfg),
            "trace": str(files[k["tracer_uid"]]),
            "log": str(sim / f"kernel-{k['tracer_uid']}.log"),
            "timeout": kernel_timeout_s,
        }
        for k in selected
        if k["tracer_uid"] in run_uids
    ]
    with ProcessPoolExecutor(max_workers=jobs or os.cpu_count() or 1) as pool:
        actual = list(pool.map(_simulate_kernel, tasks))
    results, verification = fan_out(groups, actual)
    records = [
        {
            "tracer_uid": r["uid"],
            "name": r["name"],
            "sim_cycles": r["cycles"],
            "clock_mhz": clock,
            "sim": r["sim"],
        }
        for r in results
        if r["status"] == "ok"
    ]
    counts = dict(Counter(r["status"] for r in results))
    counts.update(unique_accelsim_runs=len(actual), traced_kernel_occurrences=len(selected))
    summary = {
        "wall_s": time.monotonic() - started,
        "total_sim_wall_s": sum(r["sim_wall_s"] for r in actual),
        "counts": counts,
        "dedup_verification": verification,
        "provenance_labels": dict(Counter(r["sim"] for r in records)),
    }
    provenance = {
        "run_name": run.name,
        "alignment": None,
        "scope": "tracer_uid_only",
        "gpu_config": gpu_config,
        "clock_mhz": clock,
        "accelsim_commit": _commit(root),
        "gpgpusim_commit": _commit(root / "gpu-simulator/gpgpu-sim"),
        "dedup": dedup,
        "verify_dedup": verify_dedup,
        "isolated_cold_cache": True,
    }
    (run / "sim_durations.json").write_text(
        json.dumps({"provenance": provenance, "kernels": records}, indent=2)
    )
    _write_csv(
        sim / "kernels.csv",
        results,
        ["uid", "name", "cycles", "sim_wall_s", "status", "error", "kernel_launch_uid", "sim"],
    )
    (sim / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
