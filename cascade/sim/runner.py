from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

from cascade.sim.timeline import numeric


def config_options(path: Path) -> dict[str, str]:
    options = {}
    for line in path.read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if len(parts) >= 2 and parts[0].startswith("-"):
            options[parts[0]] = parts[1]
    return options


def config_clock(path: Path) -> float:
    value = config_options(path).get("-gpgpu_clock_domains")
    if value is None:
        raise ValueError("config is missing -gpgpu_clock_domains")
    return numeric(float(value.split(":")[0]), "clock_mhz", positive=True)


def prepare_run(framework: Path, config_name: str, kernelslist: Path, run_dir: Path) -> dict:
    """Prepare files/argv only; the worker owns execution, environment, logs and cancellation."""
    if not config_name or config_name in (".", "..") or Path(config_name).name != config_name:
        raise ValueError("config_name must be a tested-cfgs directory name")
    framework, kernelslist, run_dir = framework.resolve(), kernelslist.resolve(), run_dir.resolve()
    gpu = framework / "gpu-simulator"
    base = gpu / "gpgpu-sim/configs/tested-cfgs" / config_name
    configs = [base / "gpgpusim.config", gpu / "configs/tested-cfgs" / config_name / "trace.config"]
    binary = gpu / "bin/release/accel-sim.out"
    for path in [binary, kernelslist, *configs]:
        if not path.is_file():
            raise ValueError(f"missing simulator input: {path}")
    if not os.access(binary, os.X_OK):
        raise ValueError("simulator binary is not executable")
    clock = config_clock(configs[0])
    copies = [(p, Path(p.name)) for p in configs]
    for config in configs:
        companion = config_options(config).get("-inter_config_file")
        if companion:
            relative = Path(companion)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("interconnect config must use a relative path without traversal")
            source = config.parent / relative
            if not source.is_file():
                raise ValueError(f"missing interconnect config: {source}")
            copies.append((source, relative))
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("run directory must be empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    for source, relative in copies:
        destination = run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {
        "cwd": str(run_dir),
        "command": [
            str(binary),
            "-trace",
            str(kernelslist),
            "-config",
            str(run_dir / "gpgpusim.config"),
            "-config",
            str(run_dir / "trace.config"),
        ],
        "clock_mhz": clock,
        "launch_latency_cycles": config_options(configs[0]).get("-gpgpu_kernel_launch_latency"),
    }
