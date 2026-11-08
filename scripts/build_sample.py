"""Builds samples/opt125m-h100-e2e/ from Alessio's e2e-plan-v1 run directory.

    uv run python scripts/build_sample.py ~/Downloads/e2e-plan-v1-full/e2e-plan-v1

Copies the files Cascade's sample needs, gzipping JSON and the per-kernel Accel-Sim logs with a
fixed mtime so a rebuild is byte-identical, and checks every copy decompresses to its source.
Writes sample.json with the ids Cascade seeds and the one fact taken from an omitted file: when
Accel-Sim started, from the earliest sim/kernel-*/perf_counter_<UTC time>.csv.gz name.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "samples" / "opt125m-h100-e2e"
GZIPPED = (
    "run.json",
    "workload.json",
    "reconcile.json",
    "corrected_timeline.json",
    "sim_durations.json",
    "trace_alignment.json",
    "traced/trace_steps.json",
    "sim/summary.json",
    "sim/compare.json",
    "sim/simulated_timeline.json",
    "sim/kernels.csv",
)
PLAIN = ("traced/traces/kernelslist.g", "sim/setup.log")
PERF_COUNTER = re.compile(r"^perf_counter_(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)\.csv\.gz$")
MANIFEST = {
    "trace_id": "sample__facebook_opt-125m__h100-e2e",
    "run_id": "run-sample-opt125m-h100-e2e",
    "sim_id": "sim-sample-opt125m-h100-e2e",
    "gpu_config": "SM90_H100",
    "core_clock_mhz": 1980,
    "trace_gpu_arch": "sm_90",
    "output_files": [
        "sim/kernels.csv",
        "sim/summary.json",
        "sim/compare.json",
        "sim/setup.log",
        "sim/kernel-6207.log",
    ],
}


def gzip_copy(src: Path, dest: Path) -> None:
    data = src.read_bytes()
    packed = gzip.compress(data, compresslevel=9, mtime=0)
    packed = packed[:9] + b"\xff" + packed[10:]
    if gzip.decompress(packed) != data:
        raise SystemExit(f"{src}: gzip round trip differs")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(packed)


def simulated_uids(kernels_csv: Path) -> list[str]:
    with kernels_csv.open() as f:
        return sorted((r["uid"] for r in csv.DictReader(f) if r["sim"] == "exact"), key=int)


def sim_started_at(sim_dir: Path) -> str:
    stamps = sorted(
        m.group(1)
        for path in sim_dir.glob("kernel-*/perf_counter_*")
        if (m := PERF_COUNTER.match(path.name))
    )
    if not stamps:
        raise SystemExit(f"no perf_counter_* files under {sim_dir}/kernel-*/")
    return datetime.strptime(stamps[0], "%Y-%m-%d_%H-%M-%S").replace(tzinfo=UTC).isoformat()


def build(run_dir: Path, out: Path = OUT, configs: dict[str, Path] | None = None) -> None:
    manifest_path = out / "sample.json"
    template = json.loads(manifest_path.read_text()) if manifest_path.exists() else MANIFEST
    stats_files = sorted((run_dir / "traced/traces").glob("stats_ctx_*"))
    if len(stats_files) != 1:
        raise ValueError("expected one traced/traces/stats_ctx_* file")
    additional = configs or {}
    extra_clocks = {}
    uids = simulated_uids(run_dir / "sim/kernels.csv")
    if not uids:
        raise ValueError("sample needs at least one exact simulated kernel")
    for rel in (*GZIPPED, *PLAIN, *(f"sim/kernel-{uid}.log" for uid in uids)):
        if not (run_dir / rel).is_file():
            raise ValueError(f"missing {run_dir / rel}")
    counts = json.loads((run_dir / "sim/summary.json").read_text()).get("counts", {})
    if counts.get("failed", 0) or counts.get("timeout", 0):
        raise ValueError("primary simulation has failed or timed-out kernels")
    for config, source in additional.items():
        if config == template["gpu_config"]:
            raise ValueError(f"duplicate config: {config}")
        for rel in (
            "kernels.csv",
            "summary.json",
            "compare.json",
            "simulated_timeline.json.gz",
            "sim_durations.json.gz",
        ):
            if not (source / rel).is_file():
                raise ValueError(f"{config}: missing {source / rel}")
        counts = json.loads((source / "summary.json").read_text()).get("counts", {})
        if counts.get("failed", 0) or counts.get("timeout", 0):
            raise ValueError(f"{config}: simulation has failed or timed-out kernels")
        if not simulated_uids(source / "kernels.csv"):
            raise ValueError(f"{config}: sample needs at least one exact simulated kernel")
        extra_clocks[config] = float(
            json.loads(gzip.decompress((source / "sim_durations.json.gz").read_bytes()))[
                "provenance"
            ]["clock_mhz"]
        )
    started = sim_started_at(run_dir / "sim")
    if out.exists():
        for path in out.iterdir():
            if path.name != "README.md":
                shutil.rmtree(path) if path.is_dir() else path.unlink()
    for rel in GZIPPED:
        gzip_copy(run_dir / rel, out / f"{rel}.gz")
    stats_file = stats_files[0].relative_to(run_dir).as_posix()
    gzip_copy(stats_files[0], out / f"{stats_file}.gz")
    for rel in PLAIN:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / rel, out / rel)
    for uid in uids:
        gzip_copy(run_dir / f"sim/kernel-{uid}.log", out / f"sim/kernel-{uid}.log.gz")
    if (run_dir.parent / "README.md").is_file():
        shutil.copyfile(run_dir.parent / "README.md", out / "source-README.md")
    manifest = {
        **template,
        "status": "available",
        "sim_started_at": started,
        "stats_file": stats_file,
        "real_gpu_config": template["gpu_config"],
        "sample_set": "cascade-samples",
        "sample_sim_configs": [template["gpu_config"], *additional],
        "configs": [{"gpu_config": template["gpu_config"], "sim_dir": "sim"}],
        "output_files": [
            "sim/kernels.csv",
            "sim/summary.json",
            "sim/compare.json",
            "sim/setup.log",
            f"sim/kernel-{uids[0]}.log",
        ],
    }
    for config, source in additional.items():
        rel = f"sim/{config}"
        dest = out / rel
        dest.mkdir(parents=True)
        for name in ("kernels.csv", "summary.json", "compare.json"):
            gzip_copy(source / name, dest / f"{name}.gz")
        for name in ("simulated_timeline.json.gz", "sim_durations.json.gz"):
            shutil.copyfile(source / name, dest / name)
        manifest["configs"].append(
            {
                "gpu_config": config,
                "sim_dir": rel,
                "sim_id": f"{template['sim_id']}-{config.lower()}",
                "core_clock_mhz": extra_clocks[config],
                "stats_source": "kernels.csv",
                "sim_started_at": None,
                "output_files": [
                    f"sim/{name}" for name in ("kernels.csv", "summary.json", "compare.json")
                ],
            }
        )
    (out / "sample.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out} ({total / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="the e2e-plan-v1 run directory")
    parser.add_argument("--sample", default=OUT.name, help="bundle directory name under samples/")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="CONFIG=DIR",
        help="additional hardware-sweep result directory",
    )
    args = parser.parse_args()
    configs = dict(value.split("=", 1) for value in args.config)
    build(
        args.run_dir.expanduser(),
        OUT.parent / args.sample,
        {c: Path(p).expanduser() for c, p in configs.items()},
    )


if __name__ == "__main__":
    main()
