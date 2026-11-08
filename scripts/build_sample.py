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
    "traced/traces/stats_ctx_0x54c35c0",
    "sim/summary.json",
    "sim/compare.json",
    "sim/simulated_timeline.json",
)
PLAIN = ("traced/traces/kernelslist.g", "sim/kernels.csv", "sim/setup.log")
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


def build(run_dir: Path) -> None:
    if OUT.exists():
        for path in OUT.iterdir():
            if path.name != "README.md":
                shutil.rmtree(path) if path.is_dir() else path.unlink()
    for rel in GZIPPED:
        gzip_copy(run_dir / rel, OUT / f"{rel}.gz")
    for rel in PLAIN:
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / rel, OUT / rel)
    for uid in simulated_uids(run_dir / "sim/kernels.csv"):
        gzip_copy(run_dir / f"sim/kernel-{uid}.log", OUT / f"sim/kernel-{uid}.log.gz")
    shutil.copyfile(run_dir.parent / "README.md", OUT / "source-README.md")
    manifest = {**MANIFEST, "sim_started_at": sim_started_at(run_dir / "sim")}
    (OUT / "sample.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT} ({total / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="the e2e-plan-v1 run directory")
    build(parser.parse_args().run_dir.expanduser())


if __name__ == "__main__":
    main()
