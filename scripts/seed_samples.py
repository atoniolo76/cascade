"""Upload available cascade-samples run dirs and launch their sample configurations.

    uv run python scripts/seed_samples.py --source e2e-plan-v1=/path/to/e2e-plan-v1

Missing entries are skipped. Pending entries become available automatically when a complete
run dir is present. Only upload-layout files are staged; simulator outputs are never uploaded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("cascade-samples.json")
ROOT_FILES = ("corrected_timeline.json", "trace_alignment.json", "run.json", "workload.json")


def stage_run(source: Path, stage: Path) -> None:
    traces = source / "traced" / "traces"
    required = [*(source / name for name in ROOT_FILES), traces / "kernelslist.g"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("incomplete run dir; missing: " + ", ".join(missing))
    for name in ROOT_FILES:
        (stage / name).symlink_to((source / name).resolve())
    for path in traces.rglob("*"):
        if not path.is_file() or path.name.endswith((".trace", ".trace.xz")):
            continue
        target = stage / "traces" / path.relative_to(traces)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path.resolve())


def upload(stage: Path, entry: dict, sample_set: str, s3: str, pending: list[str]) -> None:
    metadata = {
        key: entry[key]
        for key in ("trace_gpu_arch", "model", "gpu", "real_gpu_config", "sample_sim_configs")
    }
    metadata.update(suggested_sim_config=entry["real_gpu_config"], sample_set=sample_set)
    metadata["sample_pending_configs"] = pending
    command = [
        sys.executable,
        str(ROOT / "scripts/upload_trace.py"),
        str(stage),
        "--run-name",
        entry["id"],
        "--endpoint",
        s3,
    ]
    for key, value in metadata.items():
        encoded = json.dumps(value) if isinstance(value, list) else value
        command.extend(["--meta", f"{key}={encoded}"])
    subprocess.run(command, check=True, capture_output=True, text=True)


def wait_ready(api: httpx.Client, trace_id: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        response = api.get(f"/api/traces/{trace_id}")
        if response.status_code != 404:
            response.raise_for_status()
            trace = response.json()
            if trace["state"] == "ready":
                return
            if trace["state"] == "failed":
                raise RuntimeError(f"trace {trace_id} failed: {trace.get('error')}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for trace {trace_id} to become ready")
        time.sleep(1)


def seed(api: httpx.Client, manifest: dict, sources: dict[str, Path], s3: str, timeout: float):
    runs = []
    for entry in manifest["entries"]:
        source = sources[entry["id"]]
        if not source.is_dir():
            print(f"SKIP {entry['id']}: {entry['status']}; run dir missing: {source}", flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix="cascade-samples-", dir=Path.home()) as tmp:
            stage = Path(tmp)
            stage_run(source, stage)
            pending = [
                c
                for e in manifest["entries"]
                if e["status"] == "pending"
                for c in e["sample_sim_configs"]
            ]
            upload(stage, entry, manifest["id"], s3, pending)
        print(f"uploaded {entry['id']}; waiting for ready", flush=True)
        wait_ready(api, entry["id"], timeout)
        response = api.post(
            "/api/runs",
            json={
                "trace_id": entry["id"],
                "entries": [{"gpu_config": c, "count": 1} for c in entry["sample_sim_configs"]],
            },
        )
        response.raise_for_status()
        run = response.json()
        runs.append(run)
        print(
            f"launched {entry['id']}: run {run['id']} ({', '.join(entry['sample_sim_configs'])})",
            flush=True,
        )
    return runs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=Path("."), help="parent of sample run dirs")
    p.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="ENTRY=DIR",
        help="override the local run dir for an entry; repeatable",
    )
    p.add_argument("--app", default="http://127.0.0.1:8030")
    p.add_argument("--s3", default="http://127.0.0.1:9030")
    p.add_argument("--token", default="no-auth")
    p.add_argument("--timeout", type=float, default=300)
    args = p.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    sources = {entry["id"]: args.runs_dir / entry["id"] for entry in manifest["entries"]}
    for override in args.source:
        key, separator, value = override.partition("=")
        if not separator or key not in sources or not value:
            p.error(f"invalid source {override!r}; expected ENTRY=DIR for {', '.join(sources)}")
        sources[key] = Path(value).expanduser()
    with httpx.Client(base_url=args.app, headers={"Authorization": f"Bearer {args.token}"}) as api:
        seed(api, manifest, sources, args.s3, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
