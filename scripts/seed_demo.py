"""Fill a local Cascade app with demo traces and runs, for trying the UI.

    uv run python scripts/seed_demo.py <processed trace dir> \
        --app http://127.0.0.1:8030 --s3 http://127.0.0.1:9030

Every trace reuses the same processed trace files with different metadata, and the source GPU
is labeled "(demo)". Runs replay whatever log CASCADE_SIM_CMD points at, so the numbers are not
real simulations of these models. One sim per partial run is failed through the simulator API,
the same way a crashing simulator reports itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

TRACES = [
    (
        "20261107-153012__facebook_opt-125m__model.decoder.layers.10.self_attn",
        "facebook/opt-125m",
        "model.decoder.layers.10.self_attn",
        "NVIDIA H100 80GB HBM3 (demo)",
    ),
    (
        "20261107-161540__meta-llama_Llama-3.1-8B-Instruct__model.layers.0.mlp",
        "meta-llama/Llama-3.1-8B-Instruct",
        "model.layers.0.mlp",
        "NVIDIA H100 80GB HBM3 (demo)",
    ),
    (
        "20261107-164210__Qwen_Qwen2.5-7B-Instruct__model.layers.12.self_attn",
        "Qwen/Qwen2.5-7B-Instruct",
        "model.layers.12.self_attn",
        "NVIDIA H200 (demo)",
    ),
    (
        "20261107-170455__mistralai_Mistral-7B-v0.3__model.layers.3.mlp",
        "mistralai/Mistral-7B-v0.3",
        "model.layers.3.mlp",
        "NVIDIA H100 80GB HBM3 (demo)",
    ),
]


def upload(trace_dir: Path, s3: str, run_name: str, model: str, layer: str, gpu: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/upload_trace.py"),
            str(trace_dir),
            "--run-name",
            run_name,
            "--endpoint",
            s3,
            "--meta",
            "trace_gpu_arch=sm_90",
            "--meta",
            "suggested_sim_config=SM90_H100",
            "--meta",
            f"model={model}",
            "--meta",
            f"gpu={gpu}",
            "--meta",
            f"layers={json.dumps([layer])}",
        ],
        check=True,
        capture_output=True,
    )


def wait(predicate, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise SystemExit("timed out waiting for the app")
        time.sleep(1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("trace_dir", type=Path, help="processed trace dir (holds traces/kernelslist.g)")
    p.add_argument("--app", default="http://127.0.0.1:8030")
    p.add_argument("--s3", default="http://127.0.0.1:9030")
    p.add_argument("--token", default="no-auth")
    args = p.parse_args()

    api = httpx.Client(base_url=args.app, headers={"Authorization": f"Bearer {args.token}"})
    for run_name, model, layer, gpu in TRACES:
        upload(args.trace_dir, args.s3, run_name, model, layer, gpu)
        print(f"uploaded {model}")
    wait(lambda: sum(t["state"] == "ready" for t in api.get("/api/traces").json()) == len(TRACES))

    def launch(trace: str, entries: list[tuple[str, int]]) -> dict:
        body = {"trace_id": trace, "entries": [{"gpu_config": c, "count": n} for c, n in entries]}
        res = api.post("/api/runs", json=body)
        res.raise_for_status()
        return res.json()

    def finished(run_id: str) -> bool:
        return api.get(f"/api/runs/{run_id}").json()["state"] != "active"

    opt, llama, qwen, mistral = (t[0] for t in TRACES)

    done = [
        launch(opt, [("SM90_H100", 4), ("SM90_H200", 4)]),
        launch(opt, [("SM90_H100", 2)]),
        launch(llama, [("SM90_H100", 3), ("SM90_H200", 3)]),
    ]
    partial = launch(qwen, [("SM90_H200", 4), ("SM90_H100", 2)])
    wait(
        lambda: any(
            s["state"] == "running" for s in api.get(f"/api/runs/{partial['id']}").json()["sims"]
        )
    )
    victim = next(
        s for s in api.get(f"/api/runs/{partial['id']}").json()["sims"] if s["state"] == "running"
    )
    api.post(
        f"/api/sims/{victim['id']}/heartbeat",
        json={"state": "failed", "exit_code": 134, "error": "simulator exited 134 (SIGABRT)"},
    ).raise_for_status()
    cancelled = launch(mistral, [("SM90_H100", 2)])
    api.post(f"/api/runs/{cancelled['id']}/cancel").raise_for_status()

    for run in [*done, partial]:
        wait(lambda r=run: finished(r["id"]), timeout=300)
    live = launch(llama, [("SM90_H100", 8), ("SM90_H200", 8)])
    print(f"seeded {len(TRACES)} traces and {len(done) + 3} runs; live run {live['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
