import json
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import uvicorn

from cascade.api import create_app
from cascade.settings import Settings
from tests.conftest import ACCELSIM_DIR

CASCADE = Path(sys.executable).parent / "cascade"

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def make_settings(tmp_path: Path, **env: str) -> Settings:
    return Settings.from_env(
        {
            "CASCADE_DATA_DIR": str(tmp_path / "data"),
            "CASCADE_TOKEN": TOKEN,
            "CASCADE_ACCELSIM_DIR": str(ACCELSIM_DIR),
            "CASCADE_POSTPROCESS_CMD": "/nonexistent/post-traces-processing",
            "CASCADE_RUN_S3": "false",
            **env,
        }
    )


def write_processed_trace(settings: Settings, trace_id: str, arch: str, kernels: int = 6) -> Path:
    """An upload that already carries kernelslist.g, so ingest skips post-processing."""
    run_dir = settings.trace_dir(trace_id)
    traces = run_dir / "traces"
    traces.mkdir(parents=True)
    names = [f"kernel-{k}.tracez" for k in range(1, kernels + 1)]
    (traces / "kernelslist.g").write_text("".join(f"{n}\n" for n in names))
    for n in names:
        (traces / n).write_bytes(b"z")
    real = [
        {
            "kernel_id": k,
            "kernel_name": f"k{k}",
            "duration_ns": 1000.0 * k,
            "grid": [1, 1, 1],
            "block": [32, 1, 1],
        }
        for k in range(1, kernels + 1)
    ]
    (run_dir / "real_timing.json").write_text(json.dumps(real))
    files = [
        {"key": p.relative_to(run_dir).as_posix(), "bytes": p.stat().st_size}
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
    ]
    meta = {"status": "ok", "gpu": "NVIDIA H100 80GB HBM3", "trace_gpu_arch": arch, "files": files}
    (run_dir / "metadata.json").write_text(json.dumps(meta))
    return run_dir


def wait_for(predicate, timeout: float = 15.0, every: float = 0.05):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for condition")
        time.sleep(every)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def running_app(
    tmp_path: Path, launcher=None, **env: str
) -> Iterator[tuple[httpx.Client, str, Settings]]:
    """The app on a real port, so sims started by the launcher can reach it over HTTP."""
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    settings = make_settings(tmp_path, CASCADE_PUBLIC_URL=url, **env)
    app = create_app(settings, launcher(settings) if launcher else None)
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_for(lambda: server.started)
    try:
        with httpx.Client(base_url=url, headers=AUTH, timeout=10) as client:
            yield client, url, settings
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def launch(client: httpx.Client, trace_id: str, config: str, count: int = 1) -> dict:
    body = {"trace_id": trace_id, "entries": [{"gpu_config": config, "count": count}]}
    return client.post("/api/runs", json=body).json()


def finished_run(client: httpx.Client, run_id: str, timeout: float = 60) -> dict:
    return wait_for(
        lambda: (r := client.get(f"/api/runs/{run_id}").json())["state"] != "active" and r,
        timeout=timeout,
    )
