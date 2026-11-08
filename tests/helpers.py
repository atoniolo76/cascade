import json
import time
from pathlib import Path

from cascade.settings import Settings
from tests.conftest import ACCELSIM_DIR

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
