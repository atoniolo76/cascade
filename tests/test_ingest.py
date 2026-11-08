import json
import sys
from pathlib import Path

import pytest

from cascade.ingest import Ingestor, parse_stats
from cascade.models import TraceState
from cascade.settings import Settings
from cascade.store import Store
from tests.conftest import ACCELSIM_DIR

CTX = "ctx_0x55ddb0567af0"
STATS_HEADER = (
    "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, #blocks, block_dimX,"
    " block_dimY, block_dimZ, #threads, total_insts, total_reported_insts"
)
FAKE_POSTPROCESS = f"{sys.executable} {Path(__file__).parent / 'fake_postprocess.py'}"


def settings(tmp_path, **env):
    return Settings.from_env(
        {
            "CASCADE_DATA_DIR": str(tmp_path / "data"),
            "CASCADE_TOKEN": "t",
            "CASCADE_ACCELSIM_DIR": str(ACCELSIM_DIR),
            "CASCADE_POSTPROCESS_CMD": FAKE_POSTPROCESS,
            **env,
        }
    )


@pytest.fixture
def env(tmp_path):
    s = settings(tmp_path)
    store = Store(s.db_path)
    yield s, store
    store.close()


def write(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


def upload_raw_trace(run_dir: Path, kernel_ids=(2, 10, 9)) -> None:
    for k in kernel_ids:
        write(run_dir / "traces" / f"kernel-{k}-{CTX}.trace.xz", b"x" * (100 + k))
    write(
        run_dir / "traces" / "kernelslist_ctx_0x55ddb0567af0",
        "".join(f"kernel-{k}-{CTX}.trace.xz\n" for k in sorted(kernel_ids)),
    )
    rows = [STATS_HEADER] + [
        f"kernel-{k}-{CTX}.trace.xz, _Z6kernel{k}Pfi, {k}, 1, 1, {k}, 64, 2, 1, 128, {k * 1000},"
        f"{k * 1000}"
        for k in kernel_ids
    ]
    write(run_dir / "traces" / "stats_ctx_0x55ddb0567af0", "\n".join(rows) + "\n")
    write(run_dir / "trace.log", "tracer output\n")


def finish_upload(run_dir: Path, **extra) -> None:
    files = [
        {
            "key": f"accelsim-traces/{run_dir.name}/{p.relative_to(run_dir).as_posix()}",
            "bytes": p.stat().st_size,
        }
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
    ]
    meta = {
        "status": "ok",
        "model": "facebook/opt-125m",
        "layers": ["l10"],
        "gpu": "NVIDIA H100",
        "files": files,
        **extra,
    }
    write(run_dir / "metadata.json", json.dumps(meta))


def poll(ing: Ingestor) -> None:
    ing.poll_once()
    ing.drain()


def test_upload_then_processing_then_ready(env):
    s, store = env
    run_dir = s.trace_dir("run-a")
    ing = Ingestor(s, store)
    write(run_dir / "traces" / f"kernel-2-{CTX}.trace.xz", b"x")
    write(run_dir / "traces" / f".rclone_temp_put_kernel-9-{CTX}.trace.xz", b"partial")
    poll(ing)
    t = store.get_trace("run-a")
    assert (t.state, t.kernel_count) == (TraceState.uploading, 1)

    (run_dir / "traces" / f".rclone_temp_put_kernel-9-{CTX}.trace.xz").unlink()
    upload_raw_trace(run_dir)
    finish_upload(run_dir, trace_gpu_arch="sm_90", suggested_sim_config="SM90_H100")
    write(run_dir / "traces" / f"kernel-10-{CTX}.trace.xz", b"short")
    poll(ing)
    assert store.get_trace("run-a").state == TraceState.uploading

    write(run_dir / "traces" / f"kernel-10-{CTX}.trace.xz", b"x" * 110)
    poll(ing)
    t = store.get_trace("run-a")
    assert t.state == TraceState.ready, t.error
    assert t.kernel_count == 3
    assert [k.kernel_id for k in t.kernels] == [2, 9, 10]
    assert t.kernels[2].grid == (10, 1, 1) and t.kernels[2].block == (64, 2, 1)
    assert t.kernels[2].instructions == 10000
    assert (t.model, t.source_gpu, t.layers) == ("facebook/opt-125m", "NVIDIA H100", ["l10"])
    assert t.suggested_sim_config == "SM90_H100"
    assert (run_dir / "traces" / "kernelslist.g").read_text().startswith(f"kernel-2-{CTX}.tracez")
    assert not t.has_real_timing


def test_failed_status_marks_trace_failed(env):
    s, store = env
    run_dir = s.trace_dir("run-f")
    upload_raw_trace(run_dir)
    write(run_dir / "metadata.json", json.dumps({"status": "failed"}))
    poll(Ingestor(s, store))
    t = store.get_trace("run-f")
    assert t.state == TraceState.failed
    assert "failed" in t.error
    assert not (run_dir / "traces" / "kernelslist.g").exists()


def test_older_profiler_done_needs_stats_and_a_quiet_tree(env):
    s, store = env
    run_dir = s.trace_dir("run-old")
    upload_raw_trace(run_dir)
    (run_dir / "traces" / "stats_ctx_0x55ddb0567af0").unlink()
    write(run_dir / "metadata.json", json.dumps({"status": "ok"}))
    ing = Ingestor(s, store)
    poll(ing)
    poll(ing)
    assert store.get_trace("run-old").state == TraceState.uploading
    upload_raw_trace(run_dir)
    poll(ing)
    assert store.get_trace("run-old").state == TraceState.uploading
    poll(ing)
    assert store.get_trace("run-old").state == TraceState.ready


def test_already_processed_upload_skips_postprocessing(tmp_path):
    s = settings(tmp_path, CASCADE_POSTPROCESS_CMD="/nonexistent/post-traces-processing")
    store = Store(s.db_path)
    run_dir = s.trace_dir("rodinia")
    write(
        run_dir / "traces" / "kernelslist.g", "MemcpyHtoD,0x1,8\nkernel-1.traceg\nkernel-2.traceg\n"
    )
    write(run_dir / "traces" / "kernel-1.traceg", "t")
    write(run_dir / "traces" / "kernel-2.traceg", "t")
    write(
        run_dir / "real_timing.json",
        json.dumps(
            [
                {
                    "kernel_id": 2,
                    "kernel_name": "b",
                    "duration_ns": 20.0,
                    "grid": [1, 1, 1],
                    "block": [1, 1, 1],
                },
                {
                    "kernel_id": 1,
                    "kernel_name": "a",
                    "duration_ns": 10.0,
                    "grid": [1, 1, 1],
                    "block": [1, 1, 1],
                },
            ]
        ),
    )
    finish_upload(run_dir)
    poll(Ingestor(s, store))
    t = store.get_trace("rodinia")
    assert t.state == TraceState.ready, t.error
    assert t.kernel_count == 2 and t.has_real_timing
    assert [r["kernel_id"] for r in store.real_timing("rodinia")] == [1, 2]


def test_missing_postprocessor_fails_with_clear_error(tmp_path):
    s = settings(tmp_path, CASCADE_POSTPROCESS_CMD="/nonexistent/post-traces-processing")
    store = Store(s.db_path)
    run_dir = s.trace_dir("run-m")
    upload_raw_trace(run_dir)
    finish_upload(run_dir)
    poll(Ingestor(s, store))
    t = store.get_trace("run-m")
    assert t.state == TraceState.failed
    assert "post-processor not found" in t.error and "CASCADE_POSTPROCESS_CMD" in t.error


def test_postprocessor_error_fails_trace(env):
    s, store = env
    run_dir = s.trace_dir("run-e")
    upload_raw_trace(run_dir)
    write(run_dir / "traces" / "FAIL", "")
    finish_upload(run_dir)
    poll(Ingestor(s, store))
    t = store.get_trace("run-e")
    assert t.state == TraceState.failed
    assert "exited 3" in t.error and "simulated post-processing failure" in t.error


def test_restart_mid_processing_reruns_and_restart_after_ready_is_a_no_op(env):
    s, store = env
    run_dir = s.trace_dir("run-r")
    upload_raw_trace(run_dir)
    finish_upload(run_dir)
    store.register_trace("run-r", 3, 1)
    store.update_trace("run-r", state=TraceState.processing)
    work = s.work_dir / "postprocess"
    write(work / "run-r.started", "")
    write(run_dir / "traces" / "kernelslist.g", "")
    poll(Ingestor(s, store))
    assert store.get_trace("run-r").state == TraceState.ready
    assert (run_dir / "postprocess-calls").read_text() == "1"
    before = store.get_trace("run-r")
    poll(Ingestor(s, store))
    assert store.get_trace("run-r") == before
    assert (run_dir / "postprocess-calls").read_text() == "1"


def test_parse_stats_tolerates_truncated_and_merged_rows():
    text = "\n".join(
        [
            STATS_HEADER,
            "61628",
            "kernel-3.trace, _Z6euclidPcffPfiii, 938, 1, 1, 938, 16, 1, 1, 16, 161428,161428",
            f"kernel-4-{CTX}.trace.xz, _Z1fv, 2, 3, 4, 24, 32, 1, 1, 32, ",
        ]
    )
    kernels = parse_stats(text)
    assert [(k.kernel_id, k.name, k.grid, k.block, k.instructions) for k in kernels] == [
        (3, "_Z6euclidPcffPfiii", (938, 1, 1), (16, 1, 1), 161428),
        (4, "_Z1fv", (2, 3, 4), (32, 1, 1), None),
    ]
