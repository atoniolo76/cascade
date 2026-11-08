import gzip
import json

import pytest

from scripts.build_sample import GZIPPED, build


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for rel in GZIPPED:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (root / "traced/traces").mkdir()
    (root / "traced/traces/stats_ctx_1").write_text("stats")
    (root / "traced/traces/kernelslist.g").write_text("kernel-6207.tracez\n")
    (root / "sim/kernels.csv").write_text("uid,name,cycles,sim\n6207,k,5,exact\n")
    (root / "sim/setup.log").write_text("setup")
    (root / "sim/kernel-6207.log").write_text("recorded log")
    perf = root / "sim/kernel-6207"
    perf.mkdir()
    (perf / "perf_counter_2026-11-08_06-50-10.csv.gz").touch()
    return root


@pytest.fixture
def sweep(tmp_path):
    root = tmp_path / "sweep"
    root.mkdir()
    for rel in ("summary.json", "compare.json"):
        (root / rel).write_text("{}")
    (root / "kernels.csv").write_text("uid,name,cycles,sim\n6207,k,6,exact\n")
    (root / "simulated_timeline.json.gz").write_bytes(gzip.compress(b"{}"))
    (root / "sim_durations.json.gz").write_bytes(
        gzip.compress(json.dumps({"provenance": {"clock_mhz": 1980}}).encode())
    )
    return root


def test_build_multiple_configs_is_reproducible_and_preserves_source_bytes(source, sweep, tmp_path):
    out = tmp_path / "bundle"
    for _ in range(2):
        build(source, out, {"SM90_H200": sweep})
        files = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
        if _ == 0:
            first = files
        else:
            assert files == first
    manifest = json.loads((out / "sample.json").read_text())
    assert manifest["sample_sim_configs"] == ["SM90_H100", "SM90_H200"]
    assert manifest["real_gpu_config"] == "SM90_H100"
    assert manifest["stats_file"] == "traced/traces/stats_ctx_1"
    assert manifest["configs"][1]["stats_source"] == "kernels.csv"
    assert manifest["configs"][1]["core_clock_mhz"] == 1980
    assert manifest["configs"][1]["sim_started_at"] is None
    assert (
        gzip.decompress((out / "sim/kernels.csv.gz").read_bytes())
        == (source / "sim/kernels.csv").read_bytes()
    )
    assert (
        gzip.decompress((out / "sim/SM90_H200/kernels.csv.gz").read_bytes())
        == (sweep / "kernels.csv").read_bytes()
    )
    assert (
        gzip.decompress((out / "sim/kernel-6207.log.gz").read_bytes())
        == (source / "sim/kernel-6207.log").read_bytes()
    )
    assert (out / "sim/kernel-6207.log.gz").read_bytes()[9] == 255


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "missing",
        "duplicate",
        "sweep-empty",
        "primary-failed",
        "primary-timeout",
        "sweep-failed",
        "sweep-timeout",
    ],
)
def test_bad_input_does_not_replace_an_existing_bundle(source, sweep, tmp_path, problem):
    out = tmp_path / "bundle"
    out.mkdir()
    manifest = {"gpu_config": "SM90_H100", "status": "pending"}
    (out / "sample.json").write_text(json.dumps(manifest))
    configs = {"SM90_H200": sweep}
    if problem == "empty":
        (source / "sim/kernels.csv").write_text("uid,name,cycles,sim\n")
    elif problem == "missing":
        (sweep / "compare.json").unlink()
    elif problem == "duplicate":
        configs = {"SM90_H100": sweep}
    elif problem == "sweep-empty":
        (sweep / "kernels.csv").write_text("uid,name,cycles,sim\n")
    else:
        location, kind = problem.split("-")
        path = source / "sim/summary.json" if location == "primary" else sweep / "summary.json"
        path.write_text(json.dumps({"counts": {kind: 1}}))
    with pytest.raises(ValueError):
        build(source, out, configs)
    assert json.loads((out / "sample.json").read_text()) == manifest


def test_build_turns_a_pending_native_sample_into_an_available_bundle(source, tmp_path):
    out = tmp_path / "bundle"
    out.mkdir()
    manifest = {
        "status": "pending",
        "gpu_config": "SM80_A100",
        "trace_gpu_arch": "sm_80",
        "sim_id": "a100",
        "core_clock_mhz": 1410,
    }
    (out / "sample.json").write_text(json.dumps(manifest))
    build(source, out)
    result = json.loads((out / "sample.json").read_text())
    assert result["status"] == "available"
    assert result["real_gpu_config"] == "SM80_A100"
    assert result["sample_sim_configs"] == ["SM80_A100"]
