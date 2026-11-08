import json
from unittest.mock import patch

import httpx
import pytest

from scripts.seed_samples import MANIFEST, ROOT_FILES, seed, stage_run, wait_ready


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "source"
    traces = source / "traced/traces"
    traces.mkdir(parents=True)
    (traces / "kernelslist.g").write_text("kernel-6207.tracez\n")
    (traces / "kernel-6207.tracez").write_bytes(b"processed")
    (traces / "kernel-6207.trace.xz").write_bytes(b"raw")
    for name in ROOT_FILES:
        (source / name).write_text("{}")
    return source


def test_stage_upload_layout_without_copying_binaries(source, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    stage_run(source, stage)
    assert sorted(p.name for p in stage.iterdir()) == sorted([*ROOT_FILES, "traces"])
    trace = stage / "traces/kernel-6207.tracez"
    assert trace.is_symlink() and trace.read_bytes() == b"processed"
    assert not (stage / "traces/kernel-6207.trace.xz").exists()


def test_incomplete_run_fails_before_upload(source, tmp_path):
    (source / "trace_alignment.json").unlink()
    with pytest.raises(ValueError, match="incomplete run dir; missing: .*trace_alignment.json"):
        stage_run(source, tmp_path)


def test_seed_uploads_metadata_launches_all_configs_and_skips_pending(source, capsys):
    manifest = json.loads(MANIFEST.read_text())
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"state": "ready"})
        return httpx.Response(201, json={"id": "run-sample"})

    sources = {e["id"]: source / "missing" for e in manifest["entries"]}
    sources["e2e-plan-v1"] = source
    with httpx.Client(base_url="http://app", transport=httpx.MockTransport(handler)) as api:
        with patch("scripts.seed_samples.subprocess.run") as upload:
            assert seed(api, manifest, sources, "http://s3", 1) == [{"id": "run-sample"}]
    command = upload.call_args.args[0]
    for value in [
        "sample_set=cascade-samples",
        "real_gpu_config=SM90_H100",
        'sample_sim_configs=["SM90_H100", "SM90_H200"]',
        'sample_pending_configs=["SM80_A100"]',
        "trace_gpu_arch=sm_90",
        "model=facebook/opt-125m",
        "gpu=NVIDIA H100 80GB HBM3",
    ]:
        assert value in command
    assert json.loads(requests[-1].content) == {
        "trace_id": "e2e-plan-v1",
        "entries": [{"gpu_config": c, "count": 1} for c in ["SM90_H100", "SM90_H200"]],
    }
    assert "SKIP e2e-a100-plan-v1: pending; run dir missing" in capsys.readouterr().out


def test_pending_sample_is_used_when_present(source):
    manifest = json.loads(MANIFEST.read_text())
    manifest["entries"] = [manifest["entries"][1]]
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"state": "ready"} if request.method == "GET" else {"id": "a100"}
        )
    )
    with httpx.Client(base_url="http://app", transport=transport) as api:
        with patch("scripts.seed_samples.subprocess.run"):
            assert seed(api, manifest, {"e2e-a100-plan-v1": source}, "http://s3", 1)


@pytest.mark.parametrize("state", ["failed", "uploading"])
def test_wait_ready_errors(state):
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"state": state, "error": "broken trace"})
    )
    with httpx.Client(base_url="http://app", transport=transport) as api:
        with pytest.raises((RuntimeError, TimeoutError), match="failed: broken trace|timed out"):
            wait_ready(api, "sample", 0)


def test_missing_samples_do_not_upload_or_launch(tmp_path, capsys):
    manifest = json.loads(MANIFEST.read_text())
    with httpx.Client(base_url="http://app") as api:
        with patch("scripts.seed_samples.subprocess.run") as upload:
            assert (
                seed(
                    api,
                    manifest,
                    {e["id"]: tmp_path / e["id"] for e in manifest["entries"]},
                    "http://s3",
                    1,
                )
                == []
            )
            upload.assert_not_called()
    assert capsys.readouterr().out.count("SKIP") == 2
