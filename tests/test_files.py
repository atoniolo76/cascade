"""Output files: PUT /api/sims/{id}/files/{name} and the routes that list and serve them."""

import gzip

import pytest
from fastapi.testclient import TestClient

import cascade.api
from cascade.api import create_app
from cascade.models import FILE_MAX_BYTES
from tests.helpers import AUTH, make_settings, wait_for, write_processed_trace
from tests.test_api import RecordingLauncher


@pytest.fixture
def sims(tmp_path):
    """A run with two SM90_H100 replicas and one SM90_H200 sim, none of them started."""
    settings = make_settings(tmp_path)
    write_processed_trace(settings, "trace-h100", arch="sm_90")
    with TestClient(create_app(settings, RecordingLauncher())) as client:
        client.headers.update(AUTH)
        wait_for(lambda: client.get("/api/traces/trace-h100").json().get("state") == "ready")
        entries = [{"gpu_config": "SM90_H100", "count": 2}, {"gpu_config": "SM90_H200", "count": 1}]
        run = client.post("/api/runs", json={"trace_id": "trace-h100", "entries": entries}).json()
        ids = {(s["gpu_config"], s["replica"]): s["id"] for s in run["sims"]}
        yield client, run["id"], ids, settings


def test_put_stores_the_bytes_and_a_re_put_replaces_them(sims):
    client, run_id, ids, settings = sims
    sim = ids[("SM90_H100", 0)]
    first = client.put(f"/api/sims/{sim}/files/kernel-stats.csv", content=b"uid,name\n1,a\n")
    assert first.status_code == 200
    assert {k: first.json()[k] for k in ("name", "size", "content_type")} == {
        "name": "kernel-stats.csv",
        "size": 13,
        "content_type": "text/csv",
    }
    again = client.put(f"/api/sims/{sim}/files/kernel-stats.csv", content=b"uid\n")
    assert again.json()["size"] == 4
    listed = client.get(f"/api/sims/{sim}/files").json()
    assert [(f["name"], f["size"]) for f in listed] == [("kernel-stats.csv", 4)]
    got = client.get(f"/api/sims/{sim}/files/kernel-stats.csv")
    assert got.content == b"uid\n"
    assert got.headers["content-type"].startswith("text/csv")
    assert got.headers["content-disposition"] == 'attachment; filename="kernel-stats.csv"'
    assert (settings.files_root / sim / "kernel-stats.csv").read_bytes() == b"uid\n"


def test_gz_downloads_as_an_attachment_without_content_encoding(sims):
    client, _, ids, _ = sims
    sim = ids[("SM90_H100", 0)]
    body = gzip.compress(b"x" * 1000)
    client.put(f"/api/sims/{sim}/files/perf_counter.csv.gz", content=body)
    got = client.get(f"/api/sims/{sim}/files/perf_counter.csv.gz")
    assert got.headers["content-type"] == "application/gzip"
    assert "content-encoding" not in got.headers
    assert got.content == body


def test_run_files_carry_their_sim_ordered_by_config_and_replica(sims):
    client, run_id, ids, _ = sims
    for (_, replica), sim in ids.items():
        client.put(f"/api/sims/{sim}/files/exit-status.txt", content=f"{replica}\n".encode())
    client.put(f"/api/sims/{ids[('SM90_H100', 1)]}/files/a.log", content=b"log\n")
    files = client.get(f"/api/runs/{run_id}/files").json()
    assert [(f["gpu_config"], f["replica"], f["name"], f["sim_id"]) for f in files] == [
        ("SM90_H100", 0, "exit-status.txt", ids[("SM90_H100", 0)]),
        ("SM90_H100", 1, "a.log", ids[("SM90_H100", 1)]),
        ("SM90_H100", 1, "exit-status.txt", ids[("SM90_H100", 1)]),
        ("SM90_H200", 0, "exit-status.txt", ids[("SM90_H200", 0)]),
    ]


@pytest.mark.parametrize("name", ["%2E%2E", "%2E", "a b.txt", "x" * 129, "caf\u00e9.txt", "a:b"])
def test_bad_names_are_422_and_store_nothing(sims, name):
    client, _, ids, settings = sims
    resp = client.put(f"/api/sims/{ids[('SM90_H100', 0)]}/files/{name}", content=b"x")
    assert resp.status_code == 422, resp.text
    assert not (settings.files_root / ids[("SM90_H100", 0)]).exists()


def test_names_up_to_128_allowed_characters_are_accepted(sims):
    client, _, ids, _ = sims
    sim = ids[("SM90_H100", 0)]
    for name in ("x" * 128, ".hidden-ok_1", "..."):
        assert client.put(f"/api/sims/{sim}/files/{name}", content=b"x").status_code == 200


def test_unknown_sim_is_404_and_a_missing_token_is_401(sims):
    client, run_id, ids, _ = sims
    assert client.put("/api/sims/sim-000000000000/files/a.txt", content=b"x").status_code == 404
    assert client.get("/api/sims/sim-000000000000/files").status_code == 404
    assert client.get("/api/runs/run-000000000000/files").status_code == 404
    assert client.get("/api/runs/run-000000000000/perfetto").status_code == 404
    assert client.get(f"/api/sims/{ids[('SM90_H100', 0)]}/files/none.txt").status_code == 404
    sim = ids[("SM90_H100", 0)]
    bare = {"Authorization": ""}
    assert client.put(f"/api/sims/{sim}/files/a.txt", content=b"x", headers=bare).status_code == 401
    for path in (
        f"/api/sims/{sim}/files",
        f"/api/runs/{run_id}/files",
        f"/api/runs/{run_id}/perfetto",
    ):
        assert client.get(path, headers=bare).status_code == 401
    assert client.get(f"/api/sims/{sim}/files").json() == []


def test_a_file_over_the_per_file_cap_is_413_and_stores_nothing(sims):
    client, _, ids, settings = sims
    sim = ids[("SM90_H100", 0)]
    resp = client.put(f"/api/sims/{sim}/files/big.bin", content=b"\0" * (FILE_MAX_BYTES + 1))
    assert resp.status_code == 413 and "per-file" in resp.json()["detail"]
    assert client.get(f"/api/sims/{sim}/files").json() == []
    assert list((settings.files_root / ".parts").iterdir()) == []
    exact = client.put(f"/api/sims/{sim}/files/big.bin", content=b"\0" * FILE_MAX_BYTES)
    assert exact.status_code == 200


def test_the_sim_total_cap_is_413_and_counts_a_replaced_file_once(sims, monkeypatch):
    monkeypatch.setattr(cascade.api, "SIM_FILES_MAX_BYTES", 10)
    client, _, ids, _ = sims
    sim, other = ids[("SM90_H100", 0)], ids[("SM90_H100", 1)]
    assert client.put(f"/api/sims/{sim}/files/a", content=b"x" * 6).status_code == 200
    over = client.put(f"/api/sims/{sim}/files/b", content=b"x" * 5)
    assert over.status_code == 413 and "total" in over.json()["detail"]
    assert client.put(f"/api/sims/{sim}/files/a", content=b"x" * 10).status_code == 200
    assert client.put(f"/api/sims/{other}/files/a", content=b"x" * 10).status_code == 200
    assert [f["name"] for f in client.get(f"/api/sims/{sim}/files").json()] == ["a"]
