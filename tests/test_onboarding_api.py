"""landing/api/workspaces.py, the Vercel function behind the Get started page."""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "landing" / "api"
KEY = "sk_live_do_not_echo_42"


def load():
    spec = importlib.util.spec_from_file_location("workspaces_fn", API / "workspaces.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fn(monkeypatch):
    module = load()
    calls = []

    def advance(key, name, image, budget_s):
        calls.append(("advance", key, name, image, budget_s))
        module.deploy.app_name(name)
        return module.deploy.WorkspaceInfo(name, f"cascade-{name}", module.deploy.Phase.starting)

    def delete(key, name):
        calls.append(("delete", key, name))
        return 1

    monkeypatch.setattr(module.deploy, "advance", advance)
    monkeypatch.setattr(module.deploy, "delete", delete)
    module.calls = calls
    return module


def post(fn, **body):
    status, payload = fn.handle(json.dumps(body).encode())
    assert KEY not in json.dumps(payload)
    return status, payload


def test_vendored_deploy_module_matches_cascade_deploy():
    vendored = (API / "_cascade_deploy.py").read_text()
    source = (ROOT / "cascade" / "deploy.py").read_text()
    assert vendored.split("\n", 1)[1] == source, "run uv run python landing/docs-src/build.py"


def test_deploy_steps_with_the_published_image_by_default(fn):
    status, payload = post(fn, action="deploy", api_key=f"  {KEY} ", workspace="acme")
    assert status == 200 and payload["phase"] == "starting" and payload["app"] == "cascade-acme"
    [(_, key, name, image, budget)] = fn.calls
    assert key == KEY and name == "acme" and budget == fn.STEP_BUDGET_S
    assert image == fn.deploy.FromRegistry(fn.deploy.DEFAULT_IMAGE)
    post(fn, action="deploy", api_key=KEY, workspace="acme", image="ghcr.io/me/cascade:1")
    assert fn.calls[-1][3] == fn.deploy.FromRegistry("ghcr.io/me/cascade:1")


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        (b"not json", 400),
        (b"[1, 2]", 400),
        (json.dumps({"action": "deploy", "workspace": "acme"}).encode(), 401),
        (json.dumps({"action": "deploy", "api_key": " ", "workspace": "acme"}).encode(), 401),
        (json.dumps({"action": "explode", "api_key": KEY, "workspace": "acme"}).encode(), 400),
        (json.dumps({"action": "deploy", "api_key": 7, "workspace": "acme"}).encode(), 400),
        (json.dumps({"action": "deploy", "api_key": KEY, "workspace": ["acme"]}).encode(), 400),
    ],
)
def test_request_validation(fn, raw, status):
    got, payload = fn.handle(raw)
    assert got == status and payload["error"] and KEY not in payload["error"]
    assert fn.calls == []


def test_bad_workspace_name_is_400(fn):
    status, payload = post(fn, action="deploy", api_key=KEY, workspace="Not OK")
    assert status == 400 and "lowercase" in payload["error"]


def test_delete_needs_the_name_typed_again(fn):
    assert post(fn, action="delete", api_key=KEY, workspace="acme")[0] == 400
    assert post(fn, action="delete", api_key=KEY, workspace="acme", confirm="acm")[0] == 400
    assert fn.calls == []
    status, payload = post(fn, action="delete", api_key=KEY, workspace="acme", confirm="acme")
    assert status == 200 and payload == {"workspace": "acme", "terminated": 1}


def test_deploy_errors_keep_their_status_and_message(fn, monkeypatch):
    def rejected(*a, **kw):
        raise fn.deploy.KeyRejected("Sail rejected this API key.")

    monkeypatch.setattr(fn.deploy, "advance", rejected)
    assert post(fn, action="deploy", api_key=KEY, workspace="acme") == (
        401,
        {"error": "Sail rejected this API key."},
    )


def test_unexpected_errors_echo_nothing(fn, monkeypatch, capsys):
    def boom(*a, **kw):
        raise RuntimeError(f"exploded with {KEY}")

    monkeypatch.setattr(fn.deploy, "advance", boom)
    status, payload = post(fn, action="deploy", api_key=KEY, workspace="acme")
    assert status == 500
    out = capsys.readouterr()
    assert KEY not in out.out + out.err and "RuntimeError" in out.err


def test_http_handler_speaks_json_and_post_only(fn):
    server = HTTPServer(("127.0.0.1", 0), fn.handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/api/workspaces"
    try:
        body = json.dumps({"action": "deploy", "api_key": KEY, "workspace": "acme"}).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as res:
            assert res.headers["Cache-Control"] == "no-store"
            assert json.load(res)["phase"] == "starting"
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(url)
        assert caught.value.code == 405 and caught.value.headers["Allow"] == "POST"
    finally:
        server.shutdown()
