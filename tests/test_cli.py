import pytest

from cascade import cli
from cascade.launcher import LocalLauncher, make_launcher
from tests.helpers import make_settings


@pytest.fixture
def served(monkeypatch, tmp_path):
    """Runs `cascade serve` without a server and returns the Settings it built."""
    monkeypatch.setenv("CASCADE_TOKEN", "t")
    monkeypatch.setenv("CASCADE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CASCADE_RUN_S3", "false")
    built = []
    monkeypatch.setattr(cli, "create_app", built.append)
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kw: None)

    def serve(*args: str):
        cli.main(["serve", *args])
        return built[-1]

    return serve


def test_serve_runs_sims_locally_by_default(served):
    assert served().launcher == "local"


def test_serve_sims_sail(served):
    assert served("--sims", "sail").launcher == "sail"


def test_serve_rejects_unknown_sims(served, capsys):
    with pytest.raises(SystemExit):
        served("--sims", "cloud")
    assert "invalid choice: 'cloud'" in capsys.readouterr().err


def test_launcher_env_var_is_gone(served, monkeypatch):
    monkeypatch.setenv("CASCADE_LAUNCHER", "sail")
    assert served().launcher == "local"


def test_local_settings_make_a_local_launcher(tmp_path):
    assert isinstance(make_launcher(make_settings(tmp_path)), LocalLauncher)


@pytest.fixture
def served_with_s3(served, monkeypatch):
    monkeypatch.setenv("CASCADE_RUN_S3", "true")
    monkeypatch.setattr(cli, "RcloneSupervisor", lambda settings, addr: _NoRclone())
    return served


class _NoRclone:
    def start(self):
        pass

    def stop(self):
        pass


def test_serve_derives_the_s3_endpoint_from_a_local_public_url(served_with_s3, monkeypatch):
    monkeypatch.setenv("CASCADE_PUBLIC_URL", "http://127.0.0.1:8030")
    assert served_with_s3("--s3-port", "9030").s3_public_url == "http://127.0.0.1:9030"


def test_serve_keeps_an_explicit_s3_endpoint(served_with_s3, monkeypatch):
    monkeypatch.setenv("CASCADE_PUBLIC_URL", "http://127.0.0.1:8030")
    monkeypatch.setenv("CASCADE_S3_PUBLIC_URL", "https://s3.example.com")
    assert served_with_s3().s3_public_url == "https://s3.example.com"


def test_serve_leaves_the_s3_endpoint_unknown_behind_a_portless_url(served_with_s3, monkeypatch):
    monkeypatch.setenv("CASCADE_PUBLIC_URL", "https://sb-1-8000.sail.box")
    assert served_with_s3().s3_public_url is None
