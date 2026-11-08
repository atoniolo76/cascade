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
