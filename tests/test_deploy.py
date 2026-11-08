"""cascade.deploy against an in-memory stand-in for the sail SDK.

Every call is bound against the installed SDK's real signature, so a renamed or missing
parameter fails here. Nothing talks to the Sail service.
"""

import inspect
import os
import shlex
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from cascade import deploy
from cascade.deploy import Phase

sail = pytest.importorskip("sail")

KEY = "sk_test_secret_key_123"


def bind(real, *args, **kwargs):
    inspect.signature(real).bind(*args, **kwargs)


@dataclass
class Box:
    fake: "FakeSail"
    sailbox_id: str
    app_id: str
    name: str
    status: str = "running"
    files: dict = field(default_factory=dict)
    serving: bool = False
    probes: int = 0

    def listener(self, port, **kw):
        bind(sail.Sailbox.listener, self, port, **kw)
        url = f"https://{self.sailbox_id}-{port}.sail.box/"
        return SimpleNamespace(endpoint=sail.HttpEndpoint(url))

    def run(self, command, **kw):
        bind(sail.Sailbox.run, self, command, **kw)
        self.fake.calls.append(("run", command))
        ok = sail.ExecResult(stdout="", stderr="", exit_code=0)
        bad = sail.ExecResult(stdout="", stderr="", exit_code=1)
        if command == f"test -s {deploy.ENV_FILE}":
            return ok if deploy.ENV_FILE in self.files else bad
        if command.startswith("ln "):
            tmp, target = command.split()[1:3]
            self.files.setdefault(target, self.files[tmp])
            del self.files[tmp]
            return ok
        if command == deploy._SERVING:
            return ok if self.serving else bad
        if command.endswith("printenv CASCADE_TOKEN"):
            env = dict(
                line.removeprefix("export ").split("=", 1)
                for line in self.files.get(deploy.ENV_FILE, "").splitlines()
            )
            token = shlex.split(env["CASCADE_TOKEN"])[0] if "CASCADE_TOKEN" in env else ""
            return sail.ExecResult(stdout=token + "\n", stderr="", exit_code=0)
        raise AssertionError(f"unexpected command {command!r}")

    def exec(self, command, **kw):
        bind(sail.Sailbox.exec, self, command, **kw)
        self.fake.calls.append(("exec", command, kw))
        self.serving = True

    def pause(self):
        bind(sail.Sailbox.pause, self)
        self.fake.calls.append(("pause", self.sailbox_id))
        self.status = "paused"

    def resume(self):
        bind(sail.Sailbox.resume, self)
        self.fake.calls.append(("resume", self.sailbox_id))
        self.status = "running"
        return self

    def terminate(self):
        bind(sail.Sailbox.terminate, self)
        self.fake.calls.append(("terminate", self.sailbox_id))
        self.status = "terminated"

    @property
    def fs(self):
        box = self

        class Fs:
            def write(self, path, data, **kw):
                bind(sail.SailboxFs.write, self, path, data, **kw)
                assert kw["mode"] == 0o600
                box.files[path] = data

        return Fs()


class FakeSail:
    SailError = sail.SailError
    PermissionDeniedError = sail.PermissionDeniedError
    NotFoundError = sail.NotFoundError
    InvalidArgumentError = sail.InvalidArgumentError
    TimeoutError = sail.TimeoutError
    AutoSleep = sail.AutoSleep

    def __init__(self):
        self.calls: list[tuple] = []
        self.apps: dict[str, SimpleNamespace] = {}
        self.boxes: list[Box] = []
        self.checkpoints: list[str] = []
        self.keys_seen: list[str | None] = []
        self.build_times_out = False
        self.create_error: Exception | None = None
        fake = self

        class App:
            @staticmethod
            def find(name, **kw):
                bind(sail.App.find, name, **kw)
                if os.environ.get("SAIL_API_KEY") != KEY:
                    raise sail.PermissionDeniedError(
                        f"Failed to find app: Invalid API key {os.environ['SAIL_API_KEY']}"
                    )
                if name not in fake.apps:
                    if not kw.get("mint_if_missing"):
                        raise sail.NotFoundError(f'App "{name}" not found')
                    fake.calls.append(("mint", name))
                    fake.apps[name] = SimpleNamespace(id=f"app-{name}", name=name)
                return fake.apps[name]

        class Definition:
            def __init__(self, source):
                self.source = source

            def build(self, **kw):
                bind(sail.ImageDefinition.build, self, **kw)
                fake.calls.append(("build", self.source, kw["timeout"]))
                if fake.build_times_out:
                    raise sail.TimeoutError("image build did not finish")
                return self

        class Image:
            @staticmethod
            def from_registry(ref, **kw):
                bind(sail.Image.from_registry, ref, **kw)
                return Definition(ref)

            @staticmethod
            def from_dockerfile(dockerfile, **kw):
                bind(sail.Image.from_dockerfile, dockerfile, **kw)
                return Definition(("dockerfile", dockerfile, kw["build_args"]["BASE_IMAGE"]))

        class Sailbox:
            @staticmethod
            def list(**kw):
                bind(sail.Sailbox.list, **kw)
                app_id = kw["app_id"].id
                return [
                    b
                    for b in fake.boxes
                    if b.app_id == app_id
                    and b.status in kw["statuses"]
                    and kw.get("name") in (None, b.name)
                ]

            @staticmethod
            def create(**kw):
                bind(sail.Sailbox.create, **kw)
                fake.calls.append(("create", kw))
                if fake.create_error is not None:
                    raise fake.create_error
                box = Box(fake, f"sb-{len(fake.boxes)}", kw["app"].id, kw["name"])
                fake.boxes.append(box)
                return box

        class SailboxCheckpoint:
            @staticmethod
            def list(**kw):
                bind(sail.SailboxCheckpoint.list, **kw)
                return [
                    SimpleNamespace(delete=lambda c=c: fake.checkpoints.remove(c))
                    for c in fake.checkpoints
                ]

        self.App, self.Image, self.Sailbox = App, Image, Sailbox
        self.SailboxCheckpoint = SailboxCheckpoint

    def reset_transports(self):
        bind(sail.reset_transports)
        self.keys_seen.append(os.environ.get("SAIL_API_KEY"))

    def creates(self):
        return [c for c in self.calls if c[0] == "create"]


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.delenv("SAIL_API_KEY", raising=False)
    fake = FakeSail()

    def health(url, timeout_s=5.0):
        box = next(b for b in fake.boxes if b.sailbox_id in url)
        if not box.serving or box.status != "running":
            return None
        box.probes += 1
        return {"ok": True} if box.probes > 1 else None

    monkeypatch.setattr(deploy, "probe_health", health)
    return fake


def run_to_healthy(fake, workspace="acme", **kw):
    seen = []
    for _ in range(10):
        info = deploy.advance(KEY, workspace, sail_module=fake, **kw)
        seen.append(info.phase)
        if info.phase is Phase.healthy:
            return info, seen
    raise AssertionError(f"never healthy: {seen}")


def test_first_deploy_creates_one_app_and_sailbox_then_starts_cascade(fake):
    info, phases = run_to_healthy(fake)
    assert phases == [
        Phase.creating_sandbox,
        Phase.starting,
        Phase.starting,
        Phase.starting,
        Phase.healthy,
    ]
    assert info.app == "cascade-acme" and info.sailbox_id == "sb-0"
    assert info.app_url == "https://sb-0-8000.sail.box"
    assert info.s3_url == "https://sb-0-9000.sail.box"
    assert info.token and len(info.token) >= 24
    [(_, create)] = fake.creates()
    assert create["name"] == "cascade-main" and create["ingress_ports"] == [8000, 9000]
    assert create["auto_sleep"] == sail.AutoSleep.never()
    assert [c[1] for c in fake.calls if c[0] == "build"] == [deploy.DEFAULT_IMAGE]
    env = fake.boxes[0].files[deploy.ENV_FILE]
    assert "export CASCADE_WORKSPACE=acme\n" in env
    assert f"export SAIL_API_KEY={KEY}\n" in env
    assert "export CASCADE_PUBLIC_URL=https://sb-0-8000.sail.box\n" in env
    assert "export CASCADE_S3_PUBLIC_URL=https://sb-0-9000.sail.box\n" in env
    [serve] = [c for c in fake.calls if c[0] == "exec"]
    assert serve[2] == {"background": True}, "a second start while warming up would race it"
    assert KEY not in serve[1] and "env" not in serve[2]


def test_rerun_on_a_healthy_workspace_recovers_the_same_token(fake):
    first, _ = run_to_healthy(fake)
    again = deploy.advance(KEY, "acme", sail_module=fake)
    assert again.phase is Phase.healthy and again.token == first.token
    assert len(fake.creates()) == 1
    assert len([c for c in fake.calls if c[0] == "exec"]) == 1


def test_paused_workspace_is_resumed_not_duplicated(fake):
    first, _ = run_to_healthy(fake)
    fake.boxes[0].status = "paused"
    info = deploy.advance(KEY, "acme", sail_module=fake)
    assert ("resume", "sb-0") in fake.calls
    assert info.phase is Phase.healthy and info.token == first.token
    assert len(fake.creates()) == 1


def test_cold_resume_restarts_cascade_with_the_existing_env(fake):
    first, _ = run_to_healthy(fake)
    box = fake.boxes[0]
    box.status, box.serving, box.probes = "paused", False, 0
    info, _ = run_to_healthy(fake)
    assert info.token == first.token
    assert len([c for c in fake.calls if c[0] == "exec"]) == 2
    assert len(fake.creates()) == 1


def test_build_timeout_returns_and_the_next_call_creates(fake):
    fake.build_times_out = True
    assert deploy.advance(KEY, "acme", sail_module=fake).phase is Phase.creating_sandbox
    info = deploy.advance(KEY, "acme", sail_module=fake, budget_s=7)
    assert info.phase is Phase.creating_sandbox and info.sailbox_id is None
    assert ("build", deploy.DEFAULT_IMAGE, 7) in fake.calls
    fake.build_times_out = False
    assert deploy.advance(KEY, "acme", sail_module=fake).phase is Phase.starting
    assert len(fake.creates()) == 1


def test_from_source_builds_the_dockerfile(fake, tmp_path):
    image = deploy.FromSource(tmp_path, "my/base:1")
    deploy.advance(KEY, "acme", image, sail_module=fake)
    deploy.advance(KEY, "acme", image, sail_module=fake)
    [(_, source, _)] = [c for c in fake.calls if c[0] == "build"]
    assert source == ("dockerfile", tmp_path / "Dockerfile", "my/base:1")


@pytest.mark.parametrize("name", ["ab", "a" * 33, "Acme", "acme_lab", "acme lab", "acme.lab", ""])
def test_workspace_names_are_validated_before_sail_is_called(fake, name):
    with pytest.raises(deploy.InvalidWorkspace):
        deploy.advance(KEY, name, sail_module=fake)
    assert fake.calls == [] and fake.keys_seen == []


@pytest.mark.parametrize("name", ["abc", "a" * 32, "acme-lab-2", "123"])
def test_valid_workspace_names(name):
    assert deploy.app_name(name) == f"cascade-{name}"


def test_bad_key_is_a_clear_error_without_the_key(fake):
    with pytest.raises(deploy.KeyRejected) as caught:
        deploy.advance("sk_wrong_key_999", "acme", sail_module=fake)
    assert str(caught.value) == "Sail rejected this API key."
    assert caught.value.http_status == 401
    assert caught.value.__suppress_context__


def test_quota_errors_are_verbatim_and_scrubbed(fake):
    fake.create_error = sail.SailboxCreationError(
        f"quota exceeded: 10 sailboxes for key {KEY}", status_code=429
    )
    deploy.advance(KEY, "acme", sail_module=fake)
    with pytest.raises(deploy.SailRefused) as caught:
        deploy.advance(KEY, "acme", sail_module=fake)
    assert str(caught.value) == "quota exceeded: 10 sailboxes for key [redacted]"
    assert caught.value.http_status == 429


def test_token_never_appears_in_errors(fake, monkeypatch):
    def leaky(box, values):
        raise sail.ApiError(f"write failed: {values['CASCADE_TOKEN']} {values['SAIL_API_KEY']}")

    monkeypatch.setattr(deploy, "_write_env_once", leaky)
    for _ in range(2):
        deploy.advance(KEY, "acme", sail_module=fake, token="tok_abcdefghijklmnop")
    with pytest.raises(deploy.SailRefused) as caught:
        deploy.advance(KEY, "acme", sail_module=fake, token="tok_abcdefghijklmnop")
    assert str(caught.value) == "write failed: [redacted] [redacted]"


def test_info_repr_hides_the_token(fake):
    info, _ = run_to_healthy(fake)
    assert info.token not in repr(info)
    assert info.to_json()["token"] == info.token and info.to_json()["phase"] == "healthy"


def test_key_is_set_only_for_the_call(fake):
    run_to_healthy(fake)
    assert set(fake.keys_seen) == {KEY, None}
    assert "SAIL_API_KEY" not in os.environ


def test_status_reads_without_creating_or_waking(fake):
    assert deploy.workspace_status(KEY, "acme", sail_module=fake).phase is Phase.absent
    assert fake.apps == {}
    run_to_healthy(fake)
    status = deploy.workspace_status(KEY, "acme", sail_module=fake)
    assert status.phase is Phase.healthy and status.token is None
    fake.boxes[0].status = "paused"
    assert deploy.workspace_status(KEY, "acme", sail_module=fake).phase is Phase.paused
    assert ("resume", "sb-0") not in fake.calls


def test_pause_resume_and_delete(fake):
    run_to_healthy(fake)
    assert deploy.pause(KEY, "acme", sail_module=fake).phase is Phase.paused
    assert fake.boxes[0].status == "paused"
    deploy.resume(KEY, "acme", sail_module=fake)
    assert fake.boxes[0].status == "running"
    fork = Box(fake, "sb-fork", "app-cascade-acme", "cascade-sim-1")
    fake.boxes.append(fork)
    fake.checkpoints.append("cp-run-1")
    assert deploy.delete(KEY, "acme", sail_module=fake) == 2
    assert {b.status for b in fake.boxes} == {"terminated"} and fake.checkpoints == []
    assert deploy.delete(KEY, "acme", sail_module=fake) == 0
    with pytest.raises(deploy.WorkspaceMissing):
        deploy.pause(KEY, "acme", sail_module=fake)


def test_deploy_workspace_loops_until_healthy(fake, monkeypatch):
    monkeypatch.setattr(deploy.time, "sleep", lambda s: None)
    phases = []
    info = deploy.deploy_workspace(
        KEY, "acme", sail_module=fake, timeout_s=2, on_phase=lambda i: phases.append(i.phase)
    )
    assert info.phase is Phase.healthy and phases[-1] is Phase.healthy
