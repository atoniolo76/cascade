"""SailLauncher against a recording stand-in for the sail SDK.

Every call is also bound against the installed SDK's real signature, so a renamed or missing
parameter fails here. Nothing talks to the Sail service.
"""

import inspect
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from cascade.launcher import SailLauncher
from cascade.models import SimMetrics, SimState, SimSummary
from tests.helpers import make_settings

sail = pytest.importorskip("sail")


def bind(real, *args, **kwargs):
    inspect.signature(real).bind(*args, **kwargs)


@dataclass
class Checkpoint:
    checkpoint_id: str


class FakeSail:
    def __init__(self, existing=()):
        self.calls: list[tuple] = []
        self.existing = list(existing)
        fake = self

        class Box:
            def __init__(self, sailbox_id):
                self.sailbox_id = sailbox_id

            @classmethod
            def from_id(cls, sailbox_id):
                bind(sail.Sailbox.from_id, sailbox_id)
                return cls(sailbox_id)

            @classmethod
            def from_checkpoint(cls, checkpoint_id, **kw):
                bind(sail.Sailbox.from_checkpoint, checkpoint_id, **kw)
                fake.calls.append(("from_checkpoint", checkpoint_id, kw))
                return cls(f"sb-{kw['name']}")

            def checkpoint(self, **kw):
                bind(sail.Sailbox.checkpoint, self, **kw)
                fake.calls.append(("checkpoint", self.sailbox_id, kw))
                return Checkpoint("cp-new")

            def exec(self, command, **kw):
                bind(sail.Sailbox.exec, self, command, **kw)
                fake.calls.append(("exec", self.sailbox_id, command, kw))

            def terminate(self):
                bind(sail.Sailbox.terminate, self)
                fake.calls.append(("terminate", self.sailbox_id))

        class Checkpoints:
            @staticmethod
            def list(**kw):
                bind(sail.SailboxCheckpoint.list, **kw)
                return [Checkpoint(cid) for name, cid in fake.existing if kw["name"] == name]

        def current_sailbox_id():
            bind(sail.guest.current_sailbox_id)
            return "sb-main"

        self.Sailbox = Box
        self.SailboxCheckpoint = Checkpoints
        self.guest = SimpleNamespace(current_sailbox_id=current_sailbox_id)


def sim(sim_id, run_id="run-1", ref=None):
    t = datetime.now(UTC)
    return SimSummary(
        id=sim_id,
        run_id=run_id,
        gpu_config="SM90_H100",
        replica=0,
        state=SimState.launching,
        error=None,
        launcher_ref=ref,
        created_at=t,
        started_at=None,
        finished_at=None,
        last_heartbeat_at=None,
        exit_code=None,
        kernels_done=0,
        kernels_total=1,
        current_kernel=None,
        latest=SimMetrics(),
        log_tail=[],
    )


def test_one_checkpoint_per_run_and_fresh_boot_forks(tmp_path):
    settings = make_settings(
        tmp_path, CASCADE_SIM_CMD="simulate --fast", CASCADE_PUBLIC_URL="https://main.sail.box"
    )
    fake = FakeSail()
    launcher = SailLauncher(settings, fake)
    launcher.prepare("run-1")
    launcher.prepare("run-1")
    refs = [launcher.launch(sim(i), settings.trace_dir("t")) for i in ("sim-a", "sim-b")]
    assert refs == ["sb-cascade-sim-a", "sb-cascade-sim-b"]
    checkpoints = [c for c in fake.calls if c[0] == "checkpoint"]
    assert checkpoints == [("checkpoint", "sb-main", {"name": "cascade-run-1"})]
    forks = [c for c in fake.calls if c[0] == "from_checkpoint"]
    assert [(c[1], c[2]["restore_memory"]) for c in forks] == [("cp-new", False)] * 2
    execs = [c for c in fake.calls if c[0] == "exec"]
    _, box, command, kw = execs[0]
    assert box == "sb-cascade-sim-a" and "(simulate --fast)" in command
    assert kw["background"] is True and kw["idempotency_key"] == "sim-a"
    assert kw["env"] == {
        "CASCADE_SIM_ID": "sim-a",
        "CASCADE_MAIN_URL": "https://main.sail.box",
        "CASCADE_TOKEN": settings.token,
        "CASCADE_GPU_CONFIG": "SM90_H100",
        "CASCADE_TRACE_DIR": str(settings.trace_dir("t")),
    }


def test_existing_checkpoint_is_reused_after_restart(tmp_path):
    settings = make_settings(tmp_path, CASCADE_SIM_CMD="simulate")
    fake = FakeSail(existing=[("cascade-run-1", "cp-old")])
    SailLauncher(settings, fake).launch(sim("sim-a"), settings.trace_dir("t"))
    assert not [c for c in fake.calls if c[0] == "checkpoint"]
    assert [c[1] for c in fake.calls if c[0] == "from_checkpoint"] == ["cp-old"]


def test_terminate_uses_the_recorded_sailbox(tmp_path):
    fake = FakeSail()
    launcher = SailLauncher(make_settings(tmp_path), fake)
    launcher.terminate(sim("sim-a", ref="sb-123"))
    launcher.terminate(sim("sim-b", ref=None))
    assert fake.calls == [("terminate", "sb-123")]
