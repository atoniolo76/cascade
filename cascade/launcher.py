"""Starts and stops sims. Both launchers run $CASCADE_SIM_CMD with the sim env from docs/api.md."""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import threading
from pathlib import Path
from typing import Protocol

from cascade.models import SimSummary
from cascade.settings import Settings

log = logging.getLogger(__name__)


class LaunchError(Exception):
    pass


class Launcher(Protocol):
    def prepare(self, run_id: str) -> None:
        """Called once per run before any of its sims launch. Must be idempotent."""

    def launch(self, sim: SimSummary, trace_dir: Path) -> str:
        """Starts the sim and returns its launcher_ref."""

    def terminate(self, sim: SimSummary) -> None:
        """Stops the sim if it is still running. Must be idempotent."""


def sim_env(settings: Settings, sim: SimSummary, trace_dir: Path) -> dict[str, str]:
    return {
        "CASCADE_SIM_ID": sim.id,
        "CASCADE_MAIN_URL": settings.public_url,
        "CASCADE_TOKEN": settings.token,
        "CASCADE_GPU_CONFIG": sim.gpu_config,
        "CASCADE_TRACE_DIR": str(trace_dir),
    }


def _sim_cmd(settings: Settings) -> str:
    if not settings.sim_cmd:
        raise LaunchError("CASCADE_SIM_CMD is not set, so there is no simulator to run")
    return settings.sim_cmd


class LocalLauncher:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def prepare(self, run_id: str) -> None:
        pass

    def launch(self, sim: SimSummary, trace_dir: Path) -> str:
        argv = shlex.split(_sim_cmd(self.settings))
        out_dir = self.settings.work_dir / "sims"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{sim.id}.out").open("wb") as out:
            try:
                proc = subprocess.Popen(
                    argv,
                    env={**os.environ, **sim_env(self.settings, sim, trace_dir)},
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as e:
                raise LaunchError(f"could not start {argv[0]!r}: {e}") from e
        with self._lock:
            self._procs[sim.id] = proc
        return str(proc.pid)

    def terminate(self, sim: SimSummary) -> None:
        with self._lock:
            proc = self._procs.pop(sim.id, None)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


class SailLauncher:
    """One checkpoint of main per run, shared by its sims; each sim is a fresh-boot fork of it.

    Written against the sail SDK 0.17.2 source; not exercised against the real service.
    """

    def __init__(self, settings: Settings, sail_module=None):
        if sail_module is None:
            import sail as sail_module
        self.sail = sail_module
        self.settings = settings
        self._checkpoints: dict[str, str] = {}
        self._lock = threading.Lock()

    def _checkpoint_name(self, run_id: str) -> str:
        return f"cascade-{run_id}"

    def prepare(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._checkpoints:
                return
            name = self._checkpoint_name(run_id)
            existing = self.sail.SailboxCheckpoint.list(name=name)
            if existing:
                checkpoint_id = existing[0].checkpoint_id
            else:
                # current_sailbox_id reads the identity Sail maintains across forks; SAILBOX_ID
                # in a long-lived process can name the parent.
                me = self.sail.Sailbox.from_id(self.sail.guest.current_sailbox_id())
                checkpoint_id = me.checkpoint(name=name).checkpoint_id
            self._checkpoints[run_id] = checkpoint_id

    def launch(self, sim: SimSummary, trace_dir: Path) -> str:
        cmd = _sim_cmd(self.settings)
        self.prepare(sim.run_id)
        box = self.sail.Sailbox.from_checkpoint(
            self._checkpoints[sim.run_id], name=f"cascade-{sim.id}", restore_memory=False
        )
        out = self.settings.work_dir / "sims" / f"{sim.id}.out"
        # Background exec discards output, so keep a copy on the fork's disk for debugging.
        shell = f"mkdir -p {shlex.quote(str(out.parent))} && ({cmd}) > {shlex.quote(str(out))} 2>&1"
        try:
            box.exec(
                shell,
                background=True,
                env=sim_env(self.settings, sim, trace_dir),
                idempotency_key=sim.id,
            )
        except BaseException:
            box.terminate()
            raise
        return box.sailbox_id

    def terminate(self, sim: SimSummary) -> None:
        if sim.launcher_ref:
            self.sail.Sailbox.from_id(sim.launcher_ref).terminate()


def make_launcher(settings: Settings) -> Launcher:
    return SailLauncher(settings) if settings.launcher == "sail" else LocalLauncher(settings)
