from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import get_args

import uvicorn

from cascade import deploy as workspaces
from cascade.api import create_app
from cascade.settings import LauncherKind, Settings
from cascade.sim.cli import add_commands
from cascade.worker import add_command as add_worker

log = logging.getLogger("cascade")


class RcloneSupervisor:
    """Keeps `rclone serve s3` running over $DATA/s3; restarts it if it exits."""

    def __init__(self, settings: Settings, addr: str):
        self.settings = settings
        self.addr = addr
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def command(self) -> list[str]:
        # No --auth-key: anonymous access until the profiler signs requests.
        exe = shutil.which("rclone")
        if exe is None:
            raise SystemExit("rclone is not on PATH (set CASCADE_RUN_S3=false to skip it)")
        return [
            exe,
            "serve",
            "s3",
            str(self.settings.s3_root),
            "--addr",
            self.addr,
            "--force-path-style",
        ]

    def start(self) -> None:
        (self.settings.s3_root / self.settings.s3_bucket).mkdir(parents=True, exist_ok=True)
        cmd = self.command()
        self._thread = threading.Thread(target=self._run, args=(cmd,), name="rclone", daemon=True)
        self._thread.start()

    def _run(self, cmd: list[str]) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            log.info("starting %s", " ".join(cmd))
            # Own session, so Ctrl-C reaches only main, which then stops rclone itself.
            self._proc = subprocess.Popen(cmd, start_new_session=True)
            code = self._proc.wait()
            if self._stop.is_set():
                return
            log.error("rclone exited with %s; restarting in %.0fs", code, backoff)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def serve(args: argparse.Namespace) -> None:
    settings = Settings.from_env(launcher=args.sims)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    rclone = RcloneSupervisor(settings, f":{args.s3_port}") if settings.run_s3 else None
    if rclone:
        rclone.start()
    try:
        # Open SSE streams never finish on their own, so bound the graceful shutdown.
        uvicorn.run(
            create_app(settings),
            host=args.host,
            port=args.port,
            log_level="info",
            timeout_graceful_shutdown=5,
        )
    finally:
        if rclone:
            rclone.stop()


PHASE_TEXT = {
    workspaces.Phase.creating_sandbox: "creating the sandbox (builds the image if needed)",
    workspaces.Phase.starting: "starting Cascade",
    workspaces.Phase.healthy: "healthy",
}


def deploy(args: argparse.Namespace) -> None:
    api_key = os.environ.get("SAIL_API_KEY", "")
    if not api_key:
        raise SystemExit("SAIL_API_KEY is required")
    if args.base_image and not args.from_source:
        raise SystemExit("--base-image only applies with --from-source")
    root = Path(__file__).resolve().parent.parent
    image = (
        workspaces.FromSource(root, args.base_image or "ubuntu:22.04")
        if args.from_source
        else workspaces.FromRegistry(args.image)
    )
    shown: list[workspaces.Phase] = []

    def report(info: workspaces.WorkspaceInfo) -> None:
        if not shown or shown[-1] is not info.phase:
            shown.append(info.phase)
            print(f"{info.app}: {PHASE_TEXT.get(info.phase, info.phase.value)}", flush=True)

    simulator = workspaces.Simulator(args.sim_cmd, args.accelsim_dir, args.postprocess_cmd)
    try:
        info = workspaces.deploy_workspace(
            api_key,
            args.workspace,
            image,
            simulator=simulator,
            token=os.environ.get("CASCADE_TOKEN") or None,
            disk_gib=args.disk_gib,
            on_phase=report,
        )
    except workspaces.DeployError as error:
        print(f"cascade deploy: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"sailbox:  {info.sailbox_id}")
    print(f"app:      {info.app_url}")
    print(
        f"s3:       {info.s3_url}  (bucket cascade, prefix accelsim-traces, path-style, unsigned)"
    )
    print(f"token:    {info.token}")
    print(f"open:     {info.app_url}/#token={info.token}")
    print(f"logs:     {workspaces.SERVE_LOG} inside the sailbox")


def add_deploy(sub) -> None:
    d = sub.add_parser(
        "deploy",
        help="create or resume a workspace on Sail and start the app in it",
        description="Creates the Sail app cascade-<workspace> with the sailbox cascade-main, or "
        "resumes it, starts `cascade serve --sims sail` in it, and prints its URLs and token. "
        "Reads SAIL_API_KEY, and CASCADE_TOKEN if set. Safe to rerun.",
    )
    d.add_argument("--workspace", required=True, help="3-32 lowercase letters, digits, hyphens")
    source = d.add_mutually_exclusive_group()
    source.add_argument("--image", default=workspaces.DEFAULT_IMAGE, help="prebuilt app image")
    source.add_argument(
        "--from-source",
        action="store_true",
        help="build this checkout's Dockerfile on Sail instead of pulling --image",
    )
    d.add_argument("--base-image", default="", help="Dockerfile BASE_IMAGE (--from-source)")
    d.add_argument("--sim-cmd", default=workspaces.STAND_IN_SIM_CMD, help="CASCADE_SIM_CMD")
    d.add_argument("--accelsim-dir", default=workspaces.STAND_IN_ACCELSIM_DIR)
    d.add_argument("--postprocess-cmd", default="", help="CASCADE_POSTPROCESS_CMD override")
    d.add_argument("--disk-gib", type=int, default=None, help="disk limit for the sailbox")
    d.set_defaults(func=deploy)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="run main: API, UI, trace ingest, and rclone S3")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--s3-port", type=int, default=9000, help="port for rclone serve s3")
    s.add_argument(
        "--sims",
        choices=get_args(LauncherKind),
        default="local",
        help="where sims run: local subprocesses, or one Sail fork per sim",
    )
    s.set_defaults(func=serve)
    add_deploy(sub)
    add_worker(sub)
    add_commands(sub)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = build_parser()
    args = p.parse_args(argv)
    if args.func is serve:
        args.func(args)
        return
    try:
        args.func(args)
    except (ValueError, KeyError, OSError) as error:
        p.error(str(error))
