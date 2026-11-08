from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import threading

import uvicorn

from cascade.api import create_app
from cascade.settings import Settings
from cascade.sim.cli import add_commands

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
    settings = Settings.from_env()
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="cascade")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="run main: API, UI, trace ingest, and rclone S3")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--s3-port", type=int, default=9000, help="port for rclone serve s3")
    s.set_defaults(func=serve)
    add_commands(sub)
    args = p.parse_args()
    if args.func is serve:
        args.func(args)
        return
    try:
        args.func(args)
    except (ValueError, KeyError, OSError) as error:
        p.error(str(error))
