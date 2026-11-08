"""`cascade worker`: the process each sim runs (docs/api.md, "Simulator -> main").

It runs Accel-Sim on the sim's trace, or replays a recorded Accel-Sim log with `--replay`, and
streams stdout to main in numbered chunks while heartbeating. When the sim succeeds on a trace with
step timing, it runs Alessio's pipeline (cascade/sim/) on its own log and posts the result. Before
its final heartbeat it uploads the files the simulator wrote into its run dir, and its own log.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from cascade.gpu_configs import HW_CONFIGS_REL, GpuConfigRegistry
from cascade.ingest import has_step_timeline
from cascade.models import FILE_MAX_BYTES, FILE_NAME_PATTERN, SIM_FILES_MAX_BYTES
from cascade.sim.compare import compare
from cascade.sim.importer import import_durations
from cascade.sim.timeline import load_timeline, parse_durations, simulate

BATCH_LINES = 200
BATCH_SECONDS = 0.5
HEARTBEAT_S = 5.0
RETRIES = 8
ACCELSIM_BIN = Path("gpu-simulator/bin/release/accel-sim.out")
TRACE_CONFIGS_REL = Path("gpu-simulator/configs/tested-cfgs")
DEFAULT_ACCELSIM_DIR = "/opt/accel-sim-framework"
GZIP_OVER_BYTES = 1 << 20
CHUNK_BYTES = 1 << 20
LOG_NAME = "accelsim.log"
REPLAY_SKIP = frozenset({"run.log", "setup.log"})
REQUIRED_ENV = ("CASCADE_SIM_ID", "CASCADE_MAIN_URL", "CASCADE_GPU_CONFIG", "CASCADE_TRACE_DIR")


class WorkerError(Exception):
    """A setup problem the sim reports as a `failed` heartbeat instead of running."""


def say(message: str) -> None:
    print(f"cascade worker: {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class SimEnv:
    sim_id: str
    main_url: str
    token: str
    gpu_config: str
    trace_dir: Path
    accelsim_dir: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> SimEnv:
        missing = [name for name in REQUIRED_ENV if not env.get(name)]
        if missing:
            raise ValueError(f"missing sim env {', '.join(missing)} (see docs/api.md)")
        return cls(
            sim_id=env["CASCADE_SIM_ID"],
            main_url=env["CASCADE_MAIN_URL"].rstrip("/"),
            token=env.get("CASCADE_TOKEN", ""),
            gpu_config=env["CASCADE_GPU_CONFIG"],
            trace_dir=Path(env["CASCADE_TRACE_DIR"]),
            accelsim_dir=Path(env.get("CASCADE_ACCELSIM_DIR") or DEFAULT_ACCELSIM_DIR),
        )

    @property
    def kernelslist(self) -> Path:
        return self.trace_dir / "traces" / "kernelslist.g"


class Main:
    """The sim's calls to main. Every call is safe to retry: main ignores a repeated log seq,
    heartbeats are idempotent, and a step-timeline post replaces the previous one."""

    def __init__(self, env: SimEnv, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=10)
        self.base = f"{env.main_url}/api/sims/{env.sim_id}"
        self.headers = {"Content-Type": "application/json"}
        if env.token:
            self.headers["Authorization"] = f"Bearer {env.token}"
        self.cancelled = threading.Event()

    def _send(
        self, method: str, path: str, content: Callable[[], bytes | Iterator[bytes]], headers: dict
    ) -> httpx.Response:
        """Retries 5xx and network errors with backoff; `content` makes a fresh body per try."""
        for attempt in range(RETRIES):
            try:
                resp = self.client.request(
                    method, f"{self.base}/{path}", content=content(), headers=headers
                )
                if resp.status_code < 500:
                    return resp
                error: Exception = httpx.HTTPStatusError(
                    f"{resp.status_code} from main", request=resp.request, response=resp
                )
            except httpx.TransportError as e:
                error = e
            delay = min(0.25 * 2**attempt, 10.0)
            say(f"{method} {path} failed ({error}); retry in {delay:.2f}s")
            time.sleep(delay)
        raise RuntimeError(f"{method} {path} failed after {RETRIES} attempts")

    def post(self, path: str, body: dict) -> dict | None:
        content = json.dumps(body, allow_nan=False)
        resp = self._send("POST", path, lambda: content, self.headers)
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def put_file(self, name: str, content: Callable[[], Iterator[bytes]]) -> httpx.Response:
        headers = {**self.headers, "Content-Type": "application/octet-stream"}
        return self._send("PUT", f"files/{name}", content, headers)

    def heartbeat(self, state: str, exit_code: int | None = None, error: str | None = None) -> None:
        ack = self.post("heartbeat", {"state": state, "exit_code": exit_code, "error": error})
        if ack and ack.get("cancel"):
            self.cancelled.set()


class LogSender:
    """Keeps the full log at `path` and ships lines to main as LogChunks from a thread. A chunk
    goes out at BATCH_LINES lines or BATCH_SECONDS after its first line, whichever comes first."""

    def __init__(self, main: Main, path: Path, on_error: Callable[[], None]):
        self.main = main
        self.path = path
        self.on_error = on_error
        self.error: Exception | None = None
        self._file = path.open("w")
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._seq = 0
        self._thread = threading.Thread(target=self._run, name="logs", daemon=True)
        self._thread.start()

    def put(self, line: str) -> None:
        self._file.write(line + "\n")
        self._lines.put(line)

    def flush_file(self) -> None:
        self._file.flush()

    def close(self) -> Exception | None:
        """Sends what is left and returns the error that stopped sending, if any."""
        self._lines.put(None)
        self._thread.join()
        self._file.close()
        return self.error

    def _run(self) -> None:
        batch: list[str] = []
        deadline = 0.0
        while True:
            try:
                wait = max(0.0, deadline - time.monotonic()) if batch else None
                line = self._lines.get(timeout=wait)
            except queue.Empty:
                batch = self._send(batch)
                continue
            if line is None:
                self._send(batch)
                return
            if not batch:
                deadline = time.monotonic() + BATCH_SECONDS
            batch.append(line)
            if len(batch) >= BATCH_LINES:
                batch = self._send(batch)

    def _send(self, batch: list[str]) -> list[str]:
        if batch and self.error is None:
            try:
                self.main.post("logs", {"seq": self._seq, "lines": batch})
                self._seq += 1
            except Exception as e:
                self.error = e
                self.on_error()
        return []


class Source(Protocol):
    def begin(self) -> None: ...

    def lines(self) -> Iterator[str]: ...

    def stop(self) -> None: ...

    def wait(self) -> int: ...


class AccelSim:
    def __init__(self, argv: list[str], run_dir: Path):
        self.argv = argv
        self.run_dir = run_dir

    def begin(self) -> None:
        argv = self.argv
        try:
            self.proc = subprocess.Popen(
                argv,
                cwd=self.run_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
        except OSError as e:
            raise WorkerError(f"could not start {argv[0]}: {e}") from e

    def lines(self) -> Iterator[str]:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            yield line.rstrip("\n")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def wait(self) -> int:
        return self.proc.wait()


class Replay:
    """Stands in for Accel-Sim by replaying a recorded log, pausing `delay_s` every BATCH_LINES.
    With `outputs`, it first copies that dir's files into the run dir as Accel-Sim's output."""

    def __init__(self, log: Path, delay_s: float, run_dir: Path, outputs: Path | None = None):
        if not log.is_file():
            raise WorkerError(f"replay log {log} not found")
        if outputs is not None and not outputs.is_dir():
            raise WorkerError(f"replay outputs dir {outputs} not found")
        self.log = log
        self.delay_s = delay_s
        self.run_dir = run_dir
        self.outputs = outputs
        self._stop = threading.Event()

    def begin(self) -> None:
        if self.outputs is None:
            return
        for path in sorted(self.outputs.iterdir()):
            if _is_plain_file(path) and path.name not in REPLAY_SKIP:
                shutil.copyfile(path, self.run_dir / path.name)

    def lines(self) -> Iterator[str]:
        with self.log.open(errors="replace") as f:
            for i, line in enumerate(f, 1):
                if self._stop.is_set():
                    return
                yield line.rstrip("\n")
                if i % BATCH_LINES == 0:
                    self._stop.wait(self.delay_s)

    def stop(self) -> None:
        self._stop.set()

    def wait(self) -> int:
        return 0


def accelsim_command(env: SimEnv) -> list[str]:
    binary = env.accelsim_dir / ACCELSIM_BIN
    sim_config = env.accelsim_dir / HW_CONFIGS_REL / env.gpu_config / "gpgpusim.config"
    trace_config = env.accelsim_dir / TRACE_CONFIGS_REL / env.gpu_config / "trace.config"
    for path, what in (
        (binary, "Accel-Sim binary"),
        (sim_config, f"{env.gpu_config} gpgpusim.config"),
        (trace_config, f"{env.gpu_config} trace.config"),
    ):
        if not path.is_file():
            raise WorkerError(f"{what} not found at {path} (check CASCADE_ACCELSIM_DIR)")
    return [
        str(binary),
        "-trace",
        str(env.kernelslist),
        "-config",
        str(sim_config),
        "-config",
        str(trace_config),
    ]


def fresh_run_dir(env: SimEnv) -> Path:
    run_dir = Path(tempfile.gettempdir()) / "cascade-worker" / env.sim_id
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    return run_dir


def copy_companions(argv: list[str], run_dir: Path) -> None:
    """Some configs name aux files (interconnect, power) by relative path, so Accel-Sim needs
    every file of both config dirs in its working directory."""
    for config in (Path(argv[i + 1]) for i, arg in enumerate(argv) if arg == "-config"):
        for path in config.parent.iterdir():
            if path.is_file():
                shutil.copy2(path, run_dir / path.name)


Listing = dict[str, tuple[int, int]]


def _is_plain_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def listing(run_dir: Path) -> Listing:
    """Regular files directly in run_dir, by name, with their size and mtime."""
    return {
        p.name: (st.st_size, st.st_mtime_ns)
        for p in run_dir.iterdir()
        if _is_plain_file(p) and (st := p.stat())
    }


def is_text(path: Path) -> bool:
    with path.open("rb") as f:
        head = f.read(8192)
    if b"\0" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as e:
        return e.start > len(head) - 4
    return True


def read_chunks(path: Path, gzip_it: bool) -> Iterator[bytes]:
    z = zlib.compressobj(wbits=31) if gzip_it else None
    with path.open("rb") as f:
        while block := f.read(CHUNK_BYTES):
            out = z.compress(block) if z else block
            if out:
                yield out
    if z:
        yield z.flush()


class Uploads:
    """Uploads output files within main's limits, sending at most SIM_FILES_MAX_BYTES in all.
    Never fails the sim: every skip or failed upload becomes one log line."""

    def __init__(self, main: Main, put: Callable[[str], None]):
        self.main = main
        self.put = put
        self.sent = 0

    def send(self, path: Path, gzip_it: bool) -> None:
        name = path.name + (".gz" if gzip_it else "")
        size = path.stat().st_size
        skip = None
        if not re.fullmatch(FILE_NAME_PATTERN, name):
            skip = "its name is not allowed"
        elif not gzip_it and size > FILE_MAX_BYTES:
            skip = f"it is over the {FILE_MAX_BYTES >> 20} MiB per-file limit"
        elif self.sent >= SIM_FILES_MAX_BYTES or (
            not gzip_it and self.sent + size > SIM_FILES_MAX_BYTES
        ):
            skip = f"the sim already sent {self.sent >> 20} of {SIM_FILES_MAX_BYTES >> 20} MiB"
        if skip:
            self.put(f"cascade worker: skipped output {path.name}: {skip}")
            return
        try:
            resp = self.main.put_file(name, lambda: read_chunks(path, gzip_it))
        except Exception as e:
            self.put(f"cascade worker: upload of {name} failed: {e}")
            return
        if resp.status_code == 413:
            self.put(f"cascade worker: skipped output {name}: {resp.json().get('detail')}")
        elif resp.is_error:
            self.put(f"cascade worker: upload of {name} failed: {resp.status_code} {resp.text}")
        else:
            self.sent += resp.json()["size"]
            self.put(f"cascade worker: uploaded {name} ({resp.json()['size']} bytes)")


def upload_outputs(main: Main, run_dir: Path, before: Listing, put: Callable[[str], None]) -> None:
    """Uploads every file the simulator created or changed in run_dir, gzipping text over
    GZIP_OVER_BYTES, then the worker's own log as accelsim.log.gz."""
    uploads = Uploads(main, put)
    after = listing(run_dir)
    for name in sorted(after):
        if name == LOG_NAME or before.get(name) == after[name]:
            continue
        path = run_dir / name
        uploads.send(path, gzip_it=after[name][0] > GZIP_OVER_BYTES and is_text(path))
    uploads.send(run_dir / LOG_NAME, gzip_it=True)


def step_timing(main: Main, env: SimEnv, log_path: Path, put: Callable[[str], None]) -> None:
    """Never fails the sim: a problem in the pipeline becomes a log line and no upload."""
    if not has_step_timeline(env.trace_dir):
        put("cascade worker: no corrected_timeline.json + kernel_mapping.json; no step timing")
        return
    try:
        cfg = GpuConfigRegistry(env.accelsim_dir).get(env.gpu_config)
        if cfg is None:
            raise ValueError(f"no core clock for {env.gpu_config} under {env.accelsim_dir}")
        real = load_timeline(env.trace_dir / "corrected_timeline.json")
        mapping = env.trace_dir / "kernel_mapping.json"
        if not mapping.is_file():
            mapping = env.trace_dir / "trace_alignment.json"
        durations = import_durations(log_path, mapping, cfg.core_clock_mhz)
        simulated = simulate(real, parse_durations(durations))
        report = compare(real, simulated)
        main.post("step-timeline", {"simulated": simulated, "compare": report})
    except Exception as e:
        put(f"cascade worker: step timing failed: {type(e).__name__}: {e}")
        return
    coverage = simulated["summary"]["simulation"]["coverage"]
    put(
        f"cascade worker: step timing posted, {coverage['simulated_kernels']}"
        f"/{coverage['kernels']} kernels simulated"
    )


def heartbeat_loop(
    main: Main, every_s: float, stop: threading.Event, on_cancel: Callable[[], None]
) -> None:
    while not stop.wait(every_s):
        try:
            main.heartbeat("running")
        except Exception as e:
            say(f"heartbeat failed: {e}")
        if main.cancelled.is_set():
            on_cancel()
            return


def run(env: SimEnv, start: Callable[[Path], Source], heartbeat_s: float = HEARTBEAT_S) -> int:
    """Runs one sim to its final heartbeat and returns the exit code."""
    main = Main(env)
    run_dir = fresh_run_dir(env)
    log_path = run_dir / LOG_NAME
    say(f"{env.sim_id} on {env.gpu_config}, trace {env.trace_dir}, log {log_path}")
    main.heartbeat("running")
    if main.cancelled.is_set():
        return 0
    try:
        if not env.kernelslist.is_file():
            raise WorkerError(f"{env.kernelslist} not found")
        source = start(run_dir)
        before = listing(run_dir)
        source.begin()
    except WorkerError as e:
        say(str(e))
        main.heartbeat("failed", exit_code=2, error=str(e))
        return 2

    logs = LogSender(main, log_path, on_error=source.stop)
    stop = threading.Event()
    beat = threading.Thread(target=heartbeat_loop, args=(main, heartbeat_s, stop, source.stop))
    beat.start()
    try:
        for line in source.lines():
            logs.put(line)
        code = source.wait()
        if code == 0 and not main.cancelled.is_set() and logs.error is None:
            logs.flush_file()
            step_timing(main, env, log_path, logs.put)
        if not main.cancelled.is_set():
            logs.flush_file()
            upload_outputs(main, run_dir, before, logs.put)
    finally:
        stop.set()
        beat.join()
        log_error = logs.close()

    if main.cancelled.is_set():
        say("main cancelled this sim; stopped")
        return 0
    if log_error is not None:
        error = f"sending logs to main failed: {log_error}"
        say(error)
        main.heartbeat("failed", exit_code=1, error=error)
        return 1
    if code != 0:
        main.heartbeat("failed", exit_code=code, error=f"accel-sim exited {code}")
        return code
    main.heartbeat("succeeded", exit_code=0)
    return 0


def worker(args: argparse.Namespace) -> None:
    if args.replay_outputs is not None and args.replay is None:
        raise SystemExit("cascade worker: --replay-outputs needs --replay")
    env = SimEnv.from_env(os.environ)

    def start(run_dir: Path) -> Source:
        if args.replay is not None:
            return Replay(args.replay, args.delay, run_dir, args.replay_outputs)
        argv = accelsim_command(env)
        copy_companions(argv, run_dir)
        say(f"running {' '.join(argv)} in {run_dir}")
        return AccelSim(argv, run_dir)

    raise SystemExit(run(env, start, args.heartbeat_s))


def add_command(sub: argparse._SubParsersAction) -> None:
    w = sub.add_parser("worker", help="run one sim (what CASCADE_SIM_CMD starts)")
    w.add_argument("--replay", type=Path, help="replay this recorded Accel-Sim log instead")
    w.add_argument(
        "--replay-outputs",
        type=Path,
        help="with --replay, copy this dir's files into the run dir as Accel-Sim's output files",
    )
    w.add_argument(
        "--delay", type=float, default=0.05, help="with --replay, seconds to pause every 200 lines"
    )
    w.add_argument("--heartbeat-s", type=float, default=HEARTBEAT_S, help=argparse.SUPPRESS)
    w.set_defaults(func=worker)
