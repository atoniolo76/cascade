"""Reference simulator client for the Cascade app API (docs/api.md, "Simulator -> main").

Replays a recorded Accel-Sim stdout log as if a simulator were producing it, so the whole loop
runs without Accel-Sim. Standard library only, so it runs anywhere the real simulator would.

    CASCADE_SIM_CMD="python scripts/fake_sim.py tests/fixtures/logs/babelstream_sm86_rtx3070.log"

Reads CASCADE_SIM_ID, CASCADE_MAIN_URL, CASCADE_TOKEN, CASCADE_GPU_CONFIG, CASCADE_TRACE_DIR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BATCH_LINES = 200
BATCH_SECONDS = 0.5
RETRIES = 8


class Main:
    def __init__(self, url: str, sim_id: str, token: str):
        self.base = f"{url.rstrip('/')}/api/sims/{sim_id}"
        self.token = token
        self.cancelled = threading.Event()

    def post(self, path: str, body: dict) -> dict | None:
        """POST with exponential backoff. Every request is safe to retry: main ignores a
        repeated log seq and treats heartbeats as idempotent."""
        data = json.dumps(body).encode()
        for attempt in range(RETRIES):
            req = urllib.request.Request(
                f"{self.base}/{path}",
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    raise
                err: Exception = e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                err = e
            delay = min(0.25 * 2**attempt, 10.0)
            print(f"fake_sim: POST {path} failed ({err}); retry in {delay:.2f}s", file=sys.stderr)
            time.sleep(delay)
        raise RuntimeError(f"POST {path} failed after {RETRIES} attempts")

    def heartbeat(self, state: str, exit_code: int | None = None, error: str | None = None):
        ack = self.post("heartbeat", {"state": state, "exit_code": exit_code, "error": error})
        if ack and ack.get("cancel"):
            self.cancelled.set()
        return ack


def heartbeat_loop(main: Main, every_s: float, stop: threading.Event) -> None:
    while not stop.wait(every_s):
        try:
            main.heartbeat("running")
        except Exception as e:
            print(f"fake_sim: heartbeat failed: {e}", file=sys.stderr)


def replay(main: Main, log_path: Path, delay_s: float) -> None:
    seq, batch, started = 0, [], time.monotonic()
    with log_path.open(errors="replace") as f:
        for line in f:
            if main.cancelled.is_set():
                return
            batch.append(line.rstrip("\n"))
            if len(batch) >= BATCH_LINES or time.monotonic() - started >= BATCH_SECONDS:
                main.post("logs", {"seq": seq, "lines": batch})
                seq, batch, started = seq + 1, [], time.monotonic()
                time.sleep(delay_s)
    if batch and not main.cancelled.is_set():
        main.post("logs", {"seq": seq, "lines": batch})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("log", type=Path, help="recorded Accel-Sim stdout to replay")
    p.add_argument("--delay", type=float, default=0.05, help="seconds between log chunks")
    p.add_argument("--heartbeat-s", type=float, default=5.0)
    args = p.parse_args()

    env = os.environ
    main_api = Main(env["CASCADE_MAIN_URL"], env["CASCADE_SIM_ID"], env["CASCADE_TOKEN"])
    trace_dir = Path(env["CASCADE_TRACE_DIR"])
    print(f"fake_sim: {env['CASCADE_SIM_ID']} config={env['CASCADE_GPU_CONFIG']} trace={trace_dir}")

    main_api.heartbeat("running")
    if main_api.cancelled.is_set():
        return 0
    kernelslist = trace_dir / "traces" / "kernelslist.g"
    if not kernelslist.is_file():
        main_api.heartbeat("failed", exit_code=2, error=f"{kernelslist} not found")
        return 2

    stop = threading.Event()
    beat = threading.Thread(target=heartbeat_loop, args=(main_api, args.heartbeat_s, stop))
    beat.start()
    try:
        replay(main_api, args.log, args.delay)
    except Exception as e:
        stop.set()
        beat.join()
        main_api.heartbeat("failed", exit_code=1, error=f"fake_sim: {e}")
        return 1
    stop.set()
    beat.join()
    if main_api.cancelled.is_set():
        print("fake_sim: main cancelled this sim; stopping")
        return 0
    main_api.heartbeat("succeeded", exit_code=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
