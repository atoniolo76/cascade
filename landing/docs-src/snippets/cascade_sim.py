"""Run a simulator command as a Cascade sim: send heartbeats and forward its stdout.

CASCADE_SIM_CMD="python3 cascade_sim.py accel-sim.out -trace ... -config ..."
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = f"{os.environ['CASCADE_MAIN_URL']}/api/sims/{os.environ['CASCADE_SIM_ID']}"
TOKEN = os.environ["CASCADE_TOKEN"]
cancelled = threading.Event()


def post(path, body):
    data = json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    for attempt in range(6):
        req = urllib.request.Request(f"{BASE}/{path}", data, headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2**attempt)
    raise RuntimeError(f"POST {path} kept failing")


def heartbeat(state, exit_code=None, error=None):
    ack = post("heartbeat", {"state": state, "exit_code": exit_code, "error": error})
    if ack["cancel"]:
        cancelled.set()


def keep_beating(proc):
    while proc.poll() is None:
        try:
            heartbeat("running")
        except Exception as e:
            print(f"heartbeat failed: {e}", file=sys.stderr)
        if cancelled.is_set():
            proc.terminate()
        time.sleep(5)


def main():
    heartbeat("running")
    if cancelled.is_set():
        return 0
    proc = subprocess.Popen(
        sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    )
    threading.Thread(target=keep_beating, args=(proc,), daemon=True).start()
    seq, batch, sent_at = 0, [], time.monotonic()
    for line in proc.stdout:
        batch.append(line.rstrip("\n"))
        if len(batch) >= 200 or time.monotonic() - sent_at > 1:
            post("logs", {"seq": seq, "lines": batch})
            seq, batch, sent_at = seq + 1, [], time.monotonic()
    if batch:
        post("logs", {"seq": seq, "lines": batch})
    code = proc.wait()
    if cancelled.is_set():
        return 0
    if code == 0:
        heartbeat("succeeded", exit_code=0)
    else:
        heartbeat("failed", exit_code=code, error=f"simulator exited with {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
