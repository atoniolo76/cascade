"""Recapture with: start `cascade serve` on :8020/:9020 with the fake simulator (see the Quickstart
page), then `uv run python landing/docs-src/capture_examples.py <processed trace dir>`.

Drives a running Cascade server through every route and saves each real response, trimmed, to
landing/docs-src/examples/. Trimmed spots hold {"$trimmed": "..."}, which build.py renders as a
comment, so a reader can tell captured data from elision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "examples"
CONFIG = "SM86_RTX3070"


def trim(value, *, items: int = 3, keys: int = 6, tail: int = 3, dict_keys=("stats", "metadata")):
    """Cuts long lists to `items`, `stats`/`metadata` maps to `keys`, and log tails to `tail`."""

    def walk(v, key=None):
        if isinstance(v, dict):
            out = {}
            limit = keys if key in dict_keys else None
            for i, (k, x) in enumerate(v.items()):
                if limit is not None and i >= limit:
                    out["$trimmed"] = f"{len(v) - limit} more keys"
                    break
                out[k] = walk(x, k)
            return out
        if isinstance(v, list):
            limit = tail if key in ("log_tail", "lines") else items
            if len(v) <= limit + 1:
                return [walk(x) for x in v]
            return [walk(x) for x in v[:limit]] + [{"$trimmed": f"{len(v) - limit} more items"}]
        return v

    return walk(value)


class Recorder:
    def __init__(self, base: str, token: str):
        self.base = base
        self.client = httpx.Client(base_url=base, timeout=30)
        self.auth = {"Authorization": f"Bearer {token}"}
        self.token = token

    def get(self, path: str):
        return self.client.get(path, headers=self.auth)

    def snap(self, name: str, method: str, path: str, *, auth: bool = True, body=None, **trim_kw):
        """Sends one request, saves it as examples/<name>.json, and returns the response."""
        resp = self.client.request(method, path, headers=self.auth if auth else {}, json=body)
        ctype = resp.headers.get("content-type", "")
        if resp.status_code == 204:
            payload = None
        elif ctype.startswith("application/json"):
            payload = trim(resp.json(), **trim_kw)
        else:
            lines = resp.text.splitlines()
            payload = "\n".join(lines[:12])
            if len(lines) > 12:
                payload += f"\n$trimmed {len(lines) - 12} more lines"
        save(
            name,
            {
                "request": {"method": method, "path": path, "body": body},
                "status": resp.status_code,
                "content_type": ctype.split(";")[0] if ctype else None,
                "body": payload,
            },
        )
        return resp


def save(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"saved {name}")


def wait_for(fn, what: str, timeout: float = 120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = fn()
        if result:
            return result
        time.sleep(0.5)
    raise SystemExit(f"timed out waiting for {what}")


def upload(src: Path, name: str, s3: str, *extra: str) -> None:
    script = ROOT / "scripts" / "upload_trace.py"
    cmd = [sys.executable, str(script), str(src), "--run-name", name, "--endpoint", s3, *extra]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def collect_events(rec: Recorder, run_id: str, found: dict, done: threading.Event) -> None:
    url = f"/api/runs/{run_id}/events?token={rec.token}"
    with httpx.Client(base_url=rec.base, timeout=None) as c, c.stream("GET", url) as resp:
        found["headers"] = {k: resp.headers[k] for k in ("content-type", "cache-control")}
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            kind = event["type"]
            if kind == "sim" and event["sim"]["state"] in ("succeeded", "failed"):
                kind = "sim_finished"
            found.setdefault(kind, event)
            if {"snapshot", "sim", "log", "kernel", "sim_finished"} <= found.keys():
                break
    done.set()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("trace_dir", type=Path, help="processed trace dir (traces/kernelslist.g inside)")
    p.add_argument("--app", default="http://127.0.0.1:8020")
    p.add_argument("--s3", default="http://127.0.0.1:9020")
    p.add_argument("--token", default="dev-token")
    args = p.parse_args()

    rec = Recorder(args.app, args.token)
    rec.snap("health", "GET", "/api/health", auth=False)
    rec.snap("error-401", "GET", "/api/traces", auth=False)
    rec.snap("gpu-configs", "GET", "/api/gpu-configs", items=10)

    arch = ["--meta", "trace_gpu_arch=sm_86"]
    upload(args.trace_dir, "demo", args.s3, *arch, "--meta", f"suggested_sim_config={CONFIG}")
    upload(args.trace_dir, "demo-failed", args.s3, *arch, "--status", "failed")

    def trace_state(tid):
        r = rec.get(f"/api/traces/{tid}")
        return r.json()["state"] if r.status_code == 200 else None

    wait_for(lambda: trace_state("demo") == "ready", "demo ready")
    wait_for(lambda: trace_state("demo-failed") == "failed", "demo-failed failed")
    rec.snap("traces-list", "GET", "/api/traces")
    rec.snap("trace-detail", "GET", "/api/traces/demo")
    rec.snap("error-404-trace", "GET", "/api/traces/nope")

    def run_body(trace_id: str, config: str, count: int) -> dict:
        return {"trace_id": trace_id, "entries": [{"gpu_config": config, "count": count}]}

    rec.snap("error-409", "POST", "/api/runs", body=run_body("demo-failed", CONFIG, 1))
    rec.snap("error-422-unknown-config", "POST", "/api/runs", body=run_body("demo", "SM99_X", 1))
    rec.snap("error-422-arch", "POST", "/api/runs", body=run_body("demo", "SM90_H100", 1))
    rec.snap("error-422-validation", "POST", "/api/runs", body=run_body("demo", CONFIG, 0))

    run_id = rec.snap("run-create", "POST", "/api/runs", body=run_body("demo", CONFIG, 2)).json()[
        "id"
    ]
    found: dict = {}
    done = threading.Event()
    threading.Thread(target=collect_events, args=(rec, run_id, found, done), daemon=True).start()

    def run_finished():
        r = rec.get(f"/api/runs/{run_id}").json()
        return r if r["state"] != "active" else None

    sim_id = wait_for(run_finished, "run to finish")["sims"][0]["id"]
    done.wait(30)
    events = {k: trim(v) for k, v in found.items() if k != "headers"}
    save("events", {"run_id": run_id, "headers": found.get("headers"), "events": events})

    rec.snap("runs-list", "GET", "/api/runs")
    rec.snap("run-detail", "GET", f"/api/runs/{run_id}")
    rec.snap("comparison", "GET", f"/api/runs/{run_id}/comparison")
    rec.snap("run-results", "GET", f"/api/runs/{run_id}/results", items=2, keys=4)
    rec.snap("sim-results", "GET", f"/api/sims/{sim_id}/results", items=2, keys=4)
    rec.snap("sim-log", "GET", f"/api/sims/{sim_id}/log")
    rec.snap("error-404-run", "GET", "/api/runs/run-000000000000")
    rec.snap("error-404-route", "GET", "/api/nope", auth=False)

    cancel_id = rec.snap("run-cancel-create", "POST", "/api/runs", body=run_body("demo", CONFIG, 1))
    cancel_id = cancel_id.json()["id"]
    rec.snap("run-cancel", "POST", f"/api/runs/{cancel_id}/cancel")
    cancelled = rec.get(f"/api/runs/{cancel_id}").json()["sims"][0]["id"]
    running = {"state": "running"}
    rec.snap("heartbeat-cancelled", "POST", f"/api/sims/{cancelled}/heartbeat", body=running)
    rec.snap("heartbeat-finished", "POST", f"/api/sims/{sim_id}/heartbeat", body=running)
    chunk = {"seq": 0, "lines": ["replayed line"]}
    rec.snap("logs-post", "POST", f"/api/sims/{sim_id}/logs", body=chunk)
    rec.snap("error-404-sim", "POST", "/api/sims/sim-000000000000/heartbeat", body=running)
    return 0


if __name__ == "__main__":
    sys.exit(main())
