"""Create the main Cascade sailbox from the repo's Dockerfile and start `cascade serve` in it.

    SAIL_API_KEY=... CASCADE_TOKEN=... uv run --extra sail python scripts/deploy_sail.py

With no --sim-cmd, each sim runs `cascade worker --replay` on the vendored RTX 3070 log, and
--accelsim-dir defaults to the vendored configs, so the loop works before the Accel-Sim base
image exists. With an Accel-Sim base image, pass `--sim-cmd /app/.venv/bin/cascade worker` and
its `--accelsim-dir`. Prints the app URL (port 8000) and the S3 endpoint (port 9000).
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from pathlib import Path

import sail

ROOT = Path(__file__).resolve().parent.parent
APP_PORT, S3_PORT = 8000, 9000
STAND_IN_SIM_CMD = (
    "/app/.venv/bin/cascade worker --replay /app/tests/fixtures/logs/babelstream_sm86_rtx3070.log"
)


def public_url(box: sail.Sailbox, port: int, timeout_s: float = 120) -> str:
    deadline = time.monotonic() + timeout_s
    while True:
        endpoint = box.listener(port).endpoint
        if endpoint is not None and getattr(endpoint, "url", None):
            return endpoint.url.rstrip("/")
        if time.monotonic() > deadline:
            raise SystemExit(f"port {port} never got a public URL")
        time.sleep(2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--app", default="cascade", help="Sail app name")
    p.add_argument("--name", default="cascade-main", help="name of the main sailbox")
    p.add_argument("--base-image", default="ubuntu:22.04", help="Dockerfile BASE_IMAGE")
    p.add_argument("--sim-cmd", default=STAND_IN_SIM_CMD, help="CASCADE_SIM_CMD for sims")
    p.add_argument("--accelsim-dir", default="/app/tests/fixtures/accelsim")
    p.add_argument("--postprocess-cmd", default="", help="CASCADE_POSTPROCESS_CMD override")
    p.add_argument("--disk-gib", type=int, default=None, help="disk limit for the main sailbox")
    args = p.parse_args()

    api_key = os.environ.get("SAIL_API_KEY")
    if not api_key:
        print("SAIL_API_KEY is required", file=sys.stderr)
        return 2
    token = os.environ.get("CASCADE_TOKEN") or secrets.token_urlsafe(24)

    app = sail.App.find(args.app, mint_if_missing=True)
    image = sail.Image.from_dockerfile(
        ROOT / "Dockerfile", context_dir=ROOT, build_args={"BASE_IMAGE": args.base_image}
    )
    print(f"creating sailbox {args.name!r} in app {args.app!r} (builds the image if needed)")
    box = sail.Sailbox.create(
        app=app,
        image=image,
        name=args.name,
        ingress_ports=[APP_PORT, S3_PORT],
        disk_limit_gib=args.disk_gib,
        auto_sleep=sail.AutoSleep.never(),
    )
    app_url, s3_url = public_url(box, APP_PORT), public_url(box, S3_PORT)

    env = {
        "CASCADE_TOKEN": token,
        "CASCADE_PUBLIC_URL": app_url,
        "CASCADE_DATA_DIR": "/data",
        "CASCADE_ACCELSIM_DIR": args.accelsim_dir,
        "CASCADE_SIM_CMD": args.sim_cmd,
        # SailLauncher checkpoints this sailbox and forks sims from it.
        "SAIL_API_KEY": api_key,
    }
    if args.postprocess_cmd:
        env["CASCADE_POSTPROCESS_CMD"] = args.postprocess_cmd
    serve = (
        "mkdir -p /data && cd /app"
        " && exec cascade serve --sims sail >> /data/cascade-serve.log 2>&1"
    )
    box.exec(serve, background=True, env=env)
    box.wait_for_listener(APP_PORT, timeout=120)

    print(f"sailbox:  {box.sailbox_id}")
    print(f"app:      {app_url}")
    print(f"s3:       {s3_url}  (bucket cascade, prefix accelsim-traces, path-style, unsigned)")
    if not os.environ.get("CASCADE_TOKEN"):
        print(f"token:    {token}")
    print("logs:     /data/cascade-serve.log inside the sailbox")
    return 0


if __name__ == "__main__":
    sys.exit(main())
