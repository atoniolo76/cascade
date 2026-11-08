"""One Cascade workspace on Sail: the app `cascade-<workspace>` and its sailbox `cascade-main`.

Every call reads the workspace's state from Sail and moves it one step toward a running app, so
calling again after a crash, a timeout or a pause converges instead of duplicating anything.

This file imports only the sail SDK and the standard library. The onboarding functions on Vercel
run a copy of it (landing/api/_cascade_deploy.py), which the docs build writes.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

DEFAULT_IMAGE = "ghcr.io/atoniolo76/cascade:latest"
SAILBOX_NAME = "cascade-main"
APP_PORT, S3_PORT = 8000, 9000
ENV_FILE = "/data/cascade.env"
PID_FILE = "/data/cascade-serve.pid"
SERVE_LOG = "/data/cascade-serve.log"
STAND_IN_SIM_CMD = (
    "/app/.venv/bin/cascade worker --replay /app/tests/fixtures/logs/opt125m_sm90_h100.log"
    " --replay-outputs /app/results/20261108-021007-opt125m-h100-accelsim/outputs"
)
STAND_IN_ACCELSIM_DIR = "/app/tests/fixtures/accelsim"

_SLUG = re.compile(r"[a-z0-9-]{3,32}")
_LIVE = ("creating", "running", "paused", "sleeping")
_SERVE = (
    f"mkdir -p /data && echo $$ > {PID_FILE} && . {ENV_FILE} && cd /app"
    f" && exec cascade serve --sims sail >> {SERVE_LOG} 2>&1"
)
_SERVING = (
    f"pid=$(cat {PID_FILE} 2>/dev/null)"
    " && tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q 'cascade serve'"
)
_lock = threading.Lock()


class Phase(StrEnum):
    absent = "absent"
    creating_sandbox = "creating_sandbox"
    starting = "starting"
    healthy = "healthy"
    paused = "paused"


@dataclass(frozen=True)
class FromRegistry:
    ref: str = DEFAULT_IMAGE


@dataclass(frozen=True)
class FromSource:
    """Builds the repo's Dockerfile on Sail, which uploads the checkout and takes minutes."""

    root: Path
    base_image: str = "ubuntu:22.04"


ImageSource = FromRegistry | FromSource
PUBLISHED = FromRegistry()


@dataclass(frozen=True)
class Simulator:
    sim_cmd: str = STAND_IN_SIM_CMD
    accelsim_dir: str = STAND_IN_ACCELSIM_DIR
    postprocess_cmd: str = ""


STAND_IN = Simulator()


@dataclass(frozen=True)
class WorkspaceInfo:
    workspace: str
    app: str
    phase: Phase
    sailbox_id: str | None = None
    app_url: str | None = None
    s3_url: str | None = None
    token: str | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        return {**asdict(self), "phase": self.phase.value}


class DeployError(Exception):
    """A failure whose message is safe to show: it never contains the API key or the token."""

    http_status = 502


class InvalidWorkspace(DeployError):
    http_status = 400


class KeyRejected(DeployError):
    http_status = 401


class WorkspaceMissing(DeployError):
    http_status = 404


class SailRefused(DeployError):
    """Sail refused the request (permission, quota, a failed sandbox). Sail's message, verbatim."""

    def __init__(self, message: str, http_status: int = 502):
        super().__init__(message)
        self.http_status = http_status


def app_name(workspace: str) -> str:
    if not _SLUG.fullmatch(workspace):
        raise InvalidWorkspace(
            "A workspace name is 3 to 32 characters of lowercase letters, digits and hyphens."
        )
    return f"cascade-{workspace}"


def probe_health(app_url: str, timeout_s: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"{app_url}/api/health", timeout=timeout_s) as res:
            body = json.load(res)
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) and body.get("ok") is True else None


def _scrub(text: str, hidden: list[str]) -> str:
    for value in hidden:
        if value:
            text = text.replace(value, "[redacted]")
    return text


def _translate(sail, error: Exception, hidden: list[str]) -> DeployError:
    message = _scrub(str(error), hidden) or type(error).__name__
    if isinstance(error, sail.PermissionDeniedError):
        if "api key" in message.lower():
            return KeyRejected("Sail rejected this API key.")
        return SailRefused(message, 403)
    if isinstance(error, sail.InvalidArgumentError):
        return SailRefused(message, 400)
    status = getattr(error, "status_code", None)
    return SailRefused(message, status if isinstance(status, int) and 400 <= status < 500 else 502)


@contextmanager
def _session(api_key: str, sail_module) -> Iterator[tuple]:
    """Points the SDK at `api_key` for one call. The SDK reads its key from the process env, so
    calls in one process take turns."""
    if not api_key.strip():
        raise KeyRejected("A Sail API key is required.")
    if sail_module is None:
        import sail as sail_module
    sail = sail_module
    hidden = [api_key]
    with _lock:
        previous = os.environ.get("SAIL_API_KEY")
        os.environ["SAIL_API_KEY"] = api_key
        sail.reset_transports()
        try:
            yield sail, hidden
        except sail.SailError as error:
            raise _translate(sail, error, hidden) from None
        finally:
            if previous is None:
                os.environ.pop("SAIL_API_KEY", None)
            else:
                os.environ["SAIL_API_KEY"] = previous
            sail.reset_transports()


def _find_app(sail, name: str, create: bool):
    if create:
        return sail.App.find(name, mint_if_missing=True)
    try:
        return sail.App.find(name)
    except sail.NotFoundError:
        return None


def _find_box(sail, app):
    if app is None:
        return None
    boxes = sail.Sailbox.list(app_id=app, name=SAILBOX_NAME, statuses=list(_LIVE))
    return boxes[0] if boxes else None


def _url(box, port: int) -> str | None:
    url = getattr(box.listener(port).endpoint, "url", None)
    return url.rstrip("/") if url else None


def _definition(sail, image: ImageSource):
    match image:
        case FromRegistry(ref):
            return sail.Image.from_registry(ref)
        case FromSource(root, base_image):
            return sail.Image.from_dockerfile(
                root / "Dockerfile", context_dir=root, build_args={"BASE_IMAGE": base_image}
            )


def _env_file(values: dict[str, str]) -> str:
    return "".join(f"export {k}={shlex.quote(v)}\n" for k, v in values.items() if v)


def _write_env_once(box, values: dict[str, str]) -> None:
    """Links a fresh file into place, so of two concurrent writers the first one's token wins."""
    tmp = f"{ENV_FILE}.{secrets.token_hex(4)}"
    box.fs.write(tmp, _env_file(values), mode=0o600)
    box.run(f"ln {tmp} {ENV_FILE} 2>/dev/null; rm -f {tmp}")


def _token(box) -> str | None:
    result = box.run(f". {ENV_FILE} && printenv CASCADE_TOKEN")
    return (result.stdout.strip() or None) if result.exit_code == 0 else None


def _info(workspace: str, phase: Phase, box=None, **fields) -> WorkspaceInfo:
    return WorkspaceInfo(
        workspace=workspace,
        app=app_name(workspace),
        phase=phase,
        sailbox_id=box.sailbox_id if box is not None else None,
        **fields,
    )


def advance(
    api_key: str,
    workspace: str,
    image: ImageSource = PUBLISHED,
    *,
    simulator: Simulator = STAND_IN,
    token: str | None = None,
    disk_gib: int | None = None,
    budget_s: int = 25,
    sail_module=None,
) -> WorkspaceInfo:
    """Does the next step toward a healthy workspace and returns where it got to.

    The call that creates the app returns right after, so a caller can show each step. Each
    step that waits on Sail is bounded: an image build gives up after `budget_s` and the next
    call picks it up again. `Sailbox.create` blocks until the sandbox runs, so the call that
    creates it returns right after.
    """
    name = app_name(workspace)
    with _session(api_key, sail_module) as (sail, hidden):
        app = _find_app(sail, name, create=False)
        if app is None:
            _find_app(sail, name, create=True)
            return _info(workspace, Phase.creating_sandbox)
        box = _find_box(sail, app)
        if box is None:
            try:
                built = _definition(sail, image).build(timeout=budget_s)
            except sail.TimeoutError:
                return _info(workspace, Phase.creating_sandbox)
            box = sail.Sailbox.create(
                app=app,
                image=built,
                name=SAILBOX_NAME,
                ingress_ports=[APP_PORT, S3_PORT],
                disk_limit_gib=disk_gib,
                auto_sleep=sail.AutoSleep.never(),
            )
            return _info(workspace, Phase.starting, box)
        if box.status == "creating":
            return _info(workspace, Phase.creating_sandbox, box)
        if box.status in ("paused", "sleeping"):
            box.resume()
        app_url, s3_url = _url(box, APP_PORT), _url(box, S3_PORT)
        urls = {"app_url": app_url, "s3_url": s3_url}
        if app_url is None:
            return _info(workspace, Phase.starting, box, **urls)
        if probe_health(app_url) is not None:
            return _info(workspace, Phase.healthy, box, **urls, token=_token(box))
        if box.run(f"test -s {ENV_FILE}").exit_code != 0:
            fresh = token or secrets.token_urlsafe(24)
            hidden.append(fresh)
            _write_env_once(
                box,
                {
                    "CASCADE_TOKEN": fresh,
                    "CASCADE_WORKSPACE": workspace,
                    "CASCADE_PUBLIC_URL": app_url,
                    "CASCADE_DATA_DIR": "/data",
                    "CASCADE_ACCELSIM_DIR": simulator.accelsim_dir,
                    "CASCADE_SIM_CMD": simulator.sim_cmd,
                    "CASCADE_POSTPROCESS_CMD": simulator.postprocess_cmd,
                    # SailLauncher checkpoints this sailbox and forks sims from it.
                    "SAIL_API_KEY": api_key,
                },
            )
        if box.run(_SERVING).exit_code != 0:
            box.exec(_SERVE, background=True)
        return _info(workspace, Phase.starting, box, **urls)


def deploy_workspace(
    api_key: str,
    workspace: str,
    image: ImageSource = PUBLISHED,
    *,
    timeout_s: float = 1800,
    poll_s: float = 3,
    on_phase: Callable[[WorkspaceInfo], None] = lambda info: None,
    **kwargs,
) -> WorkspaceInfo:
    """Calls `advance` until the workspace is healthy. Safe to rerun on an existing workspace."""
    deadline = time.monotonic() + timeout_s
    while True:
        budget = max(1, int(deadline - time.monotonic()))
        info = advance(api_key, workspace, image, budget_s=budget, **kwargs)
        on_phase(info)
        if info.phase is Phase.healthy:
            return info
        if time.monotonic() > deadline:
            raise DeployError(f"Workspace {workspace!r} was not healthy after {timeout_s:.0f}s.")
        time.sleep(poll_s)


def workspace_status(api_key: str, workspace: str, *, sail_module=None) -> WorkspaceInfo:
    """Reads the workspace's state without changing it or waking it."""
    name = app_name(workspace)
    with _session(api_key, sail_module) as (sail, _):
        box = _find_box(sail, _find_app(sail, name, create=False))
        if box is None:
            return _info(workspace, Phase.absent)
        if box.status == "creating":
            return _info(workspace, Phase.creating_sandbox, box)
        urls = {"app_url": _url(box, APP_PORT), "s3_url": _url(box, S3_PORT)}
        if box.status in ("paused", "sleeping"):
            return _info(workspace, Phase.paused, box, **urls)
        healthy = urls["app_url"] is not None and probe_health(urls["app_url"]) is not None
        return _info(workspace, Phase.healthy if healthy else Phase.starting, box, **urls)


def _existing_box(sail, workspace: str):
    box = _find_box(sail, _find_app(sail, app_name(workspace), create=False))
    if box is None:
        raise WorkspaceMissing(f"No workspace named {workspace!r} in this Sail account.")
    return box


def pause(api_key: str, workspace: str, *, sail_module=None) -> WorkspaceInfo:
    with _session(api_key, sail_module) as (sail, _):
        box = _existing_box(sail, workspace)
        if box.status != "paused":
            box.pause()
        return _info(workspace, Phase.paused, box)


def resume(api_key: str, workspace: str, *, sail_module=None) -> WorkspaceInfo:
    """Wakes the sailbox. A sailbox can come back cold; `advance` then restarts the app."""
    with _session(api_key, sail_module) as (sail, _):
        box = _existing_box(sail, workspace)
        if box.status in ("paused", "sleeping"):
            box.resume()
        return _info(workspace, Phase.starting, box)


def delete(api_key: str, workspace: str, *, sail_module=None) -> int:
    """Terminates every sailbox in the workspace's app (main and any sim forks) and deletes its
    run checkpoints. Returns how many sailboxes it terminated. Sail has no call to delete the
    app itself; an app with no sailboxes costs nothing."""
    name = app_name(workspace)
    with _session(api_key, sail_module) as (sail, _):
        app = _find_app(sail, name, create=False)
        if app is None:
            return 0
        boxes = sail.Sailbox.list(app_id=app, statuses=list(_LIVE))
        for box in boxes:
            box.terminate()
        for checkpoint in sail.SailboxCheckpoint.list(app_id=app):
            checkpoint.delete()
        return len(boxes)
