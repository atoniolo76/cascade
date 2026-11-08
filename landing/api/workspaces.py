"""POST /api/workspaces creates and manages a Cascade workspace in the caller's Sail account.

The JSON body is {"action", "api_key", "workspace"}, plus "image" for deploy and "confirm" (the
workspace name again) for delete. Each call is one bounded step: the page calls deploy again
until the phase is healthy. The key is used for this request only, and nothing here logs a body,
a key or a token.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _cascade_deploy as deploy  # noqa: E402

MAX_BODY = 16 * 1024
STEP_BUDGET_S = 25


def _deploy(key: str, name: str, body: dict) -> dict:
    image = deploy.FromRegistry(body.get("image") or deploy.DEFAULT_IMAGE)
    return deploy.advance(key, name, image, budget_s=STEP_BUDGET_S).to_json()


def _delete(key: str, name: str, body: dict) -> dict:
    if body.get("confirm") != name:
        raise deploy.InvalidWorkspace("Type the workspace name to confirm the delete.")
    return {"workspace": name, "terminated": deploy.delete(key, name)}


ACTIONS = {
    "deploy": _deploy,
    "status": lambda key, name, body: deploy.workspace_status(key, name).to_json(),
    "pause": lambda key, name, body: deploy.pause(key, name).to_json(),
    "resume": lambda key, name, body: deploy.resume(key, name).to_json(),
    "delete": _delete,
}
STRING_FIELDS = ("action", "api_key", "workspace", "image", "confirm")


def _bad(message: str) -> tuple[int, dict]:
    return 400, {"error": message}


def handle(raw: bytes) -> tuple[int, dict]:
    try:
        body = json.loads(raw)
    except ValueError:
        return _bad("The body must be JSON.")
    if not isinstance(body, dict):
        return _bad("The body must be a JSON object.")
    for name in STRING_FIELDS:
        if not isinstance(body.get(name, ""), str):
            return _bad(f"{name} must be a string.")
    action = ACTIONS.get(body.get("action", ""))
    if action is None:
        return _bad(f"action must be one of {', '.join(ACTIONS)}.")
    if not body.get("api_key", "").strip():
        return 401, {"error": "A Sail API key is required."}
    try:
        return 200, action(body["api_key"].strip(), body.get("workspace", ""), body)
    except deploy.DeployError as error:
        return error.http_status, {"error": str(error)}
    except Exception as error:
        print(f"workspaces: unexpected {type(error).__name__}", file=sys.stderr)
        return 500, {"error": "Something went wrong talking to Sail. Try again."}


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        if status == 405:
            self.send_header("Allow", "POST")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY:
            self._send(413 if length > MAX_BODY else 400, {"error": "Send a JSON body."})
            return
        self._send(*handle(self.rfile.read(length)))

    def do_GET(self) -> None:
        self._send(405, {"error": "Use POST."})

    def log_message(self, format: str, *args) -> None:
        pass
