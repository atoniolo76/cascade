# Cascade app decisions

Owner: Amani. Scope: the Cascade app container (main mode + simulator mode shell), its API, and its UI. See [charter.md](charter.md).

Decisions are recorded as they are made. Superseded decisions are struck through with a note.

## Process

- Work happens on the `amani` branch, committed continuously.

## Stack

- Backend: Python FastAPI. Frontend: React built with Vite into static files that FastAPI serves.
- One process, one image for both main and simulator modes. Next.js was rejected because it needs a Node server next to Python in every forked sailbox, and its SSR/API-route strengths don't apply to a Python-backed dashboard.

## Simulator → main communication

- Push over plain HTTP. Each simulator gets the main app's URL and its own sim id via env vars at fork time.
- Simulators call `POST /api/sims/{id}/heartbeat` (health/status) and `POST /api/sims/{id}/logs` (log chunks). This is the contract with Alessio's simulator side.
- Main fans updates out to the browser over SSE.
- Simulators need no public endpoint of their own.

## Sail integration

Sail facts (from docs.sailresearch.com, checked 2026-11-07):

- `pip install sail`, `import sail`, auth via `SAIL_API_KEY`.
- `sb.checkpoint(name=...)` saves disk + memory. `sail.Sailbox.from_checkpoint(checkpoint_id, name=..., restore_memory=False)` starts a copy with the same disk.
- Use `restore_memory=False` for sims, so the main app's running process is not cloned.
- `child.exec(cmd, background=True, env={...})` starts the simulator detached and passes per-sim env.
- `sb.expose(port, allowlist=None)` returns a public HTTPS URL on `sail.box`. Ports are not inherited by forks. An app-name allowlist entry admits only sailboxes in that Sail app.
- No GPU parameter; 1–8 vCPUs. Accel-Sim is a CPU simulator, so this is fine.
- Unverified: how a sailbox learns its own id (needed to checkpoint itself), and the SDK `fork()` method (its API page 404s). Check against the SDK before relying on them.

Decision:

- A `Launcher` interface with `launch(sim_id, config)` and `terminate(sim_id)`.
- `SailLauncher`: one checkpoint per run, shared by all sims in that run. Then per sim: `from_checkpoint` → background `exec` of `cascade simulate` with `CASCADE_SIM_ID` and `CASCADE_MAIN_URL` env.
- `LocalLauncher`: runs `cascade simulate` as a local subprocess. Used for laptop dev.
- Both launchers drive the same simulator-mode code path.
