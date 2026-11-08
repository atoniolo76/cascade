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

## Auth

- One shared bearer token, `CASCADE_TOKEN`, sent as `Authorization: Bearer <token>` by the profiler, sims, and UI.
- The UI prompts for it once and keeps it in localStorage.
- `SailLauncher` passes it to each sim via the `exec` env.
- Sail's app allowlist is not used: it would block the browser and the profiler.

## Simulation run shape

- A run is one trace plus a list of `{gpu_config, count}` entries, for example `[{H100: 4}, {A100: 2}, {RTX3070: 8}]`.
- `gpu_config` names one of Accel-Sim's tested configs. `count` can differ per config.
- Each entry fans out to `count` sims, all forked from the run's one checkpoint.

## Trace upload

- ~~Provisional: `POST /api/traces` multipart `.tar.gz`.~~ Superseded 2026-11-07: Alessio's profiler uploads through an S3-compatible endpoint, and the app saves the object to local disk.
- The main app runs MinIO in the same container, with its data dir on local disk (e.g. `/data/minio`).
- MinIO gets its own exposed port (9000) and its own `sail.box` URL. It is not proxied under a FastAPI path, because SigV4 signs the host and path and proxying breaks that.
- Profiler auth uses MinIO access/secret keys (`CASCADE_S3_ACCESS_KEY` / `CASCADE_S3_SECRET_KEY`), not the bearer token.
- MinIO sends a bucket notification webhook to FastAPI (`POST /internal/minio-events`) on object-created. FastAPI then registers the trace and reads its metadata.
- Because MinIO's data dir is on local disk, the run checkpoint carries the trace into every sim fork. Sims run with `restore_memory=False`, so MinIO does not run in sims.
- Still to confirm with Alessio: bucket and key layout, what the object is (single archive vs. many files), and how the app knows an upload is complete.
