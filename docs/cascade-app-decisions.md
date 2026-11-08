# Cascade app decisions

Owner: Amani. Scope: the Cascade app container (main mode + simulator mode shell), its API, and its UI. See [charter.md](charter.md).

Decisions are recorded as they are made. Superseded decisions are struck through with a note.

## Ownership boundary

- Amani (this doc): S3 ingest endpoint and trace registry, post-processing trigger, the REST API sims report to, launchers (local + Sail fork), log parsing, results/comparison, UI, and the app image's own layers.
- Alessio: the profiler and the simulator. Running Accel-Sim inside a fork and sending heartbeats/logs to the app's API, building Accel-Sim and the post-processor, `real_timing.json`.
- The seam: the app launches `$CASCADE_SIM_CMD` with the sim env in docs/api.md. `cascade worker` is that client: it runs Accel-Sim, or replays a recorded log with `--replay` for testing. The app image takes `ARG BASE_IMAGE` so Alessio can supply one with Accel-Sim.

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
- ~~Unverified: how a sailbox learns its own id.~~ Verified on live Sail 2026-11-07 (run run-28e9db36f03f): the main sailbox checkpoints itself (`guest.current_sailbox_id()`), forks start from that checkpoint, and both sims went pending → running in about 6 s and reported back over the public URL. Finished forks are terminated automatically. The run checkpoint stays until its TTL (7 days by default).
- The main sailbox carries `SAIL_API_KEY` in its env, because it creates checkpoints and forks.

Decision:

- A `Launcher` interface: `prepare(run_id)` (Sail: the run's one checkpoint), `launch(sim, trace_dir)` and `terminate(sim)`.
- `SailLauncher`: one checkpoint per run, shared by all sims in that run. Then per sim: `from_checkpoint` → background `exec` of `$CASCADE_SIM_CMD` (through a shell) with the sim env from docs/api.md.
- `LocalLauncher` (the default): runs `$CASCADE_SIM_CMD` as a local subprocess, split into arguments with no shell, so `$VARS` in it don't expand. Used for laptop dev.
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

Contract from Alessio (2026-11-07, tested end to end from a Modal H100):

- boto3 `upload_file`, raw files (no archive), path-style addressing, multipart for files over 128 MiB. Unsigned requests for now, so the endpoint must allow anonymous writes.
- Layout: `s3://<bucket>/<prefix>/<run_name>/` with `metadata.json`, `traces/kernel-<N>-ctx_<0x..>.trace.xz`, `traces/kernelslist_ctx_<0x..>`, `traces/stats_ctx_<0x..>`, `spinlock_detection/`, `*.log`. Kernel files trickle in during tracing. Kernel numbers have gaps; sort numerically.
- Done signal (planned by Alessio): `metadata.json` uploaded last, with `status` and `files: [{key, bytes}]`. Today's profiler (b348d7d) uploads in sorted path order, so `metadata.json` lands before `traces/kernelslist_ctx_*` and `stats_ctx_*` and has no `files`. Until that changes, ingest treats a trace as done when `metadata.json` exists, every `kernelslist_ctx_*` has its `stats_ctx_*`, and two scans 2 s apart see no change.
- Uploads are RAW. The simulator needs processed `kernelslist.g` + `kernel-*.tracez`, so main runs `post-traces-processing <run>/traces -j 8` once per trace, before any run checkpoint.
- Sizes: ~14 MiB raw for the small test, GBs to hundreds of GBs for real models. Disk needed is ~2.2x uploaded bytes.

Decision:

- ~~Provisional: `POST /api/traces` multipart `.tar.gz`.~~ Superseded: profiler uploads through S3.
- ~~MinIO in the same container with a webhook.~~ Superseded 2026-11-07: MinIO stores objects in its own format, not plain files, so sims could not read traces off the forked disk. Use **`rclone serve s3`** instead (what Alessio tested). It serves a plain directory, so objects land as normal files at `/data/s3/<bucket>/<prefix>/<run_name>/...`.
- rclone gets its own exposed port (9000) and `sail.box` URL, not proxied under FastAPI (SigV4 signs host and path). Anonymous for now; add `--auth-key` when the profiler sends keys.
- Main watches the bucket directory, registers a trace when its first file appears (state `uploading`), and marks it `complete` per the done signal above. Then it post-processes (`processing` → `ready` or `failed`).
- Runs can only launch on `ready` traces.

## Simulator configs

- These traces contain Hopper-only instructions, so they are valid only on sm_90 configs (`SM90_H100`, likely `SM90_H200`).
- `metadata.json` will carry `trace_gpu_arch` and `suggested_sim_config`. The New run screen only offers configs matching the trace's arch (default `sm_90` when missing) and preselects the suggested one.
- Simulator: Accel-Sim tag v2.0.1 (`1c9cc831`), built with CMake (the legacy make fails). The profiler still pins dev `7ba54ded`; per Alessio the only difference is job-launcher scripts, and he plans to pin v2.0.1.
- The profiler doesn't write `suggested_sim_config` or `trace_gpu_arch` yet, so the app assumes `sm_90` and the UI preselects the first compatible config.

## Real vs. simulated comparison

- Charter update: save the simulator's output and compare it one-to-one to the job on real hardware.
- Each sim's results are saved as a JSON document (`GET /api/sims/{id}/results`): config, per-kernel stats, totals, and the raw log is downloadable.
- Alessio will add `<run_name>/real_timing.json`: `[{kernel_id, kernel_name, duration_ns, grid, block}]`, pointed to by `metadata.json` `real_timing`. `kernel_id` matches `kernel-<N>`.
- Comparison: simulated time = `gpu_sim_cycle / core_clock_mhz` from the sim config's `gpgpusim.config` (`-gpgpu_clock_domains`, first value). Matched by kernel order.
- Until `real_timing.json` exists, the UI shows simulated metrics plus original-hardware context (`metadata.json`, `stats_ctx_*` launch dims), and the comparison panel says real timing is not captured yet.

## UI (v1)

Three screens:

- **Traces:** uploaded traces with original hardware info from `metadata.json` (GPU, model, layers, kernel count).
- **New run:** pick a trace, add `{gpu_config, count}` rows, launch.
- **Run detail:** a grid of sim cards (status, heartbeat age, kernel progress i/N, live log tail via SSE) plus a chart of simulated cycles per GPU config.

Superseded 2026-11-07: per-kernel real-vs-simulated comparison is now in scope (see above), shown when `real_timing.json` exists. Cost/timing panel still deferred.

## Metrics parsing

- Main parses. Sims forward raw Accel-Sim log lines only.
- Main's parser extracts per-kernel stats (`kernel_name`, `gpu_sim_cycle`, `gpu_ipc`, ...) from the stream and stores them.
- Raw logs are stored too, so a fixed parser can re-parse and backfill.

## Defaults (not separately discussed)

- State lives in SQLite at `/data/cascade.db` on main's local disk.
- A `scripts/deploy_sail.py` creates the main sailbox from the image, sets env/secrets, and exposes ports 8000 (app) and 9000 (rclone S3).
- The sim command is configurable (`CASCADE_SIM_CMD`). `cascade worker --replay <log>` replays a real Accel-Sim log against the API for local dev, and `cascade worker` runs real Accel-Sim.

## Lost sims

- A sim with no heartbeat for `CASCADE_HEARTBEAT_TIMEOUT_S` (60s) becomes `lost`. A later heartbeat revives it.
- A sim still lost after 10x that timeout is marked `failed` and its sandbox is terminated, so runs always finish.

## Local dev mode

- `CASCADE_NO_AUTH=1` turns off the token check for local play only. `/api/health` reports `auth: false` and the UI skips the token screen. A missing token without this flag still refuses to start, so a deployment can't end up open by accident.
- `scripts/seed_demo.py` fills a local app with demo traces and runs (one processed trace re-labeled, source GPU marked "(demo)").

## Subcommands (2026-11-07, agreed with Alessio)

Ownership change: Amani now also owns the simulator *process* (`cascade worker`). Alessio keeps the profiler and the step-timing model (`cascade simulate`, `cascade compare`, `cascade import-sim-durations`, `cascade/sim/`).

- `cascade serve --sims local|sail` picks where simulators run. It replaces the `CASCADE_LAUNCHER` env var (migrated and removed in the same change).
- `cascade worker` is what every sim runs: a local subprocess, or a background exec in a Sail fork. Same image, second mode, as the charter wants. It replaces `scripts/fake_sim.py` (migrated and deleted).
  - Real mode: runs `$CASCADE_ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out` on `$CASCADE_TRACE_DIR/traces/kernelslist.g` with the config's `gpgpusim.config` and `trace.config`, in its own run dir.
  - `--replay <log>`: the stand-in, replaying a recorded Accel-Sim log.
  - Both stream log chunks and heartbeats to main and stop on a cancel ack (the contract in docs/api.md).
- Step timing: if the trace dir holds `corrected_timeline.json` (from the profiler's phases run) and `kernel_mapping.json` (uid/name/repeat/step/seq), the worker runs Alessio's importer and `simulate`, plus `compare` against the corrected timeline, and posts the result to main. Otherwise it skips that step. Building the mapping automatically stays Alessio's.
- Contract additions: `TraceSummary.has_step_timeline`, `POST /api/sims/{id}/step-timeline` (worker → main), `GET /api/runs/{id}/step-timelines`.

## Output files (2026-11-07)

Real Accel-Sim writes files into its run dir (`kernel-stats.csv`, `perf_counter.csv.gz`, `gpgpu_inst_stats.txt`, a Perfetto JSON, `exit-status.txt`), and a Sail fork's disk is gone when the fork ends.

- `cascade worker` uploads the files the simulator created or changed, plus its own log as `accelsim.log.gz`, with `PUT /api/sims/{id}/files/{name}` before its final heartbeat. Text over 1 MiB is gzipped on the way.
- Main stores metadata in SQLite (`sim_files`) and bytes on disk under `$CASCADE_DATA_DIR/files/<sim id>/`. Limits: 64 MiB per file, 256 MiB per sim. Deleting files is not a feature.
- `GET /api/runs/{id}/perfetto` is generated by main from the parsed kernel stats, so every run gets a Perfetto view whether or not Accel-Sim wrote one. Slice names use the same kernel-name simplifier as the UI (`cascade/kernel_names.py`, twin of `web/src/domain/kernelName.ts`).
- `cascade worker --replay-outputs DIR` lets the stand-in demonstrate the whole flow with Alessio's real OPT-125m outputs.
- Contract additions: `SimFile`, `RunFile`, `PUT /api/sims/{id}/files/{name}`, `GET /api/sims/{id}/files`, `GET /api/sims/{id}/files/{name}`, `GET /api/runs/{id}/files`, `GET /api/runs/{id}/perfetto`.
