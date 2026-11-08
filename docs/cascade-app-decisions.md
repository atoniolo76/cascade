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
- ~~A `scripts/deploy_sail.py` creates the main sailbox from the image, sets env/secrets, and exposes ports 8000 (app) and 9000 (rclone S3).~~ Superseded 2026-11-07 by `cascade deploy` and the Get started page (see Self-serve onboarding).
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

## Self-serve onboarding (2026-11-07)

A visitor gets their own running app from the landing site's Get started page (`landing/start.html`) with a workspace name and a Sail API key. No clone.

- A workspace is the Sail app `cascade-<name>` (3-32 lowercase letters, digits, hyphens) holding one sailbox, `cascade-main`, with ingress 8000 and 9000 and auto-sleep off. `/api/health` returns `CASCADE_WORKSPACE` and the UI shows it in the rail.
- One module, `cascade/deploy.py`, does every lifecycle step: `advance` (one bounded step toward healthy), `deploy_workspace` (loops `advance`), `workspace_status`, `pause`, `resume`, `delete`. `cascade deploy` and the Vercel function `landing/api/workspaces.py` both call it. It imports only `sail` and the standard library, and the docs build copies it to `landing/api/_cascade_deploy.py` because Vercel deploys `landing/` alone. A test fails when the copy drifts. `scripts/deploy_sail.py` is deleted.
- Idempotent: every call reads state from Sail. An existing sailbox is reused and resumed if paused, never duplicated. The app's settings are written once to `/data/cascade.env` (mode 600, linked into place so a concurrent writer can't replace the token), and `cascade serve` is started only when its pid file doesn't name a live `cascade serve`. A rerun recovers the token by sourcing that file and running `printenv CASCADE_TOKEN` in the sailbox. A paused sailbox resumes with its processes. One that comes back cold gets `cascade serve` restarted from the same file.
- Trust model for the key: the browser keeps it in JS memory only (never localStorage) and sends it over HTTPS in a POST body on each call. The function sets it as `SAIL_API_KEY` for that one call (calls in one process take turns behind a lock) and removes it afterwards. Nothing logs bodies, and every error message is scrubbed of the key and token. The key is copied into the workspace, because `SailLauncher` checkpoints main and forks sims with it, so it also sits on the disk that sims fork from. Sail Secrets can't carry it: in SDK 0.17.2 a Secret is only inserted into egress HTTPS requests by an egress policy, not into an exec's env, so the key goes into the env file instead.
- Polling: Vercel functions are time-limited, so the page calls `deploy` repeatedly (the key re-sent from memory) and each call does one step. The call that creates the app returns right after, so the page can show each step. An image build gives up after 25 s and the next call picks it up. `Sailbox.create` blocks until the sandbox runs, so the call that creates it returns right after.
- Image source: `.github/workflows/image.yml` builds the `Dockerfile` for linux/amd64 and pushes `ghcr.io/atoniolo76/cascade:latest` and `:<short sha>` on pushes to `amani` that touch the app. Sail pulls only public images in SDK 0.17.2, so the package must be public. Sail pins a tag to a digest per organization on first use, so a workspace created later can keep an older `:latest` until someone force-builds it; pass a `:<sha>` tag under advanced to pick a version. `cascade deploy --from-source` still builds the Dockerfile on Sail, for simulator images.
- Delete terminates every sailbox in the workspace's app (main and sim forks) and deletes its checkpoints. Sail has no API to delete an app; an empty app costs nothing.

## Output files (2026-11-07)

Real Accel-Sim writes files into its run dir (`kernel-stats.csv`, `perf_counter.csv.gz`, `gpgpu_inst_stats.txt`, a Perfetto JSON, `exit-status.txt`), and a Sail fork's disk is gone when the fork ends.

- `cascade worker` uploads the files the simulator created or changed, plus its own log as `accelsim.log.gz`, with `PUT /api/sims/{id}/files/{name}` before its final heartbeat. Text over 1 MiB is gzipped on the way.
- Main stores metadata in SQLite (`sim_files`) and bytes on disk under `$CASCADE_DATA_DIR/files/<sim id>/`. Limits: 64 MiB per file, 256 MiB per sim. Deleting files is not a feature.
- `GET /api/runs/{id}/perfetto` is generated by main from the parsed kernel stats, so every run gets a Perfetto view whether or not Accel-Sim wrote one. Slice names use the same kernel-name simplifier as the UI (`cascade/kernel_names.py`, twin of `web/src/domain/kernelName.ts`).
- `cascade worker --replay-outputs DIR` lets the stand-in demonstrate the whole flow with Alessio's real OPT-125m outputs.
- Contract additions: `SimFile`, `RunFile`, `PUT /api/sims/{id}/files/{name}`, `GET /api/sims/{id}/files`, `GET /api/sims/{id}/files/{name}`, `GET /api/runs/{id}/files`, `GET /api/runs/{id}/perfetto`.

## Samples (2026-11-08)

Profiling takes about 30 minutes (the NVBit trace of the OPT-125m run alone took 1,739 s), so a new workspace had nothing to look at for half an hour.

- `cascade serve` seeds each bundle under `samples/` on startup. Today that is `samples/opt125m-h100-e2e`, a 3.3 MB trimmed copy of Alessio's `e2e-plan-v1` run directory, rebuilt byte for byte by `scripts/build_sample.py`. `CASCADE_SAMPLES=false` skips it. The image copies `samples/`.
- `TraceSummary.sample` and `RunSummary.sample` mark it, stored in a `sample` column on `traces` and `runs` (migrated like `has_step_timeline`). The UI shows a neutral Sample badge with a tooltip wherever the trace or run appears.
- Seeding writes the files first, then every row in one transaction keyed on the trace id, so a second start does nothing and a crash in between redoes it. It reuses the store's insert helpers for kernel stats, the step timeline, output files and the log chunk.
- Kernel stats are one per kernel Accel-Sim simulated (93), parsed by the app's own `LogParser` from each kernel's log, in launch order. Each kernel ran in its own Accel-Sim process, so its log's launch uid is 1; `kernel_launch_uid` holds the tracer's uid instead. The 619 dedup copies in `kernels.csv` are not stats, because Accel-Sim didn't run them.
- The trace's real timing is the nsys duration of those 93 kernels, joined through `trace_alignment.json` to `corrected_timeline.json`, so Compare and Perfetto pair real and simulated kernel by kernel. The trace's kernel launches are the 712 traced kernels' rows of the tracer's `stats_ctx_*`, with their instruction counts. The sim's totals are the sums of its kernels' cycles and instructions, IPC and sim rate are null, and its exit code is null.
- Timestamps: the trace's `created_at` is the first traced step's wall clock (`trace_steps.json`). The run starts when Accel-Sim started (the earliest `perf_counter_<UTC time>` name, kept in `sample.json`) and finishes 719 s later (`summary.json` `wall_s`). `run.json` has no timestamp.
- Honesty about Run: without the 415 MB of `.tracez` files Accel-Sim can't rerun the sample, so the UI disables Run on sample traces with the reason in a tooltip, and `POST /api/runs` returns 409 for one. A stand-in rerun would show numbers that aren't this trace's, so it isn't offered. Cancel on the sample run is a no-op, since its sim is already `succeeded`, and the UI hides the button.
- Ingest skips sample trace dirs: the sample has no `kernel_mapping.json`, and the scan would otherwise clear its `has_step_timeline`.
