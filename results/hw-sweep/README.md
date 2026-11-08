# OPT-125m hardware sweep

Pinned sources: Accel-Sim `7ba54ded60c4ba0083410c37d22d719166acd6dc`;
GPGPU-Sim `4195dd4f229cbf03f4f38c49139ae94d01d23f03`.
Input: the existing `phases/e2e-plan-v1` run, `facebook/opt-125m`, prompt
`Hello, my name is`, 32 output tokens, four repeats. No profiling was repeated.
There are 32 steps/request (one prefill and 31 decode), not 33 decode steps.
The instruction traces cover repeat 1, local steps 0, 2, 10, 26: 712 aligned
occurrences, 178 kernels/step. Header binary version is 90.

## Config research

`inventory.csv` / `inventory.json` list every `gpgpusim.config` in the pinned
GPGPU-Sim `configs/`: 12 tested and nine deprecated. Tested SM versions come
from `-gpgpu_compute_capability_major/minor`; deprecated entries lack those
fields, so their column reports `-gpgpu_ptx_force_max_capability`, **not an
independently validated hardware specification**. This distinction matters:
the deprecated GTX750Ti file forces 5.2 and the GTX1080 file forces 6.0.

| Tested config | Config SM | Core MHz |
|---|---:|---:|
| SM2_GTX480 | 2.0 | 700 |
| SM3_KEPLER_TITAN | 3.5 | 837 |
| SM6_TITANX | 6.1 | 1417 |
| SM75_RTX2060 | 7.5 | 1365 |
| SM75_RTX2060_S | 7.5 | 1905 |
| SM7_GV100 | 7.0 | 1447 |
| SM7_QV100 | 7.0 | 1132 |
| SM7_TITANV | 7.0 | 1200 |
| SM80_A100 | 8.0 | 1410 |
| SM86_RTX3070 | 8.6 | 1132 |
| SM90_H100 | 9.0 | 1980 |
| SM90_H200 | 9.0 | 1980 |

Accel-Sim ships matching `gpu-simulator/configs/tested-cfgs/*/trace.config`
files for all these except SM2_GTX480. Its other config directory is
`util/job_launching/configs/`: `define-standard-cfgs.yml` and
`define-LRC-cfgs.yml` register base-file aliases and composable modifiers,
not additional Blackwell architectures. Some legacy aliases reference base
directories absent from the pinned GPGPU-Sim tree.

`git ls-remote origin refs/heads/dev` on both upstream checkouts returned the
exact pins above on 2026-11-08 UTC. Searching both committed file trees found
**no SM100, B200, B100, GB200 or other Blackwell hardware config**. H200 is the
only shipped Hopper variant beyond H100. No approximation was fabricated.
The two Hopper configs share SM count (132), clocks (`1980:3960:3960:3106`),
and simple DRAM modeling; H200 changes memory-controller count from 40 to 48.
This is the shipped model, not a complete measured H100-versus-H200 hardware
specification or a capacity/bandwidth validation.

## Execution route: real Cascade app on Sail

The final sweep uses `scripts/deploy_sail.py`, the repository `Dockerfile`
with `BASE_IMAGE=nvidia/cuda:12.9.0-devel-ubuntu22.04`, and `cascade serve`
**inside a Sail sailbox**, with `CASCADE_LAUNCHER=sail`.
This VM is not a Sail guest: `sail.guest.current_sailbox_id()` raises
`InvalidArgumentError: the current sailbox is only available inside a sailbox`.
`SailLauncher.prepare()` must checkpoint its own Sail guest, so a local
`cascade serve` cannot correctly fork these simulators.

`drive_sail.py` deploys the unmodified app image, builds the pinned simulator
once on the main sailbox's disk using the known-good CPU build recipe, uploads
the existing processed traces and contract inputs, then submits `/api/runs`.
The existing `SailLauncher` checkpoints main and boots one independent sim
sailbox per compatible GPU config, disk-only (`restore_memory=False`).
`CASCADE_SIM_CMD` is the `sail_worker.py` adapter: it calls the existing
`cascade.sim.accelsim.simulate_accelsim`, posts real simulator logs and
heartbeats to main, and uploads artifacts through main's repo-provided S3
endpoint **before the terminal heartbeat**, because main terminates completed
sim sailboxes. It does not replay the fake fixture log.

The small `metadata.json` ingest envelope is generated for the app's upload
contract (`gpu`, `trace_gpu_arch=sm_90`, file sizes); corrected timeline and
alignment are copied unchanged. No synthetic alignment or traces are made.
Worker options: eight CPU processes, dedup on, `verify_dedup=0`, 900-second
per-kernel timeout, `propagate=interp`. Each distinct header group is simulated
once; the dedup implementation is unchanged.

`sail-incompatible-configs.json` records actual API attempts for the older
tested configurations. The app's architecture gate rejects them with HTTP
422 before scheduling: config architecture differs from trace `sm_90`.
Those are **rejections, not fabricated simulation numbers**. Deprecated
configs are not registered by the app and are not valid SM90 targets.

### Preliminary Modal work, superseded by Sail

Before the execution-backend change arrived, two Modal full sweeps were
launched in app `ap-jQrrDTr4SXBS3c6zM9tUYP`. They were stopped immediately
on the change; no partial H100/H200 data from them is used in Sail tables.
The separate cheap incompatibility probe had already completed in Modal app
`ap-FigYXifWoVoYJY3UgWmVy7`: nine older configs failed on the same SM90
attention trace (UID 6224), and SM2_GTX480 had no trace config. Results and
last 40 simulator log lines per config are preserved in
`incompatible-probes.json`, **preliminary Modal evidence only**, not Sail runs.
The first failure in several older models is resource admission (109568-byte
shared memory / 255 registers), not proof that Hopper opcode execution itself
was reached. Kepler also rejects the trace option `-smem_latency`.

## Results and interpretation

Both Sail simulators succeeded with **zero failed kernels or timeouts**.
Each executed 31 unique simulations for 712 traced occurrences and propagated
16,376 further durations. Replay covered 17,088/17,088 steady kernels (100%),
covering 98.13% of measured GPU time; the remaining 1.87% is retained residual.

Both configs have provenance counts **31 exact, 681 dedup, 16,376 identical,
zero interp**. Although `propagate=interp` was enabled, header dedup made all
sampled decode contexts identical, so propagation used `identical`, not
interpolation. The broad sweep therefore does **not** measure context scaling;
it deliberately trades that detail for 31 unique executions/config.

| Config | Accel-Sim wall s | Prefill GPU busy µs (mean) | Decode GPU busy µs (mean) | E2E repeat 1/2/3 ms |
|---|---:|---:|---:|---|
| SM90_H100 | 804.428951 | 822.678949 | 816.980001 | 261.133918 / 262.302298 / 262.987898 |
| SM90_H200 | 777.060434 | 823.135515 | 817.619900 | 261.154211 / 262.322591 / 263.008191 |

The shipped H200 model is **not faster** on this trace: lm_head increases from
52,002 to 55,296 cycles. Do not generalize this sub-percent request difference
to physical H200 performance; it is the outcome of the shipped simplified
models and this fixed SM90 instruction/memory stream. Simulation wall seconds
measure the CPU computation, not modeled GPU execution latency.

See [tables.md](tables.md) for the per-config tables and each config directory
for raw durations, simulator summary/cycles/wall timings, replay, comparisons,
and per-kernel/group/step/request CSVs. Kernel tables report cycles and
`µs = cycles / core_clock_mhz`, using each file's first
`-gpgpu_clock_domains` domain, not a common hard-coded clock.

Role names are **inferred**, not NVTX labels: names identify attention, KV
write, split-K reduction and elementwise work. For this trace's `nvjet` GEMMs,
grid Y of 9/3/12/6/66 identifies QKV/out-proj/MLP-up/MLP-down/lm_head, consistent
with OPT's QKV→attention→out-proj→fc1→activation→fc2 execution order and the
observed repeated 13-kernel decoder-layer block. Unknown GEMM shapes fail
report generation rather than silently receiving a label. Layer norm and
setup/index/reduction remain separate, so no kernels disappear from totals.
Tables show per-occurrence means and sums across all layers; `per-kernel.csv`
preserves every traced occurrence and full name. `kernel-groups.csv` contains
all four traced contexts, not just the two shown in Markdown.

Step GPU busy is the replay's total `gpu_busy_ns`, including its unchanged
measured residual for activities without matched kernel records. Thus it may
exceed the sum of kernel durations. Per-step CSVs include all 96 steady steps;
request metrics include repeats 1–3. Warmup has no comparable profiled kernels
and is excluded. End-to-end replay holds measured host time, inter-step gaps
and request envelopes fixed; it is **not an independently simulated serving
stack or predicted B200 request latency**.

Deduplication compares trace headers and may reuse values despite different
instruction/memory streams. Earlier verification of these inputs found up to
39.73% cycle disagreement; this cost-minimized sweep does not re-verify dedup.
Interpolation/clamping fills untraced contexts. Full kernel coverage, if
reported, is provenance coverage, **not demonstrated timing accuracy**.
H100 SASS is not a valid cross-architecture benchmark for Ampere/Volta or
Blackwell. Those require native binaries/traces and supported hardware models.

## Reproduction and cleanup

Bind `SAIL_API_KEY` from the provisioned secret, never a literal in a file:

```sh
uv sync --frozen --extra sail
export CASCADE_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run python results/hw-sweep/drive_sail.py
```

The driver deliberately reads `SAIL_API_KEY` and `CASCADE_TOKEN` only from the
environment and never prints either. Running it creates chargeable Sail
resources. Its `finally` block terminates all sailboxes in its dedicated app
and deletes the run checkpoint, including on an exception. `sail-resources.json`
records created/terminated box IDs and deleted checkpoints. This API/SDK
version has **no app-delete operation** (`/apps` supports GET, `/apps/find`
supports POST); the empty app grouping cannot be deleted programmatically.
No running or paused sailboxes should be left behind.

Actual teardown was verified: main `sb_cf6382a1-1608-4cef-b08a-439fb163cba8`,
H100 `sb_2457431e-4da3-42d4-826b-6c8060c39450`, and H200
`sb_e0aade19-0b98-417f-b8e6-9036e9554e00` all report `terminated`;
checkpoint `sbcp_34c6ed54-b25c-4ca5-bd31-1f3ebf81607b` was deleted (zero
matching checkpoints remain). Empty app `cascade-hw-sweep`
(`app_01a1ba10-7eed-7d8a-adf3-005355b2a53b`) remains because the service
does not expose deletion. No compute resources remain active.

During setup, restarting main after the runtime build left the original
rclone server alive; the replacement supervisor logged `bind: address already
in use`. The existing server continued serving artifact uploads successfully;
it was destroyed with main. This did not affect simulation or collection.

Replay a collected configuration (expand any `.gz` artifacts first):

```sh
RUN=/home/ubuntu/runs/phases/e2e-plan-v1
CFG=results/hw-sweep/SM90_H100
gzip -dk "$CFG/sim_durations.json.gz" "$CFG/simulated_timeline.json.gz"
uv run cascade simulate "$RUN/corrected_timeline.json" \
  --durations "$CFG/sim_durations.json" --output "$CFG/simulated_timeline.json"
uv run cascade compare "$RUN/corrected_timeline.json" \
  "$CFG/simulated_timeline.json" --output "$CFG/compare.json"
uv run python results/hw-sweep/report.py --source "$RUN"
uv run pytest -q results/hw-sweep/test_report.py
uv run ruff check results/hw-sweep/*.py
```

Large raw JSONs, verbatim build/serve/driver logs, and config snapshots are
committed with deterministic gzip compression. CSV line endings are normalized
to LF; numeric values are unchanged. There is
no need to upload the instruction traces again to reproduce replay/reporting.
For `report.py --source`, expand both H100 and H200 artifact pairs first.

Checks performed: ruff on the four sweep scripts (`All checks passed!`);
report tests plus existing Sail launcher, registry, API, Accel-Sim and timeline
tests (`83 passed, 1 warning, 9 subtests passed in 5.56s`); both real CLI
replays/compares; saved-artifact validation and `git diff --check`.
The one pytest warning is Starlette's preexisting `httpx` deprecation.
