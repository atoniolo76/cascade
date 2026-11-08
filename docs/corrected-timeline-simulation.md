# Corrected-host timeline simulation (offline CLI)

The timing-model commands ship alongside `cascade serve`; they do not need CUDA, Modal, S3,
Sail, a running API, or an Accel-Sim installation. They consume previously simulated kernel
durations, not execute Accel-Sim. The separate preparation helper validates an existing
Accel-Sim installation and returns a run plan without executing it. No scheduler, batching,
or request-level vLLM execution model is implemented.

## Worker integration and automated preparation

Ownership follows `docs/cascade-app-decisions.md` on `amani`: the worker owns execution,
environment setup, heartbeats, cancellation, log streaming and posting results. This branch
provides library functions and CLI commands only; it does not implement or modify the worker.

`cascade.sim.runner.prepare_run(framework, config_name, kernelslist, run_dir)` returns a JSON-
compatible plan with `command` (an argv list), `cwd`, `clock_mhz` and the configured launch
latency. The run directory must be empty. It locates:

- `gpu-simulator/gpgpu-sim/configs/tested-cfgs/<name>/gpgpusim.config`;
- `gpu-simulator/configs/tested-cfgs/<name>/trace.config`;
- each config's relative `-inter_config_file` companion.

It copies those configs/companions into the run directory and keeps the configured launch
latency unchanged. Trace paths and argv are absolute, with no shell interpolation. It does
not build the binary or source `gpu-simulator/setup_environment.sh`: the worker must provide
the simulator's runtime environment and execute the returned argv with the returned cwd.

```bash
cascade prepare-sim-run --framework "$CASCADE_ACCELSIM_DIR" \
  --config-name SM90_H100 --kernelslist "$CASCADE_TRACE_DIR/traces/kernelslist.g" \
  --run-dir /data/sim-run --output /data/run-plan.json
```

### Generate kernel_mapping.json and sim_durations.json

`cascade.sim.mapping.build_mapping(stats_path, kernelslist, timeline_path, clock_mhz, ...)`
returns `mapping`, `durations`, and kernel/GPU-time `coverage`. Input stdout must have the
completion marker; CSV must have `uid,name,gpu_sim_cycle`. This supports fresh single-process
launch UIDs 1..N, one per kernel entry in list order (Memcpy commands/comments are skipped).
Logs with aggregate/concurrent kernel snapshots are rejected; generated mappings additionally
reject missing/reordered UID/name associations.

Headers are read without loading instruction bodies: plain `.trace`, `.trace.xz`, `.gz`,
`.bz2`, and `.tracez`'s text header are supported. Required fields are kernel name/id and
grid/block dimensions. Kernel names and numerical dimensions match exactly; legacy `NxNxN`
geometry strings are normalized to triplets.

**Identity is never inferred from shape or filename alone.** Exact matching requires either:

- a unique `trace_kernel_id` in a corrected-timeline kernel, equal to the capture header ID; or
- `--identity capture_identity.json`, a trusted capture-side list of
  `uid/name/repeat/step/seq` records linking these artifacts. UID/name/geometry are checked,
  and one launch UID cannot claim several exact occurrences.

The current profiler's corrected timelines do not contain capture IDs. Without a capture
link, headers/step-local seq cannot prove instruction identity even for a nominally same-run
capture. Have the producer export IDs/the sidecar, or use the explicitly labeled fallback.

Shape transfer is **opt-in** (`--allow-shape-transfer`) and requires known source phases:
`--source-phases` takes a list `[{"uid":1,"kind":"prefill"}, ...]`. Exact identities can also
establish phases for their source UIDs. Unknown phases are never guessed. Transfer groups use
kind (prefill/decode), full mangled name, grid and block, take median source cycles at the
provided/configured core clock, and apply to every matching occurrence not already identified.

```bash
cascade import-sim-durations --stats /data/sim-run/run.log \
  --kernelslist "$CASCADE_TRACE_DIR/traces/kernelslist.g" \
  --timeline "$CASCADE_TRACE_DIR/corrected_timeline.json" \
  --source-phases /data/source_phases.json --allow-shape-transfer \
  --config /data/sim-run/gpgpusim.config \
  --mapping-output "$CASCADE_TRACE_DIR/kernel_mapping.json" \
  --output /data/sim_durations.json
```

Use `--clock-mhz` instead of `--config` to supply a clock explicitly. No matches is an error,
not a claim of useful coverage. The generated list preserves the agreed worker fields:

```json
[{"uid":6,"name":"gemm","repeat":1,"step":33,"seq":4,
  "method":"shape_transfer","source_uids":[6,11,16],"source_group":5}]
```

Identity records have `method:"identity"` and a single source UID. Legacy mapping lists without
`method` are still accepted. The importer uses the median of `source_uids` for transfer entries,
not just the representative `uid`. Thus the worker's existing sequence remains valid:

```bash
cascade import-sim-durations --stats /data/sim-run/run.log \
  --mapping "$CASCADE_TRACE_DIR/kernel_mapping.json" --clock-mhz 1980 \
  --output /data/sim_durations.json
cascade simulate "$CASCADE_TRACE_DIR/corrected_timeline.json" \
  --durations /data/sim_durations.json --output /data/simulated_timeline.json
cascade compare "$CASCADE_TRACE_DIR/corrected_timeline.json" /data/simulated_timeline.json \
  --output /data/compare.json
```

The existing `import_durations(stats_path, mapping_path, clock_mhz)` API is unchanged. No API,
worker, shared types, container configuration, or `amani` code is modified here.

### Six-configuration replay in one command

```bash
cascade simulate-batch /data/corrected_timeline.json --durations /data/sim_durations.json \
  --output results/my-run
```

Library equivalent: `cascade.sim.pipeline.run_pipeline(input_path, durations_path, output)`.
This copies the inputs and writes measured-only and Accel-Sim modes, each with measured, mean
and linear gaps, under `<mode>/<gap>/simulated_timeline.json` and `compare.json`. It also writes
README tables, `run.log`, and `sanity.json`. Existing named outputs are overwritten on rerun;
choose a separate directory to preserve earlier results. Launch subtraction defaults to 0
(`--launch-latency-cycles` overrides only Accel-Sim modes). Sanity uses measured-only measured
gaps, retains the 1% target and separates e2e/ITL from TTFT. Missing percentages/empty request
sets do not count as passing. No fitted correction hides an in-step first-token stamp.

### Reproduce the historical gaps-final transfer

The old source phase inference is now explicit data in
`tests/fixtures/timeline/h100_source_phases.json`, not hard-coded UID ranges in the matcher.
Its annotations remain inferred (6-token reshape + neighboring attention sequence), not
verified step labels. On the research VM:

```bash
SOURCE=/home/ubuntu/repos/opt125m
FRAMEWORK=/home/ubuntu/repos/accel-sim-framework
OUT=/data/automated-gaps-final
cascade import-sim-durations --stats "$SOURCE/sim_H100/kernel-stats.csv" \
  --kernelslist "$SOURCE/run/traces/kernelslist.g" \
  --timeline results/gaps-final/corrected_timeline.json \
  --source-phases tests/fixtures/timeline/h100_source_phases.json --allow-shape-transfer \
  --config "$FRAMEWORK/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config" \
  --mapping-output "$OUT/kernel_mapping.json" --output "$OUT/sim_durations.json"
cascade simulate-batch results/gaps-final/corrected_timeline.json \
  --durations "$OUT/sim_durations.json" --output "$OUT"
```

This reproduces all 5,856 duration records and 10 source groups exactly. Provenance is newly
computed from actual input/stat/header/annotation hashes rather than copied historical prose
and a profiler SHA absent from the input metadata. All 12 timeline/compare files and sanity.json
are byte-identical to the committed results; README/run.log intentionally become generic
generated reports rather than historical hand-written notes/subprocess logs.

### Artifacts and transport

The gaps-final timing input is 7,590,537 bytes. One generated timeline is 7,885,203 bytes
(about 1.04× input); compare.json is 42,310 bytes and mapped durations about 0.84 MB. These are
ordinary JSON files/dicts and can be transported independently. Main can receive compact
comparison/coverage data while full timelines remain artifacts. The worker owns posting and
observability; no network transport or API-limit assumptions are implemented/tested here.

The latest timing replays did **not** produce fresh raw Accel-Sim instruction traces. Their
source is the earlier `opt125m/run/traces/kernelslist.g` plus every referenced `.tracez` file;
the original raw simulator outputs remain in `opt125m/sim_H100/` (`run.log`, `kernel-stats.csv`,
performance counters). Configs, binary and runtime environment are additional execution
requirements. The committed timing-model outputs are separate under `results/gaps-final/`.

The profiler phases artifacts live on Modal volume `accelsim-traces`, under
`phases/gaps-final/` (mounted as `/traces/phases/gaps-final/` inside its function):

```bash
modal volume get accelsim-traces phases/gaps-final ./gaps-final
```

That phases run supplies timing/nsys artifacts, not a new NVBit instruction trace capture.
For actual Accel-Sim execution, download the separately captured instruction-trace bundle,
including kernelslist and every referenced file; corrected_timeline.json alone is insufficient.

### Audit of remaining research-only steps (not automated here)

The earlier `opt125m/aligned/align_timeline.py` and `opt125m/calib/` also contain manual
latency sweeps and calibration/fitting, nsys SQLite API/correlation/transfer/clock analysis,
host-driven replay with measured launch delays/proxy step boundaries, and TraceProcessor
overlap/spill validation. These are capture-conditioned research diagnostics, not prerequisites
of the corrected-host worker contract. They were reported separately and deliberately not
ported into the production pipeline. Trace capture/preprocessing, clock locking, and workload
phase labeling remain profiler responsibilities. No scheduler or causal host-feedback model
was added.

## Inputs and schema

Input is the profiler's `phases/<run>/corrected_timeline.json`:

```json
{
  "meta": {"run_name": "example", "model": "facebook/opt-125m"},
  "steps": [{
    "step": 0, "repeat": 0, "warmup": false, "kind": "prefill",
    "num_reqs": 1, "num_scheduled_tokens": 6,
    "baseline_wall_ns": 4790000, "profiled_wall_ns": 6040000,
    "gpu_busy_ns": 570000, "cuda_api_ns": 30000, "host_ns": 4220000,
    "kernels": [{"seq": 656, "name": "gemm", "grid": [4, 9, 1],
                 "block": [384, 1, 1], "duration_ns": 570000,
                 "start_offset_ns": 200000}]
  }],
  "requests": [{"repeat": 0, "warmup": false, "queued_ms": 0.2,
                "ttft_ms": 4.99, "itl_ms_mean": null, "itl_ms_p50": null,
                "e2e_ms": 4.99, "num_output_tokens": 1}],
  "summary": {}
}
```

Other profiler metadata/step/kernel fields are preserved. Non-warm-up `(repeat, step)` keys and
kernel `seq` within each step must be unique. Warm-ups are copied unchanged, including null
profiled fields and empty kernels; they are excluded from fitting, coverage, comparison and
request derivation. Values must be finite/nonnegative, host + GPU busy must match baseline wall
within 1 ns, and kernel duration sums must not exceed GPU busy. Out-of-order kernels are
rejected. Observed overlaps are serialized with zero intervening gap and reported per step as
`measured_kernel_overlap_ns`; this is a serial budget reconstruction, not a concurrency model.

### Simulated duration file

Use JSON with a `kernels` list (or just a list):

```json
{"kernels": [
  {"repeat": 0, "step": 0, "seq": 656, "sim_cycles": 10120, "clock_mhz": 1980},
  {"seq": 657, "duration_ns": 1226.77}
]}
```

Supply either `sim_cycles` + positive `clock_mhz`, or `duration_ns`, never both. A scoped entry
matches `(repeat, step, seq)`. An unscoped entry matches a globally unique kernel seq; ambiguous
seqs require scoping. Missing entries fall back to measured durations; unused or duplicate keys
are errors so a wrong mapping cannot silently imply useful coverage.

`--launch-latency-cycles N` (`--subtract-launch-cycles` remains an alias) defaults to zero and
converts cycles as
`(sim_cycles - N) / clock_mhz * 1000` nanoseconds. It requires cycle-based entries and rejects
negative resulting cycles. This exposes calibration uncertainty: the H100 experiments found
that removing all 5,000 configured cycles systematically underestimates GPU execution, while
keeping all of them overestimates tiny kernels. Do not automatically subtract a particular
constant or claim a fit on the same workload is independent validation.

### Import Accel-Sim stdout or kernel-stats CSV

```bash
cascade import-sim-durations \
  --stats /data/sim/run.log --mapping /data/kernel_mapping.json \
  --clock-mhz 1980 --output /data/sim_durations.json
```

The same helper accepts `kernel-stats.csv` with `uid,name,gpu_sim_cycle`. JSON mapping is a list:

```json
[{"uid": 1, "name": "gemm", "repeat": 0, "step": 0, "seq": 656}]
```

Names must match exactly, UIDs must exist, and log import requires a completion marker.
It reuses `cascade.stats.LogParser`. Aggregate/concurrent stats blocks cannot reliably identify
individual durations and are rejected. A mapping may intentionally reuse a simulated UID for
several scoped kernel occurrences, but this is an explicit transfer assumption, not inferred
operator equivalence. The output validates through the same duration loader.

## Host model and time conventions

Default `--host-model scaled-gaps` uses corrected `host_ns`, **never the sum of profiled gaps**.
For each step it computes measured gaps from offsets relative to that step's start:

1. Leading gap = first kernel's `start_offset_ns`.
2. Internal gap = next start minus preceding kernel end.
3. Trailing gap = `profiled_wall_ns` minus last kernel end.

All gaps are scaled proportionally so their sum is exactly corrected `host_ns`. Simulated or
fallback durations are then inserted sequentially. If all measured gaps are zero, host time is
distributed uniformly over the leading/internal/trailing slots. If kernels are absent, wall is
`host_ns + gpu_busy_ns`; the busy time is retained as measured, with zero simulated coverage.

`--host-model parametric` fits `host = a + b * number_of_kernels` separately for prefill/decode
from non-warm-up input steps by ordinary least squares. With constant kernel count (including
a single sample), slope is unidentifiable: b=0, a=mean host. Fits and identifiability are reported
under `summary.simulation.host_fit`; negative fitted host predictions are clamped to zero.
The same gap weights distribute this fitted host budget. This in-sample model is not a vLLM
scheduler and may change measured-only step walls; the baseline sanity check uses scaled-gaps.

Output `simulated_timeline.json` keeps the top-level `meta/steps/requests/summary` schema.
For simulated steps, `baseline_wall_ns` is the simulated wall, `host_ns` is corrected/fitted host,
`gpu_busy_ns` is the sum of replacement/fallback durations, and kernel offsets are reconstructed.
`profiled_wall_ns` is set to synthesized wall so the offsets remain internally consistent;
`cuda_api_ns` becomes null because CUDA API cost is not separately predicted. Original profiler
timings remain in the input, not mislabeled as simulated measurements. `summary` is recomputed
with per-kind means, fits, launch subtraction and aggregate/per-step coverage.

Coverage uses both kernel count and **original measured GPU time** of covered kernels divided
by original total measured GPU time; it is independent of whether simulated kernels get faster.
No-kernel steps contribute their measured GPU busy budget to the denominator, not the numerator.
The real profiler defines GPU busy as the union of kernels, memcpy and memset, but only exports
kernel slices. Any positive residual `gpu_busy_ns - sum(kernel durations)` is preserved as a
net unmodeled GPU budget (reported in coverage), never reclassified as corrected host time or
counted as simulated. This is activity absent from the kernel list minus overlap adjustments. It is
included in step wall/busy totals; event placement is unavailable, so optional Perfetto output
shows kernel slices and full steps, not invented transfer slices. Inputs with kernel sums larger
than GPU busy need a real concurrency model and are rejected.

### Request metrics for legacy files (derivation only)

Supports one non-warm-up request per repeat and `num_reqs=1`. Steps are taken in input order;
all prefill walls for that repeat are summed, as are decode walls:

- TTFT = preserved `queued_ms` + sum of prefill walls.
- ITL mean / p50 = mean / median of decode walls; null if no decode steps.
- e2e = queue + prefill walls + decode walls.

Warm-up request rows are unchanged. Multiple active requests are rejected rather than inventing
a scheduler. No CPU sampling/token-delivery delay outside step walls is predicted. Request metric
errors may therefore remain even with measured-only durations; this indicates those boundaries
do not fully explain observed request timing. Comparison uses supplied unprofiled request metrics,
not re-derived real values, so that discrepancy is visible.

### Inter-step host gaps and arrival-referenced requests

Simulator work lives on branch **`simulator`**. Profiler code lives on **`profiler`**; merge that
branch only if code is needed. The corrected JSON remains the interface.

New files may include `baseline_wall_start_ns`, `baseline_wall_end_ns`,
`baseline_gap_before_ns`, and their `profiled_*` counterparts. Gaps are time outside the step
wall, separate from the intra-step corrected host budget. Only **baseline** gaps drive the
reconstruction; profiled gaps are not substituted for missing baseline gaps.

`--gap-model measured|mean|linear` defaults to **measured**. Activation is per repeat: if no
baseline gap fields exist, legacy request derivation and output are unchanged. If any baseline
gap field exists in a captured repeat, every captured step in that repeat must supply it:
first step null, subsequent gaps finite/nonnegative. Step order is input order. Provided
start/end pairs must be ordered; baseline adjacent timestamps must match the gap. Scalar wall
durations and timestamp spans can be independently sampled: baseline wall uses wall-clock calls
while boundaries use monotonic calls; profiled wall uses NVTX while boundaries wrap the step.
They are not required to be identical. Scalar walls remain the simulation budgets, unchanged.
Warm-up rows, including null fields, are copied unchanged.

- `measured`: preserve each measured `baseline_gap_before_ns` exactly.
- `mean`: replace each non-first gap with the mean observed gap for its destination step kind.
- `linear`: fit `gap = a + b*num_reqs` per destination kind using non-first gaps. With constant
  request count, use b=0 and a=mean; report the unidentifiable slope. Clamp negative predictions
  to zero. These are in-sample host-gap fits, not a scheduler or a prediction of batching.

Fits are in `summary.simulation.gap_fit`; simulated gap n/mean/p50/p95 statistics are under
`summary.per_kind.<kind>.gap_before_ns`. First-step null gaps are excluded from both.

For gap-aware repeats, use **queued_ns** as the request reference when supplied, otherwise
fall back to **arrival_ns** (or zero). Start the timeline at **reference + pre_first_step_ns**, then place
each subsequent step after its preceding simulated wall and selected gap. Output baseline
start/end/gap fields are reconstructed; existing profiled start/end/gap fields are normalized
to the synthesized timeline, consistently with the synthesized `profiled_wall_ns`.

Request boundaries and metrics are reconstructed as follows (ns converted to ms for metrics):

- First-step start = reference + pre-first-step. Pre-first-step already includes queue/dispatch
  time; **do not add queued_ms again**.
- TTFT = pre-first-step + all prefill walls and gaps through the final prefill end. Prefill
  must precede decode. First-token timestamp is that final prefill end.
- Each ITL sample = decode wall + that decode step's gap-before; mean/p50 summarize samples.
- e2e = pre-first-step + all step walls + gaps between steps + post-last-step.
- Last-token timestamp = last-step end + post-last-step. Post-last-step contributes to e2e,
  not the wall-plus-gap ITL samples defined above.

The first-step gap is null, never an additional pre-first-step budget. If optional request
boundaries are absent, arrival defaults to 0, pre-first-step defaults to queued_ms (or 0), and
post-last-step to 0. Provided arrival/queued/pre values must be finite/nonnegative; post-last-step
must be finite but may be negative when the last token is stamped before the final step ends.
The diagnostic fields `arrival_to_first_step_ns` and `last_step_end_to_return_ns` are preserved
but ignored. TTFT/e2e use the reference above; first token remains the modeled prefill end,
not an offset fitted to the real in-step stamp. No request
timing fields are required for legacy files without baseline gaps. No multi-request scheduler
is introduced: request derivation still requires one active request per repeat and num_reqs=1.

The extended offline fixture uses baseline decode gaps 0.8/1.0/1.2 ms, pre-first-step 0.2 ms,
and post-last-step 0. Measured-only + measured gaps reproduces its TTFT/ITL/e2e exactly, including
ITL mean 5.78 ms and e2e 24.04 ms. Mean/linear (constant num_reqs) preserve total gaps but can
change ITL p50. This synthetic fixture is not a replacement for the forthcoming real artifact.

```bash
uv run cascade simulate tests/fixtures/timeline/corrected_timeline.json \
  --measured-only --gap-model measured --output /data/gap-sanity.json
uv run cascade compare tests/fixtures/timeline/corrected_timeline.json \
  /data/gap-sanity.json --output /data/gap-sanity-compare.json
# Substitute --gap-model mean or linear to exercise fitted gaps.
```

The prior real artifact lacks the new fields, so its measured-only and unsubtracted Accel-Sim
outputs remain byte-identical. The new real gap-aware file must still be supplied before the
request-level ≤1% sanity criterion can be checked on actual data.

## Commands and offline fixture

Install the app with `uv sync` (Python 3.12) or use the main container image:

```bash
uv run cascade import-sim-durations \
  --stats tests/fixtures/timeline/kernel-stats.csv \
  --mapping tests/fixtures/timeline/mapping.json --clock-mhz 1980 \
  --output /data/demo/sim_durations.json

uv run cascade simulate tests/fixtures/timeline/corrected_timeline.json \
  --durations /data/demo/sim_durations.json --output /data/demo/simulated_timeline.json

uv run cascade compare tests/fixtures/timeline/corrected_timeline.json \
  /data/demo/simulated_timeline.json --output /data/demo/compare.json \
  --perfetto /data/demo/compare_trace.json

uv run cascade simulate tests/fixtures/timeline/corrected_timeline.json \
  --measured-only --output /data/demo/measured_only.json
uv run cascade compare tests/fixtures/timeline/corrected_timeline.json \
  /data/demo/measured_only.json --output /data/demo/sanity.json

uv run pytest tests/test_timeline.py
uv run ruff check .
uv run pytest
```

For an already-built Docker image, replace `uv run cascade` with
`docker run --rm -v "$PWD/tests/fixtures/timeline:/input:ro" -v "$PWD/demo:/data/demo" IMAGE cascade`
and use `/input/...` paths. No image or simulator build is required just to test this module.

`compare.json` includes per-step signed wall/GPU-busy errors and percentages; aggregate signed
means plus absolute mean/p50/p95 errors and percentages (linear-interpolated quantiles); and
per-request TTFT/ITL/e2e errors. Zero reference and nonzero prediction yields null percentage,
not infinity. The printed table is the compact absolute-error summary. Empty comparisons report
count 0 and null metrics. Warm-ups are excluded.

Optional Chrome JSON has real-corrected-host and simulated processes, separate step/kernel
tracks per repeat, and integer-ns boundary quantization (duration = quantized end minus start).
The real track uses measured durations plus corrected gaps, not the inflated profiled offsets.
Repeat tracks start at zero independently; no cross-repeat scheduling or absolute wall-clock
capture reconstruction is implied.

## Fixture provenance and limits

`tests/fixtures/timeline` uses actual H100 OPT-125m kernel names, grid/block sizes and durations
from 683 nsys launches, and actual cycles for 20 traced layer-10 attention kernels. Warm-up,
prefill/decode assignment, queue/API fields and baseline/profiled walls are synthetic. Measured
gap weights are capped and rescaled to representative profiled step budgets; the first decode
baseline/profiled walls are 4.79/6.04 ms. Other walls vary to exercise fitting. This is an offline
schema fixture, **not** a real corrected artifact and not an accuracy claim.

No merge of the profiler branch is necessary: the artifact is the integration boundary. For
new runs, use the same commands with the corresponding UID/seq mapping. This work
does not change the profiler's optional S3/Modal-volume storage behavior.

Real artifact runs, including measured-only request-level discrepancies and transfer assumptions,
are committed under [`results/20261108-043849`](../results/20261108-043849/README.md).
The endpoint-aligned gap-aware run and all six configurations are under
[`results/gaps-final`](../results/gaps-final/README.md), including signed post-step budgets
and separate e2e/ITL versus in-step TTFT sanity outcomes.
