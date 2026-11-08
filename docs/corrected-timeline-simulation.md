# Corrected-host timeline simulation (offline CLI)

These commands ship in the main app image alongside `cascade serve`; they do not need CUDA,
Modal, S3, Sail, a running API, or an Accel-Sim installation. They consume previously simulated
kernel durations, not execute Accel-Sim. No scheduler, batching, or request-level vLLM execution
model is implemented.

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

`--subtract-launch-cycles N` defaults to zero and converts cycles as
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

### Request metrics (derivation only)

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
schema fixture, **not** the forthcoming real corrected artifact and not an accuracy claim.

No merge of the profiler branch is necessary: the artifact is the integration boundary. When
the real file arrives, run the same commands with its corresponding UID/seq mapping. This work
does not change the profiler's optional S3/Modal-volume storage behavior.

Real artifact runs, including measured-only request-level discrepancies and transfer assumptions,
are committed under [`results/20261108-043849`](../results/20261108-043849/README.md).
