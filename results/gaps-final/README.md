# Final endpoint-aligned H100 replay: gaps-final

Input: Modal `phases/gaps-final`, profiler commit `2adf744`, OPT-125m eager on H100,
32 tokens × 4 repeats. The original attachment is copied unchanged as
`corrected_timeline.json`: 128 steps, 96 captured steps, 17,088 kernels.

Input SHA-256: `710f5f847f6695e624bfcdd209d916e05a64e94279c7dcef80746954d8d2580f`.

## Reproduce on simulator

From the repository root, after `uv sync` with Python 3.12:

```bash
ROOT=results/gaps-final
for MODE in measured-only accelsim; do
  if [ "$MODE" = measured-only ]; then
    SOURCE=(--measured-only)
  else
    SOURCE=(--durations "$ROOT/sim_durations.json")
  fi
  for GAP in measured mean linear; do
    DEST="$ROOT/$MODE/$GAP"
    uv run cascade simulate "$ROOT/corrected_timeline.json" "${SOURCE[@]}" \
      --gap-model "$GAP" --output "$DEST/simulated_timeline.json"
    uv run cascade compare "$ROOT/corrected_timeline.json" "$DEST/simulated_timeline.json" \
      --output "$DEST/compare.json"
  done
done
```

All six output pairs, original input, duration mapping, actual commands/output (`run.log`),
and measured-only measured-gap sanity assessment (`sanity.json`) are committed here.
Launch-latency subtraction remains **0 cycles**.

## Endpoint handling and sanity

The fixed profiler defines pre-first-step relative to **queued_ns**. Reconstruction now uses
that reference when present, falling back to arrival for older inputs without queued_ns.
The original arrival field is preserved. Post-last-step is a finite **signed** budget;
negative values correctly place the final token inside the final step. The older diagnostic
fields `arrival_to_first_step_ns` and `last_step_end_to_return_ns` are preserved but ignored.
No fitted stamp offset, clamping, or threshold adjustment was applied.

The unchanged **1%** target gives:

|Repeat|e2e error ms|e2e error %|Mean ITL error %|TTFT error ms|TTFT error %|
|---:|---:|---:|---:|---:|---:|
|1|−0.001187|−0.000967|0.041433|0.114600|2.717988|
|2|−0.001043|−0.000852|0.031089|0.104795|2.549911|
|3|−0.001105|−0.000899|0.055761|0.102435|2.503064|

**e2e and mean ITL PASS. TTFT FAILS 1%**, as expected from the known in-step first-token stamp.
`sanity.json` records both outcomes separately; the combined all-request-metrics flag remains
false rather than silently excluding TTFT. Step-wall and GPU-busy errors are exactly zero.

### Verified TTFT cause

All three measured request first-token timestamps fall strictly inside their prefill steps.
The actual prefill-end-minus-first-token offsets are **0.114620, 0.104762, 0.102413 ms**.
The model intentionally stamps first token at prefill end. Adding the independently sampled
scalar-wall discrepancies (−20, +33, +22 ns) reproduces every TTFT error above exactly.
As a disconfirming check against a remaining queue/reference mismatch, each reconstructed
first-step start equals both the measured first-step start and queued_ns + pre_first_step_ns
exactly. No GPU replacement is involved in this measured-only check.

The small e2e residual (1.043–1.187 µs) is the aggregate discrepancy between independently
sampled scalar walls and monotonic spans. The final-token tail is applied with its original
negative sign, with no invented return-time budget.

## Gap models

Decode gap p50 is **29,751 ns ≈ 0.030 ms**. Measured total gaps per captured repeat are
**0.928321, 0.926102, 0.927750 ms**, with no growth across repeats. The pooled mean fits
**29,915.838710 ns** over 93 decode gaps. All steps have num_reqs=1, so linear reports an
unidentifiable slope and falls back to the identical mean (b=0). These are in-sample fits,
not scheduler predictions or held-out validation.

## Accel-Sim timing transfer

The scoped occurrence mapping is rebuilt from the same 10 groups derived from the earlier
20 H100 attention instruction traces. Match: exact kind + full mangled name + grid + block;
duration: median source cycles / 1,980 MHz. Source groups and cycles are embedded in
`sim_durations.json`. Coverage is **5,856 / 17,088 kernels = 34.27%**: 183 prefill and 5,673
decode occurrences. GPU-time coverage is **20,213,243 / 55,036,957 ns = 36.73%**.

Unmatched kernels retain measured durations. The old source prefill grouping remains inferred
from reshape grids/adjacency rather than verified step annotations. This is cross-run and
cross-layer timing transfer, not 5,856 new instruction traces or a fresh simulator execution.

## Checks and changes

- `cascade/sim/gaps.py`: signed finite post-step validation and queued-reference reconstruction.
- `cascade/sim/timeline.py`: update the request-assumption metadata to describe that reference.
- `tests/test_timeline.py`: queued reference, signed/zero/missing tail, ignored diagnostics,
  round-trip loading, and invalid/nonfinite tail regression tests.
- `docs/corrected-timeline-simulation.md`: updated endpoint semantics.

```text
uv run pytest
90 passed, 1 skipped, 1 warning in 16.93s

uv run ruff check .
All checks passed!
```

Skip: optional Sail. Warning: existing upstream Starlette/httpx deprecation. All six outputs
reload and their comparison reports recompute identically. Source signatures/cycles, unchanged
host budgets/warmups/diagnostics, replacement-duration deltas, preserved measured gaps,
queued-reference starts, signed final-token tails and zero launch subtraction were checked.
No scheduler, profiler code, new instruction capture, or interactive UI work was performed.

## Compare tables

Step/GPU errors are absolute; request errors are signed simulated-minus-reference in ms.
Real ITL p50 remains unavailable. Every full table is also captured in `run.log`.

### Measured-only / measured gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   0.000    0.000    0.000    0.000
GPU busy    96   0.000    0.000    0.000    0.000
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.114600    0.001585    n/a    -0.001187
2         0.104795    0.001186    n/a    -0.001043
3         0.102435    0.002138    n/a    -0.001105
```

### Measured-only / mean gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   0.000    0.000    0.000    0.000
GPU busy    96   0.000    0.000    0.000    0.000
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.114600    0.001555    n/a    -0.002117
2         0.104795    0.001228    n/a    0.000246
3         0.102435    0.002126    n/a    -0.001464
```

### Measured-only / linear gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   0.000    0.000    0.000    0.000
GPU busy    96   0.000    0.000    0.000    0.000
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.114600    0.001555    n/a    -0.002117
2         0.104795    0.001228    n/a    0.000246
3         0.102435    0.002126    n/a    -0.001464
```

### Accel-Sim / measured gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   41906.055    41871.677    43121.177    1.102
GPU busy    96   41906.055    41871.677    43121.177    7.310
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.158184    0.043394    n/a    1.338483
2         0.147739    0.043146    n/a    1.342644
3         0.146307    0.043936    n/a    1.338518
```

### Accel-Sim / mean gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   41906.055    41871.677    43121.177    1.102
GPU busy    96   41906.055    41871.677    43121.177    7.310
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.158184    0.043364    n/a    1.337553
2         0.147739    0.043187    n/a    1.343933
3         0.146307    0.043924    n/a    1.338159
```

### Accel-Sim / linear gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   41906.055    41871.677    43121.177    1.102
GPU busy    96   41906.055    41871.677    43121.177    7.310
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.158184    0.043364    n/a    1.337553
2         0.147739    0.043187    n/a    1.343933
3         0.146307    0.043924    n/a    1.338159
```
