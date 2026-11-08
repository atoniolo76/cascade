# Real gap-aware H100 replay: gaps-nosync

Input: Modal `phases/gaps-nosync`, profiler commit `2641389`, `--no-step-sync`.
H100, OPT-125m eager, 32 tokens × 4 repeats. The original attachment is copied unchanged as
`corrected_timeline.json` (128 steps, 96 captured, 17,088 measured kernels).

SHA-256: `fa7c13a0cd46337f3d9ecf36302ab9074e41e5393a5b335ba2e1e4611f6fca59`.

## Reproduce on branch simulator

From the repository root, with Python 3.12 and `uv sync`:

```bash
ROOT=results/gaps-nosync
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

All runs use **0 launch-latency subtraction** and corrected intra-step host budgets. Six pairs
of `simulated_timeline.json` / `compare.json` are committed. `run.log` contains every actual
command and comparison table; `sanity.json` contains the unchanged 1% target and its failures.

## Duration transfer and coverage

The 20 earlier H100 attention trace samples are reused through the same 10 source groups as
`results/20261108-043849/sim_durations.json`. This run's scoped occurrence mapping was rebuilt
by exact **kind + full mangled name + grid + block**, using median source cycles at 1,980 MHz.
All source groups and cycles are embedded in the new `sim_durations.json`.

Coverage: **5,856 / 17,088 kernels = 34.27%**, consisting of 183 prefill and 5,673 decode
occurrences. Measured GPU-time coverage is **20,230,830 / 55,012,025 ns = 36.78%**. Unmatched
kernels retain measured durations. As before, source prefill/decode grouping is inferred from
reshape token grids and sequence adjacency, not verified old step annotations. Timings are
transferred across matching layers and runs, not newly simulated instruction traces.

## Sanity result: FAIL for e2e; PASS for mean ITL

Measured-only + measured gaps reproduces all 96 step walls and GPU-busy budgets exactly.
Mean ITL error is below 0.06%. However, **request e2e exceeds the 1% target** on every repeat;
TTFT also differs substantially. The target and request formulas were not changed to pass.

This is a **reference-endpoint mismatch**, established by the artifact arithmetic and the
producer code at `2641389:accelsim_trace.py`:

- `_corrected_timeline` lines 1455–1473 uses converted `arrival_time` for `arrival_ns`, and
  `generate_return_monotonic_ns` to derive `post_last_step_ns`.
- The same function reports TTFT/e2e from **queued_ts**, ending at first/last token timestamps,
  not arrival-to-generate-return. Lines 1589–1591 pass baseline request metrics into it.
- The simulator intentionally retains the approved formula: pre-first-step + walls + measured
  gaps + post-last-step, referenced to arrival. It therefore includes time not present in the
  queued-to-last-token reference metric.

For repeat 1, the +2.219840 ms e2e error decomposes exactly into:

```text
queued_ns - arrival_ns                     0.677819 ms
last_step_end_ns - last_token_ns            0.220796 ms
post_last_step_ns (through generate return) 1.320986 ms
scalar-wall versus timestamp-span delta    0.000239 ms
TOTAL                                     2.219840 ms
```

The other repeats have the same endpoint mismatch; exact components and percentage errors are
in `sanity.json`. The producer's request e2e equals `(last_token_ns - queued_ns)/1e6` on all three
repeats. As a disconfirming check, **walls + gaps alone** match reference e2e within 0.046 ms,
consistent with the supplied observation. The extra pre/post/reference terms, not the gaps or
simulated durations, account for the measured-only failure. The positive post field is host
time until generate returns; actual last-token timestamps precede the final step end.

Recommended next action: agree on one request reference/end event and export compatible
boundary budgets and reference metrics. Do not fit extra overhead or silently remove budgets
just to satisfy the threshold. No scheduler or request-formula change was made here.

## Mean/linear gap model limitations

Measured total gaps grow by repeat: **32.745459 → 45.172591 → 60.491633 ms**. The per-kind pooled
mean assigns the same total gap budget to each request, so it overpredicts repeat 1 and
underpredicts repeat 3. Every step has `num_reqs=1`; the linear slope is unidentifiable and
falls back to the same mean, explaining the identical mean/linear outputs. This is an in-sample
model comparison, not held-out prediction accuracy.

## Loader compatibility correction

The original loader required scalar wall durations and absolute timestamp spans to match to
1 ns. This artifact uses independently sampled boundaries: baseline scalar walls use wall-clock
calls while boundaries use monotonic calls; profiled scalar walls use NVTX spans while the
boundaries wrap the instrumented step. Baseline span-minus-wall ranges from −110 to +89 ns;
profiled differences range from 22,816 to 54,635 ns. These are not the same measurement.

The loader now validates ordered start/end pairs without asserting scalar equality. It still
checks baseline adjacency against baseline gaps, finite/nonnegative budgets and other existing
invariants. No input timings are normalized or changed. A regression test covers independent
samples. The sub-microsecond baseline discrepancy cannot explain the millisecond request
failure, as the exact decomposition above shows.

## Verification

```text
83 passed, 1 skipped, 1 warning in 17.45s
All checks passed!
```

Commands: `uv run pytest`, `uv run ruff check .`. Skip: optional Sail dependency. Warning:
existing upstream Starlette/httpx deprecation. All six outputs were loaded again; every
comparison recomputed identically; all 5,856 transferred signatures/cycles were checked; host
budgets and warm-ups were preserved; GPU/wall deltas equaled kernel replacements; measured-gap
runs retained every baseline gap. No interactive UI or new simulator execution was performed.

## Compare tables

Complete tables follow below and are also in `run.log`. Step/GPU columns show absolute errors;
request columns show signed simulated-minus-reference errors. Real ITL p50 is unavailable and
remains n/a.

### Measured-only / measured gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   0.000    0.000    0.000    0.000
GPU busy    96   0.000    0.000    0.000    0.000
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.799429    0.003207    n/a    2.219840
2         0.670852    0.001973    n/a    3.065603
3         0.612244    0.002437    n/a    2.809626
```

### Measured-only / mean gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   0.000    0.000    0.000    0.000
GPU busy    96   0.000    0.000    0.000    0.000
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.799429    0.435178    n/a    15.610942
2         0.670852    0.033069    n/a    4.029573
3         0.612244    -0.460630    n/a    -11.545446
```

### Measured-only / linear gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   0.000    0.000    0.000    0.000
GPU busy    96   0.000    0.000    0.000    0.000
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.799429    0.435178    n/a    15.610942
2         0.670852    0.033069    n/a    4.029573
3         0.612244    -0.460630    n/a    -11.545446
```

### Accel-Sim / measured gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   41722.857    41807.677    42757.677    0.784
GPU busy    96   41722.857    41807.677    42757.677    7.281
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.841890    0.044870    n/a    3.553838
2         0.714594    0.043673    n/a    4.402054
3         0.655506    0.044104    n/a    4.144570
```

### Accel-Sim / mean gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   41722.857    41807.677    42757.677    0.784
GPU busy    96   41722.857    41807.677    42757.677    7.281
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.841890    0.476841    n/a    16.944940
2         0.714594    0.074769    n/a    5.366024
3         0.655506    -0.418963    n/a    -10.210502
```

### Accel-Sim / linear gaps

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   41722.857    41807.677    42757.677    0.784
GPU busy    96   41722.857    41807.677    42757.677    7.281
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.841890    0.476841    n/a    16.944940
2         0.714594    0.074769    n/a    5.366024
3         0.655506    -0.418963    n/a    -10.210502
```
