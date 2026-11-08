# Native sample: facebook/opt-125m on A100

Part of `cascade-samples`, seeded at startup as a separate native `sm_80` trace and recorded
SM80_A100 run. This is not a counterfactual simulation of H100's instruction stream.

## Source and execution

- Source run: Modal volume `accelsim-traces`, `phases/e2e-a100-plan-v1/`.
- Workload: OPT-125m, prompt "Hello, my name is", 32 output tokens, four repeats, eager vLLM.
- Accel-Sim: `7ba54ded60c4ba0083410c37d22d719166acd6dc`;
  GPGPU-Sim: `4195dd4f229cbf03f4f38c49139ae94d01d23f03`.
- Config SM80_A100, 1410 MHz; eight CPU workers, dedup enabled, `verify_dedup=0`,
  `propagate=interp`, 900-second per-kernel timeout. No GPU or synthetic simulator replay used.
- Modal app `cascade-hw-sweep`, run `ap-91JYUfkTyIo4WZXimfvCL9`; results on the same volume at
  `hw-sweep/e2e-a100-plan-v1-sm80-v1/`. Executed using `cascade.sim.modal_support.accelsim_image`
  and `cascade.sim.accelsim.simulate_accelsim`, the repository's pinned CPU simulation path.
- 33 unique exact simulations, 624 traced kernel occurrences, 14,100 propagated launches;
  zero failed kernels, timeouts or unmatched untraced kernels. No dedup verification copies
  were run, so this does not independently validate context scaling.

The simulated step timeline and comparison were generated from those recorded durations using
`cascade.sim.timeline.simulate` (defaults) and `cascade.sim.compare.compare` against this
native run's `corrected_timeline.json`. Host time and gaps remain the measured A100 values.

## Mean steady request timings (ms)

| Metric | Measured A100 | SM80_A100 simulated |
|---|---:|---:|
| E2E | 222.083118 | 229.580065 |
| TTFT | 7.817560 | 8.280832 |
| ITL | 6.911792 | 7.144044 |

Signed E2E error is **+3.38%**, against A100's own baseline only. H100's sample page also
includes these native A100 rows from this trace's latest succeeded run. H200 remains metrics-only.

## Bundle

`scripts/build_sample.py` made this trimmed run directory. It contains the same gzipped JSON,
tracer stats, exact per-kernel logs and selected downloadable outputs as the H100 bundle;
the approximately 1.4 GB of `.tracez` instruction traces and simulator workspaces are omitted.
Every compressed file decompresses to its source bytes. Native A100's trace context and first
simulated tracer UID are read from its files rather than copied from H100.

After obtaining a complete native run directory including its real simulation outputs:

```sh
uv run python scripts/build_sample.py /path/to/e2e-a100-plan-v1 --sample opt125m-a100-e2e
```

The bundle is about 1.7 MB. Startup seeds it once per trace id; use a fresh data directory when
inspecting rebuilt samples. Sample traces cannot launch new simulator runs without the omitted
instruction traces. Set `CASCADE_SAMPLES=false` to disable built-in samples.
