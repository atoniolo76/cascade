# Sample: facebook/opt-125m on H100, with H100/H200 Accel-Sim

Every Cascade workspace opens with this sample, so you can explore a finished trace and run before
your own trace profiles. Profiling takes about 30 minutes. The NVBit trace alone took 1,739 s.

Alessio profiled and simulated this run. The files here are a trimmed copy of his complete
`e2e-plan-v1` run directory. The numbers are his. `scripts/build_sample.py` rebuilds this
directory from that run directory, and every gzipped file decompresses to its source byte for byte.

## The run

- Workload: facebook/opt-125m, prompt "Hello, my name is", 32 output tokens (1 prefill and 31
  decode steps), 4 repeats, eager mode, vLLM 0.21.0, on an NVIDIA H100 80GB HBM3.
- Traced: steps 0, 2, 10 and 26 of repeat 1 (prefill and decode contexts 8, 16 and 32), 712
  kernels, with NVBit.
- Simulator: Accel-Sim 7ba54de, config SM90_H100, cycles converted to time at 1980 MHz.
- Simulated: 31 distinct kernels, 93 Accel-Sim runs including 2 verification copies of each.
  Durations propagate to all 17,088 kernel launches. Host time and the gaps between steps are
  replayed from the real, unprofiled run.

## Results (simulated minus real)

| Request | Real e2e | Sim e2e | Error |
|---|---|---|---|
| 1 | 253.44 ms | 261.11 ms | +3.03% |
| 2 | 254.61 ms | 262.28 ms | +3.01% |
| 3 | 255.31 ms | 262.96 ms | +3.00% |

TTFT +3.9 to +4.9%; ITL mean +3.0%; step wall +3.0% (+240 us/step); GPU busy per step +42% decode,
+34% prefill. Accel-Sim over-predicts small kernels (about a 2.7 us floor), which drives the GPU-busy
error. In eager mode host time dominates (about 7.9 ms per decode step against 0.57 ms on the GPU),
so the end-to-end error is diluted. `source-README.md` has the full method and caveats.

## Included

Paths match the run directory. JSON and the per-kernel logs are gzipped (`.gz`).

- `run.json`, `workload.json`, `reconcile.json`: run metadata, stage wall times, and the baseline
  against profiled reconciliation.
- `corrected_timeline.json`: the real timeline. Unprofiled step walls, gaps and request endpoints,
  plus the nsys kernels of every step.
- `trace_alignment.json`: each traced kernel matched to its timeline kernel (712 of 712).
- `sim_durations.json`: Accel-Sim cycles for every kernel launch, with how each was obtained.
- `traced/trace_steps.json`, `traced/traces/kernelslist.g` and
  `traced/traces/stats_ctx_0x54c35c0`: the traced steps, the kernel list Accel-Sim ran, and the
  tracer's launch dimensions and instruction counts for all 23,294 launches of the traced run.
- `sim/kernels.csv`, `sim/summary.json`: Accel-Sim's per-kernel results and the dedup check.
- `sim/simulated_timeline.json`, `sim/compare.json`: the simulated timeline and its errors against
  the real one.
- `sim/kernel-<uid>.log`: the Accel-Sim output of each of the 93 simulated kernels, and
  `sim/setup.log`.
- `source-README.md`: the run directory's own README.
- `sample.json`: the ids Cascade seeds, and when Accel-Sim started (the earliest
  `sim/kernel-*/perf_counter_<UTC time>.csv.gz` name).

## Omitted

- The 712 `traced/traces/kernel-*.tracez` instruction traces (415 MB). Without them Accel-Sim
  can't rerun, so Cascade doesn't offer Run on the sample trace.
- The per-kernel Accel-Sim workspaces under `sim/kernel-*/` (configs, perf counters, instruction
  stats; 35 MB). Their logs are kept.
- `baseline/`, `profiled/` (including the nsys database), the spinlock detection outputs and the
  tracer logs.

## How Cascade shows it

- The trace `sample__facebook_opt-125m__h100-e2e` is `ready`. Its kernel launches are the 712 traced
  kernels' rows of `stats_ctx_0x54c35c0`, read with the parser ingest uses for any trace.
- The run `run-sample-opt125m-h100-e2e` has one SM90_H100 sim with one kernel stat per simulated
  kernel (93, in launch order), parsed from that kernel's log. Its `kernel_launch_uid` is the
  tracer's kernel uid, as in `kernels.csv`.
- The trace's real timing is the nsys duration of those 93 kernels, joined through
  `trace_alignment.json` to `corrected_timeline.json`, so Compare lines them up one to one.
- The run's step timeline is `sim/simulated_timeline.json` with `sim/compare.json`.
- Its output files are `kernels.csv`, `summary.json`, `compare.json`, `setup.log` and
  `kernel-6207.log`.
- The sim's start time comes from `sample.json`, and its end is that plus the Accel-Sim wall time
  in `summary.json` (719 s).
- Set `CASCADE_SAMPLES=false` to start a workspace without it.

## Multi-architecture sample set

This bundle belongs to `cascade-samples`. The native H100 recording above is unchanged:
mean request E2E is 262.118311 ms against measured H100 254.452178 ms (+3.01%). It is the
original 93-simulation recording, not the later 31-simulation H100 hardware sweep (262.141371 ms).

`sim/SM90_H200/` adds the recorded hardware-sweep result from
`origin/simulator:results/hw-sweep/SM90_H200/`: mean E2E 262.161664 ms, TTFT 9.000004 ms,
ITL 8.173451 ms. Both simulations use the same native H100 timeline. H200 is counterfactual:
the dashboard shows its metrics, never an error against H100.

The H200 bundle keeps the original compressed duration/timeline exports and losslessly gzipped
CSV, comparison and summary. Its 31 exact kernel stats come from `kernels.csv`, not invented
logs. Instruction counts, IPC and execution start time are unknown and stay unset. Kernel
indices are shared by tracer UID across configs, so the smaller H200 subset is not shifted
against H100's measured kernels. The original H100 logs and artifacts remain byte-identical.

To rebuild both configs, fetch a complete H100 run directory and the hardware-sweep exports:

```sh
uv run python scripts/build_sample.py /path/to/e2e-plan-v1 \
  --config SM90_H200=/path/to/hw-sweep/SM90_H200
```

[`samples/opt125m-a100-e2e/`](../opt125m-a100-e2e/README.md) contains the native A100 entry.
Its measured A100 baseline and real SM80_A100 simulation are now available. Rebuild it with:

```sh
uv run python scripts/build_sample.py /path/to/e2e-a100-plan-v1 --sample opt125m-a100-e2e
```

For future pending entries, startup skips `status: "pending"` manifests, and building a complete
run directory replaces that status with `available`. H100's `sample_pending_configs` hint is
hidden once the A100 native trace has been seeded. Startup seeds bundles once per
trace id, so use a fresh data directory to inspect rebuilt samples. Each native trace's
`real_gpu_config` identifies the only simulation eligible for error against that trace's
measured baseline. Other traces in the set are included from their latest succeeded run.
