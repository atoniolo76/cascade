# Sample: facebook/opt-125m on H100, real vs Accel-Sim

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
- `traced/trace_steps.json` and `traced/traces/kernelslist.g`: the traced steps and the kernel list
  Accel-Sim ran.
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

- The trace `sample__facebook_opt-125m__h100-e2e` is `ready`, with the 712 traced kernels and their
  launch dimensions from `trace_alignment.json`.
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
