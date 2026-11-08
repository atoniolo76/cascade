# cascade e2e-plan-v1: opt-125m full request, real vs Accel-Sim

Workload: facebook/opt-125m, prompt "Hello, my name is", 32 output tokens (1 prefill + 31 decode steps),
4 repeats, eager mode, vLLM 0.21.0, NVIDIA H100 80GB HBM3. Profiler: cascade-profiler 0.1.0
(profiler @ 3fc9384 pipeline; run executed on 80214cf code). Simulator: simulator @ 54545b2.
Accel-Sim 7ba54de, config SM90_H100, cycles converted at 1980 MHz.

## Files
- corrected_timeline.json   real trace: unprofiled baseline step walls/gaps/request endpoints + nsys kernels per step
- simulated_timeline.json   simulator output in the same format, kernel durations from Accel-Sim
- compare.json              per-step and per-request real vs simulated errors
- sim_durations.json        Accel-Sim cycles per (repeat, step, seq) with provenance label
- trace_alignment.json      NVBit tracer kernel uid -> timeline kernel (712/712 matched)
- kernels.csv, summary.json Accel-Sim per-kernel results, wall seconds, dedup verification
- run.json, workload.json, reconcile.json  run metadata, stage wall times, baseline/profiled reconciliation

## Method
Traced steps 0, 2, 10, 26 of repeat 1 (prefill + decode contexts 8/16/32) with NVBit; 31 distinct kernels
simulated (93 Accel-Sim runs incl. 2 verification copies each); durations propagated to all 17,088 kernel
launches: 93 exact, 619 dedup, 15,206 identical, 1,170 interp. Host time and inter-step gaps are replayed
from the real unprofiled run.

## Results (simulated - real)
| Request | Real e2e | Sim e2e | Error |
|---|---|---|---|
| 1 | 253.44 ms | 261.11 ms | +3.03% |
| 2 | 254.61 ms | 262.28 ms | +3.01% |
| 3 | 255.31 ms | 262.96 ms | +3.00% |

TTFT +3.9 to +4.9%; ITL mean +3.0%; step wall +3.0% (+240 us/step); GPU busy per step +42% decode, +34% prefill.

## Caveats
- Accel-Sim over-predicts small kernels (~2.7 us floor), which drives the GPU-busy error.
- Dedup verification max difference: 4,426 cycles / 39.7% relative between grouped kernels.
- Eager mode: host time dominates (~7.9 ms decode step vs 0.57 ms GPU), so e2e error is diluted.
- Stage wall: baseline 124 s, profiled 30 s, NVBit trace 1,739 s, post-process 34 s, Accel-Sim 719 s.

## Full run directory (this bundle)
This bundle is the complete `e2e-plan-v1` run directory, in the layout of `docs/run-dir-contract.md`, so the simulation can be reproduced from it directly.

- `traced/traces/kernelslist.g` + `kernel-*-ctx_0x54c35c0.tracez` (712): NVBit instruction traces Accel-Sim executes. The raw `kernel-*.trace.xz` NVBit outputs (420 MB) are omitted; the `.tracez` files are post-processed from them and are what Accel-Sim reads.
- `traced/trace_steps.json`, `metrics.json`, spinlock detection outputs and logs.
- `corrected_timeline.json`, `trace_alignment.json`, `run.json`, `workload.json`, `reconcile.*`, `baseline/`, `profiled/` (incl. nsys sqlite-derived data), `*_iterations.jsonl`.
- `sim_durations.json` and `sim/`: Accel-Sim outputs: per-kernel workspaces with `gpgpusim.config` (SM90_H100), logs, `kernels.csv`, `summary.json`, `simulated_timeline.json`, `compare.json`.

### Reproduce (simulator branch checkout)
```sh
# Accel-Sim (CPU, needs built Accel-Sim; the Modal helper builds it):
#   python: cascade.sim.accelsim.simulate_accelsim("e2e-plan-v1", gpu_config="SM90_H100", dedup=True, verify_dedup=2, propagate="interp")
uv run cascade simulate e2e-plan-v1/corrected_timeline.json --durations e2e-plan-v1/sim_durations.json --output e2e-plan-v1/sim/simulated_timeline.json
uv run cascade compare e2e-plan-v1/corrected_timeline.json e2e-plan-v1/sim/simulated_timeline.json --output e2e-plan-v1/sim/compare.json
```
