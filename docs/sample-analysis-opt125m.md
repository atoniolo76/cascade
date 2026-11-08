# OPT-125m sample: GPU profiling → instruction simulation → request replay

> Copied verbatim from [`simulator@8cbcd04`](https://github.com/atoniolo76/cascade/blob/8cbcd04d46daa2c57089193c76f3c25584d485cc/docs/sample-analysis-opt125m.md); only the hardware-sweep link was made absolute. Optimization (1), local NVBit staging, is done on `profiler` in 0b54e42.

## Scope and evidence

This analysis describes the **existing H100 `e2e-plan-v1` sample**, not a
generic model benchmark. All measured values below are computed from:

* `/home/ubuntu/runs/phases/e2e-plan-v1/`: `run.json`, `workload.json`,
  `corrected_timeline.json`, `trace_alignment.json`, phase logs/step logs,
  `sim/kernels.csv`, `sim/summary.json`, `sim_durations.json`, `sim/compare.json`.
* Committed [hardware sweep](https://github.com/atoniolo76/cascade/blob/8cbcd04d46daa2c57089193c76f3c25584d485cc/results/hw-sweep/README.md): inventory,
  per-kernel/step/request CSVs, H100/H200 summaries and comparisons.

Absolute paths identify the collected evidence on the execution VM, not
portable dependencies of Cascade. The hardware-sweep artifacts are committed;
the large profiling run is retained in the run store. See the
[run-directory contract](run-dir-contract.md) for file ownership and semantics.
Numbers marked **derived** are arithmetic on these artifacts; **inferred**
means interpretation, not an additional measurement.

**A100 status:** unavailable. The launch stopped before image build with
`Token missing. Could not authenticate client.` No A100 baseline, trace,
per-step trace timing, native SM80 simulation or comparison exists yet.
Missing values are not zero and no H100 traces are substituted for A100 SASS.

## Workload and environment

| Item | Saved value |
|---|---|
| Model / architecture | `facebook/opt-125m` / `OPTForCausalLM` |
| Prompt | `Hello, my name is` (six scheduled prefill tokens) |
| Generation | 32 output tokens, four repeats; repeat 0 is warm-up |
| Execution | eager (`enforce_eager=true`), async scheduling off |
| Model shape | 12 layers, hidden 768, 12 attention/KV heads, head dimension 64 |
| Numeric format / parallelism | float16, tensor parallel size 1, no quantization |
| vLLM / measured GPU | 0.21.0 / NVIDIA H100 80GB HBM3 |
| Profiled median SM / memory clocks | 1980 MHz / 2619 MHz |
| Accel-Sim / GPGPU-Sim pins | `7ba54de` / `4195dd4` |
| Simulation core clocks | H100 and H200: 1980 MHz, read from config |

There are **32 steps/request: one prefill and 31 decode**, not 33 steps.
Repeat-local traced steps are 0, 2, 10, 26, at contexts 6, 8, 16, 32.
The requested plan contains context 1, but this six-token prompt has no such
decode context. All 128 timing steps exist; the three steady repeats have
96 profiled steps and 17,088 kernels (178 per step). Repeat 1 supplies the
four instruction-traced steps: 712 aligned occurrences.

## Pipeline cost

These are stage wall seconds from the **original** run, not the later Sail
sweep. The trace stage includes both spinlock-detection passes and the NVBit
trace process. Compilation/image build, scheduling and downloads are excluded.

| Stage | Wall seconds | Share of summed stages (derived) |
|---|---:|---:|
| Baseline process | 123.582 | 4.67% |
| Nsight-profiled process | 30.140 | 1.14% |
| NVBit + spinlock processes | 1738.577 | 65.73% |
| Trace post-processing | 34.171 | 1.29% |
| Accel-Sim orchestration | 718.709 | 27.17% |
| Total of the above | 2645.179 | 100% |

NVBit step envelopes come from `traced/trace_steps.json` monotonic timestamps:

| Local step | Context | Instrumented step seconds |
|---|---:|---:|
| 0 | 6 | 390.943 |
| 2 | 8 | 375.588 |
| 10 | 16 | 375.926 |
| 26 | 32 | 372.553 |

The sum is 1515.010 seconds (derived), not the full 1738.577-second stage.
Spinlock pass 0 and pass 1 step-log spans, from first step start to last step
end, are 50.968 and 44.285 seconds. These spans **exclude process/engine
startup**; they are not complete pass wall timings. Trace-process step span
is 1541.794 seconds, also excluding its startup/teardown.

The logs explicitly report engine initialization (profile/KV-cache/warmup)
of 85.80 seconds in baseline, 2.00 in profiled, 13.72 in spinlock pass 0,
13.68 in pass 1 and 13.78 in tracing. These overlap the stage totals and
must not be added again. Baseline weight loading alone is only 0.41 seconds.
**Inference:** engine initialization and cold/warm caches are significant;
the difference between processes does not isolate any one cache's effect.

### Local-disk optimization and artifact compatibility

The old trace logs show `TRACES_FOLDER`/working paths under
`/__modal/volumes/.../phases/e2e-plan-v1/traced`; the approximately 375-second
decode envelopes above are measured on that path. This alone does not prove
the fraction attributable to volume I/O. The separate earlier local-disk
comparison mentioned in the task is not in the source artifacts above, so
no measured speedup factor is asserted here.

Profiler commit `0b54e42` now runs both spinlock passes, tracing and
post-processing under `/tmp/cascade-trace-*/traced`, then copies the result
once into the run directory, measuring `stage_wall_s.trace_collect`.
Failures also collect diagnostic files before scratch cleanup. This commit
was tested and pushed **before** the A100 launch; `f190c99` adds `--gpu`
and builds NVBit with `ARCH=all` rather than an H100-only binary.

**No bundle was introduced.** The collected directory preserves the
existing `traced/traces/kernel-*.trace.xz`, `kernelslist.g` and processed
`.tracez` layout, plus `traced/trace_steps.json`, metrics, spinlock data and
logs. The pipeline's post-processor provides the existing compressed trace
format; collection is a directory copy, not tar/zstd. No extraction command
or app ingestion change is needed. The separate legacy streaming S3 path is
unchanged. The real A100 layout and performance still require a successful
run; the launch did not reach a GPU.

## Real timeline anatomy

The following means cover three steady prefills and 93 steady decodes.
`host_ns = baseline_wall_ns − gpu_busy_ns` is a **residual decomposition**:
it is not independently instrumented exclusive CPU time. CUDA API time can
overlap GPU work and is not an additional summand.

| Step kind | N | Baseline wall mean ms | Wall p50 ms | GPU busy mean ms | Residual host mean ms | CUDA API mean ms |
|---|---:|---:|---:|---:|---:|---:|
| Prefill | 3 | 8.498641 | 8.458830 | 0.615279 | 7.883362 | 1.637063 |
| Decode | 93 | 7.894492 | 7.848008 | 0.575611 | 7.318881 | 1.656008 |

Decode gap-before mean/p50 is 0.036950/0.035911 ms. First-step gaps are
null, not an extra request boundary delay. Request endpoint fields retain
signed post-last-step intervals; they must not be replaced by zero.

| Steady repeat | Real TTFT ms | Real ITL mean ms | Real E2E ms | Sum walls ms | Sum gaps ms |
|---|---:|---:|---:|---:|---:|
| 1 | 8.636344 | 7.896743 | 253.435384 | 252.179409 | 1.158236 |
| 2 | 8.382869 | 7.942919 | 254.613360 | 253.426680 | 1.109083 |
| 3 | 8.864829 | 7.949773 | 255.307791 | 254.077620 | 1.169005 |

Walls + gaps are not the whole envelope: the contract also preserves
pre-first-step and signed post-last-step terms. ITL p50 is absent in the
original request metrics, so none is invented.

### Real kernel mix per step

Counts and summed kernel durations below are averaged over **all steady
steps of each kind**, not just the four instruction samples. Names are
classified with `results/hw-sweep/report.py::group`: KV/attention names,
`nvjet` GEMM grid-Y values 9/3/12/6/66 for QKV/out/up/down/lm_head, and
elementwise/layer-norm/reduction names. Role labels are **inferred**, not NVTX
annotations. Kernel sums need not equal step GPU busy (unmatched residual
activity and the busy-time aggregation are different quantities).

| Group | Kernels/step | Prefill summed GPU µs | Decode summed GPU µs |
|---|---:|---:|---:|
| QKV GEMM | 12 | 56.682 | 55.910 |
| Attention | 12 | 95.584 | 69.689 |
| KV write | 12 | 26.645 | 23.904 |
| Out-proj | 12 | 49.258 | 49.028 |
| MLP up GEMM | 12 | 58.900 | 58.622 |
| MLP down GEMM | 12 | 60.288 | 59.908 |
| MLP down split-K reduction | 12 | 24.096 | 23.163 |
| lm_head | 1 | 30.773 | 30.474 |
| Small elementwise | 63 | 88.585 | 86.520 |
| Layer norm | 25 | 85.151 | 85.398 |
| Other setup/index/reduction | 5 | 28.533 | 22.196 |
| Total count | 178 | — | — |

## Accel-Sim by kernel group

These tables use the **later Sail hardware sweep**, where both configs ran
31 representatives with verification disabled. Means are over the same 712
aligned occurrences (one prefill and three sampled decodes); they are not
weighted by the full request's 31 decodes. Real values are matched by
`(repeat, step, seq)`, not approximate name matching. `cycles / MHz = µs`.
Error is signed `(mean simulated µs / mean real µs − 1) × 100`, not MAPE.
The reference is Nsight-measured kernel time, not independently measured
unprofiled per-kernel time; only step/request baselines are unprofiled.

| Group | Real mean µs | H100 mean cycles | H100 µs | H100 error % | H200 mean cycles | H200 µs | H200 vs H100-reference error % |
|---|---:|---:|---:|---:|---:|---:|---:|
| QKV GEMM | 4.689312 | 9786.000 | 4.942424 | +5.398 | 9771.000 | 4.934848 | +5.236 |
| Attention | 6.326604 | 13589.750 | 6.863510 | +8.486 | 13590.250 | 6.863763 | +8.490 |
| KV write | 2.052000 | 8073.750 | 4.077652 | +98.716 | 8073.250 | 4.077399 | +98.704 |
| Out-proj | 4.079292 | 8832.000 | 4.460606 | +9.348 | 8831.000 | 4.460101 | +9.335 |
| MLP up GEMM | 4.875271 | 10821.000 | 5.465152 | +12.099 | 10684.000 | 5.395960 | +10.680 |
| MLP down GEMM | 5.007292 | 10273.000 | 5.188384 | +3.617 | 10282.000 | 5.192929 | +3.707 |
| Split-K reduction | 1.940000 | 7269.000 | 3.671212 | +89.238 | 7269.000 | 3.671212 | +89.238 |
| lm_head | 30.623500 | 52002.000 | 26.263636 | −14.237 | 55296.000 | 27.927273 | −8.804 |
| Small elementwise | 1.385139 | 6397.849 | 3.231237 | +133.279 | 6397.044 | 3.230830 | +133.250 |
| Layer norm | 3.404100 | 10103.750 | 5.102904 | +49.905 | 10099.750 | 5.100884 | +49.845 |
| Other setup/index/reduction | 4.753550 | 13534.600 | 6.835657 | +43.801 | 13486.800 | 6.811515 | +43.293 |

H200's error column is a **counterfactual against the real H100 reference**,
not accuracy against measured H200 hardware. No real H200 baseline exists.
The shipped H100/H200 configs both have 132 SMs, identical clocks and simple
DRAM modeling; memory-controller count changes from 40 to 48. Near-identical
request predictions do not imply physical H100/H200 equivalence.

| Native workload / modeled config | Real baseline | Sim wall s | Unique executions / traced occurrences | Failures / timeouts |
|---|---|---:|---:|---:|
| H100 traces / H100, original verification run | H100 | 718.709 | 93 / 712 | 0 / 0 |
| H100 traces / H100, Sail sweep | H100 | 804.429 | 31 / 712 | 0 / 0 |
| H100 traces / H200, Sail sweep | H100 counterfactual only | 777.060 | 31 / 712 | 0 / 0 |
| A100 traces / SM80_A100 | unavailable: launch authentication failed | — | — | not run |

## Coverage, dedup verification and context scaling

Both original and Sail replay cover **17,088/17,088 steady kernels (100%)**;
each captured request has 5696/5696. Measured GPU-time coverage is **98.13%**;
the unmatched measured remainder is retained, not modeled by Accel-Sim.
Warm-up has no comparable profiled kernels. Coverage is not an accuracy score.

| Run | Exact | Dedup | Identical propagation | Interpolation | Verification |
|---|---:|---:|---:|---:|---|
| Original H100 | 93 | 619 | 15206 | 1170 | 2 extra members/group |
| Sail H100 | 31 | 681 | 16376 | 0 | disabled |
| Sail H200 | 31 | 681 | 16376 | 0 | disabled |

The original has 31 header groups, 31 representatives plus 62 verification
executions. The sum of individual simulation wall times is 2866.019 seconds,
versus 718.709 orchestration seconds because kernels execute concurrently.
Do not sum reused dedup-row wall fields as additional CPU executions.

Dedup keys contain launch/header properties, **not full instruction/memory
trace contents**. Original verification therefore matters:

| Group / representative UID | Maximum cycle difference | Maximum relative difference |
|---|---:|---:|
| Index-select setup, 6217 | 4426 | 39.734% |
| Small elementwise, 6379 | 604 | 6.404% |
| Prefill layer norm, 6221 | 621 | 6.145% |
| QKV GEMM, 6222 | 360 | 3.679% |
| Out-proj, 6226 | 201 | 2.276% |
| Decode layer norm, 6577 | 213 | 2.108% |
| MLP down GEMM, 6231 | 127 | 1.236% |
| MLP up GEMM, 6229 | 132 | 1.220% |

The largest mismatch is UID 6573 versus representative 6217
(`indexSelectSmallIndex`), not attention or a large GEMM. It occurs between
different prefill/decode input contexts sharing a header. This demonstrates
that header identity is not cycle identity; it does not quantify error for
every unverified group member. A zero reference did not produce nonzero
verification cycles in this sample.

First-layer original duration records show:

| Local step / context | KV cycles | KV provenance | Attention cycles | Attention provenance |
|---|---:|---|---:|---|
| 0 / 6 | 8229 | exact | 14501 | exact |
| 2 / 8 | 8022 | exact | 13286 | exact |
| 10 / 16 | 8022 | dedup:6579 | 13286 | dedup:6580 |
| 26 / 32 | 8022 | dedup:6579 | 13286 | dedup:6580 |

Thus this pipeline's output is flat across sampled decode contexts **because
it reuses representative cycles**. This is not an independent observation of
physical/context-invariant attention. The original decode attention/KV
verification comparisons have zero difference, but those two checked members
do not establish arbitrary context scaling. In the verification-free Sail
sweep, all sampled decode groups satisfy identical propagation, so enabling
`propagate=interp` actually yields **zero interpolation labels**. Inference:
independent context simulations and a stronger cache/dedup key are required
before making scaling claims; the current results do not provide them.

## Replay errors and eager-mode dilution

`cascade simulate` replaces modeled kernel durations while retaining host
residuals, gaps, unmatched GPU activity and request envelopes. Both original
and later runs use the same real H100 baseline. Positive signed errors below
mean simulation is longer; the original step/GPU mean errors are all positive.

| Run | Mean abs step error µs / % | Mean abs GPU-busy error µs / % |
|---|---:|---:|
| Original H100 | 239.587 / 3.031 | 239.587 / 41.551 |
| Sail H100 | 240.307 / 3.040 | 240.307 / 41.676 |
| Sail H200 counterfactual | 240.942 / 3.048 | 240.942 / 41.786 |

Original H100 p50/p95 absolute step errors are 240.540/242.402 µs.

| Repeat | Original TTFT error ms | ITL mean error ms | E2E error ms | Sail H100 E2E error ms | Sail H200 E2E error ms |
|---|---:|---:|---:|---:|---:|
| 1 | +0.350397 | +0.243503 | +7.675474 | +7.698534 | +7.718827 |
| 2 | +0.326948 | +0.243288 | +7.665878 | +7.688938 | +7.709231 |
| 3 | +0.436559 | +0.239997 | +7.657047 | +7.680107 | +7.700400 |

Later sweep GPU busy decode/prefill mean is 816.980/822.679 µs for H100 and
817.620/823.136 µs for H200. Predicted E2E repeats 1/2/3 are
261.133918/262.302298/262.987898 ms and
261.154211/262.322591/263.008191 ms respectively.

**Derived explanation:** only about 7.29% of the mean decode step is GPU busy
(0.575611 / 7.894492); about 92.71% is the retained host residual. Adding
roughly 240 µs to GPU busy can therefore be a roughly 42% GPU error but only
3% step/E2E error. Eager-mode request agreement is a weak validation of GPU
model fidelity. It is not evidence that a graph-mode or compute-dominated
workload would have the same small request error. Keeping host/gaps fixed
also makes the counterfactual insensitive to hardware-dependent host changes.

## Architecture compatibility

Trace binary version is 90; the stream contains Hopper-specific kernels,
including `FlashAttnFwdSm90` and SM90 GEMMs. H100 and H200 are the valid
registered native targets. Older configs are rejected by the Sail app's
architecture gate; preliminary pre-Sail probes also failed (for example,
A100 hit the specialized-unit scheduler assertion). These are **not valid
A100 predictions**. A real A100 run must choose native kernels and collect
its own SM80 traces and unprofiled baseline.

The pinned and checked upstream-dev trees have no Blackwell/SM100/B200/B100/
GB200 configuration. H200 is the only shipped Hopper variant beyond H100.
No H100-number-edited approximation was created. See the sweep inventory for
all 12 tested and nine deprecated config files and their SM/clock evidence.

## Bottlenecks and optimization order

| Priority / proposal | Measured motivation | Status / qualification |
|---|---|---|
| 1. Local-disk trace and post-process, collect once | NVBit 1738.577 s, 65.73% of stage sum; decode envelopes 372.553–375.926 s on volume | Implemented/pushed `0b54e42`; new A100 measurement blocked by auth. Not a bundle or S3 protocol change. |
| 2. Trace one instance per distinct kernel, using NVBit kernel-range limits | 712 traced occurrences collapse to 31 header groups | Proposal only. Need a prepass/range mapping and alignment preserving omitted identities; headers are not sufficient proof of semantic identity. |
| 3. Cache spinlock detection | Two step spans of 50.968/44.285 s plus initialization, every trace invocation | Proposal only. Cache must include tool/binary/model/input compatibility; do not disable required spinlock handling blindly. |
| 4. CTA sampling for large kernels | Original lm_head executions 459.661/454.428/397.684 s; next slowest 46.701 s | Proposal only. Needs estimator/error validation, especially persistent scheduling and memory contention. |
| 5. Persistent simulator cache keyed by trace hash + config | Repeated H100/H200 sweep recomputes 31 representatives/config | Proposal only. Include simulator/build/trace-parser versions, complete effective config and auxiliary config files; no header-only identity. |
| 6. Fewer verification copies after validation | 93 versus 31 executions; verification lm_head copies include the 459.661/397.684 s jobs | Proposal/trade-off. Disabling verification hides observed 39.734% mismatch and changes propagation; not an accuracy-preserving optimization by default. |
| 7. Warm caches / amortize engine init | Baseline engine init 85.80 s versus 2.00 s profiled, weights only 0.41 s | Proposal. Difference suggests warming effects but is not a controlled measurement of cache speedup. |

The lm_head is only one of 178 kernels/step and about 30.5 µs of real decode
GPU time, yet its trace-driven CPU simulation takes hundreds of seconds.
**Inference:** these jobs are likely critical-path drivers of the original
719-second orchestration, but CSVs do not store execution overlap intervals,
so an exact critical-path decomposition is not available. The later Sail
sweep likewise spends 773.080 s (H100) / 745.353 s (H200) in its single
lm_head representative, compared with total walls 804.429/777.060 s.
These CPU wall measurements must never be confused with modeled GPU latency.

## Reproduce and finish the A100 comparison

Profiler (already includes both pushed fixes):

```sh
modal run accelsim_trace.py::e2e_main --gpu A100-80GB \
  --run-name e2e-a100-plan-v1 --model facebook/opt-125m \
  --prompt 'Hello, my name is' --max-tokens 32 --repeats 4 \
  --trace-steps plan --trace-repeat 1
modal volume get accelsim-traces phases/e2e-a100-plan-v1 \
  /home/ubuntu/runs/phases/e2e-a100-plan-v1
```

After a successful profile, deploy the same Sail app route as the hardware
sweep, ingest the **A100** timeline/alignment/traces with architecture `sm_80`,
and execute `SM80_A100` with `dedup=True`, `verify_dedup=2`,
`propagate="interp"`. Upload artifacts before the terminal heartbeat. Always
terminate main/child sailboxes and delete the run checkpoint in cleanup.
No new Sail resources were created for the authentication-blocked attempt.

Then, from the simulator checkout:

```sh
RUN=/home/ubuntu/runs/phases/e2e-a100-plan-v1
uv run cascade simulate "$RUN/corrected_timeline.json" \
  --durations "$RUN/sim_durations.json" --output "$RUN/sim/simulated_timeline.json"
uv run cascade compare "$RUN/corrected_timeline.json" \
  "$RUN/sim/simulated_timeline.json" --output "$RUN/sim/compare.json"
```

Update this document only after the native A100 results exist. In particular,
do not compare SM80 simulation with the real H100 baseline and call it an
A100 accuracy result.
