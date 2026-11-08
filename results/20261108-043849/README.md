# Real Modal corrected-timeline replay

Input: `phases/20261108-043849/corrected_timeline.json`, profiler commit `e424578`.
H100, OPT-125m eager, 32 tokens × 4 repeats, no layerwise NVTX. 128 steps: 32 warm-up
steps preserved unchanged, 96 captured steps (17,088 kernels), 3 non-warm-up requests.

Input attachment (download separately; not duplicated here):
https://app.devin.ai/attachments/917d4c73-7ff0-4191-8b43-c3652481187a/corrected_timeline.json

SHA-256: `cc244b7650ab09640b1c7df86ac44e036ead8a873f1a8024de093f2aa0a721e4`.

## Reproduce

From the repo root, with Python 3.12 and `uv sync`:

```bash
INPUT=/home/ubuntu/attachments/70eeaf6f-efe8-4815-ba9d-a9c1c5a08516/corrected_timeline.json
OUT=results/20261108-043849

uv run cascade simulate "$INPUT" --measured-only \
  --output "$OUT/measured-only/simulated_timeline.json"
uv run cascade compare "$INPUT" "$OUT/measured-only/simulated_timeline.json" \
  --output "$OUT/measured-only/compare.json"

uv run cascade simulate "$INPUT" --durations "$OUT/sim_durations.json" \
  --output "$OUT/accelsim/simulated_timeline.json"
uv run cascade compare "$INPUT" "$OUT/accelsim/simulated_timeline.json" \
  --output "$OUT/accelsim/compare.json" --perfetto /data/real_compare_trace.json
```

Change INPUT to the downloaded attachment path. Outputs from both modes and the explicit
duration-transfer file are committed here. Default launch-cycle subtraction is **0**; the source
simulations include their configured 5,000-cycle launch latency at 1,980 MHz.

## How the 20-kernel source is transferred

Source: earlier H100 OPT-125m replay `sim_H100/kernel-stats.csv`, UIDs 1–20, corresponding
to old trace IDs 656–660, 833–837, 1010–1014, 1187–1191. Names, grids and blocks come from
the corresponding old nsys `kernels.csv`. The 20 source cycle values and 10 grouped signatures
are embedded in `sim_durations.json`, so the CLI does not need the old external directories.

- Exact full mangled name, grid `[x,y,z]`, block `[x,y,z]`, and phase-kind match only.
- UIDs 1–5 are **prefill-like**, inferred from their 6-token reshape grid and adjacency in the
  five-kernel attention sequence; UIDs 6–20 are **decode-like**, with 1-token reshape grid.
  The old capture lacked verified step NVTX labels, so this phase assignment is an assumption,
  not a recovered annotation. New prefill/decode grids and attention specializations agree.
- For each phase/signature group use the median of its source `gpu_sim_cycle` samples.
  The file includes all source UIDs/cycles and each mapped occurrence's `source_group`.
- Apply that group to **every** matching occurrence: 183 prefill and 5,673 decode kernels
  (61 of 178 per captured step). This includes other layers plus the step's initial fill kernel,
  not only layer 10. Everything unmatched uses measured duration unchanged.

This is **cross-run/cross-layer shape-based timing transfer**, not 5,856 newly traced/simulated
kernels and not proof of identical instructions/data/cache/KV lengths. Name/grid/block cannot
establish those. This supplies an explicit partial-coverage demonstration, not a hardware
accuracy validation. Coverage is **5,856/17,088 = 34.27% of kernels**, and
**20,247,976/55,288,296 ns = 36.62% of original measured GPU busy time**.

## Measured-only sanity output

```text
Simulated kernels: 0/17088 (0.00%); measured GPU-time coverage: 0.00%
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   0.000    0.000    0.000    0.000
GPU busy    96   0.000    0.000    0.000    0.000
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.127942    -1.115891    n/a    -34.464687
2         0.142573    -1.536301    n/a    -47.482747
3         0.128082    -1.847182    n/a    -57.134573
```

Maximum step-wall error is **0%**, meeting the ≤1% sanity requirement.

## Accel-Sim timing-transfer output

```text
Simulated kernels: 5856/17088 (34.27%); measured GPU-time coverage: 36.62%
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   41544.253    41500.677    42788.927    0.539
GPU busy    96   41544.253    41500.677    42788.927    7.215
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.168451    -1.074267    n/a    -33.133840
2         0.184647    -1.494809    n/a    -46.154436
3         0.169103    -1.805632    n/a    -55.805484
```

Maximum step-wall error is 0.574834%. The small wall error is host-dominated and must not be
misread as full request prediction accuracy or high simulator coverage.

## Important observations and limits

**Request timing does not match even in measured-only mode.** The required derivation
`TTFT = queue + prefill`, `ITL = decode wall`, `e2e = queue + all step walls` gives request
e2e 253.793140 / 247.170604 / 243.257650 ms, while the artifact reports
288.257827 / 294.653351 / 300.392223 ms. The discrepancy is present before simulated durations
are used. Profiler `_corrected_timeline` takes request metrics from `request_metrics["baseline"]`
at its call site, so this is not caused by comparing against profiled request metrics. Exact
causes of the extra elapsed time (inter-step work, token delivery, etc.) are **unverified** without
the corresponding timing/dependency data. No scheduler or extra fitted request overhead is added.

Real `itl_ms_p50` is null throughout: comparisons show **n/a**, not zero. Simulated p50 is
derived from decode walls and retained in the output, but no unavailable real percentile is invented.

The profiler's `_phase_nsys_stats` computes GPU busy from a union of kernel/memcpy/memset
events, while the exported timeline lists only kernels selected by start time within the NVTX
range. GPU busy exceeds the exported duration sums by 10,496–11,776 ns per step. The net
**1,036,483 ns residual** is preserved as unmodeled GPU busy, not inflated host time or simulated
coverage. Four measured kernel overlaps total **608 ns** (max 160 ns); they are serialized at
zero gap, with their overlap amounts reported per step. No concurrent scheduling is predicted.
The net residual and unknown event locations are not fabricated as transfer slices in Perfetto.

Optional real comparison trace was imported with TraceProcessor:

```text
[{'name': 'Real unprofiled model', 'slices': 17184},
 {'name': 'Simulated corrected-host timeline', 'slices': 17184}]
[{'overlaps': 0}]
[]
```

The last row is nonzero `stats` with name containing overlap/spill. UI layout is visually
unverified. Full simulator usage, schema conventions, fitting and offline tests are documented
in [`docs/corrected-timeline-simulation.md`](../../docs/corrected-timeline-simulation.md).

## Configured launch-latency subtraction experiment

The same mapping and clock were replayed with **5,000 cycles subtracted per covered kernel**.
Both additional outputs are committed under `accelsim-nolat/`:

```bash
uv run cascade simulate "$INPUT" --durations "$OUT/sim_durations.json" \
  --launch-latency-cycles 5000 --output "$OUT/accelsim-nolat/simulated_timeline.json"
uv run cascade compare "$INPUT" "$OUT/accelsim-nolat/simulated_timeline.json" \
  --output "$OUT/accelsim-nolat/compare.json"
```

Coverage remains 5,856/17,088 kernels (34.27%) and 36.62% of measured GPU time. No new
Accel-Sim execution is implied; the option removes latency from the existing simulated cycles.

| Subtracted cycles | Step-wall mean absolute error ns | Step-wall mean absolute % | GPU-busy mean absolute error ns | GPU-busy mean absolute % |
|---:|---:|---:|---:|---:|
| 0 | 41544.253 | 0.539 | 41544.253 | 7.215 |
| 5000 | 112496.152 | 1.460 | 112496.152 | 19.535 |

```text
Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %
Step wall   96   112496.152    112539.727    113531.409    1.460
GPU busy    96   112496.152    112539.727    113531.409    19.535
Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms
1         0.014411    -1.228308    n/a    -38.063133
2         0.030607    -1.648850    n/a    -51.083729
3         0.015063    -1.959672    n/a    -60.734777
```

Subtraction worsens this capture-conditioned timing transfer: GPU busy changes from an
overestimate to an underestimate. This experiment exposes the configured latency contribution;
it does not prove a physically separable 5,000-cycle overhead should always be removed.
This old input lacks inter-step gaps, so these request errors retain the old missing-time issue.
Simulator work now lives on branch `simulator`; profiler code lives on `profiler`.
