# Profiler → simulator run-directory contract

The installable profiler owns only GPU measurement, instruction tracing and alignment.
Our non-packaged `accelsim_local.py` supplies CPU execution for our repository-local app;
the separate simulator checkout's `cascade.sim` owns CPU
Accel-Sim execution and timeline replay. Neither package needs to import the other.
Paths below are relative to one run directory, normally
`accelsim-traces:/phases/<run-name>/` on Modal.

Ground truth: profiler `cascade_profiler/{pipeline,timeline,align,workload}.py`;
simulator `cascade/sim/{accelsim,accelsim_contract,timeline,gaps,compare}.py`.
Schemas and the trimmed examples below were checked against the downloaded
`/home/ubuntu/runs/phases/e2e-plan-v1` run. Examples omit fields/records, not
defaults; they are illustrations, not complete loader fixtures.

## Layout, ownership and required inputs

| Path | Writer | Consumer / requirement |
| --- | --- | --- |
| `corrected_timeline.json` | Profiler GPU pipeline | Required for aligned Accel-Sim execution, replay and comparison. Baseline + profiled phases must both exist. |
| `trace_alignment.json` | Profiler | Required by aligned Accel-Sim execution; maps tracer UIDs to timeline keys. |
| `traced/traces/kernelslist.g`, `kernel-<uid>-ctx_<context>.tracez` | Profiler postprocessor | Postprocessed instruction inputs for Accel-Sim. Wrapper can attempt postprocessing if missing, but a completed GPU handoff should supply them. |
| `traced/traces/stats_ctx_*`, raw `*.trace[.xz]` | Profiler NVBit tracer | Used to build alignment/postprocess traces; not timing sources for replay. |
| `traced/trace_steps.json` | Profiler recorder | Required when building alignment; informational after alignment is supplied. |
| `run.json` | Profiler pipeline | Manifest/status/provenance; required when profiler builds alignment, not read by current simulator loaders. |
| `traced/metadata.json` | Profiler optional S3 handoff | Non-secret destination fields and final S3 URI; emitted for traced runs with S3 output. |
| `workload.json` | Profiler | Serializable workload configuration; informational to simulator. |
| `reconcile.json` (also `reconcile.csv`) | Profiler | Timing/clock diagnostics; informational to simulator. |
| `baseline/`, `profiled/` | Profiler | Source logs, step/request metrics, clock samples; `profiled/profile.sqlite` and Nsight capture data support timing reconciliation. Not required once corrected timeline is available. |
| `baseline_iterations.jsonl`, `profiled_iterations.jsonl` | Profiler | Parsed iteration logs; informational. |
| `sim_durations.json` | Simulator Accel-Sim runner | Required for `cascade simulate --durations`; not required for `--measured-only`. |
| `sim/kernels.csv`, `sim/summary.json` | Simulator Accel-Sim runner | Per-kernel diagnostics and execution summary; informational, not replay inputs. |
| `sim/kernel-<uid>/`, `sim/kernel-<uid>.log`, `sim/setup.log` | Simulator | Isolated execution workspaces/logs; informational. |
| `sim/simulated_timeline.json` | Simulator replay CLI | Required alongside real corrected timeline for comparison. Output path is caller-selected. |
| `sim/compare.json` | Simulator comparison CLI | Comparison report; output path is caller-selected. |

Timing-only runs intentionally lack instruction traces/alignment. A partial or
failed run is not a completed handoff: inspect `run.json.status` and required
files. The directory name alone is not evidence of completion.

### Identity, units and compatibility

- `repeat`: integer workload repetition (zero-based); repeat 0 is warm-up here.
- `step`: integer global recorder step ID, **not** the position in a repeat.
  `step_in_repeat` is that zero-based position. Alignment joins captures by
  `(repeat, step_in_repeat)`, then emits the target timeline's global `step`.
- `seq`: integer kernel position unique within a timeline step. Replay identity
  is **(repeat, step, seq)**. A tracer UID is not a timeline seq or simulator's
  `kernel_launch_uid`; each isolated simulator process may report launch UID 1.
- `*_ns` are nanoseconds; `*_ms` milliseconds; `*_s` seconds.
  `sim_cycles`/`cycles` are core cycles, and `clock_mhz` is MHz:
  `duration_ns = sim_cycles / clock_mhz * 1000`.
  Measured JSON timestamps are integers; synthesized durations/timestamps and
  distributions may be floating-point. Numeric loader inputs must be finite,
  nonnegative unless explicitly signed; booleans do not count as numbers.
- Grid/block dimensions are three integers. Writers emit arrays; kernel-key
  helpers also accept `"1x1x1"` strings. Name identity uses `mangled_name` if
  present, otherwise `name`; current corrected writer stores mangled names in
  `name`. Do not demangle names before joining.
- There is **no run-directory schema-version field or version negotiation** in
  these writers/loaders. `run.json.package_version`, git/tool commits and trace
  header binary/tracer versions are provenance, not a JSON schema version.
  Changes to this contract must be made in both repositories together, with
  producer/consumer tests and identical copies of this document. Preserve or
  explicitly migrate old artifacts; do not assume arbitrary old runs are compatible.
  No version field is added by this documentation.

## Profiler files

### `corrected_timeline.json`

Top-level: `meta: object`, `steps: object[]`, `requests: object[]`,
`summary: object`, `gc_summary: object`.

- `meta`: `run_name, model, prompt, exec_mode, gpu_name, vllm_version` strings;
  `max_tokens, repeats` integers; `layerwise_nvtx` boolean;
  `sm_clock_mhz_median, mem_clock_mhz` numbers or null; `model_config` object
  (architecture/dtype strings, model dimensions and tensor parallel size integers,
  quantization nullable). `clock` records `step_clock`, offset sampling `how`
  strings and `wall_minus_monotonic_ns, profiled_wall_minus_monotonic_ns,
  sample_bracket_ns` numeric clock-conversion metadata. Corrected step/request
  endpoints are on the monotonic clock; do not subtract clocks from different runs.
- Each step: `repeat, step, step_in_repeat` integers, `warmup` boolean,
  `kind` string, `context_length, num_reqs, num_scheduled_tokens` integers or null.
  Steady replay supports only `prefill|decode`.
  `baseline_wall_ns` is the uninstrumented step wall duration;
  `profiled_wall_ns` is the Nsight span, **not** necessarily
  `profiled_wall_end_ns - profiled_wall_start_ns`.
  `gpu_busy_ns, cuda_api_ns, host_ns` are numbers or null when uncaptured;
  `host_ns = baseline_wall_ns - gpu_busy_ns`.
  `kernels` is an ordered array of
  `{seq:int, name:string, grid:int[3], block:int[3], duration_ns:number,
  start_offset_ns:number}`; offsets are relative to the profiled step span.
- Step endpoints/gaps: `baseline_wall_start_ns, baseline_wall_end_ns,
  profiled_wall_start_ns, profiled_wall_end_ns` are integer timestamps or null.
  `baseline_gap_before_ns, profiled_gap_before_ns` are current start minus
  previous end **within the same repeat**; first-step gap is null, not zero.
  For gap-aware steady replay, every later baseline gap must be nonnegative and
  agree with supplied baseline endpoints within 1 ns.
- GC step fields: for each `baseline_`/`profiled_` prefix,
  `gc_before, gc_after` are nullable objects containing `count, collections`
  integer arrays; `gc_pause_ns_in_step, gc_pause_ns_in_gap_before` are nullable
  numeric pause totals.
- Each request: `repeat:int, warmup:bool, num_output_tokens:int|null`;
  `queued_ms, ttft_ms, itl_ms_mean, itl_ms_p50, e2e_ms:number|null`.
  `queued_ms` means scheduled minus queued, not an absolute timestamp.
  `first_step_start_ns, last_step_end_ns, arrival_ns, queued_ns,
  first_token_ns, last_token_ns` are nullable integer timestamps.
  `pre_first_step_ns = first_step_start_ns - queued_ns`.
  **`post_last_step_ns = last_token_ns - last_step_end_ns` is signed**:
  it can be negative because the token endpoint precedes recorder exit.
  `arrival_to_first_step_ns` and `last_step_end_to_return_ns` are nullable
  diagnostic durations, not substitutes for these request envelopes.
  E2E is last-token minus queued; TTFT is first-token minus queued.
  Replay assumes one request per repeat and reconstructs E2E from
  pre + step walls + inter-step gaps + signed post. It approximates TTFT using
  pre + first-step wall; recorder exit need not equal the token timestamp.
- `summary` maps kind to `n:int`, `profiler_overhead_pct:number|null`,
  and distribution objects for `baseline_wall_ns, gpu_busy_ns, host_ns,
  baseline_gap_before_ns, profiled_gap_before_ns` (mean/p50, with p95 where
  emitted; values nullable). `gc_summary` maps `baseline|profiled` to arrays
  of `{repeat:int, gc_collect_before_ns:number, total_gc_pause_ns:number}`.

Loader constraints: non-warmup step keys and kernel seqs must be unique;
wall = busy + host within 1 ns; kernel offsets are ordered; kernels fit within
profiled span; serial duration sum cannot exceed GPU busy. Warm-up steps can
have empty kernels and null profiled fields and are excluded from replay.

Trimmed real example:
<!-- example: corrected_timeline.json -->
```json
{"meta":{"run_name":"e2e-plan-v1"},"steps":[{"repeat":1,"step":32,"step_in_repeat":0,"warmup":false,"kind":"prefill","baseline_wall_ns":8458830,"gpu_busy_ns":615355,"host_ns":7843475,"baseline_gap_before_ns":null,"kernels":[{"seq":0,"grid":[1,1,1],"block":[128,1,1],"duration_ns":928,"start_offset_ns":1243298}]}],"requests":[{"repeat":1,"pre_first_step_ns":320787,"post_last_step_ns":-223525}],"summary":{"prefill":{"n":3}}}
```

### `trace_alignment.json`

Top-level: `provenance:object, steps:object[], kernels:object[], unmatched:object`.

Alignment sorts tracer launches by UID, selects the half-open recorder range
`[first_kernel_id, end_kernel_id)` per traced step, then performs ordered
sequence matching on **mangled name + grid + block**. Within remaining
replacement gaps it falls back to ordered name-only matching. Such dimension
mismatches have `dims_match=false`; they are not dimension-exact matches.
Duplicate tracer UIDs and overlapping traced ranges are rejected.

- `provenance`: `trace_repeat:int, run_name:string|null,
  traced_steps:int[2][], traced_timing_used:bool` (false).
  Pairs in `traced_steps` are repeat/local-step, not global step.
- `steps[]`: `repeat, step_in_repeat, n_tracer, n_timeline, matched,
  unmatched_tracer, unmatched_timeline:int`, `step:int|null`, `exact:bool`.
  `exact` means the entire ordered key sequences are equal.
- `kernels[]`: `tracer_uid, repeat, step, seq:int`, `trace_file, name:string`,
  `grid, block:int[3]`, `dims_match:bool`.
  `trace_file` can name the original `.trace.xz`; CPU code resolves it to the
  corresponding `.tracez`. Dimension-mismatched results can be used on the
  aligned kernel, but do not seed dimension-exact propagation.
- `unmatched`: `n_tracer, n_timeline:int`, `tracer, timeline:object[]`.
  Tracer records retain launch fields plus repeat/local step; timeline records
  have `repeat, step, seq:int, name:string`. Inspect these instead of inventing joins.

<!-- example: trace_alignment.json -->
```json
{"provenance":{"run_name":"e2e-plan-v1","traced_timing_used":false},"steps":[{"repeat":1,"step":32,"step_in_repeat":0,"n_tracer":178,"n_timeline":178,"matched":178,"exact":true}],"kernels":[{"tracer_uid":6207,"trace_file":"kernel-6207-ctx_0x54c35c0.trace.xz","repeat":1,"step":32,"seq":0,"grid":[1,1,1],"block":[128,1,1],"dims_match":true}],"unmatched":{"n_tracer":0,"n_timeline":0,"tracer":[],"timeline":[]}}
```

### `traced/trace_steps.json` and instruction files

Step log is a JSON **array**, not an object. Records contain
`step, repeat, step_in_repeat:int`, `warmup, traced:bool`, `nvtx_range:string`;
`wall_start_ns, wall_end_ns` epoch timestamps and
`monotonic_start_ns, monotonic_end_ns` monotonic timestamps;
`pre_step_sync_ns, post_step_sync_ns` numeric durations;
`gc_before, gc_after` snapshots and `gc_pause_ns_in_step,
gc_pause_ns_in_gap_before` nullable numeric durations;
`first_kernel_id, end_kernel_id:int|null` delimit the UID range.
Scheduling metadata: `kind:string`, `num_scheduled_tokens, num_reqs,
num_prefill_reqs, num_decode_reqs:int`,
`tokens_per_req, new_req_num_computed_tokens:object<string,int>`.
NVBit durations are instrumentation costs, never corrected GPU timings.

<!-- example: traced/trace_steps.json -->
```json
[{"step":32,"repeat":1,"step_in_repeat":0,"warmup":false,"traced":true,"first_kernel_id":6207,"end_kernel_id":6385,"monotonic_start_ns":363432168100,"monotonic_end_ns":754375203810,"kind":"prefill","num_scheduled_tokens":6}]
```

`kernelslist.g` is a text launch listing (may include non-kernel commands);
postprocessed kernel lines name `kernel-<uid>-ctx_<context>.tracez`.
Example line from this run:

```text
kernel-6207-ctx_0x54c35c0.tracez
```

A `.tracez` is **binary, not JSON**: null-terminated text header followed by
compressed instruction data. Relevant header strings include `kernel name,
kernel id, grid dim, block dim, shmem, nregs, binary version, nvbit version,
accelsim tracer version, enable lineinfo`. Header example (trimmed):

```text
-grid dim = (1,1,1)
-block dim = (128,1,1)
-nregs = 14
```

### `run.json`, `workload.json`, `reconcile.json`

`run.json` top-level fields:
`workload:object` (same schema as workload file), `run_name:string`,
`phases:string[]`, `trace_steps:int[]|null` (resolved local selections),
`trace_repeat:int`, `stage_wall_s:object<string,number>`,
`package_version:string, git_sha:string|null, traced_timing_used:bool,
status:string` (`running|completed|failed`);
`effective_llm_kwargs:object<phase,object>`,
`exec_mode_defaults_overridden:string[]`,
`runner_flags:object` with `layerwise_nvtx, no_step_sync,
gc_between_repeats:bool`. Optional `trace_plan` has `context_lengths:int[]`
and `selected:int[2][]` (repeat/local-step pairs).

<!-- example: run.json -->
```json
{"run_name":"e2e-plan-v1","phases":["baseline","profiled","traced"],"trace_steps":[0,2,10,26],"trace_repeat":1,"stage_wall_s":{"baseline":123.582354665,"profiled":30.139649288,"traced":1738.576739129,"post_process":34.1713620779999},"package_version":"0.1.0","git_sha":null,"traced_timing_used":false,"status":"completed"}
```

`workload.json` top-level: `model:string, prompts:string[], max_tokens:int,
repeats:int, exec_mode:string` (`eager|compile|graphs`), `max_model_len:int`,
`llm_kwargs, sampling_kwargs:object`, `driver:string|null` (module:function).
These are workload settings, not a promise that output has max_tokens + 1 steps.

<!-- example: workload.json -->
```json
{"model":"facebook/opt-125m","prompts":["Hello, my name is"],"max_tokens":32,"repeats":4,"exec_mode":"eager","max_model_len":2048,"llm_kwargs":{},"sampling_kwargs":{},"driver":null}
```

`reconcile.json` top-level: `run_name, phase, model, prompt, exec_mode:string`;
`max_tokens, repeats, max_model_len:int`; `layerwise_nvtx:bool`;
`diagnostics:object<phase,object>`; `table:object[]`.
Diagnostics include `iteration_count, step_count:int`,
`iteration_step_mismatch:bool`, `clock_sampler_error:string|null`;
`clocks` has sample count, SM/memory MHz min/median/max and throttle strings.
Table rows have `kind:string`, metric distribution columns
`<metric>_n, _mean, _p50, _p95`, and `delta_*_ns, delta_*_pct` numbers/null.
Metric suffixes indicate ns or ms; delta columns ending in ns are nanoseconds.

<!-- example: reconcile.json -->
```json
{"run_name":"e2e-plan-v1","phase":"both","diagnostics":{"baseline":{"iteration_count":128,"step_count":128,"iteration_step_mismatch":false}},"table":[{"kind":"prefill","baseline_wall_ns_n":3,"baseline_wall_ns_mean":8498641.0,"baseline_wall_ns_p50":8458830.0,"baseline_wall_ns_p95":8779165.2}]}
```

## Simulator files

### `sim_durations.json`

Top-level: `provenance:object, kernels:object[]`.

Aligned kernel records have `repeat, step, seq:int`, `name:string`,
`sim_cycles:number`, `clock_mhz:number>0`,
`source:string` (`traced|propagated`), `sim:string`;
`tracer_uid:int` is present on traced records; `context_length:int` can be
present on interpolated records. Failed/timeout results are omitted.

Provenance includes `run_name, gpu_config, clock_mhz_source, mode, propagate,
dedup_scope:string`; `gpu_mapping:object`; `accelsim_commit,
gpgpusim_commit:string|null`; `clock_mhz:number`;
`measured_sm_clock_mhz:number|null`; `post_process_error:string|null`;
`counts, provenance_labels, dedup_verification:object`;
`dedup, isolated_cold_cache:bool`; `verify_dedup, dims_mismatch:int`;
`ambiguous_propagation:object[]` (kind/name/grid/block, observed cycles, median).
Clock conversion uses the simulator config's core frequency, not the measured
GPU clock. Execution uses isolated cold-cache kernels, not a simulated full GPU request.

Labels:
- `exact`: that UID was actually simulated (including verification members).
- `dedup:<uid>`: reused representative UID's result; groups use stable trace
  header + mangled name/grid/block across selected traced steps.
- `identical:<uid>`: propagated median cycles for matching kind/name/dimensions;
  interpolation mode additionally scopes sample matches by seq.
- `interp:<a>,<b>[:clamped]`: decode context interpolation between sampled
  cycles at matching seq/name/dimensions. Outside sampled contexts endpoints
  are clamped. Labels reference tracer source UIDs, not target seqs.

Verification records cycle differences; nonzero disagreement does **not**
automatically split/reject a group. Coverage therefore does not prove accuracy.

Replay loader also accepts a bare array and records with `duration_ns` instead
of cycles (not both). Scoped records require repeat and step together.
Legacy unscoped `seq` records apply across matching steps; prefer scoped keys.
Duplicate keys and unused duration keys are errors. Missing duration keys retain
measured kernel durations, so inspect coverage.

<!-- example: sim_durations.json -->
```json
{"provenance":{"run_name":"e2e-plan-v1","gpu_config":"SM90_H100","clock_mhz":1980.0,"dedup":true,"verify_dedup":2},"kernels":[{"repeat":1,"step":32,"seq":0,"tracer_uid":6207,"sim_cycles":5473,"clock_mhz":1980.0,"source":"traced","sim":"exact"},{"repeat":1,"step":33,"seq":0,"sim_cycles":5473,"clock_mhz":1980.0,"source":"propagated","sim":"identical:6207"}]}
```

Legacy trace-only execution (`scripts/simulate_traces.py`) instead consumes
`<run>/traces/` without timing alignment. Its durations explicitly use
`provenance.alignment=null, scope="tracer_uid_only"` and records
`tracer_uid, name, sim_cycles, clock_mhz, sim` with **no seq/repeat/step**.
They are not valid timeline-replay inputs. Never fabricate alignment to use them.

### `sim/kernels.csv` and `sim/summary.json`

CSV columns: `uid:int, name:string, cycles:number|null, sim_wall_s:number,
status:string` (`ok|failed|timeout`), `error:string,
kernel_launch_uid:int|null, sim:string`. Null CSV cells are empty.
Fan-out rows include all occurrences; dedup-reused rows have sim_wall_s = 0.
Sum of actual execution walls differs from concurrent orchestration wall time.

Real header and first row:
```csv
uid,name,cycles,sim_wall_s,status,error,kernel_launch_uid,sim
6207,_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEESt5arrayIPcLm1EEEEviT0_T1_,5473,14.720270473,ok,,1,exact
```

Summary top-level:
`coverage:object[]` with `repeat, step, simulated, total:int` for traced steps;
`counts:object<string,int>` (aligned/selected/ok/failed/timeout/propagated/
unique_accelsim_runs/traced_kernel_occurrences/unmatched_untraced);
`total_sim_wall_s, wall_s:number`;
`sim_wall_s_mean, sim_wall_s_max:number|null`;
`dedup_verification, provenance_labels:object`;
`post_process_error:string|null`.
The unique count includes verification executions, not just representatives.
Legacy trace-only summary omits aligned coverage and some aligned-only fields.

`dedup_verification` contains `groups:object[]`,
`max_absolute_cycles:number, max_relative:number, zero_reference_nonzero:bool`.
Groups include representative UID, member UIDs, header/name/grid/block,
`comparisons:[{uid:int, absolute_cycles:number, relative:number|null}]`
and the same maximum/zero-reference fields. Relative difference divides by the
representative's absolute cycles; a zero reference with nonzero difference
has relative null and sets the flag. `provenance_labels` maps full labels to counts.

<!-- example: sim/summary.json -->
```json
{"coverage":[{"repeat":1,"step":32,"simulated":178,"total":178}],"counts":{"aligned":712,"selected":712,"ok":712,"failed":0,"timeout":0,"propagated":16376,"unique_accelsim_runs":93,"traced_kernel_occurrences":712,"unmatched_untraced":0},"total_sim_wall_s":2866.0192408609996,"wall_s":718.709229989,"dedup_verification":{"max_absolute_cycles":4426,"max_relative":0.3973426698985546,"zero_reference_nonzero":false},"post_process_error":null}
```

### `sim/simulated_timeline.json`

Same top-level keys as corrected timeline; replay copies metadata/GC/warm-up
records, rewrites steady kernel durations/offsets, wall/busy/host and request
endpoints, sets profiled wall to synthesized wall and cuda_api_ns to null.
Copied GC metadata is historical, not simulated collection behavior.

`summary` becomes `{simulation:object, per_kind:object}`.
Simulation includes `host_model:string, host_fit:object, request_assumption:string,
coverage:object` and gap-model metadata when applicable.
`host_model` is `scaled-gaps|parametric`; host fits are per-kind coefficients
`a_ns, b_ns_per_kernel:number, samples:int, slope_identifiable:bool`.
Gap metadata includes
`gap_model:string` (`measured|mean|linear`), `gap_fit:object` (per-kind
`a_ns, b_ns_per_req:number, samples:int, slope_identifiable:bool`), and
`subtract_launch_cycles:number` (core cycles removed before conversion).
Coverage has `kernels, simulated_kernels:int`, `kernel_fraction,
gpu_time_fraction:number`, `measured_gpu_ns, covered_measured_gpu_ns,
unmodeled_gpu_ns:number`, and per-step coverage records.
Each coverage step has `repeat, step, kernels, simulated_kernels:int` and
`measured_gpu_ns, simulated_coverage_gpu_ns, unmodeled_gpu_ns,
measured_kernel_overlap_ns:number`.
Per-kind entries contain steps and wall/busy means plus gap distributions.
Kernel coverage can reach 100% while GPU-time coverage is lower because
non-kernel measured GPU work is retained.

<!-- example: sim/simulated_timeline.json -->
```json
{"meta":{"run_name":"e2e-plan-v1"},"summary":{"simulation":{"host_model":"scaled-gaps","coverage":{"kernels":17088,"simulated_kernels":17088,"kernel_fraction":1.0,"gpu_time_fraction":0.9812807226710433}}}}
```

### `sim/compare.json`

Top-level: `steps:object[], requests:object[], summary:object`.
Steps have `repeat, step:int, kind:string, wall_ns, gpu_busy_ns:error-object`.
Requests have `repeat:int` and `ttft_ms, itl_ms_mean, itl_ms_p50,
e2e_ms:error-object|null`. Error objects:
`real, simulated, error:number, error_pct:number|null`;
signed error = simulated - real, percentage = error / real * 100.
Null marks unavailable request metrics.
Summary maps `wall_ns|gpu_busy_ns` to aggregates:
`count:int`, `mean_error, mean_abs_error, p50_abs_error, p95_abs_error`
in ns and `mean_error_pct, mean_abs_error_pct,
p50_abs_error_pct, p95_abs_error_pct` in percent (nullable when unavailable).
Comparison requires identical non-warmup step keys/kinds and request repeats.

<!-- example: sim/compare.json -->
```json
{"steps":[{"repeat":1,"step":32,"kind":"prefill","wall_ns":{"real":8458830,"simulated":8665953.626262626,"error":207123.62626262568,"error_pct":2.448608451318039}}],"requests":[{"repeat":1,"itl_ms_p50":null}],"summary":{"wall_ns":{"count":96,"mean_abs_error":239586.723169192}}}
```

## End-to-end commands

Use matching branch checkouts and the same Modal account/volume. Authenticate
through the normal Modal credential mechanism; do not put tokens in run files.

```sh
# Profiler checkout (GPU): skip our app's optional internal simulation.
python -m pip install -e '.[modal]'
modal run accelsim_trace.py::e2e_main --run-name my-run \
  --model facebook/opt-125m --prompt "Hello, my name is" \
  --max-tokens 32 --repeats 4 --trace-steps plan --trace-repeat 1 --profile-only

# Simulator checkout (CPU): reads phases/my-run on accelsim-traces.
uv sync --python 3.12 --extra modal
uv run --python 3.12 modal run scripts/simulate_accelsim.py \
  --run-name my-run --gpu-config SM90_H100 --verify-dedup 2 --propagate interp

# Download after CPU execution completes; parent directory must exist.
mkdir -p /home/ubuntu/runs/phases
uv run --python 3.12 modal volume get --force accelsim-traces \
  phases/my-run /home/ubuntu/runs/phases/

# Still in simulator checkout: replay then compare.
uv run cascade simulate /home/ubuntu/runs/phases/my-run/corrected_timeline.json \
  --durations /home/ubuntu/runs/phases/my-run/sim_durations.json \
  --output /home/ubuntu/runs/phases/my-run/sim/simulated_timeline.json
uv run cascade compare /home/ubuntu/runs/phases/my-run/corrected_timeline.json \
  /home/ubuntu/runs/phases/my-run/sim/simulated_timeline.json \
  --output /home/ubuntu/runs/phases/my-run/sim/compare.json
```

The CPU Python API is `cascade.sim.accelsim.simulate_accelsim(run_dir,
gpu_config="SM90_H100", dedup=True, verify_dedup=2, propagate="interp")` and
requires built Accel-Sim tools locally. The Modal helper supplies those tools.
Our repo-local `e2e_main` can also run CPU simulation through `accelsim_local.py`;
use `--profile-only` to stop after profiling/tracing, or `--timing-only` for timing without NVBit.
Neither CPU runner nor its image is shipped in `cascade_profiler`.
The inspected e2e-plan-v1 artifact predates this entrypoint split
but has this same run-directory format.

Verified e2e-plan-v1 facts: 128 total steps, four requests, 712 aligned trace
occurrences at repeat-local steps 0/2/10/26; repeat 1 has 5,696/5,696 derived
durations across all 32 steps. Across steady repeats replay covers
17,088/17,088 kernels. The 93 actual simulations include verification members
of 31 groups. Maximum verification disagreement is 4,426 cycles / 39.73%:
the contract preserves provenance and diagnostics, not a guarantee of dedup
equivalence or simulation accuracy.
