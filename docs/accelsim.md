# Accel-Sim execution

CPU execution belongs to `cascade.sim.accelsim`; the GPU-only `cascade_profiler`
package is not a runtime dependency. The run directory is the contract:
`corrected_timeline.json`, `trace_alignment.json`, `run.json`, and
`traced/traces/` come from the profiler. Simulation writes `sim_durations.json`,
`sim/kernels.csv`, `sim/summary.json`, and per-kernel logs/work directories.

```python
from cascade.sim.accelsim import simulate_accelsim

summary = simulate_accelsim("/data/phases/my-run", gpu_config="SM90_H100",
                            jobs=32, dedup=True, verify_dedup=2, propagate="interp")
```

The caller needs the built Accel-Sim tools. The optional Modal helper supplies
a pinned CPU image (no vLLM, no GPU request):

```sh
uv sync --python 3.12 --extra modal
uv run --python 3.12 modal run scripts/simulate_accelsim.py --run-name my-run \
  --gpu-config SM90_H100 --verify-dedup 2 --propagate interp
```

This reads `phases/my-run` from the `accelsim-traces` Modal volume. For a custom
app, use `cascade.sim.modal_support.register(app, volume, mount="/data", run_subdir="phases")`;
it returns the remote CPU function and commits even on errors. Serialized
functions require matching client/container Python minor versions (default 3.12).

The image pins Accel-Sim `7ba54ded60c4ba0083410c37d22d719166acd6dc` and
GPGPU-Sim `4195dd4f229cbf03f4f38c49139ae94d01d23f03`. It exposes the tracer's
`tracez_format.h` through `CPLUS_INCLUDE_PATH` and links zstd after the object files.

Deduplication groups stable trace headers and name/grid/block; it is an assumption
about instruction equivalence, not proof. `verify_dedup=N` independently runs up
to N additional members per group and records maximum cycle differences.
`propagate="identical"` reuses matching kind/name/dimensions on untraced steps;
`"interp"` also interpolates matching decode sequence positions across measured
context lengths, clamping outside the range. Missing matches remain uncovered.
Durations label exact, dedup, identical and interp provenance. Each kernel runs
in isolation with cold caches. Individual failures/timeouts are retained in
the CSV and omitted from durations; setup/configuration failures raise.

After downloading the run:

```sh
uv run cascade simulate /data/phases/my-run/corrected_timeline.json \
  --durations /data/phases/my-run/sim_durations.json \
  --output /data/phases/my-run/sim/simulated_timeline.json
uv run cascade compare /data/phases/my-run/corrected_timeline.json \
  /data/phases/my-run/sim/simulated_timeline.json \
  --output /data/phases/my-run/sim/compare.json
```

Legacy trace-only runs without corrected timing can instead use:

```sh
uv run --python 3.12 --extra modal modal run scripts/simulate_traces.py \
  --run-name legacy-run --gpu-config SM90_H100 --verify-dedup 2
```

This reads `legacy-run/traces/` at the volume root and writes per-UID cycles to
`sim_durations.json` with `scope="tracer_uid_only"` and `alignment=null`. These
records deliberately have no repeat/step/seq and cannot be used for timeline
replay until a real timing alignment exists. No timeline or alignment is fabricated.
