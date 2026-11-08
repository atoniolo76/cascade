# Cascade Profiler

GPU workload profiling, Nsight Systems timing reconciliation, and NVBit instruction
tracing for vLLM. Python 3.10+ can import and inspect runs; GPU workloads require
CUDA, vLLM, and the tracing tools. Modal images use Python 3.12.

```sh
pip install 'cascade-profiler[modal]'
```

```python
from cascade_profiler import Workload, profile

workload = Workload("facebook/opt-125m", ["Hello, my name is"], max_tokens=32, repeats=4)
result = profile(workload, "/data/phases", trace_steps="plan", trace_repeat=1)
print(result.run_dir, result.stage_wall_s)
```

`profile` needs a GPU environment with vLLM, Nsight Systems and the NVBit tracer
installed. `cascade_profiler.image()` supplies the optional Modal GPU image;
`register(app, volume, gpu="H100!")` registers only the GPU `.profile` function.
See [PROFILER_RUNS.md](PROFILER_RUNS.md) and [examples/modal_app.py](examples/modal_app.py).

## Profile, then simulate

From this checkout:

```sh
modal run accelsim_trace.py::e2e_main --run-name my-run --trace-steps plan
# Timing only (no CPU simulator image is built):
modal run accelsim_trace.py::e2e_main --run-name timing-run --timing-only
```

Accel-Sim CPU execution lives in the separate `cascade.sim` simulator package on
the `simulator` branch of [atoniolo76/cascade](https://github.com/atoniolo76/cascade).
From that checkout, using the same Modal volume:

```sh
modal run scripts/simulate_accelsim.py --run-name my-run --gpu-config SM90_H100 --propagate interp
```

The run directory is the interface: `run.json`, `corrected_timeline.json`,
`trace_alignment.json`, and `traced/traces/` are produced by the profiler. The
simulator consumes them and writes `sim_durations.json` and `sim/`.
The profiler does not build or execute the CPU simulator.

See the [OPT-125m sample analysis](docs/sample-analysis-opt125m.md) for a worked end-to-end example.

See the [run-directory contract](docs/run-dir-contract.md) for ownership, schemas,
identity/units, compatibility rules, real examples, and end-to-end commands.
