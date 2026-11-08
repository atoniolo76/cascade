# Cascade Profiler

GPU workload profiling, Nsight Systems timing reconciliation, and NVBit instruction
tracing for vLLM. Python 3.10+ can import and inspect runs; GPU workloads require
CUDA, vLLM, and the tracing tools. Modal images use Python 3.12.
The installable package only profiles; it does not provide CPU simulation.

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

Download the example's run with `modal volume get my-profiler-runs <run_name> ./<run_name>`.

Optional S3-compatible output uploads the finished run while retaining the volume copy:

```python
prof = register(app, runs, gpu="H100!", hf_cache=hf,
    s3={"endpoint": "https://s3.example.com", "bucket": "my-runs", "prefix": "profiles",
        "region": "auto", "addressing_style": "path"},
    secrets=[modal.Secret.from_name("my-s3-credentials")])
```

The same `s3` dictionary can be passed to `profile()` or `.profile.remote()`.
Install `cascade-profiler[modal]` (or `boto3>=1.36`) for S3 uploads. Credentials
come only from AWS environment variables or the caller's Modal secret, never
destination options or files. `run.json` and `traced/metadata.json` record the
destination and final `s3://my-runs/profiles/<run_name>` URI.

## Our repository-local app

From this checkout:

```sh
modal run accelsim_trace.py::e2e_main --run-name my-run --trace-steps plan
# Timing only (no NVBit or simulation):
modal run accelsim_trace.py::e2e_main --run-name timing-run --timing-only
# Profiling and tracing only, without simulation:
modal run accelsim_trace.py::e2e_main --run-name trace-run --trace-steps plan --profile-only
```

Our app's CPU execution, image build, deduplication and interpolation live in
`accelsim_local.py`, which is not shipped by pip. These are internal tools for
our own hardware, not a supported `cascade_profiler` API. To simulate an existing
run in our app's `accelsim-traces` volume:

```sh
modal run accelsim_trace.py::simulate --run-name my-run --gpu-config SM90_H100 --propagate interp
```

The run directory is the interface: `run.json`, `corrected_timeline.json`,
`trace_alignment.json`, and `traced/traces/` are produced by the profiler. The
simulator consumes them and writes `sim_durations.json` and `sim/`.
The installable profiler does not build or execute the CPU simulator.

See the [OPT-125m sample analysis](docs/sample-analysis-opt125m.md) for a worked end-to-end example.

See the [run-directory contract](docs/run-dir-contract.md) for ownership, schemas,
identity/units, compatibility rules, real examples, and end-to-end commands.
