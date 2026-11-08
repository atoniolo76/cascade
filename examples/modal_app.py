"""Run with Python 3.12: modal run examples/modal_app.py (requires your Modal credentials)."""

import modal
from cascade_profiler import Workload, register

app = modal.App("my-cascade-profiler")
volume = modal.Volume.from_name("my-profiler-runs", create_if_missing=True)
cache = modal.Volume.from_name("my-hf-cache", create_if_missing=True)
fns = register(app, volume, gpu="H100!", hf_cache=cache)


@app.local_entrypoint()
def main():
    workload = Workload("facebook/opt-125m", ["Hello, my name is"], max_tokens=32, repeats=4)
    result = fns.profile.remote(workload.to_dict(), trace_steps=[0, 5])
    print(result["run_name"], result["stage_wall_s"])
    summary = fns.simulate.remote(result["run_name"])
    print(summary["coverage"], summary["wall_s"])
    print(summary["counts"], summary["provenance_labels"])
