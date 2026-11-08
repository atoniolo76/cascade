"""Run after profiling: modal run scripts/simulate_accelsim.py --run-name <run>."""

import modal

from cascade.sim.modal_support import register

app = modal.App("cascade-accelsim")
volume = modal.Volume.from_name("accelsim-traces", create_if_missing=True)
simulate = register(app, volume, mount="/traces", run_subdir="phases")


@app.local_entrypoint()
def main(
    run_name: str,
    gpu_config: str = "",
    propagate: str = "none",
    dedup: bool = True,
    verify_dedup: int = 2,
):
    summary = simulate.remote(
        run_name,
        gpu_config=gpu_config or None,
        propagate=propagate,
        dedup=dedup,
        verify_dedup=verify_dedup,
    )
    print("Simulation wall seconds:", summary["wall_s"], "coverage:", summary["coverage"])
    print(
        "Simulation counts:", summary["counts"], "provenance labels:", summary["provenance_labels"]
    )
    print(
        "Dedup verification:",
        {k: v for k, v in summary["dedup_verification"].items() if k != "groups"},
    )
