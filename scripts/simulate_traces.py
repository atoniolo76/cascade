"""CPU-only legacy run without a corrected timeline or fabricated alignment."""

import modal

from cascade.sim.modal_support import register

app = modal.App("cascade-legacy-accelsim")
volume = modal.Volume.from_name("accelsim-traces", create_if_missing=True)
simulate = register(app, volume, mount="/traces")


@app.local_entrypoint()
def main(run_name: str, gpu_config: str = "SM90_H100", verify_dedup: int = 2):
    summary = simulate.remote(
        run_name, gpu_config=gpu_config, dedup=True, verify_dedup=verify_dedup, traces_only=True
    )
    print("Simulation wall seconds:", summary["wall_s"])
    print("Simulation counts:", summary["counts"])
    print("Provenance labels:", summary["provenance_labels"])
    print(
        "Dedup verification:",
        {k: v for k, v in summary["dedup_verification"].items() if k != "groups"},
    )
