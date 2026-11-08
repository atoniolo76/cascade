"""Optional CPU-only Modal image and function for the profiler run-directory contract."""

import shlex
from pathlib import Path

ACCELSIM_COMMIT = "7ba54ded60c4ba0083410c37d22d719166acd6dc"
GPGPUSIM_REF = "4195dd4f229cbf03f4f38c49139ae94d01d23f03"
ROOT = "/opt/accel-sim-framework"
REPO = "https://github.com/accel-sim/accel-sim-framework.git"


def accelsim_image(*, accelsim_commit=ACCELSIM_COMMIT, gpgpusim_ref=GPGPUSIM_REF):
    import modal

    build = (
        f"source {ROOT}/gpu-simulator/setup_environment.sh release && "
        f"make -j$(nproc) -C {ROOT}/gpu-simulator/ && "
        f"make -C {ROOT}/util/tracer_nvbit/tracer_tool/traces-processing/ && "
        f"test -x {ROOT}/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing && "
        f"test -x {ROOT}/gpu-simulator/bin/release/accel-sim.out"
    )
    return (
        modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .apt_install(
            "git",
            "wget",
            "build-essential",
            "xutils-dev",
            "bison",
            "flex",
            "zlib1g-dev",
            "libglu1-mesa-dev",
            "libssl-dev",
            "libxml2-dev",
            "libboost-all-dev",
            "libzstd-dev",
            "python3-dev",
            "python3-setuptools",
            "python3-pip",
            "bc",
            "cmake",
        )
        .uv_pip_install("pyyaml", "plotly", "psutil")
        .env(
            {
                "CUDA_INSTALL_PATH": "/usr/local/cuda",
                "GPGPUSIM_BRANCH": gpgpusim_ref,
                "CPLUS_INCLUDE_PATH": f"{ROOT}/util/tracer_nvbit/tracer_tool",
            }
        )
        .run_commands(
            f"git clone {REPO} {ROOT}",
            f"git -C {ROOT} checkout {shlex.quote(accelsim_commit)}",
            "sed -i 's|-lzstd -pthread $(BUILD_DIR)/[*].o|-pthread $(BUILD_DIR)/*.o -lzstd|' "
            f"{ROOT}/gpu-simulator/Makefile",
            "bash -c " + shlex.quote(build),
        )
        .add_local_python_source("cascade")
    )


def register(
    app, volume, *, image=None, mount="/traces", run_subdir="", cpu=32, timeout=24 * 60 * 60
):
    from cascade.sim.accelsim import simulate_accelsim

    if not Path(mount).is_absolute():
        raise ValueError("mount must be absolute")
    if Path(run_subdir).is_absolute() or ".." in Path(run_subdir).parts:
        raise ValueError("run_subdir must stay inside the volume")

    @app.function(
        name="cascade_simulate",
        serialized=True,
        image=image if image is not None else accelsim_image(),
        cpu=cpu,
        memory=64 * 1024,
        timeout=timeout,
        volumes={mount: volume},
    )
    def simulate_remote(
        run_name,
        gpu_config=None,
        accelsim_root=ROOT,
        jobs=None,
        kernel_timeout_s=None,
        propagate="none",
        uids=None,
        dedup=True,
        verify_dedup=0,
    ):
        if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
            raise ValueError("run_name must be a single directory name")
        volume.reload()
        try:
            return simulate_accelsim(
                Path(mount) / run_subdir / run_name,
                gpu_config=gpu_config,
                accelsim_root=accelsim_root,
                jobs=cpu if jobs is None else jobs,
                kernel_timeout_s=kernel_timeout_s,
                propagate=propagate,
                uids=uids,
                dedup=dedup,
                verify_dedup=verify_dedup,
            )
        finally:
            volume.commit()

    return simulate_remote
