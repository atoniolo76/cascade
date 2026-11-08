"""Optional Modal image factories and serialized function registration; no global Modal objects."""

import shlex
from pathlib import Path
from types import SimpleNamespace

ACCELSIM_COMMIT = "7ba54ded60c4ba0083410c37d22d719166acd6dc"
GPGPUSIM_REF = "4195dd4f229cbf03f4f38c49139ae94d01d23f03"
ROOT = "/opt/accel-sim-framework"
REPO = "https://github.com/accel-sim/accel-sim-framework.git"


def image(base=None, *, accelsim_commit=ACCELSIM_COMMIT, nvbit=True, nsys=True, arch="all"):
    """Custom bases must supply vLLM and CUDA 12.x devel/Ubuntu 22.04 apt repositories."""
    import modal

    if base is None:
        base = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
                .entrypoint([]).uv_pip_install("vllm==0.21.0", "huggingface-hub", "nvtx"))
    result = (base.apt_install("git", "wget", "build-essential", "bc", "libzstd-dev", "zlib1g-dev")
              .env({"HF_XET_HIGH_PERFORMANCE": "1", "CUDA_INSTALL_PATH": "/usr/local/cuda"})
              .run_commands(f"git clone {REPO} {ROOT}",
                            f"git -C {ROOT} checkout {shlex.quote(accelsim_commit)}"))
    if nvbit:
        result = result.run_commands(
            f"cd {ROOT} && ./util/tracer_nvbit/install_nvbit.sh",
            f"cd {ROOT} && ARCH={shlex.quote(arch)} make -j$(nproc) -C util/tracer_nvbit/",
            f"test -f {ROOT}/util/tracer_nvbit/tracer_tool/tracer_tool.so",
            f"test -f {ROOT}/util/tracer_nvbit/others/spinlock_tool/spinlock_tool.so",
            f"test -x {ROOT}/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing")
    result = result.uv_pip_install("boto3>=1.36")
    if nsys:
        result = (result.apt_install("cuda-nsight-systems-12-9").run_commands(
            'command -v nsys || ln -sf "$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys | tail -1)" /usr/local/bin/nsys',
            "nsys --version"))
    return result.add_local_python_source("cascade_profiler")


def accelsim_image(*, accelsim_commit=ACCELSIM_COMMIT, gpgpusim_ref=GPGPUSIM_REF):
    """CPU simulator image; CUDA toolkit is for building, not a runtime GPU request."""
    import modal

    build = (f"source {ROOT}/gpu-simulator/setup_environment.sh release && "
             f"make -j$(nproc) -C {ROOT}/gpu-simulator/ && "
             f"make -C {ROOT}/util/tracer_nvbit/tracer_tool/traces-processing/ && "
             f"test -x {ROOT}/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing && "
             f"test -x {ROOT}/gpu-simulator/bin/release/accel-sim.out")
    return (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
            .entrypoint([]).apt_install(
                "git", "wget", "build-essential", "xutils-dev", "bison", "flex", "zlib1g-dev",
                "libglu1-mesa-dev", "libssl-dev", "libxml2-dev", "libboost-all-dev", "libzstd-dev",
                "python3-dev", "python3-setuptools", "python3-pip", "bc", "cmake")
            .uv_pip_install("pyyaml", "plotly", "psutil")
            .env({"CUDA_INSTALL_PATH": "/usr/local/cuda", "GPGPUSIM_BRANCH": gpgpusim_ref,
                  "CPLUS_INCLUDE_PATH": f"{ROOT}/util/tracer_nvbit/tracer_tool"})
            .run_commands(f"git clone {REPO} {ROOT}",
                          f"git -C {ROOT} checkout {shlex.quote(accelsim_commit)}",
                          f"sed -i 's|-lzstd -pthread $(BUILD_DIR)/[*].o|-pthread $(BUILD_DIR)/*.o -lzstd|' {ROOT}/gpu-simulator/Makefile",
                          "bash -c " + shlex.quote(build))
            .add_local_python_source("cascade_profiler"))


def register(app, volume, *, gpu, image=None, sim_image=None, hf_cache=None, mount="/data",
             timeout=4 * 60 * 60, sim_cpu=32, sim_timeout=24 * 60 * 60):
    """Register nested serialized functions; client and container Python minor versions must match."""
    from cascade_profiler import profile, simulate_accelsim
    from cascade_profiler.modal_support import image as make_image

    if not Path(mount).is_absolute():
        raise ValueError("mount must be absolute")
    volumes = {mount: volume}
    if hf_cache is not None:
        volumes["/root/.cache/huggingface"] = hf_cache
    gpu_image = image if image is not None else make_image()
    cpu_image = sim_image if sim_image is not None else accelsim_image()

    @app.function(name="cascade_profile", serialized=True, image=gpu_image, gpu=gpu, cpu=8,
                  memory=64 * 1024, ephemeral_disk=512 * 1024, timeout=timeout, volumes=volumes)
    def profile_remote(workload_dict, run_name=None, trace_steps=None, trace_repeat=1, layerwise_nvtx=False,
                       no_step_sync=False, gc_between_repeats=True, keep_nsys_rep=False,
                       phases=("baseline", "profiled"), accelsim_root=ROOT, nsys_bin="nsys", trace_plan=None):
        from cascade_profiler import Workload

        volume.reload()
        try:
            result = profile(Workload.from_dict(workload_dict), mount, run_name=run_name,
                             trace_steps=trace_steps, trace_repeat=trace_repeat, layerwise_nvtx=layerwise_nvtx,
                             no_step_sync=no_step_sync, gc_between_repeats=gc_between_repeats,
                             keep_nsys_rep=keep_nsys_rep, phases=phases, accelsim_root=accelsim_root, nsys_bin=nsys_bin,
                             trace_plan=trace_plan)
            return {"run_name": result.run_dir.name, "run_dir": str(result.run_dir),
                    "stage_wall_s": result.stage_wall_s}
        finally:
            volume.commit()

    @app.function(name="cascade_simulate", serialized=True, image=cpu_image, cpu=sim_cpu,
                  memory=64 * 1024, timeout=sim_timeout, volumes={mount: volume})
    def simulate_remote(run_name, gpu_config=None, accelsim_root=ROOT, jobs=None,
                        kernel_timeout_s=None, propagate="none", uids=None, dedup=True, verify_dedup=0):
        if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
            raise ValueError("run_name must be a single directory name")
        volume.reload()
        try:
            return simulate_accelsim(Path(mount) / run_name, gpu_config=gpu_config, accelsim_root=accelsim_root,
                                     jobs=sim_cpu if jobs is None else jobs, kernel_timeout_s=kernel_timeout_s,
                                     propagate=propagate, uids=uids, dedup=dedup, verify_dedup=verify_dedup)
        finally:
            volume.commit()

    return SimpleNamespace(profile=profile_remote, simulate=simulate_remote)
