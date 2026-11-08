"""Generate Accel-Sim 2.0 SASS traces of a vLLM model on a Modal H100.

Follows the "Tracing LLMs with vLLM" section of the accel-sim-framework README:
  1. build NVBit + the tracer (tracer_tool.so) and install vLLM
  2. hook NVBit onto selected layers via llm.collective_rpc (see vllm_trace.py)
  3. run the script through util/tracer_nvbit/others/torch_hook/run.sh, preceded by the
     two-phase spinlock detection that run_hw_trace.py does (required for mark_region)
  4. post-process the raw traces into .tracez + kernelslist.g

Traces are saved to the `accelsim-traces` Modal Volume under /<run_name>/traces.

Usage:
  # discover layer names
  modal run accelsim_trace.py --list-layers
  # trace a layer (defaults: facebook/opt-125m, model.decoder.layers.10.self_attn)
  modal run accelsim_trace.py --model facebook/opt-125m \
      --layers model.decoder.layers.10.self_attn --max-tokens 4
  # skip the spinlock-detection passes (faster; mbarrier spin loops are traced as-is)
  modal run accelsim_trace.py --spinlock-handling none
  # download results
  modal volume get accelsim-traces <run_name> ./traces
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

ACCELSIM_REPO = "https://github.com/accel-sim/accel-sim-framework.git"
ACCELSIM_COMMIT = "7ba54ded60c4ba0083410c37d22d719166acd6dc"  # dev branch, 2026-10-27
ACCELSIM_DIR = "/opt/accel-sim-framework"
TORCH_HOOK_DIR = f"{ACCELSIM_DIR}/util/tracer_nvbit/others/torch_hook"
POST_PROCESS_BIN = (
    f"{ACCELSIM_DIR}/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"
)
TRACER_SO = f"{ACCELSIM_DIR}/util/tracer_nvbit/tracer_tool/tracer_tool.so"
SPINLOCK_SO = f"{ACCELSIM_DIR}/util/tracer_nvbit/others/spinlock_tool/spinlock_tool.so"
SPINLOCK_MODES = {"none": 0, "fast_forward": 1, "mark_region": 2}
TRACE_SCRIPT = "/root/vllm_trace.py"
TRACES_MOUNT = "/traces"
FUNCTION_TIMEOUT_S = 4 * 60 * 60  # NVBit instrumentation is slow
TRACE_TIMEOUT_S = FUNCTION_TIMEOUT_S - 20 * 60  # leave time to post-process and save

# Modal's vLLM image (modal-examples/06_gpu_and_ml/llm-serving/vllm_inference.py),
# extended with the Accel-Sim tracer build. The CUDA *devel* base provides nvcc for NVBit.
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "huggingface-hub", "nvtx")
    .apt_install("git", "wget", "build-essential", "bc", "libzstd-dev", "zlib1g-dev")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "CUDA_INSTALL_PATH": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    .run_commands(
        f"git clone {ACCELSIM_REPO} {ACCELSIM_DIR}",
        f"cd {ACCELSIM_DIR} && git checkout {ACCELSIM_COMMIT}",
        # README step 1: build the tracer. ARCH=sm_90 targets H100 (and is much faster
        # to compile than the default ARCH=all).
        f"cd {ACCELSIM_DIR} && ./util/tracer_nvbit/install_nvbit.sh",
        f"cd {ACCELSIM_DIR} && ARCH=sm_90 make -j$(nproc) -C ./util/tracer_nvbit/",
        f"test -f {TRACER_SO}",
        f"test -f {SPINLOCK_SO}",
        f"test -x {POST_PROCESS_BIN}",
    )
    .add_local_file(Path(__file__).parent / "vllm_trace.py", TRACE_SCRIPT)
)

app = modal.App("accelsim-vllm-trace", image=image)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
traces_vol = modal.Volume.from_name("accelsim-traces", create_if_missing=True)


@app.function(
    gpu="H100!",  # "!" pins H100; plain "H100" may be upgraded to H200, which would mis-tag traces
    timeout=FUNCTION_TIMEOUT_S,
    cpu=8,
    memory=64 * 1024,
    volumes={"/root/.cache/huggingface": hf_cache_vol, TRACES_MOUNT: traces_vol},
)
def trace_vllm(
    model: str = "facebook/opt-125m",
    layers: list[str] | None = None,
    prompts: list[str] | None = None,
    max_tokens: int = 4,
    max_model_len: int = 2048,
    run_name: str | None = None,
    list_layers: bool = False,
    spinlock_handling: str = "mark_region",
) -> str | None:
    """Trace `layers` of `model` under vLLM and save the traces to the volume.

    spinlock_handling: "mark_region" (Accel-Sim default for Hopper), "fast_forward", or "none".
    Any mode other than "none" first runs Accel-Sim's two-phase spinlock detection
    (spinlock_tool.so), which the tracer requires: with SPINLOCK_HANDLING_MODE>0 and no
    spinlock_detection/spinlock_instructions.txt, tracer_tool.so dereferences a null
    pointer (segfault in create_counter) on the first traced kernel.

    Returns the run directory name in the `accelsim-traces` volume (None for list_layers).
    """
    if spinlock_handling not in SPINLOCK_MODES:
        raise ValueError(f"spinlock_handling must be one of {list(SPINLOCK_MODES)}")
    layers = layers or ["model.decoder.layers.10.self_attn"]
    prompts = prompts or ["Hello, my name is"]
    workload = [
        sys.executable,
        TRACE_SCRIPT,
        "--model", model,
        "--layers", ",".join(layers),
        "--prompts", json.dumps(prompts),
        "--max-tokens", str(max_tokens),
        "--max-model-len", str(max_model_len),
    ]
    if list_layers:
        workload.append("--list-layers")

    if run_name is None:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{model}__{'+'.join(layers)}")
        run_name = f"{time.strftime('%Y%m%d-%H%M%S')}__{slug}"[:200]

    # Write traces to local disk first (many small writes); copy to the volume at the end.
    work_dir = Path("/tmp/accelsim") / run_name
    work_dir.mkdir(parents=True, exist_ok=True)
    # Both tools write under $TRACES_FOLDER (traces/, spinlock_detection/). The spinlock tool
    # concatenates paths without a separator, so the trailing slash is required.
    base_env = {**os.environ, "TRACES_FOLDER": f"{work_dir}/"}
    deadline = time.monotonic() + TRACE_TIMEOUT_S

    def run_step(name: str, extra_env: dict[str, str]):
        # run.sh exports its own CUDA_INJECTION64_PATH / SPINLOCK_HANDLING_MODE, so overrides
        # go through `env` *after* run.sh, right before the workload is exec'd.
        overrides = [f"{k}={v}" for k, v in extra_env.items()]
        cmd = [f"{TORCH_HOOK_DIR}/run.sh", "env", *overrides, *workload]
        _run_logged(name, cmd, base_env, work_dir, deadline)

    subprocess.run(["nvidia-smi"], check=False)
    status = "failed"
    try:
        if list_layers:
            run_step("list_layers", {})
            status = "ok"
            return None

        mode = SPINLOCK_MODES[spinlock_handling]
        if mode > 0:
            # Same as run_hw_trace.py's run_spinlock_detection.sh: run the workload twice under
            # spinlock_tool.so; phase 1 diffs the instruction histograms and writes
            # spinlock_detection/spinlock_instructions.txt. The torch hook drives this tool
            # too, so only the selected layers are profiled.
            for phase in (0, 1):
                run_step(f"spinlock_phase{phase}", {
                    "CUDA_INJECTION64_PATH": SPINLOCK_SO,
                    "SPINLOCK_PHASE": str(phase),
                })
            spin_file = work_dir / "spinlock_detection" / "spinlock_instructions.txt"
            if not spin_file.is_file() or spin_file.stat().st_size == 0:
                raise RuntimeError(
                    f"Spinlock detection produced no {spin_file}; see spinlock_phase*.log"
                )
            print(f"Spinlock detection found entries for "
                  f"{sum(1 for l in spin_file.read_text().splitlines() if l and l[0] != '#')} kernels",
                  flush=True)

        run_step("trace", {
            "CUDA_INJECTION64_PATH": TRACER_SO,
            "SPINLOCK_HANDLING_MODE": str(mode),
        })

        traces_dir = work_dir / "traces"
        if not traces_dir.is_dir() or not any(traces_dir.glob("kernel-*")):
            raise RuntimeError(
                f"No kernel traces were produced in {traces_dir}; check that the layer name is "
                "correct and that 'Writing results to ...' appears in trace.log"
            )

        # README step 4: post-process into .tracez and generate kernelslist.g
        subprocess.run(
            [POST_PROCESS_BIN, str(traces_dir), "-j", str(os.cpu_count() or 8)], check=True
        )
        status = "ok"
        return run_name
    finally:
        if not list_layers:
            _save_to_volume(work_dir, run_name, status, {
                "model": model,
                "layers": layers,
                "prompts": prompts,
                "max_tokens": max_tokens,
                "max_model_len": max_model_len,
                "spinlock_handling": spinlock_handling,
            })


def _run_logged(name: str, cmd: list[str], env: dict, cwd: Path, deadline: float):
    """Run cmd, streaming output to stdout and <cwd>/<name>.log; kill it at `deadline`."""
    print(f"=== [{name}] Running: {' '.join(cmd)}", flush=True)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError(f"[{name}] no time left before the function timeout")
    with open(cwd / f"{name}.log", "w") as log:
        proc = subprocess.Popen(
            cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        # Kill the workload before Modal's function timeout so partial traces + logs still get saved.
        killer = threading.Timer(remaining, proc.kill)
        killer.start()
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()
        finally:
            killer.cancel()
    if proc.returncode != 0:
        raise RuntimeError(f"[{name}] failed with exit code {proc.returncode}")


def _save_to_volume(work_dir: Path, run_name: str, status: str, params: dict):
    """Copy everything in work_dir (traces, log, metadata) to the volume, even on failure."""
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    (work_dir / "metadata.json").write_text(
        json.dumps({**params, "status": status, "accelsim_commit": ACCELSIM_COMMIT, "gpu": gpu}, indent=2)
    )
    dest = Path(TRACES_MOUNT) / run_name
    shutil.copytree(work_dir, dest, dirs_exist_ok=True)
    traces_vol.commit()
    n_files = sum(1 for f in dest.rglob("*") if f.is_file())
    print(f"[{status}] Saved {n_files} files to volume 'accelsim-traces' at /{run_name}", flush=True)


@app.local_entrypoint()
def main(
    model: str = "facebook/opt-125m",
    layers: str = "model.decoder.layers.10.self_attn",
    prompt: str = "Hello, my name is",
    max_tokens: int = 4,
    max_model_len: int = 2048,
    run_name: str = "",
    list_layers: bool = False,
    spinlock_handling: str = "mark_region",
):
    result = trace_vllm.remote(
        model=model,
        layers=[l.strip() for l in layers.split(",") if l.strip()],
        prompts=[prompt],
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        run_name=run_name or None,
        list_layers=list_layers,
        spinlock_handling=spinlock_handling,
    )
    if result:
        print(f"\nDone. Download with:\n  modal volume get accelsim-traces {result} ./{result}")
        print(
            "Simulate with:\n  ./gpu-simulator/bin/release/accel-sim.out "
            "-config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config "
            "-config ./gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config "
            f"-trace ./{result}/traces/kernelslist.g"
        )
