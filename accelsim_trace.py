"""Generate Accel-Sim 2.0 SASS traces of a vLLM model on a Modal H100, streamed to S3.

Follows the "Tracing LLMs with vLLM" section of the accel-sim-framework README:
  1. build NVBit + the tracer (tracer_tool.so) and install vLLM
  2. hook NVBit onto selected layers via llm.collective_rpc (see vllm_trace.py)
  3. run the script through util/tracer_nvbit/others/torch_hook/run.sh, preceded by the
     two-phase spinlock detection that run_hw_trace.py does (required for mark_region)
  4. post-process the raw traces into .tracez + kernelslist.g -- done *off* Modal (see below)

Traces go to an S3-compatible bucket while tracing is still running. The tracer writes one
`kernel-N-ctx_*.trace.xz` per kernel and fully closes it before the next kernel launches, so
a background thread uploads each finished kernel file (streaming multipart, constant RAM) and
deletes it locally. Local disk only ever holds the in-progress kernel plus any upload backlog.

Post-processing is skipped on Modal: it loads a whole kernel trace into RAM and needs every
file locally, so run it where the traces land (it's a CPU-only tool):
  ./util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing <dir>/traces -j 8

S3 config is passed on the command line (--s3-endpoint, --s3-bucket, --s3-prefix). Credentials
are not wired up yet: if AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY happen to be set in the
container they're used, otherwise requests are sent unsigned.

Usage:
  modal run accelsim_trace.py::main --list-layers
  modal run accelsim_trace.py::main --layers model.decoder.layers.10.self_attn --max-tokens 4
      # -> accelsim-traces volume (post-processed); add --s3-endpoint/--s3-bucket to stream to S3
  modal run accelsim_trace.py::main ... --spinlock-handling none   # skip spinlock detection
  modal run accelsim_trace.py::main ... --trace-scope steps [--trace-steps 0,1] [--exec-mode compile]
  modal run accelsim_trace.py::nsys_main --trace-run <run> [--exec-mode graphs]
  modal run accelsim_trace.py::microbench_main
  (see PROFILER_RUNS.md)
Objects land at s3://<bucket>/<prefix>/<run_name>/{traces/,spinlock_detection/,*.log,metadata.json}
Without --s3-bucket the run dir goes to the accelsim-traces Modal volume at /<run_name>/ instead,
post-processed on Modal (kernelslist.g + .tracez) unless --no-post-process.
"""

import json
import os
import re
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
MICROBENCH_SCRIPT = "/root/kernel_microbench.py"
TRACE_SCOPES = ("layers", "steps")
EXEC_MODES = ("eager", "compile", "graphs")
FUNCTION_TIMEOUT_S = 4 * 60 * 60  # NVBit instrumentation is slow
TRACE_TIMEOUT_S = FUNCTION_TIMEOUT_S - 30 * 60  # leave time to drain the upload backlog
# Local scratch for the in-progress kernel + upload backlog. Modal's max is 3 TiB; disk is billed
# by raising the memory request 20:1 (512 GiB -> 25.6 GiB, below the 64 GiB requested anyway).
EPHEMERAL_DISK_MIB = 512 * 1024
# S3 allows 10,000 parts per multipart upload: 128 MiB parts -> objects up to ~1.2 TiB.
# Upload RAM ~= part size x concurrency = 1 GiB.
MULTIPART_CHUNK = 128 * 1024 * 1024
MULTIPART_CONCURRENCY = 8

KERNEL_FILE_RE = re.compile(r"^kernel-(\d+)-(ctx_0x[0-9a-f]+)\.trace(\.xz)?$")

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
    # Separate layer so adding it doesn't rebuild vLLM / the tracer.
    .uv_pip_install("boto3>=1.36")
    # Nsight Systems CLI for the real-hardware timing pass (profile_nsys). Own layer, same reason.
    .apt_install("cuda-nsight-systems-12-9")
    .run_commands(
        "command -v nsys || ln -sf \"$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys | tail -1)\" /usr/local/bin/nsys",
        "nsys --version",
    )
    .add_local_file(Path(__file__).parent / "vllm_trace.py", TRACE_SCRIPT)
    .add_local_file(Path(__file__).parent / "kernel_microbench.py", MICROBENCH_SCRIPT)
)

app = modal.App("accelsim-vllm-trace", image=image)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
traces_vol = modal.Volume.from_name("accelsim-traces", create_if_missing=True)
TRACES_MOUNT = "/traces"


# --------------------------------------------------------------------------------------
# S3 upload
# --------------------------------------------------------------------------------------
class S3Dest:
    """S3-compatible destination. Credentials: AWS_* env vars if present, else unsigned."""

    def __init__(self, run_name: str, endpoint: str, bucket: str, prefix: str = "accelsim-traces",
                 region: str = "auto", addressing_style: str = "auto"):
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore import UNSIGNED
        from botocore.config import Config

        self.bucket = bucket
        prefix = prefix.strip("/")
        self.prefix = f"{prefix}/{run_name}" if prefix else run_name
        has_creds = bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint or None,
            region_name=region,
            config=Config(
                signature_version=None if has_creds else UNSIGNED,
                retries={"max_attempts": 10, "mode": "adaptive"},
                # boto3>=1.36 adds CRC checksums by default, which many S3-compatible
                # endpoints (MinIO/Ceph/older R2 setups) reject. Only send when required.
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                s3={"addressing_style": addressing_style},
            ),
        )
        self.transfer = TransferConfig(
            multipart_threshold=MULTIPART_CHUNK,
            multipart_chunksize=MULTIPART_CHUNK,
            max_concurrency=MULTIPART_CONCURRENCY,
        )

    def uri(self, rel: str = "") -> str:
        return f"s3://{self.bucket}/{self.prefix}/{rel}".rstrip("/")

    def upload(self, path: Path, rel: str):
        # upload_file streams from disk in MULTIPART_CHUNK pieces; it never reads the whole file.
        self.client.upload_file(
            str(path), self.bucket, f"{self.prefix}/{rel}", Config=self.transfer
        )


class VolumeDest:
    """The accelsim-traces Modal volume as a destination (used when no S3 bucket is given)."""

    def __init__(self, run_name: str):
        self.root = Path(TRACES_MOUNT) / run_name
        self.run_name = run_name

    def uri(self, rel: str = "") -> str:
        return f"modal-volume://accelsim-traces/{self.run_name}/{rel}".rstrip("/")

    def upload(self, path: Path, rel: str):
        import shutil
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)


class KernelTraceUploader(threading.Thread):
    """Uploads finished kernel-N-*.trace.xz files while the tracer is still running.

    tracer_tool.so traces one kernel at a time: leave_kernel_launch() syncs the device and
    pclose()s kernel N's xz pipe before kernel N+1's file is opened. So within a context,
    every kernel file except the highest-numbered one is complete; the last one is complete
    once the traced process exits (call finish()).
    """

    def __init__(self, dest: S3Dest, traces_dir: Path, poll_s: float = 2.0):
        super().__init__(daemon=True, name="s3-uploader")
        self.dest, self.traces_dir, self.poll_s = dest, traces_dir, poll_s
        self._final = threading.Event()
        self.uploaded: list[str] = []
        self.bytes_uploaded = 0
        self.error: BaseException | None = None

    def _ready_files(self, include_latest: bool) -> list[Path]:
        if not self.traces_dir.is_dir():
            return []
        by_ctx: dict[str, list[tuple[int, Path]]] = {}
        for p in self.traces_dir.iterdir():
            m = KERNEL_FILE_RE.match(p.name)
            if m:
                by_ctx.setdefault(m.group(2), []).append((int(m.group(1)), p))
        ready = []
        for files in by_ctx.values():
            files.sort()
            ready += [p for _, p in (files if include_latest else files[:-1])]
        return ready

    def _drain(self, include_latest: bool):
        for path in self._ready_files(include_latest):
            size = path.stat().st_size
            t0 = time.monotonic()
            self.dest.upload(path, f"traces/{path.name}")
            path.unlink()  # free local disk; the object is now in S3
            self.uploaded.append(path.name)
            self.bytes_uploaded += size
            dt = max(time.monotonic() - t0, 1e-6)
            print(f"[s3] {path.name} {size / 2**20:.1f} MiB in {dt:.1f}s "
                  f"({size / 2**20 / dt:.0f} MiB/s) -> {self.dest.uri('traces/' + path.name)}",
                  flush=True)

    def run(self):
        try:
            while not self._final.is_set():
                self._drain(include_latest=False)
                self._final.wait(self.poll_s)
            self._drain(include_latest=True)
        except BaseException as e:  # surfaced by finish(); local files are kept on failure
            self.error = e

    def finish(self, timeout: float | None = None):
        """Call after the traced process has exited: uploads the last kernel and joins."""
        self._final.set()
        self.join(timeout)
        if self.is_alive():
            raise RuntimeError("S3 upload did not finish before the timeout")
        if self.error:
            raise RuntimeError(f"S3 upload failed: {self.error!r}") from self.error


# --------------------------------------------------------------------------------------
# Modal function
# --------------------------------------------------------------------------------------
@app.function(
    gpu="H100!",  # "!" pins H100; plain "H100" may be upgraded to H200, which would mis-tag traces
    timeout=FUNCTION_TIMEOUT_S,
    cpu=8,
    memory=64 * 1024,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
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
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "accelsim-traces",
    s3_region: str = "auto",
    s3_addressing_style: str = "auto",
    trace_scope: str = "layers",
    trace_steps: list[int] | None = None,
    exec_mode: str = "eager",
    post_process: bool = True,
) -> str | None:
    """Trace `layers` of `model` under vLLM, streaming traces to S3 as kernels finish.

    spinlock_handling: "mark_region" (Accel-Sim default for Hopper), "fast_forward", or "none".
    Any mode other than "none" first runs Accel-Sim's two-phase spinlock detection
    (spinlock_tool.so), which the tracer requires: with SPINLOCK_HANDLING_MODE>0 and no
    spinlock_detection/spinlock_instructions.txt, tracer_tool.so dereferences a null
    pointer (segfault in create_counter) on the first traced kernel.

    trace_scope: "layers" traces only `layers`; "steps" traces every kernel of the engine steps
    in `trace_steps` (0 = prefill; None/[] = all steps) -- whole forward passes incl. sampler.
    exec_mode: "eager" (enforce_eager), "compile" (torch.compile, no CUDA graphs: same Triton
    kernels as "graphs" but traceable), or "graphs" (torch.compile + CUDA graphs). NVBit only
    instruments graph kernels at capture time (vLLM captures inside LLM(), while tracing is
    off) and gives replayed kernels no launch ids, so "graphs" traces are incomplete; trace
    with "compile" and time with profile_nsys(exec_mode="graphs") instead.

    Storage: with s3_bucket, kernel traces stream to S3 while tracing (no post-processing).
    Without it, the whole run dir is written to the accelsim-traces volume under /<run_name>/
    and, if post_process, post-traces-processing runs first (kernelslist.g + .tracez).

    Every run logs per-engine-step tracer kernel-id ranges to <run>/trace_steps.json.

    Returns the s3:// URI of the run (None for list_layers).
    """
    if spinlock_handling not in SPINLOCK_MODES:
        raise ValueError(f"spinlock_handling must be one of {list(SPINLOCK_MODES)}")
    if trace_scope not in TRACE_SCOPES:
        raise ValueError(f"trace_scope must be one of {TRACE_SCOPES}")
    if exec_mode not in EXEC_MODES:
        raise ValueError(f"exec_mode must be one of {EXEC_MODES}")
    layers = layers or ["model.decoder.layers.10.self_attn"]
    prompts = prompts or ["Hello, my name is"]
    params = {
        "model": model,
        "layers": layers,
        "prompts": prompts,
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "spinlock_handling": spinlock_handling,
        "trace_scope": trace_scope,
        "trace_steps": sorted(trace_steps or []),
        "exec_mode": exec_mode,
        "async_scheduling": False,
    }
    workload = _workload(params)
    if list_layers:
        workload.append("--list-layers")

    if run_name is None:
        what = "+".join(layers) if trace_scope == "layers" else (
            "steps-" + ("-".join(map(str, params["trace_steps"])) or "all"))
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{model}__{what}__{exec_mode}")
        run_name = f"{time.strftime('%Y%m%d-%H%M%S')}__{slug}"[:200]

    work_dir = Path("/tmp/accelsim") / run_name
    work_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = work_dir / "traces"
    # Both tools write under $TRACES_FOLDER (traces/, spinlock_detection/). The spinlock tool
    # concatenates paths without a separator, so the trailing slash is required.
    base_env = {**os.environ, "TRACES_FOLDER": f"{work_dir}/"}
    deadline = time.monotonic() + TRACE_TIMEOUT_S

    def run_step(name: str, extra_env: dict[str, str]):
        # run.sh exports its own CUDA_INJECTION64_PATH / SPINLOCK_HANDLING_MODE, so overrides
        # go through `env` *after* run.sh, right before the workload is exec'd.
        overrides = [f"{k}={v}" for k, v in
                     {"VLLM_STEP_LOG": str(work_dir / f"{name}_steps.json"), **extra_env}.items()]
        cmd = [f"{TORCH_HOOK_DIR}/run.sh", "env", *overrides, *workload]
        _run_logged(name, cmd, base_env, work_dir, deadline)

    subprocess.run(["nvidia-smi"], check=False)
    _write_clock_snapshot(work_dir / "clocks_trace_host.txt")
    if list_layers:
        run_step("list_layers", {})
        return None

    use_s3 = bool(s3_bucket)
    if use_s3:
        dest = S3Dest(run_name, s3_endpoint, s3_bucket, s3_prefix, s3_region, s3_addressing_style)
        # Fail fast on an unreachable endpoint / missing bucket before spending GPU time.
        dest.client.head_bucket(Bucket=dest.bucket)
    else:
        dest = VolumeDest(run_name)
    params["storage"] = "s3" if use_s3 else "modal_volume"
    params["post_processed"] = False
    print(f"Saving to {dest.uri()}", flush=True)

    status = "failed"
    uploader: KernelTraceUploader | None = None
    try:
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

        if use_s3:
            uploader = KernelTraceUploader(dest, traces_dir)
            uploader.start()
        trace_err: Exception | None = None
        try:
            run_step("trace", {
                "CUDA_INJECTION64_PATH": TRACER_SO,
                "SPINLOCK_HANDLING_MODE": str(mode),
            })
        except Exception as e:
            trace_err = e  # still upload whatever kernels were traced before re-raising
            print(f"Trace step failed: {e!r}; uploading partial traces", flush=True)
        # The traced process has exited (or been killed), so every kernel file is closed.
        if uploader:
            uploader.finish(timeout=max(deadline - time.monotonic(), 0) + 25 * 60)
        if trace_err:
            raise trace_err

        n_kernels = len(uploader.uploaded) if uploader else len(_kernel_files(traces_dir))
        if not n_kernels:
            raise RuntimeError(
                "No kernel traces were produced; check that the layer name is correct and "
                "that 'Writing results to ...' appears in trace.log"
            )
        if not use_s3 and post_process:
            # Writes kernelslist.g + one .tracez per kernel next to the raw traces.
            _run_logged("post_process", [POST_PROCESS_BIN, str(traces_dir), "-j", "8"],
                        dict(os.environ), work_dir, deadline + 25 * 60)
            params["post_processed"] = True
        status = "ok"
        return dest.uri()
    finally:
        _upload_run_files(dest, work_dir, status, uploader, params)
        if not use_s3:
            traces_vol.commit()


def _kernel_files(traces_dir: Path) -> list[Path]:
    if not traces_dir.is_dir():
        return []
    return [p for p in traces_dir.iterdir() if KERNEL_FILE_RE.match(p.name)]


def _workload(meta: dict, exec_mode: str | None = None, script: str = TRACE_SCRIPT) -> list[str]:
    """vllm_trace.py command line for a run's parameters (metadata.json).

    Keys missing from older metadata fall back to the original behaviour (layer scope, eager,
    vLLM's default async scheduling), so old trace runs are re-profiled with identical settings.
    """
    cmd = [
        sys.executable, script,
        "--model", meta["model"],
        "--layers", ",".join(meta.get("layers") or []),
        "--prompts", json.dumps(meta["prompts"]),
        "--max-tokens", str(meta["max_tokens"]),
        "--max-model-len", str(meta["max_model_len"]),
        "--trace-scope", meta.get("trace_scope", "layers"),
        "--trace-steps", ",".join(map(str, meta.get("trace_steps") or [])),
        "--exec-mode", exec_mode or meta.get("exec_mode", "eager"),
    ]
    if meta.get("async_scheduling") is False:
        cmd.append("--no-async-scheduling")
    return cmd


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
        # Kill the workload before Modal's function timeout so uploads can still drain.
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


def _upload_run_files(
    dest: "S3Dest | VolumeDest", work_dir: Path, status: str, uploader: "KernelTraceUploader | None", params: dict
):
    """Upload everything except kernel traces (logs, kernelslist/stats, spinlock file, metadata).

    Runs even on failure. Kernel trace files still on disk at this point (only if the uploader
    failed) are retried here too, so a partial run is as complete in S3 as possible.
    """
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    meta = {
        **params,
        "status": status,
        "accelsim_commit": ACCELSIM_COMMIT,
        "gpu": gpu,
        "kernel_traces_uploaded": len(uploader.uploaded) if uploader else 0,
        "bytes_uploaded": uploader.bytes_uploaded if uploader else 0,
        "post_processed": params.get("post_processed", False),
        "post_process_cmd": "post-traces-processing <run>/traces -j 8  # produces kernelslist.g + .tracez",
    }
    (work_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    n, failed = 0, []
    for path in sorted(p for p in work_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(work_dir).as_posix()
        if rel.startswith("spinlock_detection/") and rel != "spinlock_detection/spinlock_instructions.txt":
            continue  # intermediate histograms; not needed by the tracer/simulator
        try:
            dest.upload(path, rel)
            n += 1
        except Exception as e:
            failed.append(f"{rel}: {e!r}")
    print(f"[{status}] Saved {n} run files (+{meta['kernel_traces_uploaded']} streamed kernel traces, "
          f"{meta['bytes_uploaded'] / 2**30:.2f} GiB) to {dest.uri()}", flush=True)
    if failed:
        print("FAILED uploads:\n  " + "\n  ".join(failed), flush=True)


# --------------------------------------------------------------------------------------
# GPU clocks
# --------------------------------------------------------------------------------------
CLOCK_SMI_FIELDS = ("timestamp,clocks.sm,clocks.mem,clocks.gr,clocks.max.sm,clocks.max.mem,"
                    "clocks.applications.graphics,clocks.applications.memory,power.draw,"
                    "temperature.gpu,clocks_throttle_reasons.active")


def _write_clock_snapshot(path: Path):
    """nvidia-smi clock/throttle snapshot (supported, current, max, application clocks)."""
    out = subprocess.run(["nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE,POWER"],
                         capture_output=True, text=True)
    path.write_text(out.stdout + out.stderr)


class ClockSampler(threading.Thread):
    """Samples SM/mem clocks + throttle reasons via NVML every `period_s` into a CSV.

    Each row carries wall_ns (time.time_ns, same clock as vllm_trace.py's step log) so samples
    can be mapped onto the nsys timeline through the step NVTX ranges.
    """

    COLS = ["wall_ns", "sm_mhz", "mem_mhz", "gr_mhz", "power_w", "temp_c", "throttle_reasons_hex",
            "throttle_reasons"]

    def __init__(self, path: Path, period_s: float = 0.05):
        super().__init__(daemon=True, name="clock-sampler")
        self.path, self.period_s = path, period_s
        self._stop = threading.Event()
        self.rows: list[dict] = []
        self.error: str | None = None

    def run(self):
        try:
            import pynvml as nv  # nvidia-ml-py, a vLLM dependency
            nv.nvmlInit()
            h = nv.nvmlDeviceGetHandleByIndex(0)
            get_reasons = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
                nv.nvmlDeviceGetCurrentClocksThrottleReasons
            names = {getattr(nv, a): a.split("Reason", 1)[1] for a in dir(nv)
                     if a.startswith(("nvmlClocksThrottleReason", "nvmlClocksEventReason"))
                     and "All" not in a and "None" not in a and isinstance(getattr(nv, a), int)}
            while not self._stop.is_set():
                r = get_reasons(h)
                self.rows.append({
                    "wall_ns": time.time_ns(),
                    "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                    "mem_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                    "gr_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_GRAPHICS),
                    "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000,
                    "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                    "throttle_reasons_hex": hex(r),
                    "throttle_reasons": "|".join(sorted({n for b, n in names.items() if r & b})),
                })
                self._stop.wait(self.period_s)
        except Exception as e:
            self.error = repr(e)
            print(f"[clocks] NVML sampling failed: {e!r}", flush=True)

    def stop(self):
        self._stop.set()
        self.join(10)
        import csv
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.COLS)
            w.writeheader()
            w.writerows(self.rows)


def _lock_clocks(mhz: int) -> dict:
    """Try to lock SM clocks (needs root on the host; usually refused in containers)."""
    res = {"requested_sm_mhz": mhz}
    for key, cmd in (("lgc", ["nvidia-smi", "-lgc", f"{mhz},{mhz}"]),):
        r = subprocess.run(cmd, capture_output=True, text=True)
        res[key] = {"returncode": r.returncode, "output": (r.stdout + r.stderr).strip()}
    res["locked"] = res["lgc"]["returncode"] == 0
    return res


def _clock_stats(rows: list[dict], t0_ns: int | None, t1_ns: int | None) -> dict:
    import statistics
    sel = [r for r in rows if t0_ns is None or t0_ns <= r["wall_ns"] <= t1_ns] or rows
    if not sel:
        return {}
    out = {"samples": len(sel)}
    for k in ("sm_mhz", "mem_mhz"):
        v = [r[k] for r in sel]
        out[k] = {"min": min(v), "median": statistics.median(v), "max": max(v)}
    out["throttle_reasons_seen"] = sorted({x for r in sel for x in r["throttle_reasons"].split("|") if x})
    return out


# --------------------------------------------------------------------------------------
# Real-hardware kernel timing (Nsight Systems), matched to the tracer's kernel ids
# --------------------------------------------------------------------------------------
NSYS_TIMEOUT_S = 60 * 60
STEP_RE = re.compile(r"^vllm_step_(\d+)$")
STEP_KIND_RE = re.compile(r"^(prefill|decode|mixed|empty|unknown)_(\d+)$")


def _nsys_base_cmd(gpu_metrics: bool = False) -> list[str]:
    """nsys profile flags; optional ones are only added if this nsys version knows them."""
    help_text = subprocess.run(["nsys", "profile", "--help"], capture_output=True, text=True).stdout
    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        # No CPU sampling / context-switch tracing: needs perf, unavailable in gVisor, not needed.
        "--sample=none", "--cpuctxsw=none",
        "--force-overwrite=true", "--export=sqlite",
    ]
    # One CUPTI record per kernel *node* of a CUDA graph (default: one per graph launch).
    if "--cuda-graph-trace" in help_text:
        cmd.append("--cuda-graph-trace=node")
    # cudaEventRecord/cudaStreamWaitEvent etc. as activity records (synchronization table).
    if "--cuda-event-trace" in help_text:
        cmd.append("--cuda-event-trace=true")
    if gpu_metrics:
        flag = "--gpu-metrics-devices" if "--gpu-metrics-devices" in help_text else "--gpu-metrics-device"
        cmd.append(f"{flag}=all")
    return cmd


def _nsys_env(extra: dict | None = None) -> dict:
    env = {
        **os.environ,
        # Same process layout as the traced run (run.sh sets these); no CUDA_INJECTION64_PATH,
        # so the NVBit hooks are no-ops and nsys owns CUDA injection.
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "PYTHONPATH": f"{ACCELSIM_DIR}/util/tracer_nvbit/others:"
                      f"{ACCELSIM_DIR}/util/tracer_nvbit/tracer_tool:{os.environ.get('PYTHONPATH', '')}",
        **(extra or {}),
    }
    env.pop("CUDA_INJECTION64_PATH", None)
    return env


def _write_csv(path: Path, rows: list[dict], cols: list[str] | None = None):
    import csv
    cols = cols or (list(rows[0]) if rows else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


@app.function(
    gpu="H100!",
    timeout=NSYS_TIMEOUT_S,
    cpu=8,
    memory=64 * 1024,
    volumes={"/root/.cache/huggingface": hf_cache_vol, TRACES_MOUNT: traces_vol},
)
def profile_nsys(trace_run: str, exec_mode: str = "", lock_clocks_mhz: int = 0,
                 gpu_metrics: bool = False, out_name: str = "") -> dict:
    """Re-run a traced workload *without* the tracer, under nsys, to get real kernel timings.

    The Accel-Sim tracer records no timestamps, only launch order (kernel-N = N-th kernel launch
    in the process). This pass runs the identical vLLM workload (same args, read from the trace
    run's metadata.json; greedy + ignore_eos so the kernel sequence is deterministic), records
    every kernel with nsys, and aligns the nsys kernels to the tracer's launch sequence
    (stats_ctx_*: all launches, traced or not) by mangled name, globally and per engine step.

    exec_mode: override the trace run's exec mode (e.g. trace with "compile", time with
      "graphs"); output then goes to nsys_<exec_mode>/ unless out_name is given.
    lock_clocks_mhz: try `nvidia-smi -lgc <mhz>,<mhz>` first (result recorded; reset after).
    gpu_metrics: also sample nsys GPU metrics (SM clock, occupancy...); may need privileges.

    Writes to the volume under /<trace_run>/<out_name or nsys>/:
      profile.nsys-rep, profile.sqlite   raw nsys output (open the .nsys-rep in the Nsight GUI)
      kernels.csv     one row per kernel launch, in launch order: trace_kernel_id, step, GPU
                      start/end/duration, launch API name/start/end, correlation id, pid/tid
      events.csv      every GPU activity (kernel, memcpy, memset, synchronization) in time order
                      with stream, bytes/copy kind, sync type/event id, and its API call
      api_calls.csv   every CUDA runtime API call (name, start/end, correlation id, tid, step)
      steps.csv       per engine step (NVTX vllm_step_<i>): kind, span, GPU busy/idle, counts,
                      and per-step tracer alignment
      steps.json      vllm_trace.py's step log for this run (wall clock, scheduled tokens)
      clocks.csv      NVML SM/mem clock + throttle-reason samples every 50 ms (wall_ns and
                      nsys-timeline ns), clocks_*.txt nvidia-smi snapshots
      summary.json    alignment (global + per step), clocks, GPU, versions
      nsys.log        profiler + vLLM output
    """
    run_dir = Path(TRACES_MOUNT) / trace_run
    meta = json.loads((run_dir / "metadata.json").read_text())
    stats_files = sorted((run_dir / "traces").glob("stats_ctx_*"))
    if len(stats_files) != 1:
        raise RuntimeError(f"expected 1 stats_ctx_* in {run_dir}/traces, found {stats_files}")
    if exec_mode and exec_mode not in EXEC_MODES:
        raise ValueError(f"exec_mode must be one of {EXEC_MODES}")
    run_mode = exec_mode or meta.get("exec_mode", "eager")
    out_name = out_name or ("nsys" if run_mode == meta.get("exec_mode", "eager") else f"nsys_{run_mode}")

    work = Path("/tmp/nsys") / trace_run / out_name
    work.mkdir(parents=True, exist_ok=True)
    env = _nsys_env({"VLLM_STEP_LOG": str(work / "steps.json")})
    cmd = [*_nsys_base_cmd(gpu_metrics), "-o", str(work / "profile"), *_workload(meta, run_mode)]

    subprocess.run(["nvidia-smi"], check=False)
    clock_lock = _lock_clocks(lock_clocks_mhz) if lock_clocks_mhz else None
    _write_clock_snapshot(work / "clocks_before.txt")
    sampler = ClockSampler(work / "clocks.csv")
    sampler.start()
    try:
        _run_logged("nsys", cmd, env, work, time.monotonic() + NSYS_TIMEOUT_S - 10 * 60)
    finally:
        sampler.stop()
        _write_clock_snapshot(work / "clocks_after.txt")
        if clock_lock and clock_lock["locked"]:
            subprocess.run(["nvidia-smi", "-rgc"], check=False)

    db = work / "profile.sqlite"
    kernels = _nsys_kernels(db)
    steps = _nsys_steps(db)
    step_log = json.loads((work / "steps.json").read_text()) if (work / "steps.json").exists() else []
    trace_step_log_path = run_dir / "trace_steps.json"
    trace_step_log = json.loads(trace_step_log_path.read_text()) if trace_step_log_path.exists() else []
    _assign_steps(kernels, steps)
    traced = _tracer_launches(stats_files[0])
    rows, alignment = _align(kernels, traced)
    per_step = _per_step(rows, steps, step_log, trace_step_log, db)

    # Map NVML samples onto the nsys timeline via step start (wall clock vs nsys clock).
    offsets = sorted(st["wall_start_ns"] - steps[st["step"]]["start"]
                     for st in step_log if st["step"] in steps)
    offset = offsets[len(offsets) // 2] if offsets else None
    for r in sampler.rows:
        r["nsys_ns"] = "" if offset is None else r["wall_ns"] - offset
    _write_csv(work / "clocks.csv", sampler.rows, ClockSampler.COLS + ["nsys_ns"])

    _write_csv(work / "kernels.csv", rows, [
        "launch_seq", "trace_kernel_id", "traced", "launch_dims_match", "step", "step_kind",
        "start_ns", "end_ns", "duration_ns", "gap_from_prev_end_ns", "abs_start_ns",
        "api_name", "api_start_ns", "api_end_ns", "launch_to_start_ns", "correlation_id",
        "pid", "tid", "device_id", "stream_id", "graph_node_id", "grid", "block",
        "registers_per_thread", "shared_mem_bytes", "short_name", "mangled_name"])
    t0 = kernels[0]["start"] if kernels else 0
    events = _nsys_events(db, t0, steps)
    _write_csv(work / "events.csv", events)
    _write_csv(work / "api_calls.csv", _nsys_api_calls(db, t0, steps))
    _write_csv(work / "steps.csv", per_step)

    traced_rows = [r for r in rows if r["traced"]]
    w0 = min((s["wall_start_ns"] for s in step_log), default=None)
    w1 = max((s["wall_end_ns"] for s in step_log), default=None)
    summary = {
        "trace_run": trace_run,
        "out_name": out_name,
        "workload": {k: meta.get(k) for k in ("model", "layers", "prompts", "max_tokens",
                                              "max_model_len", "trace_scope", "trace_steps",
                                              "exec_mode", "async_scheduling")},
        "nsys_exec_mode": run_mode,
        "gpu": _gpu_name(),
        "nsys_version": subprocess.run(["nsys", "--version"], capture_output=True, text=True).stdout.strip(),
        "nsys_cmd": cmd[:len(cmd) - len(_workload(meta, run_mode))],
        "nsys_kernels": len(kernels),
        "tracer_launches": len(traced),
        "streams": sorted({k["stream_id"] for k in kernels}),
        "event_counts": {t: sum(1 for e in events if e["event_type"] == t)
                         for t in ("kernel", "memcpy", "memset", "sync")},
        "alignment": alignment,
        "per_step": per_step,
        "clock_lock": clock_lock,
        "clocks_during_steps": _clock_stats(sampler.rows, w0, w1),
        "clocks_sampler_error": sampler.error,
        "wall_to_nsys_offset_ns": offset,
        "traced_kernels_matched": sum(1 for r in traced_rows if r["duration_ns"] != ""),
        "traced_kernels_total": sum(1 for t in traced if t["traced"]),
        "traced_kernel_durations_ns": {
            r["trace_kernel_id"]: r["duration_ns"] for r in traced_rows if r["duration_ns"] != ""
        },
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2))

    dest = run_dir / out_name
    dest.mkdir(exist_ok=True)
    for p in work.iterdir():
        if p.is_file():
            (dest / p.name).write_bytes(p.read_bytes())
    traces_vol.commit()
    print(json.dumps({k: v for k, v in summary.items() if k != "traced_kernel_durations_ns"}, indent=2))
    return summary


def _has_table(con, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone() is not None


def _cols(con, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _enum(con, table: str) -> dict:
    """nsys ENUM_* table as {id: label}."""
    if not _has_table(con, table):
        return {}
    c = _cols(con, table)
    label = "label" if "label" in c else "name"
    return {i: v for i, v in con.execute(f"SELECT id, {label} FROM {table}")}


def _rt_join(con, alias: str) -> tuple[str, str]:
    """(select columns, join clause) for the runtime API call that issued `alias`'s activity."""
    if not _has_table(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return "NULL, NULL, NULL, NULL", ""
    return ("r.start, r.end, rn.value, r.globalTid",
            f"LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = {alias}.correlationId "
            f"AND r.globalTid / 16777216 = {alias}.globalPid / 16777216 "
            "LEFT JOIN StringIds rn ON rn.id = r.nameId")


def _pid_tid(global_tid) -> tuple:
    if global_tid is None:
        return "", ""
    return (global_tid >> 24) & 0xFFFFFF, global_tid & 0xFFFFFF


def _nsys_kernels(sqlite_path: Path) -> list[dict]:
    """All GPU kernels from an nsys sqlite export, in host launch order, with their API call."""
    import sqlite3

    con = sqlite3.connect(sqlite_path)
    kcols = _cols(con, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_col = "mangledName" if "mangledName" in kcols else "demangledName"
    rt_sel, rt_join = _rt_join(con, "k")
    graph_node = "k.graphNodeId" if "graphNodeId" in kcols else "NULL"
    # Launch order = time of the host-side launch API call (what the tracer numbers by);
    # fall back to GPU start time if the API call wasn't captured. Kernels replayed from one
    # cudaGraphLaunch share its API call and are ordered by GPU start.
    launch_ts = "COALESCE(r.start, k.start)" if rt_join else "k.start"
    q = f"""
      SELECT k.start, k.end, k.streamId, k.deviceId, k.correlationId, {graph_node},
             k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
             k.registersPerThread, k.staticSharedMemory + k.dynamicSharedMemory,
             sm.value, ss.value, {rt_sel}
      FROM CUPTI_ACTIVITY_KIND_KERNEL k
      JOIN StringIds sm ON sm.id = k.{name_col}
      JOIN StringIds ss ON ss.id = k.shortName
      {rt_join}
      ORDER BY {launch_ts}, k.start
    """
    out = []
    for (start, end, stream, dev, corr, gnode, gx, gy, gz, bx, by, bz, regs, smem, mangled, short,
         api_start, api_end, api_name, gtid) in con.execute(q):
        pid, tid = _pid_tid(gtid)
        out.append({
            "start": start, "end": end, "stream_id": stream, "device_id": dev,
            "correlation_id": corr, "graph_node_id": "" if gnode is None else gnode,
            "grid": f"{gx}x{gy}x{gz}", "block": f"{bx}x{by}x{bz}",
            "regs": regs, "smem": smem, "mangled": mangled, "short": short,
            "api_start": api_start, "api_end": api_end, "api_name": api_name or "",
            "pid": pid, "tid": tid,
        })
    return out


def _nsys_steps(sqlite_path: Path) -> dict[int, dict]:
    """vllm_step_<i> NVTX ranges (and their nested kind range) -> {i: {start, end, kind}}."""
    import sqlite3

    con = sqlite3.connect(sqlite_path)
    if not _has_table(con, "NVTX_EVENTS"):
        return {}
    text = "COALESCE(n.text, s.value)" if "textId" in _cols(con, "NVTX_EVENTS") else "n.text"
    join = "LEFT JOIN StringIds s ON s.id = n.textId" if "textId" in _cols(con, "NVTX_EVENTS") else ""
    steps, kinds = {}, {}
    for start, end, name in con.execute(
            f"SELECT n.start, n.end, {text} FROM NVTX_EVENTS n {join} WHERE n.end IS NOT NULL"):
        if not name:
            continue
        if m := STEP_RE.match(name):
            steps[int(m.group(1))] = {"start": start, "end": end}
        elif m := STEP_KIND_RE.match(name):
            kinds[int(m.group(2))] = m.group(1)
        elif name == "vllm_init":
            steps.setdefault("init", {"start": start, "end": end})
    init = steps.pop("init", None)
    for i, st in steps.items():
        st["kind"] = kinds.get(i, "unknown")
    if init:
        steps[-1] = {**init, "kind": "init"}
    return steps


def _step_of(ts, steps: dict[int, dict]):
    if ts is None:
        return ""
    for i, st in steps.items():
        if i >= 0 and st["start"] <= ts <= st["end"]:
            return i
    init = steps.get(-1)
    return "init" if init and init["start"] <= ts <= init["end"] else ""


def _assign_steps(kernels: list[dict], steps: dict[int, dict]):
    for k in kernels:
        k["step"] = _step_of(k["api_start"] if k["api_start"] is not None else k["start"], steps)
        k["step_kind"] = steps[k["step"]]["kind"] if isinstance(k["step"], int) else k["step"]


def _nsys_events(sqlite_path: Path, t0: int, steps: dict[int, dict]) -> list[dict]:
    """Kernels, memcpys, memsets and synchronizations in GPU time order, with their API call."""
    import sqlite3

    con = sqlite3.connect(sqlite_path)
    copy_kind = _enum(con, "ENUM_CUDA_MEMCPY_OPER")
    mem_kind = _enum(con, "ENUM_CUDA_MEM_KIND")
    sync_kind = _enum(con, "ENUM_CUPTI_SYNC_TYPE")
    out = []

    def add(etype, alias, table, extra_sel, fmt):
        if not _has_table(con, table):
            return
        rt_sel, rt_join = _rt_join(con, alias)
        cols = _cols(con, table)
        stream = f"{alias}.streamId" if "streamId" in cols else "NULL"
        dev = f"{alias}.deviceId" if "deviceId" in cols else "NULL"
        q = (f"SELECT {alias}.start, {alias}.end, {stream}, {dev}, {alias}.correlationId, "
             f"{rt_sel}, {extra_sel(cols)} FROM {table} {alias} {rt_join}")
        for row in con.execute(q):
            start, end, stream_id, dev_id, corr, api_start, api_end, api_name, gtid = row[:9]
            pid, tid = _pid_tid(gtid)
            ts = api_start if api_start is not None else start
            st = _step_of(ts, steps)
            out.append({
                "event_type": etype,
                "start_ns": start - t0, "end_ns": end - t0, "duration_ns": end - start,
                "abs_start_ns": start, "stream_id": stream_id, "device_id": dev_id,
                "correlation_id": corr, "api_name": api_name or "",
                "api_start_ns": "" if api_start is None else api_start - t0,
                "api_end_ns": "" if api_end is None else api_end - t0,
                "pid": pid, "tid": tid, "step": st,
                "step_kind": steps[st]["kind"] if isinstance(st, int) else st,
                "name": "", "bytes": "", "copy_kind": "", "src_kind": "", "dst_kind": "",
                "memset_value": "", "sync_type": "", "event_id": "",
                **fmt(row[9:]),
            })

    def col(cols, c, alias):
        return f"{alias}.{c}" if c in cols else "NULL"

    add("kernel", "k", "CUPTI_ACTIVITY_KIND_KERNEL",
        lambda c: "(SELECT value FROM StringIds WHERE id = k.shortName)",
        lambda r: {"name": r[0]})
    add("memcpy", "m", "CUPTI_ACTIVITY_KIND_MEMCPY",
        lambda c: f"m.bytes, m.copyKind, {col(c, 'srcKind', 'm')}, {col(c, 'dstKind', 'm')}",
        lambda r: {"bytes": r[0], "copy_kind": copy_kind.get(r[1], r[1]),
                   "src_kind": mem_kind.get(r[2], r[2] if r[2] is not None else ""),
                   "dst_kind": mem_kind.get(r[3], r[3] if r[3] is not None else ""),
                   "name": f"memcpy {copy_kind.get(r[1], r[1])}"})
    add("memset", "s", "CUPTI_ACTIVITY_KIND_MEMSET",
        lambda c: f"s.bytes, {'s.value' if 'value' in c else 'NULL'}",
        lambda r: {"bytes": r[0], "memset_value": "" if r[1] is None else r[1], "name": "memset"})
    add("sync", "y", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        lambda c: f"y.syncType, {'y.eventId' if 'eventId' in c else 'NULL'}",
        lambda r: {"sync_type": sync_kind.get(r[0], r[0]),
                   "event_id": "" if r[1] is None else r[1],
                   "name": f"sync {sync_kind.get(r[0], r[0])}"})
    out.sort(key=lambda e: (e["abs_start_ns"], e["end_ns"]))
    return out


def _nsys_api_calls(sqlite_path: Path, t0: int, steps: dict[int, dict]) -> list[dict]:
    import sqlite3

    con = sqlite3.connect(sqlite_path)
    if not _has_table(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    out = []
    rv_col = "r.returnValue" if "returnValue" in _cols(con, "CUPTI_ACTIVITY_KIND_RUNTIME") else "NULL"
    for start, end, corr, name, gtid, rv in con.execute(
            f"SELECT r.start, r.end, r.correlationId, s.value, r.globalTid, {rv_col} "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r LEFT JOIN StringIds s ON s.id = r.nameId "
            "ORDER BY r.start"):
        pid, tid = _pid_tid(gtid)
        st = _step_of(start, steps)
        out.append({"api_name": name, "start_ns": start - t0, "end_ns": end - t0,
                    "duration_ns": end - start, "abs_start_ns": start, "correlation_id": corr,
                    "return_value": rv, "pid": pid, "tid": tid, "step": st,
                    "step_kind": steps[st]["kind"] if isinstance(st, int) else st})
    return out


def _tracer_launches(stats_path: Path) -> list[dict]:
    """Every kernel launch the tracer saw (traced or not), in launch order."""
    out = []
    for line in stats_path.read_text().splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 12:
            continue
        m = re.match(r"kernel-(\d+)-", parts[0])
        out.append({
            "id": int(m.group(1)),
            "mangled": parts[1],
            "grid": f"{parts[2]}x{parts[3]}x{parts[4]}",
            "block": f"{parts[6]}x{parts[7]}x{parts[8]}",
            "traced": int(parts[10]) > 0,
        })
    return out


def _align(kernels: list[dict], traced: list[dict]) -> tuple[list[dict], dict]:
    """Match nsys kernels to tracer launch ids by mangled-name sequence.

    Matching is by name only: launch dims can legitimately differ between the traced and the
    profiled run (e.g. vLLM sizes the KV cache from free GPU memory, and NVBit takes some, so
    the KV-cache fill kernels get different grids). Differences are flagged per row instead.
    """
    import difflib

    a, b = [k["mangled"] for k in kernels], [t["mangled"] for t in traced]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    nsys_to_trace = {}
    for blk in sm.get_matching_blocks():
        for i in range(blk.size):
            nsys_to_trace[blk.a + i] = blk.b + i
    exact = a == b
    t0 = kernels[0]["start"] if kernels else 0

    def rel(ts):
        return "" if ts is None else ts - t0

    rows, prev_end = [], None
    for i, k in enumerate(kernels):
        t = traced[nsys_to_trace[i]] if i in nsys_to_trace else None
        rows.append({
            "launch_seq": i + 1,
            "trace_kernel_id": t["id"] if t else "",
            "traced": bool(t and t["traced"]),
            "launch_dims_match": "" if not t else (t["grid"], t["block"]) == (k["grid"], k["block"]),
            "step": k.get("step", ""), "step_kind": k.get("step_kind", ""),
            "start_ns": k["start"] - t0, "end_ns": k["end"] - t0,
            "duration_ns": k["end"] - k["start"],
            "gap_from_prev_end_ns": "" if prev_end is None else k["start"] - prev_end,
            "abs_start_ns": k["start"],
            "api_name": k.get("api_name", ""),
            "api_start_ns": rel(k.get("api_start")), "api_end_ns": rel(k.get("api_end")),
            "launch_to_start_ns": "" if k.get("api_start") is None else k["start"] - k["api_start"],
            "correlation_id": k.get("correlation_id", ""),
            "pid": k.get("pid", ""), "tid": k.get("tid", ""), "device_id": k.get("device_id", ""),
            "stream_id": k["stream_id"], "graph_node_id": k.get("graph_node_id", ""),
            "grid": k["grid"], "block": k["block"],
            "registers_per_thread": k["regs"], "shared_mem_bytes": k["smem"],
            "short_name": k["short"], "mangled_name": k["mangled"],
        })
        prev_end = k["end"]
    # Traced kernels that found no nsys partner are listed so nothing is silently dropped.
    matched_ids = {r["trace_kernel_id"] for r in rows if r["trace_kernel_id"] != ""}
    blank = {c: "" for c in rows[0]} if rows else {}
    for t in traced:
        if t["traced"] and t["id"] not in matched_ids:
            rows.append(blank | {"trace_kernel_id": t["id"], "traced": True, "grid": t["grid"],
                                 "block": t["block"], "mangled_name": t["mangled"]})
    alignment = {
        "exact_sequence_match": exact,
        "matched": len(nsys_to_trace),
        "nsys_unmatched": len(kernels) - len(nsys_to_trace),
        "tracer_unmatched": len(traced) - len(nsys_to_trace),
        "launch_dims_mismatched": sum(1 for r in rows if r["launch_dims_match"] is False),
        "traced_launch_dims_mismatched": sum(
            1 for r in rows if r["traced"] and r["launch_dims_match"] is False),
    }
    return rows, alignment


def _per_step(rows: list[dict], steps: dict[int, dict], step_log: list[dict],
              trace_step_log: list[dict], sqlite_path: Path) -> list[dict]:
    """Per engine step: timing from nsys + kernel matching against the tracer's step ranges."""
    import sqlite3

    con = sqlite3.connect(sqlite_path)
    busy_iv: dict[int, list] = {}
    memcpy: dict[int, list] = {}
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
                  "CUPTI_ACTIVITY_KIND_MEMSET"):
        if not _has_table(con, table):
            continue
        sel = "start, end" + (", bytes" if table.endswith("MEMCPY") else "")
        for r in con.execute(f"SELECT {sel} FROM {table}"):
            st = _step_of(r[0], steps)
            if isinstance(st, int) and st >= 0:
                busy_iv.setdefault(st, []).append((r[0], r[1]))
                if table.endswith("MEMCPY"):
                    memcpy.setdefault(st, []).append(r[2])
    log = {s["step"]: s for s in step_log}
    tlog = {s["step"]: s for s in trace_step_log}
    out = []
    for i in sorted(s for s in steps if s >= 0):
        st = steps[i]
        ks = [r for r in rows if r["step"] == i]
        iv = sorted(busy_iv.get(i, []))
        busy, cur_s, cur_e = 0, None, None
        for s0, e0 in iv:
            if cur_e is None or s0 > cur_e:
                busy += 0 if cur_e is None else cur_e - cur_s
                cur_s, cur_e = s0, e0
            else:
                cur_e = max(cur_e, e0)
        busy += 0 if cur_e is None else cur_e - cur_s
        span = st["end"] - st["start"]
        tr = tlog.get(i, {})
        lo, hi = tr.get("first_kernel_id"), tr.get("end_kernel_id")
        in_range = (lambda kid: kid != "" and lo is not None and hi is not None and lo <= kid < hi)
        matched = sum(1 for r in ks if in_range(r["trace_kernel_id"]))
        out.append({
            "step": i, "kind": st["kind"],
            "num_scheduled_tokens": log.get(i, {}).get("num_scheduled_tokens", ""),
            "num_reqs": log.get(i, {}).get("num_reqs", ""),
            "nvtx_start_ns": st["start"], "nvtx_end_ns": st["end"], "span_ns": span,
            "gpu_busy_ns": busy, "gpu_idle_ns": span - busy,
            "kernel_sum_ns": sum(r["duration_ns"] for r in ks),
            "first_launch_to_first_start_ns": ks[0]["launch_to_start_ns"] if ks else "",
            "nsys_kernels": len(ks),
            "memcpys": len(memcpy.get(i, [])), "memcpy_bytes": sum(memcpy.get(i, [])),
            "tracer_kernels": "" if lo is None or hi is None else hi - lo,
            "tracer_first_kernel_id": "" if lo is None else lo,
            "tracer_end_kernel_id": "" if hi is None else hi,
            "tracer_traced": tr.get("traced", ""),
            "matched_in_step": matched,
            "exact_step_match": bool(ks) and matched == len(ks) == (hi - lo if lo and hi else -1),
            "traced_matched": sum(1 for r in ks if r["traced"] and in_range(r["trace_kernel_id"])),
        })
    return out


# --------------------------------------------------------------------------------------
# Launch-overhead calibration microbenchmark (no tracer)
# --------------------------------------------------------------------------------------
@app.function(
    gpu="H100!",
    timeout=NSYS_TIMEOUT_S,
    cpu=8,
    memory=32 * 1024,
    volumes={TRACES_MOUNT: traces_vol},
)
def microbench_nsys(n: int = 200, tokens: str = "1,6", ops: str = "", run_name: str = "") -> dict:
    """Run kernel_microbench.py under nsys; per (op, tokens, eager|graph) kernel stats.

    Writes to the volume under /microbench/<run_name>/: profile.{nsys-rep,sqlite},
    microbench_kernels.csv (every kernel with its range, duration, gap to previous kernel,
    launch API start/end), microbench.csv (per range: count, median/mean/p10/p90 duration,
    median start-to-start interval and gap, median API call duration), clocks.csv, summary.json.
    """
    import sqlite3
    import statistics

    run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
    work = Path("/tmp/microbench") / run_name
    work.mkdir(parents=True, exist_ok=True)
    cmd = [*_nsys_base_cmd(), "-o", str(work / "profile"), sys.executable, MICROBENCH_SCRIPT,
           "--n", str(n), "--tokens", tokens, "--ops", ops]
    subprocess.run(["nvidia-smi"], check=False)
    sampler = ClockSampler(work / "clocks.csv")
    sampler.start()
    try:
        _run_logged("nsys", cmd, _nsys_env(), work, time.monotonic() + NSYS_TIMEOUT_S - 10 * 60)
    finally:
        sampler.stop()

    con = sqlite3.connect(work / "profile.sqlite")
    text = "COALESCE(n.text, s.value)"
    ranges = [(a, b, t) for a, b, t in con.execute(
        f"SELECT n.start, n.end, {text} FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId "
        "WHERE n.end IS NOT NULL") if t and t.count("|") == 2]
    kernels = _nsys_kernels(work / "profile.sqlite")
    per_kernel, stats = [], []

    def pct(v, q):
        v = sorted(v)
        return v[min(int(q * len(v)), len(v) - 1)]

    for a, b, name in sorted(ranges):
        op, tok, mode = name.split("|")
        # Kernels whose launch API call (eager) or graph launch (graph) is inside the range.
        ks = [k for k in kernels if a <= (k["api_start"] or k["start"]) <= b]
        prev = None
        for k in ks:
            per_kernel.append({
                "range": name, "op": op, "tokens": tok.split("=")[1], "mode": mode,
                "short_name": k["short"], "duration_ns": k["end"] - k["start"],
                "start_to_start_ns": "" if prev is None else k["start"] - prev["start"],
                "gap_from_prev_end_ns": "" if prev is None else k["start"] - prev["end"],
                "api_name": k["api_name"],
                "api_duration_ns": "" if k["api_start"] is None else k["api_end"] - k["api_start"],
                "grid": k["grid"], "block": k["block"], "mangled_name": k["mangled"]})
            prev = k
        rows = [r for r in per_kernel if r["range"] == name]
        d = [r["duration_ns"] for r in rows]
        s2s = [r["start_to_start_ns"] for r in rows if r["start_to_start_ns"] != ""]
        gaps = [r["gap_from_prev_end_ns"] for r in rows if r["gap_from_prev_end_ns"] != ""]
        apis = [r["api_duration_ns"] for r in rows if r["api_duration_ns"] != ""]
        stats.append({
            "range": name, "op": op, "tokens": tok.split("=")[1], "mode": mode,
            "kernels": len(rows), "distinct_kernels": len({r["mangled_name"] for r in rows}),
            "kernel_names": "|".join(sorted({r["short_name"] for r in rows})),
            "range_ns": b - a,
            "dur_median_ns": statistics.median(d) if d else "",
            "dur_mean_ns": statistics.fmean(d) if d else "",
            "dur_p10_ns": pct(d, 0.1) if d else "", "dur_p90_ns": pct(d, 0.9) if d else "",
            "start_to_start_median_ns": statistics.median(s2s) if s2s else "",
            "gap_median_ns": statistics.median(gaps) if gaps else "",
            "api_median_ns": statistics.median(apis) if apis else "",
        })
    _write_csv(work / "microbench_kernels.csv", per_kernel)
    _write_csv(work / "microbench.csv", stats)
    summary = {"run_name": run_name, "n": n, "tokens": tokens, "gpu": _gpu_name(),
               "clocks": _clock_stats(sampler.rows, None, None), "ranges": stats}
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    dest = Path(TRACES_MOUNT) / "microbench" / run_name
    dest.mkdir(parents=True, exist_ok=True)
    for p in work.iterdir():
        if p.is_file():
            (dest / p.name).write_bytes(p.read_bytes())
    traces_vol.commit()
    return summary


@app.local_entrypoint()
def microbench_main(n: int = 200, tokens: str = "1,6", ops: str = "", run_name: str = ""):
    """modal run accelsim_trace.py::microbench_main [--n 200 --tokens 1,6 --ops fill,copy]"""
    s = microbench_nsys.remote(n=n, tokens=tokens, ops=ops, run_name=run_name)
    print(f"\nSaved to volume accelsim-traces:/microbench/{s['run_name']}/  "
          f"(modal volume get accelsim-traces microbench/{s['run_name']} ./microbench)")
    print(f"{'range':45s} {'kern':>5s} {'dur_med':>9s} {'s2s_med':>9s} {'api_med':>9s}")
    for r in s["ranges"]:
        print(f"{r['range']:45s} {r['kernels']:5d} {r['dur_median_ns']!s:>9} "
              f"{r['start_to_start_median_ns']!s:>9} {r['api_median_ns']!s:>9}")


@app.local_entrypoint()
def nsys_main(trace_run: str = "20261108-004021__facebook_opt-125m__model.decoder.layers.10.self_attn",
              exec_mode: str = "", lock_clocks_mhz: int = 0, gpu_metrics: bool = False,
              out_name: str = ""):
    """modal run accelsim_trace.py::nsys_main --trace-run <run dir on the accelsim-traces volume>
    [--exec-mode graphs] [--lock-clocks-mhz 1980] [--gpu-metrics]"""
    s = profile_nsys.remote(trace_run, exec_mode=exec_mode, lock_clocks_mhz=lock_clocks_mhz,
                            gpu_metrics=gpu_metrics, out_name=out_name)
    print(f"\nSaved to volume accelsim-traces:/{trace_run}/{s['out_name']}/  "
          f"(modal volume get accelsim-traces {trace_run}/{s['out_name']} ./{s['out_name']})")
    for st in s["per_step"]:
        print(f"  step {st['step']} {st['kind']:8s} span={st['span_ns']}ns busy={st['gpu_busy_ns']}ns "
              f"nsys={st['nsys_kernels']} tracer={st['tracer_kernels']} matched={st['matched_in_step']}")
    durs = s["traced_kernel_durations_ns"]
    if durs and len(durs) <= 50:
        print("Real durations of the traced kernels (ns):")
        for kid, d in sorted(durs.items(), key=lambda kv: int(kv[0])):
            print(f"  kernel-{kid}: {d}")


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
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "accelsim-traces",
    s3_region: str = "auto",
    s3_addressing_style: str = "auto",
    trace_scope: str = "layers",
    trace_steps: str = "",
    exec_mode: str = "eager",
    post_process: bool = True,
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
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_region=s3_region,
        s3_addressing_style=s3_addressing_style,
        trace_scope=trace_scope,
        trace_steps=[int(s) for s in trace_steps.split(",") if s.strip()],
        exec_mode=exec_mode,
        post_process=post_process,
    )
    if result and not s3_bucket:
        run = result.rsplit("/", 1)[1]
        print(f"\nDone. Run saved to volume accelsim-traces:/{run}/\n"
              f"  modal volume get accelsim-traces {run} ./{run}\n"
              f"  modal run accelsim_trace.py::nsys_main --trace-run {run}")
    elif result:
        print(f"\nDone. Traces at {result}")
        print("Next, wherever you download them (CPU-only):\n"
              "  post-traces-processing <run>/traces -j 8\n"
              "  accel-sim.out -config .../SM90_H100/gpgpusim.config "
              "-config .../SM90_H100/trace.config -trace <run>/traces/kernelslist.g")
