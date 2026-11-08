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
  modal run accelsim_trace.py --list-layers
  modal run accelsim_trace.py --s3-endpoint https://s3.example.com --s3-bucket my-bucket \
      --layers model.decoder.layers.10.self_attn --max-tokens 4
  modal run accelsim_trace.py ... --spinlock-handling none   # skip spinlock detection
Objects land at s3://<bucket>/<prefix>/<run_name>/{traces/,spinlock_detection/,*.log,metadata.json}
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
    volumes={"/root/.cache/huggingface": hf_cache_vol},
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
) -> str | None:
    """Trace `layers` of `model` under vLLM, streaming traces to S3 as kernels finish.

    spinlock_handling: "mark_region" (Accel-Sim default for Hopper), "fast_forward", or "none".
    Any mode other than "none" first runs Accel-Sim's two-phase spinlock detection
    (spinlock_tool.so), which the tracer requires: with SPINLOCK_HANDLING_MODE>0 and no
    spinlock_detection/spinlock_instructions.txt, tracer_tool.so dereferences a null
    pointer (segfault in create_counter) on the first traced kernel.

    Returns the s3:// URI of the run (None for list_layers).
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
        overrides = [f"{k}={v}" for k, v in extra_env.items()]
        cmd = [f"{TORCH_HOOK_DIR}/run.sh", "env", *overrides, *workload]
        _run_logged(name, cmd, base_env, work_dir, deadline)

    subprocess.run(["nvidia-smi"], check=False)
    if list_layers:
        run_step("list_layers", {})
        return None

    if not s3_bucket:
        raise ValueError("s3_bucket is required (and s3_endpoint for non-AWS S3)")
    dest = S3Dest(run_name, s3_endpoint, s3_bucket, s3_prefix, s3_region, s3_addressing_style)
    # Fail fast on an unreachable endpoint / missing bucket before spending GPU time.
    dest.client.head_bucket(Bucket=dest.bucket)
    print(f"Uploading to {dest.uri()}", flush=True)

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
        uploader.finish(timeout=max(deadline - time.monotonic(), 0) + 25 * 60)
        if trace_err:
            raise trace_err

        if not uploader.uploaded:
            raise RuntimeError(
                "No kernel traces were produced; check that the layer name is correct and "
                "that 'Writing results to ...' appears in trace.log"
            )
        status = "ok"
        return dest.uri()
    finally:
        _upload_run_files(dest, work_dir, status, uploader, {
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
    dest: S3Dest, work_dir: Path, status: str, uploader: "KernelTraceUploader | None", params: dict
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
        "post_processed": False,
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
    print(f"[{status}] Uploaded {n} run files (+{meta['kernel_traces_uploaded']} kernel traces, "
          f"{meta['bytes_uploaded'] / 2**30:.2f} GiB) to {dest.uri()}", flush=True)
    if failed:
        print("[s3] FAILED uploads:\n  " + "\n  ".join(failed), flush=True)


# --------------------------------------------------------------------------------------
# Real-hardware kernel timing (Nsight Systems), matched to the tracer's kernel ids
# --------------------------------------------------------------------------------------
NSYS_TIMEOUT_S = 60 * 60


@app.function(
    gpu="H100!",
    timeout=NSYS_TIMEOUT_S,
    cpu=8,
    memory=64 * 1024,
    volumes={"/root/.cache/huggingface": hf_cache_vol, TRACES_MOUNT: traces_vol},
)
def profile_nsys(trace_run: str) -> dict:
    """Re-run a traced workload *without* the tracer, under nsys, to get real kernel timings.

    The Accel-Sim tracer records no timestamps, only launch order (kernel-N = N-th kernel launch
    in the process). This pass runs the identical vLLM workload (same args, read from the trace
    run's metadata.json; greedy + ignore_eos so the kernel sequence is deterministic), records
    every kernel with nsys, and aligns the nsys kernels to the tracer's launch sequence
    (stats_ctx_*: all launches, traced or not) by (mangled name, grid, block).

    Writes to the volume under /<trace_run>/nsys/:
      profile.nsys-rep, profile.sqlite   raw nsys output (open the .nsys-rep in the Nsight GUI)
      kernels.csv                        one row per kernel launch, in launch order, with
                                         trace_kernel_id + real start/end/duration
      summary.json                       alignment result, GPU, versions
      nsys.log                           profiler + vLLM output
    """
    run_dir = Path(TRACES_MOUNT) / trace_run
    meta = json.loads((run_dir / "metadata.json").read_text())
    stats_files = sorted((run_dir / "traces").glob("stats_ctx_*"))
    if len(stats_files) != 1:
        raise RuntimeError(f"expected 1 stats_ctx_* in {run_dir}/traces, found {stats_files}")

    work = Path("/tmp/nsys") / trace_run
    work.mkdir(parents=True, exist_ok=True)
    workload = [
        sys.executable, TRACE_SCRIPT,
        "--model", meta["model"],
        "--layers", ",".join(meta["layers"]),
        "--prompts", json.dumps(meta["prompts"]),
        "--max-tokens", str(meta["max_tokens"]),
        "--max-model-len", str(meta["max_model_len"]),
    ]
    env = {
        **os.environ,
        # Same process layout as the traced run (run.sh sets these); no CUDA_INJECTION64_PATH,
        # so the NVBit hooks are no-ops and nsys owns CUDA injection.
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "PYTHONPATH": f"{ACCELSIM_DIR}/util/tracer_nvbit/others:"
                      f"{ACCELSIM_DIR}/util/tracer_nvbit/tracer_tool:{os.environ.get('PYTHONPATH', '')}",
    }
    env.pop("CUDA_INJECTION64_PATH", None)
    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        # No CPU sampling / context-switch tracing: needs perf, unavailable in gVisor, not needed.
        "--sample=none", "--cpuctxsw=none",
        "--force-overwrite=true", "--export=sqlite",
        "-o", str(work / "profile"),
        *workload,
    ]
    subprocess.run(["nvidia-smi"], check=False)
    _run_logged("nsys", cmd, env, work, time.monotonic() + NSYS_TIMEOUT_S - 10 * 60)

    kernels = _nsys_kernels(work / "profile.sqlite")
    traced = _tracer_launches(stats_files[0])
    rows, alignment = _align(kernels, traced)

    import csv
    cols = ["launch_seq", "trace_kernel_id", "traced", "launch_dims_match", "start_ns", "end_ns", "duration_ns",
            "gap_from_prev_end_ns", "abs_start_ns", "stream_id", "grid", "block",
            "registers_per_thread", "shared_mem_bytes", "short_name", "mangled_name"]
    with open(work / "kernels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    traced_rows = [r for r in rows if r["traced"]]
    summary = {
        "trace_run": trace_run,
        "workload": {k: meta[k] for k in ("model", "layers", "prompts", "max_tokens", "max_model_len")},
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip(),
        "nsys_version": subprocess.run(["nsys", "--version"], capture_output=True, text=True).stdout.strip(),
        "nsys_kernels": len(kernels),
        "tracer_launches": len(traced),
        "streams": sorted({k["stream_id"] for k in kernels}),
        "alignment": alignment,
        "traced_kernels_matched": sum(1 for r in traced_rows if r["duration_ns"] != ""),
        "traced_kernels_total": sum(1 for t in traced if t["traced"]),
        "traced_kernel_durations_ns": {
            r["trace_kernel_id"]: r["duration_ns"] for r in traced_rows if r["duration_ns"] != ""
        },
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2))

    dest = run_dir / "nsys"
    dest.mkdir(exist_ok=True)
    for p in work.iterdir():
        if p.is_file():
            (dest / p.name).write_bytes(p.read_bytes())
    traces_vol.commit()
    print(json.dumps({k: v for k, v in summary.items() if k != "traced_kernel_durations_ns"}, indent=2))
    return summary


def _nsys_kernels(sqlite_path: Path) -> list[dict]:
    """All GPU kernels from an nsys sqlite export, in host launch order."""
    import sqlite3

    con = sqlite3.connect(sqlite_path)
    kcols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    has_rt = con.execute(
        "SELECT 1 FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_RUNTIME'").fetchone()
    name_col = "mangledName" if "mangledName" in kcols else "demangledName"
    # Launch order = time of the host-side launch API call (what the tracer numbers by);
    # fall back to GPU start time if the API call wasn't captured.
    launch_ts = "COALESCE(r.start, k.start)" if has_rt else "k.start"
    rt_join = ("LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId "
               "AND r.globalTid / 16777216 = k.globalPid / 16777216") if has_rt else ""
    q = f"""
      SELECT k.start, k.end, k.streamId, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
             k.registersPerThread, k.staticSharedMemory + k.dynamicSharedMemory,
             sm.value, ss.value
      FROM CUPTI_ACTIVITY_KIND_KERNEL k
      JOIN StringIds sm ON sm.id = k.{name_col}
      JOIN StringIds ss ON ss.id = k.shortName
      {rt_join}
      ORDER BY {launch_ts}, k.start
    """
    out = []
    for (start, end, stream, gx, gy, gz, bx, by, bz, regs, smem, mangled, short) in con.execute(q):
        out.append({
            "start": start, "end": end, "stream_id": stream,
            "grid": f"{gx}x{gy}x{gz}", "block": f"{bx}x{by}x{bz}",
            "regs": regs, "smem": smem, "mangled": mangled, "short": short,
        })
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

    rows, prev_end = [], None
    for i, k in enumerate(kernels):
        t = traced[nsys_to_trace[i]] if i in nsys_to_trace else None
        rows.append({
            "launch_seq": i + 1,
            "trace_kernel_id": t["id"] if t else "",
            "traced": bool(t and t["traced"]),
            "launch_dims_match": "" if not t else (t["grid"], t["block"]) == (k["grid"], k["block"]),
            "start_ns": k["start"] - t0, "end_ns": k["end"] - t0,
            "duration_ns": k["end"] - k["start"],
            "gap_from_prev_end_ns": "" if prev_end is None else k["start"] - prev_end,
            "abs_start_ns": k["start"], "stream_id": k["stream_id"],
            "grid": k["grid"], "block": k["block"],
            "registers_per_thread": k["regs"], "shared_mem_bytes": k["smem"],
            "short_name": k["short"], "mangled_name": k["mangled"],
        })
        prev_end = k["end"]
    # Traced kernels that found no nsys partner are listed so nothing is silently dropped.
    matched_ids = {r["trace_kernel_id"] for r in rows if r["trace_kernel_id"] != ""}
    for t in traced:
        if t["traced"] and t["id"] not in matched_ids:
            rows.append({c: "" for c in ("launch_seq", "start_ns", "end_ns", "duration_ns",
                                         "gap_from_prev_end_ns", "abs_start_ns", "stream_id",
                                         "registers_per_thread", "shared_mem_bytes", "short_name")}
                        | {"trace_kernel_id": t["id"], "traced": True, "launch_dims_match": "",
                           "grid": t["grid"],
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


@app.local_entrypoint()
def nsys_main(trace_run: str = "20261108-004021__facebook_opt-125m__model.decoder.layers.10.self_attn"):
    """modal run accelsim_trace.py::nsys_main --trace-run <run dir on the accelsim-traces volume>"""
    s = profile_nsys.remote(trace_run)
    print(f"\nSaved to volume accelsim-traces:/{trace_run}/nsys/  "
          f"(modal volume get accelsim-traces {trace_run}/nsys ./nsys)")
    durs = s["traced_kernel_durations_ns"]
    if durs:
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
    )
    if result:
        print(f"\nDone. Traces at {result}")
        print("Next, wherever you download them (CPU-only):\n"
              "  post-traces-processing <run>/traces -j 8\n"
              "  accel-sim.out -config .../SM90_H100/gpgpusim.config "
              "-config .../SM90_H100/trace.config -trace <run>/traces/kernelslist.g")
