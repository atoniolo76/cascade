import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from cascade_profiler.tracing import post_process_command, trace_command
from cascade_profiler.utils import _unique_run_name, _workload, _run_logged, _write_csv
from cascade_profiler.clocks import _write_clock_snapshot, ClockSampler, _lock_clocks, _clock_stats, _gpu_name
from cascade_profiler.nsys import (
    _nsys_base_cmd, _nsys_env, _scrub_nsys, _nsys_kernels, _nsys_steps, _assign_steps,
    _nsys_events, _nsys_api_calls, _tracer_launches, _align, _per_step,
)

ACCELSIM_COMMIT = "7ba54ded60c4ba0083410c37d22d719166acd6dc"  # dev branch, 2026-10-27
SPINLOCK_MODES = {"none": 0, "fast_forward": 1, "mark_region": 2}
TRACE_SCOPES = ("layers", "steps")
EXEC_MODES = ("eager", "compile", "graphs")
FUNCTION_TIMEOUT_S = 4 * 60 * 60  # NVBit instrumentation is slow
TRACE_TIMEOUT_S = FUNCTION_TIMEOUT_S - 30 * 60  # leave time to drain the upload backlog
# S3 allows 10,000 parts per multipart upload: 128 MiB parts -> objects up to ~1.2 TiB.
# Upload RAM ~= part size x concurrency = 1 GiB.
MULTIPART_CHUNK = 128 * 1024 * 1024
MULTIPART_CONCURRENCY = 8

KERNEL_FILE_RE = re.compile(r"^kernel-(\d+)-(ctx_0x[0-9a-f]+)\.trace(\.xz)?$")

NSYS_TIMEOUT_S = 60 * 60


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

    def __init__(self, run_name: str, out_dir: Path, output_uri: str | None = None):
        self.root = Path(out_dir) / run_name
        self.output_uri = output_uri or str(out_dir)
        self.run_name = run_name

    def uri(self, rel: str = "") -> str:
        return f"{self.output_uri}/{self.run_name}/{rel}".rstrip("/")

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
    *, out_dir: Path, work_root: Path, accelsim_root: str = "/opt/accel-sim-framework",
    output_uri: str | None = None, trace_script: str | None = None,
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
    workload = _workload(params, script=trace_script)
    if list_layers:
        workload.append("--list-layers")

    if run_name is None:
        what = "+".join(layers) if trace_scope == "layers" else (
            "steps-" + ("-".join(map(str, params["trace_steps"])) or "all"))
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{model}__{what}__{exec_mode}")
        run_name = f"{_unique_run_name()}__{slug}"[:200]

    work_dir = Path(work_root) / run_name
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
        cmd = trace_command(workload, accelsim_root, dict(item.split("=", 1) for item in overrides))
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
        dest = VolumeDest(run_name, out_dir, output_uri)
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
                    "CUDA_INJECTION64_PATH": str(Path(accelsim_root) / "util/tracer_nvbit/others/spinlock_tool/spinlock_tool.so"),
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
                "CUDA_INJECTION64_PATH": str(Path(accelsim_root) / "util/tracer_nvbit/tracer_tool/tracer_tool.so"),
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
            _run_logged("post_process", post_process_command(accelsim_root, traces_dir),
                        dict(os.environ), work_dir, deadline + 25 * 60)
            params["post_processed"] = True
        status = "ok"
        return dest.uri()
    finally:
        _upload_run_files(dest, work_dir, status, uploader, params)


def _kernel_files(traces_dir: Path) -> list[Path]:
    if not traces_dir.is_dir():
        return []
    return [p for p in traces_dir.iterdir() if KERNEL_FILE_RE.match(p.name)]


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


def profile_nsys(trace_run: str, exec_mode: str = "", lock_clocks_mhz: int = 0,
                 gpu_metrics: bool = False, out_name: str = "",
                 keep_nsys_rep: bool = False, *, out_dir: Path, work_root: Path,
                 accelsim_root: str = "/opt/accel-sim-framework", trace_script: str | None = None) -> dict:
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
      profile.sqlite   scrubbed nsys export (.nsys-rep only with keep_nsys_rep=True)
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
    run_dir = Path(out_dir) / trace_run
    meta = json.loads((run_dir / "metadata.json").read_text())
    stats_files = sorted((run_dir / "traces").glob("stats_ctx_*"))
    if len(stats_files) != 1:
        raise RuntimeError(f"expected 1 stats_ctx_* in {run_dir}/traces, found {stats_files}")
    if exec_mode and exec_mode not in EXEC_MODES:
        raise ValueError(f"exec_mode must be one of {EXEC_MODES}")
    run_mode = exec_mode or meta.get("exec_mode", "eager")
    out_name = out_name or f"nsys_{run_mode}_{_unique_run_name()}"

    work = Path(work_root) / trace_run / out_name
    work.mkdir(parents=True, exist_ok=True)
    env = _nsys_env({"VLLM_STEP_LOG": str(work / "steps.json")}, accelsim_root)
    cmd = [*_nsys_base_cmd(gpu_metrics), "-o", str(work / "profile"), *_workload(meta, run_mode, script=trace_script)]

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
        "nsys_cmd": cmd[:len(cmd) - len(_workload(meta, run_mode, script=trace_script))],
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

    _scrub_nsys(work, keep_nsys_rep)
    dest = run_dir / out_name
    dest.mkdir(exist_ok=True)
    if not keep_nsys_rep:
        (dest / "profile.nsys-rep").unlink(missing_ok=True)
    for p in work.iterdir():
        if p.is_file():
            (dest / p.name).write_bytes(p.read_bytes())
    print(json.dumps({k: v for k, v in summary.items() if k != "traced_kernel_durations_ns"}, indent=2))
    return summary


def microbench_nsys(n: int = 200, tokens: str = "1,6", ops: str = "", run_name: str = "",
                    keep_nsys_rep: bool = False, *, out_dir: Path, work_root: Path, microbench_script: str) -> dict:
    """Run kernel_microbench.py under nsys; per (op, tokens, eager|graph) kernel stats.

    Writes to the volume under /microbench/<run_name>/: scrubbed profile.sqlite,
    profile.nsys-rep only with keep_nsys_rep=True,
    microbench_kernels.csv (every kernel with its range, duration, gap to previous kernel,
    launch API start/end), microbench.csv (per range: count, median/mean/p10/p90 duration,
    median start-to-start interval and gap, median API call duration), clocks.csv, summary.json.
    """
    import sqlite3
    import statistics

    run_name = run_name or _unique_run_name()
    work = Path(work_root) / run_name
    work.mkdir(parents=True, exist_ok=True)
    cmd = [*_nsys_base_cmd(), "-o", str(work / "profile"), sys.executable, microbench_script,
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
    con.close()
    _scrub_nsys(work, keep_nsys_rep)
    dest = Path(out_dir) / "microbench" / run_name
    dest.mkdir(parents=True, exist_ok=True)
    if not keep_nsys_rep:
        (dest / "profile.nsys-rep").unlink(missing_ok=True)
    for p in work.iterdir():
        if p.is_file():
            (dest / p.name).write_bytes(p.read_bytes())
    return summary
