"""Run baseline, Nsight Systems, and optional instruction tracing in one GPU container."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cascade_profiler.align import write_alignment
from cascade_profiler.storage import s3_destination, upload_run
from cascade_profiler.planning import trace_plan as make_trace_plan
from cascade_profiler.clocks import ClockSampler, _clock_stats, _gpu_name
from cascade_profiler.nsys import (
    _assign_steps, _nsys_api_calls, _nsys_base_cmd, _nsys_env, _nsys_events,
    _nsys_kernels, _nsys_nvtx, _nsys_steps, _per_step, _scrub_nsys,
)
from cascade_profiler.timeline import _corrected_timeline, _parse_iterations, _phase_nsys_stats, _reconcile
from cascade_profiler.tracing import post_process_command, trace_command
from cascade_profiler.utils import _run_logged, _unique_run_name, _write_csv
from cascade_profiler.workload import Workload, execution_kwargs


@dataclass
class RunResult:
    run_dir: Path
    paths: dict[str, Path]
    stage_wall_s: dict[str, float]


def _git_sha() -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        return None


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _trace(command: list[str], work: Path, accelsim_root: str, env: dict, deadline: float):
    spinlock = str(Path(accelsim_root) / "util/tracer_nvbit/others/spinlock_tool/spinlock_tool.so")
    for phase in (0, 1):
        name = f"spinlock_phase{phase}"
        spin_command = [*command, "--step-log", str(work / f"{name}_steps.json"),
                        "--metrics-out", str(work / f"{name}_metrics.json")]
        _run_logged(name, trace_command(spin_command, accelsim_root, {
            "CUDA_INJECTION64_PATH": spinlock, "SPINLOCK_PHASE": str(phase)}), env, work, deadline)
    spin_file = work / "spinlock_detection/spinlock_instructions.txt"
    if not spin_file.is_file() or spin_file.stat().st_size == 0:
        raise RuntimeError(f"Spinlock detection produced no {spin_file}; see spinlock_phase*.log")
    _run_logged("trace", trace_command(command, accelsim_root), env, work, deadline)


def profile(workload: Workload, out_dir: str | Path, *, run_name: str | None = None,
            phases=("baseline", "profiled"), trace_steps: list[int] | str | None = None,
            trace_repeat: int = 1, layerwise_nvtx: bool = False, no_step_sync: bool = False,
            gc_between_repeats: bool = True, keep_nsys_rep: bool = False,
            accelsim_root: str = "/opt/accel-sim-framework", nsys_bin: str = "nsys", trace_plan=None, s3=None) -> RunResult:
    """Profile in the caller's GPU container. NVBit timing is never used for reconciliation."""
    workload = Workload.from_dict(workload.to_dict())
    workload.resolve_driver()
    workload.merged_llm_kwargs({**execution_kwargs(workload.exec_mode, True),
                                "enable_logging_iteration_details": True, "disable_log_stats": False})
    phases = tuple(phases)
    if len(set(phases)) != len(phases) or any(p not in {"baseline", "profiled"} for p in phases):
        raise ValueError("phases must contain baseline and/or profiled without duplicates")
    selected = [p for p in ("baseline", "profiled") if p in phases]
    planned = trace_steps == "plan" or trace_plan is not None
    if planned:
        if "baseline" not in selected:
            raise ValueError("trace planning requires a baseline phase")
        if trace_steps not in (None, "plan"):
            raise ValueError("choose explicit trace_steps or trace_plan, not both")
        trace_steps = []
    if not selected and trace_steps is None:
        raise ValueError("select at least one phase or request tracing")
    if "profiled" in selected and workload.repeats < 2:
        raise ValueError("profiled capture requires at least two repeats")
    if trace_steps is not None:
        if not isinstance(trace_steps, list) or any(isinstance(i, bool) or not isinstance(i, int) or i < 0
                                                  for i in trace_steps):
            raise ValueError("trace_steps must be a list of nonnegative repeat-local step indices")
        if isinstance(trace_repeat, bool) or not isinstance(trace_repeat, int) or not 0 <= trace_repeat < workload.repeats:
            raise ValueError("trace_repeat must be in [0, repeats)")
        selected.append("traced")
    for required in ({"profiler_config": {"profiler": "cuda"}} if "profiled" in selected else {},
                     {"enable_layerwise_nvtx_tracing": True} if layerwise_nvtx else {}):
        workload.merged_llm_kwargs(required)
    run_name = _unique_run_name() if run_name is None else run_name
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single nonempty directory name")
    if s3 is not None and keep_nsys_rep:
        raise ValueError("S3 output requires keep_nsys_rep=False to avoid uploading an unscrubbed environment")
    destination, s3_meta = s3_destination(s3, run_name) if s3 is not None else (None, None)
    run_dir = Path(out_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    workload_file = run_dir / "workload.json"
    workload_file.write_text(json.dumps(workload.to_dict(), indent=2))
    stage_wall_s = {}
    manifest = {"workload": workload.to_dict(), "run_name": run_name, "phases": selected,
                "trace_steps": trace_steps, "trace_repeat": trace_repeat,
                "stage_wall_s": stage_wall_s, "package_version": _version("cascade-profiler") or "0.1.1",
                "git_sha": _git_sha(), "traced_timing_used": False, "status": "running"}
    manifest["effective_llm_kwargs"] = {name: workload.merged_llm_kwargs({
        "enable_logging_iteration_details": True, "disable_log_stats": False,
        **({"profiler_config": {"profiler": "cuda"}} if name == "profiled" else {}),
        **({"enable_layerwise_nvtx_tracing": True} if name == "profiled" and layerwise_nvtx else {})})
        for name in selected}
    manifest["exec_mode_defaults_overridden"] = [key for key, value in execution_kwargs(workload.exec_mode, True).items()
                                                 if key in workload.llm_kwargs and workload.llm_kwargs[key] != value]
    manifest["runner_flags"] = {"layerwise_nvtx": layerwise_nvtx, "no_step_sync": no_step_sync,
                                "gc_between_repeats": gc_between_repeats}
    manifest_path = run_dir / "run.json"
    if destination is not None:
        manifest.update(s3=s3_meta, s3_upload_status="pending")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    parsed_phases, diagnostics, step_logs, metrics = {}, {}, {}, {}
    nsys_stats, ranges, kernels = {}, {}, []
    import_paths = [str(Path(p or os.getcwd()).resolve()) for p in sys.path]
    env = _nsys_env({"PYTHONPATH": os.pathsep.join(dict.fromkeys(
        [*import_paths, os.environ.get("PYTHONPATH", "")]))})
    timing_deadline = time.monotonic() + 50 * 60
    trace_scratch = None
    try:
        for name in selected:
            work = run_dir / name
            if name == "traced":
                trace_scratch = tempfile.TemporaryDirectory(prefix="cascade-trace-", dir="/tmp")
                work = Path(trace_scratch.name) / name
            work.mkdir()
            step_path = work / ("trace_steps.json" if name == "traced" else "steps.json")
            command = [sys.executable, "-m", "cascade_profiler.runner", "--workload-json", str(workload_file),
                       "--trace-scope", "steps", "--no-async-scheduling", "--iteration-details",
                       "--metrics-out", str(work / "metrics.json"), "--step-log", str(step_path)]
            if no_step_sync:
                command.append("--no-step-sync")
            if not gc_between_repeats:
                command.append("--no-gc-between-repeats")
            if name == "profiled":
                command += ["--cuda-profiler-from-repeat", "1"]
                if layerwise_nvtx:
                    command.append("--layerwise-nvtx")
                command = [*_nsys_base_cmd(capture_range=True, cuda_events=False, nsys_bin=nsys_bin),
                           "-o", str(work / "profile"), *command]
            started = time.monotonic()
            try:
                if name == "traced":
                    command += ["--trace-repeat", str(trace_repeat), "--trace-steps-in-repeat",
                                ",".join(map(str, trace_steps))]
                    _trace(command, work, accelsim_root, {**env, "TRACES_FOLDER": f"{work}/"},
                           time.monotonic() + 4 * 60 * 60)
                    trace_log = json.loads(step_path.read_text())
                    seen = {r["step_in_repeat"] for r in trace_log if r["repeat"] == trace_repeat}
                    if set(trace_steps) - seen:
                        raise ValueError(f"trace_steps were not reached in repeat {trace_repeat}: {sorted(set(trace_steps) - seen)}")
                    if not any(r["traced"] for r in trace_log):
                        raise RuntimeError("traced run did not instrument any selected steps")
                else:
                    sampler = ClockSampler(work / "clocks.csv")
                    sampler.start()
                    try:
                        _run_logged(name, command, {**env, "VLLM_STEP_LOG": str(step_path)},
                                    work, timing_deadline)
                    finally:
                        sampler.stop()
                        _scrub_nsys(work, keep_nsys_rep if name == "profiled" else False)
                    step_logs[name] = json.loads(step_path.read_text())
                    metrics[name] = json.loads((work / "metrics.json").read_text())
                    parsed, diag = _parse_iterations((work / f"{name}.log").read_text(), step_logs[name])
                    parsed_phases[name] = parsed
                    diagnostics[name] = {**diag, "clocks": _clock_stats(sampler.rows, None, None),
                                         "clock_sampler_error": sampler.error}
                    (run_dir / f"{name}_iterations.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in parsed))
                    if name == "baseline" and planned:
                        lengths = (1, 8, 16, 32) if trace_plan is None or trace_plan is True else tuple(trace_plan)
                        plan = make_trace_plan(workload, lengths, baseline_steps=step_logs[name], iterations=parsed,
                                               repeat=trace_repeat)
                        trace_steps = [local for repeat, local in plan]
                        manifest.update(trace_steps=trace_steps, trace_plan={"context_lengths": lengths, "selected": plan})
                        manifest_path.write_text(json.dumps(manifest, indent=2))
                    if name == "profiled":
                        db = work / "profile.sqlite"
                        kernels, ranges = _nsys_kernels(db), _nsys_steps(db)
                        _assign_steps(kernels, ranges)
                        events, apis = _nsys_events(db, 0, ranges), _nsys_api_calls(db, 0, ranges)
                        nvtx = _nsys_nvtx(db)
                        nsys_stats = _phase_nsys_stats(nvtx, events, apis)
                        for filename, rows in (("kernels", kernels), ("events", events),
                                               ("api_calls", apis), ("nvtx", nvtx)):
                            _write_csv(work / f"{filename}.csv", rows)
                        aligned = [{**k, "trace_kernel_id": "", "traced": False,
                                    "duration_ns": k["end"] - k["start"], "launch_to_start_ns": ""}
                                   for k in kernels]
                        per_step = _per_step(aligned, ranges, step_logs[name], [], db)
                        for row in per_step:
                            row.update(nsys_stats.get(row["step"], {}))
                        _write_csv(work / "steps.csv", per_step)
            finally:
                stage_wall_s[name] = time.monotonic() - started
            if name == "traced":
                started = time.monotonic()
                try:
                    _run_logged("post_process", post_process_command(accelsim_root, work / "traces"),
                                env, work, time.monotonic() + 25 * 60)
                finally:
                    stage_wall_s["post_process"] = time.monotonic() - started
                started = time.monotonic()
                shutil.copytree(work, run_dir / "traced")
                stage_wall_s["trace_collect"] = time.monotonic() - started
                trace_scratch.cleanup()
                trace_scratch = None
        summary = {"run_name": run_name, "phase": "both" if len(parsed_phases) == 2 else next(iter(parsed_phases), None),
                   "model": workload.model, "prompt": workload.prompts[0] if len(workload.prompts) == 1 else workload.prompts,
                   "max_tokens": workload.max_tokens, "repeats": workload.repeats,
                   "max_model_len": workload.max_model_len, "exec_mode": workload.exec_mode,
                   "layerwise_nvtx": layerwise_nvtx, "diagnostics": diagnostics,
                   "table": _reconcile(parsed_phases, nsys_stats)}
        (run_dir / "reconcile.json").write_text(json.dumps(summary, indent=2))
        _write_csv(run_dir / "reconcile.csv", summary["table"])
        if "baseline" in step_logs and "profiled" in step_logs:
            clocks = diagnostics["profiled"]["clocks"]
            meta = {**summary, "gpu_name": _gpu_name().split(",")[0].strip(), "vllm_version": _version("vllm"),
                    "model_config": metrics["baseline"].get("model_config"),
                    "sm_clock_mhz_median": clocks.get("sm_mhz", {}).get("median"),
                    "mem_clock_mhz": clocks.get("mem_mhz", {}).get("median")}
            timeline = _corrected_timeline(meta, step_logs["baseline"], step_logs["profiled"],
                                           parsed_phases["baseline"], metrics["baseline"], nsys_stats,
                                           ranges, kernels, metrics["profiled"])
            (run_dir / "corrected_timeline.json").write_text(json.dumps(timeline, indent=2))
            if "traced" in selected:
                write_alignment(run_dir)
        else:
            print("Skipping corrected_timeline.json: both baseline and profiled phases are required")
        manifest["status"] = "completed"
    finally:
        if trace_scratch is not None:
            try:
                shutil.copytree(Path(trace_scratch.name) / "traced", run_dir / "traced", dirs_exist_ok=True)
            finally:
                trace_scratch.cleanup()
        if manifest["status"] == "running":
            manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if destination is not None:
        if (run_dir / "traced").is_dir():
            (run_dir / "traced/metadata.json").write_text(json.dumps({"run_name": run_name, "s3": s3_meta,
                                                                   "traced_timing_used": False}, indent=2))
        upload_run(run_dir, destination, manifest)
    paths = {p.relative_to(run_dir).as_posix(): p for p in run_dir.rglob("*") if p.is_file()}
    return RunResult(run_dir, paths, stage_wall_s)
