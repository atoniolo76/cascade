"""CPU-only per-kernel Accel-Sim execution and simulator duration export."""

import json
import lzma
import math
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median
from collections import Counter

from cascade_profiler.align import kernel_key
from cascade_profiler.planning import step_contexts
from cascade_profiler.tracing import post_process_command
from cascade_profiler.utils import _write_csv


def parse_log(text: str) -> dict:
    if "break due to reaching the maximum" in text:
        raise ValueError("simulation reached its cycle/instruction limit")
    name, uid, blocks = None, None, []
    for line in text.splitlines():
        if match := re.fullmatch(r"\s*kernel_name\s*=\s*(.*?)\s*", line):
            name, uid = match[1], None
        elif match := re.fullmatch(r"\s*kernel_launch_uid\s*=\s*(.*?)\s*", line):
            if not re.fullmatch(r"\d+", match[1]):
                raise ValueError("expected one kernel_launch_uid, not a concurrent stats block")
            uid = int(match[1])
        elif match := re.fullmatch(r"\s*gpu_sim_cycle\s*=\s*(\d+)\s*", line):
            if not name or uid is None:
                raise ValueError("gpu_sim_cycle without kernel identity")
            blocks.append({"name": name, "kernel_launch_uid": uid, "sim_cycles": int(match[1])})
    if not blocks or len({(b["name"], b["kernel_launch_uid"]) for b in blocks}) != 1:
        raise ValueError("expected completed statistics for exactly one kernel")
    return blocks[-1]


def core_clock(config: Path) -> float:
    values = re.findall(r"(?m)^\s*-gpgpu_clock_domains\s+([\d.]+):", config.read_text())
    if not values or not math.isfinite(value := float(values[-1])) or value <= 0:
        raise ValueError(f"missing/invalid -gpgpu_clock_domains in {config}")
    return value


def write_kernel_list(directory: Path, trace: Path) -> Path:
    """The parser accepts kernel-only lists; omit copies to isolate cold-cache kernel cycles."""
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / trace.name
    if link.is_symlink():
        link.unlink()
    if not link.exists():
        link.symlink_to(trace.resolve())
    path = directory / "kernelslist.g"
    path.write_text(trace.name + "\n")
    return path


def _simulate_kernel(task: dict) -> dict:
    started = time.monotonic()
    result = {"uid": task["uid"], "name": task["name"], "cycles": None,
              "status": "failed", "error": "", "kernel_launch_uid": None}
    try:
        directory = Path(task["directory"])
        shutil.copytree(task["cfgdir"], directory, dirs_exist_ok=True)
        listing = write_kernel_list(directory, Path(task["trace"]))
        with Path(task["log"]).open("w") as stream:
            completed = subprocess.run([task["binary"],
                                        "-trace", str(listing), "-config", str(directory / "gpgpusim.config"),
                                        "-config", task["tracecfg"]], cwd=directory, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=task["timeout"], env=task["env"])
        if completed.returncode:
            raise RuntimeError(f"accel-sim exited {completed.returncode}")
        parsed = parse_log(Path(task["log"]).read_text())
        if parsed["name"] != task["name"]:
            raise ValueError(f"simulator name {parsed['name']!r} differs from trace {task['name']!r}")
        result.update(cycles=parsed["sim_cycles"], kernel_launch_uid=parsed["kernel_launch_uid"], status="ok")
    except subprocess.TimeoutExpired:
        result.update(status="timeout", error="kernel timeout")
    except (OSError, ValueError, RuntimeError) as exc:
        result["error"] = str(exc)
    result["sim_wall_s"] = time.monotonic() - started
    return result


def trace_header(path: Path) -> dict:
    """tracez starts with null-terminated text; cap reads before compressed instruction data."""
    try:
        opener = lzma.open if path.suffix == ".xz" else open
        with opener(path, "rb") as stream:
            prefix = stream.read(64 * 1024)
            if path.suffix == ".tracez" and b"\0" not in prefix:
                return {}
            data = prefix.split(b"\0", 1)[0].decode(errors="replace")
    except (OSError, EOFError, lzma.LZMAError):
        return {}
    fields = {}
    for line in data.splitlines():
        if line.startswith("-") and "=" in line:
            key, value = line[1:].split("=", 1)
            key = key.strip()
            if key not in {"kernel id", "cuda stream id", "shmem base_addr", "local mem base_addr"}:
                fields[key] = value.strip()
        elif line and not line.startswith("#"):
            break
    return fields


def dedup_groups(kernels: list[dict], files: dict, dedup=True) -> list[dict]:
    groups = {}
    for kernel in kernels:
        uid = kernel["tracer_uid"]
        header = trace_header(files[uid])
        # Unknown/truncated headers are never grounds for reusing an instruction trace.
        key = (*kernel_key(kernel), tuple(sorted(header.items()))) if dedup and header else (uid,)
        group = groups.setdefault(key, {"representative": uid, "members": [], "header": header,
                                        "name": kernel["name"], "grid": kernel["grid"], "block": kernel["block"]})
        group["members"].append(uid)
    return list(groups.values())


def fan_out(groups: list[dict], results: list[dict]) -> tuple[list[dict], dict]:
    by_uid = {r["uid"]: r for r in results}
    expanded, verification = [], []
    for group in groups:
        rep = group["representative"]
        reference = by_uid[rep]
        comparisons = []
        for uid in group["members"]:
            own = uid in by_uid
            result = dict(by_uid[uid] if own else reference)
            result.update(uid=uid, sim="exact" if own else f"dedup:{rep}", representative_uid=uid if own else rep)
            if not own:
                result["sim_wall_s"] = 0
            expanded.append(result)
            if own and uid != rep and result["status"] == reference["status"] == "ok":
                absolute = abs(result["cycles"] - reference["cycles"])
                relative = absolute / abs(reference["cycles"]) if reference["cycles"] else (0 if not absolute else None)
                comparisons.append({"uid": uid, "absolute_cycles": absolute, "relative": relative})
        verification.append({**group, "comparisons": comparisons,
                             "max_absolute_cycles": max((c["absolute_cycles"] for c in comparisons), default=0),
                             "max_relative": max((c["relative"] for c in comparisons if c["relative"] is not None), default=0),
                             "zero_reference_nonzero": any(c["relative"] is None for c in comparisons)})
    return expanded, {"groups": verification,
                      "max_absolute_cycles": max((v["max_absolute_cycles"] for v in verification), default=0),
                      "max_relative": max((v["max_relative"] for v in verification), default=0),
                      "zero_reference_nonzero": any(v["zero_reference_nonzero"] for v in verification)}


def duration_records(timeline: dict, alignment: dict, results: list[dict], clock_mhz: float,
                     propagate="none") -> tuple[list[dict], list[dict]]:
    mode = "identical" if propagate is True else "none" if propagate is False else propagate
    if mode not in {"none", "identical", "interp"}:
        raise ValueError("propagate must be none, identical, or interp")
    successes = {r["uid"]: r for r in results if r["status"] == "ok"}
    steps = {(s["repeat"], s["step"]): s for s in timeline["steps"]}
    contexts = step_contexts(timeline["steps"])
    records, samples, by_seq, interpolation = [], {}, {}, {}
    for mapping in alignment["kernels"]:
        if mapping["tracer_uid"] not in successes:
            continue
        result = successes[mapping["tracer_uid"]]
        record = {k: mapping[k] for k in ("repeat", "step", "seq", "tracer_uid", "name")}
        record.update(sim_cycles=result["cycles"], clock_mhz=clock_mhz, source="traced", sim=result.get("sim", "exact"))
        records.append(record)
        step = steps[mapping["repeat"], mapping["step"]]
        kernel = next(k for k in step["kernels"] if k["seq"] == mapping["seq"])
        if mapping["dims_match"]:
            uid = result.get("representative_uid", result["uid"])
            samples.setdefault((step["kind"], *kernel_key(kernel)), []).append((result["cycles"], uid))
            by_seq.setdefault((step["kind"], kernel["seq"], *kernel_key(kernel)), []).append((result["cycles"], uid))
            context = contexts.get((step["repeat"], step["step"]))
            if step["kind"] == "decode" and context is not None:
                interpolation.setdefault((kernel["seq"], *kernel_key(kernel)), {}).setdefault(context, []).append((result["cycles"], uid))
    ambiguous = [{"kind": key[0], "name": key[1], "grid": key[2], "block": key[3],
                  "cycles": [v for v, _ in values], "median": median(v for v, _ in values)}
                 for key, values in samples.items() if len({v for v, _ in values}) > 1]
    if mode != "none":
        traced_steps = {(s["repeat"], s["step"]) for s in alignment["steps"]}
        for step in timeline["steps"]:
            if (step["repeat"], step["step"]) in traced_steps:
                continue
            for kernel in step["kernels"]:
                values = samples.get((step["kind"], *kernel_key(kernel)))
                if mode == "interp":
                    values = by_seq.get((step["kind"], kernel["seq"], *kernel_key(kernel)))
                extra = {}
                label = None
                cycles = None
                if values:
                    cycles = median(v for v, _ in values)
                    uid = min(values, key=lambda item: (abs(item[0] - cycles), item[1]))[1]
                    label = f"identical:{uid}"
                points = interpolation.get((kernel["seq"], *kernel_key(kernel)), {})
                context = contexts.get((step["repeat"], step["step"]))
                if mode == "interp" and step["kind"] == "decode" and context is not None and len(points) >= 2:
                    points = sorted((x, median(v for v, _ in values), min(uid for _, uid in values))
                                    for x, values in points.items())
                    if len({p[1] for p in points}) > 1:
                        lower = max((p for p in points if p[0] <= context), default=points[0])
                        upper = min((p for p in points if p[0] >= context), default=points[-1])
                        cycles = lower[1] if lower[0] == upper[0] else lower[1] + (
                            upper[1] - lower[1]) * (context - lower[0]) / (upper[0] - lower[0])
                        label = f"interp:{lower[2]},{upper[2]}"
                        if context < points[0][0] or context > points[-1][0]:
                            label += ":clamped"
                        extra["context_length"] = context
                if label is not None:
                    records.append({"repeat": step["repeat"], "step": step["step"], "seq": kernel["seq"],
                                    "name": kernel["name"], "sim_cycles": cycles, "sim": label,
                                    "clock_mhz": clock_mhz, "source": "propagated", **extra})
    return records, ambiguous


def _commit(directory: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        return None


def _runtime_environment(root: Path, log: Path) -> dict:
    # Setup writes shared symlinks: source once before launching concurrent workers.
    script = ('set -e; unset GPGPUSIM_ROOT GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN; '
              'source "$1/gpu-simulator/setup_environment.sh" release >&2; env -0')
    with log.open("w") as stream:
        result = subprocess.run(["bash", "-c", script, "accelsim", str(root)], stdout=subprocess.PIPE,
                                stderr=stream, check=True)
    return dict(item.decode(errors="surrogateescape").split("=", 1)
                for item in result.stdout.split(b"\0") if item)


def choose_gpu_config(gpu_name: str | None, explicit: str | None = None) -> tuple[str, dict]:
    matched = None
    if explicit is None:
        for name, config in (("H200", "SM90_H200"), ("H100", "SM90_H100"), ("A100", "SM80_A100"),
                             ("RTX 3070", "SM86_RTX3070"), ("RTX 2060", "SM75_RTX2060")):
            if re.search(r"\b" + re.escape(name) + r"\b", (gpu_name or "").upper()):
                explicit, matched = config, name
                break
        if explicit is None:
            raise ValueError(f"unknown measured GPU {gpu_name!r}; supply gpu_config explicitly")
    return explicit, {"measured_gpu_name": gpu_name, "gpu_config": explicit,
                      "selection": "name mapping" if matched else "explicit", "matched_name": matched}


def simulate_accelsim(run_dir, gpu_config=None, accelsim_root="/opt/accel-sim-framework",
                      jobs=None, kernel_timeout_s=None, propagate="none", uids=None,
                      dedup=True, verify_dedup=0) -> dict:
    started = time.monotonic()
    if jobs is not None and (isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1):
        raise ValueError("jobs must be a positive integer")
    if kernel_timeout_s is not None and (not math.isfinite(kernel_timeout_s) or kernel_timeout_s <= 0):
        raise ValueError("kernel_timeout_s must be positive and finite")
    if isinstance(verify_dedup, bool) or not isinstance(verify_dedup, int) or verify_dedup < 0:
        raise ValueError("verify_dedup must be a nonnegative integer")
    if not isinstance(dedup, bool):
        raise ValueError("dedup must be boolean")
    mode = "identical" if propagate is True else "none" if propagate is False else propagate
    if mode not in {"none", "identical", "interp"}:
        raise ValueError("propagate must be none, identical, or interp")
    if gpu_config is not None and (not gpu_config or Path(gpu_config).name != gpu_config or gpu_config in {".", ".."}):
        raise ValueError("gpu_config must be a config directory name")
    root, run = Path(accelsim_root).resolve(), Path(run_dir).resolve()
    timeline = json.loads((run / "corrected_timeline.json").read_text())
    gpu_config, gpu_mapping = choose_gpu_config(timeline["meta"].get("gpu_name"), gpu_config)
    alignment = json.loads((run / "trace_alignment.json").read_text())
    selected = alignment["kernels"]
    if uids is not None:
        uids = set(uids)
        if any(isinstance(i, bool) or not isinstance(i, int) for i in uids) or uids - {k["tracer_uid"] for k in selected}:
            raise ValueError("uids must identify aligned tracer kernels")
        selected = [k for k in selected if k["tracer_uid"] in uids]
    cfgdir = root / "gpu-simulator/gpgpu-sim/configs/tested-cfgs" / gpu_config
    tracecfg = root / "gpu-simulator/configs/tested-cfgs" / gpu_config / "trace.config"
    binary = root / "gpu-simulator/bin/release/accel-sim.out"
    for required in (cfgdir / "gpgpusim.config", tracecfg, binary, root / "gpu-simulator/setup_environment.sh"):
        if not required.is_file():
            raise FileNotFoundError(required)
    clock = core_clock(cfgdir / "gpgpusim.config")
    sim = run / "sim"
    sim.mkdir(exist_ok=True)
    traces = run / "traced/traces"
    trace_files = {k["tracer_uid"]: traces / (Path(k["trace_file"]).name.removesuffix(".xz") + "z") for k in selected}
    post_process_error = None
    if selected and (not (traces / "kernelslist.g").is_file() or any(not p.is_file() for p in trace_files.values())):
        with (sim / "post_process.log").open("w") as log:
            try:
                subprocess.run(post_process_command(str(root), traces), check=True, stdout=log, stderr=subprocess.STDOUT)
            except (OSError, subprocess.CalledProcessError) as exc:
                post_process_error = str(exc)
    target_kernels = {(s["repeat"], s["step"], k["seq"]): k for s in timeline["steps"] for k in s["kernels"]}
    selected = [{**target_kernels[k["repeat"], k["step"], k["seq"]], **k} for k in selected]
    groups = dedup_groups(selected, trace_files, dedup)
    run_uids = {uid for group in groups for uid in group["members"][:1 + verify_dedup]}
    env = _runtime_environment(root, sim / "setup.log") if selected else {}
    tasks = [{"uid": k["tracer_uid"], "name": k["name"], "env": env, "binary": str(binary),
              "directory": str(sim / f"kernel-{k['tracer_uid']}"), "cfgdir": str(cfgdir),
              "tracecfg": str(tracecfg), "trace": str(trace_files[k["tracer_uid"]]),
              "log": str(sim / f"kernel-{k['tracer_uid']}.log"), "timeout": kernel_timeout_s}
             for k in selected if k["tracer_uid"] in run_uids]
    with ProcessPoolExecutor(max_workers=jobs or os.cpu_count() or 1) as pool:
        actual_results = list(pool.map(_simulate_kernel, tasks))
    results, verification = fan_out(groups, actual_results)
    records, ambiguous = duration_records(timeline, alignment, results, clock, mode)
    counts = {"aligned": len(alignment["kernels"]), "selected": len(selected),
              "ok": sum(r["status"] == "ok" for r in results),
              "failed": sum(r["status"] == "failed" for r in results),
              "timeout": sum(r["status"] == "timeout" for r in results),
              "propagated": sum(r["source"] == "propagated" for r in records)}
    traced_steps = {(s["repeat"], s["step"]) for s in alignment["steps"]}
    counts.update(unique_accelsim_runs=len(actual_results), traced_kernel_occurrences=len(selected),
                  unmatched_untraced=sum(len(s["kernels"]) for s in timeline["steps"]
                                         if (s["repeat"], s["step"]) not in traced_steps) - counts["propagated"])
    labels = dict(Counter(r["sim"] for r in records))
    provenance = {"run_name": timeline["meta"].get("run_name"), "gpu_config": gpu_config,
                  "gpu_mapping": gpu_mapping,
                  "post_process_error": post_process_error,
                  "accelsim_commit": _commit(root), "gpgpusim_commit": _commit(root / "gpu-simulator/gpgpu-sim"),
                  "clock_mhz": clock, "clock_mhz_source": str(cfgdir / "gpgpusim.config") + ":-gpgpu_clock_domains[0]",
                  "measured_sm_clock_mhz": timeline["meta"].get("sm_clock_mhz_median"),
                  "mode": "exact+propagated" if mode != "none" else "exact", "counts": counts,
                  "propagate": mode, "provenance_labels": labels,
                  "dedup": dedup, "dedup_scope": "global across selected traced steps, header + name/grid/block",
                  "verify_dedup": verify_dedup, "dedup_verification": verification,
                  "ambiguous_propagation": ambiguous, "isolated_cold_cache": True,
                  "dims_mismatch": sum(not k["dims_match"] for k in selected)}
    (run / "sim_durations.json").write_text(json.dumps({"provenance": provenance, "kernels": records}, indent=2))
    _write_csv(sim / "kernels.csv", results, ["uid", "name", "cycles", "sim_wall_s", "status", "error", "kernel_launch_uid", "sim"])
    covered = {(r["repeat"], r["step"], r["seq"]) for r in records if r["source"] == "traced"}
    coverage = [{"repeat": s["repeat"], "step": s["step"], "simulated": sum(
        (s["repeat"], s["step"], k["seq"]) in covered for k in s["kernels"]), "total": len(s["kernels"])}
        for s in timeline["steps"] if (s["repeat"], s["step"]) in
        {(a["repeat"], a["step"]) for a in alignment["steps"]}]
    walls = [r["sim_wall_s"] for r in actual_results]
    summary = {"coverage": coverage, "counts": counts, "total_sim_wall_s": sum(walls),
               "sim_wall_s_mean": mean(walls) if walls else None, "sim_wall_s_max": max(walls) if walls else None,
               "dedup_verification": verification, "provenance_labels": labels,
               "post_process_error": post_process_error,
               "wall_s": time.monotonic() - started}
    (sim / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
