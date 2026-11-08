from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from cascade.sim.gaps import apply_gaps, derive_gap_request, gap_stats, validate_gaps
from cascade.sim.gaps import number as numeric


def key(step: dict) -> tuple[int, int]:
    for field in ("repeat", "step"):
        if isinstance(step.get(field), bool) or not isinstance(step.get(field), int):
            raise ValueError(f"{field} must be an integer")
    return step["repeat"], step["step"]


def load_timeline(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("meta"), dict):
        raise ValueError("timeline requires meta, steps and requests")
    if not isinstance(data.get("steps"), list) or not isinstance(data.get("requests"), list):
        raise ValueError("steps and requests must be lists")
    seen = set()
    for step in data["steps"]:
        if not isinstance(step, dict):
            raise ValueError("each step must be an object")
        identity = key(step)
        if not isinstance(step.get("warmup"), bool):
            raise ValueError("warmup must be boolean")
        if step["warmup"]:
            continue
        if identity in seen:
            raise ValueError(f"duplicate step {identity}")
        seen.add(identity)
        if step.get("kind") not in ("prefill", "decode"):
            raise ValueError("non-warmup kind must be prefill or decode")
        wall = numeric(step.get("baseline_wall_ns"), "baseline_wall_ns")
        busy = numeric(step.get("gpu_busy_ns"), "gpu_busy_ns")
        host = numeric(step.get("host_ns"), "host_ns")
        if abs(wall - busy - host) > 1:
            raise ValueError("host_ns must equal baseline_wall_ns - gpu_busy_ns (within 1 ns)")
        kernels = step.get("kernels")
        if not isinstance(kernels, list):
            raise ValueError("kernels must be a list")
        seqs = set()
        end = 0.0
        previous_start = 0.0
        for kernel in kernels:
            if not isinstance(kernel, dict) or not isinstance(kernel.get("name"), str):
                raise ValueError("each kernel requires a name")
            seq = kernel.get("seq")
            if isinstance(seq, bool) or not isinstance(seq, int) or seq in seqs:
                raise ValueError("kernel seq must be a unique integer within its step")
            seqs.add(seq)
            start = numeric(kernel.get("start_offset_ns"), "start_offset_ns")
            duration = numeric(kernel.get("duration_ns"), "duration_ns")
            if start < previous_start:
                raise ValueError("out-of-order kernels are unsupported")
            previous_start = start
            end = max(end, start + duration)
        if kernels:
            profile_wall = numeric(step.get("profiled_wall_ns"), "profiled_wall_ns")
            if end > profile_wall + 1:
                raise ValueError("kernels exceed profiled_wall_ns")
            if sum(k["duration_ns"] for k in kernels) > busy + 1:
                raise ValueError("serial kernel duration sum cannot exceed gpu_busy_ns")
    for request in data["requests"]:
        if not isinstance(request, dict) or not isinstance(request.get("warmup"), bool):
            raise ValueError("requests must be objects with boolean warmup")
        if isinstance(request.get("repeat"), bool) or not isinstance(request.get("repeat"), int):
            raise ValueError("request repeat must be an integer")
    validate_gaps(data)
    return data


def load_durations(path: Path | None, subtract_cycles: float = 0) -> dict:
    return parse_durations(json.loads(path.read_text()) if path else [], subtract_cycles)


def parse_durations(data: dict | list, subtract_cycles: float = 0) -> dict:
    numeric(subtract_cycles, "subtract launch cycles")
    records = data.get("kernels") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("simulated durations require a kernels list")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("duration records must be objects")
        seq = record.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int):
            raise ValueError("duration seq must be an integer")
        if ("step" in record) != ("repeat" in record):
            raise ValueError("scoped durations require both repeat and step")
        identity = (*key(record), seq) if "step" in record else (None, None, seq)
        if identity in result:
            raise ValueError(f"duplicate simulated duration {identity}")
        if "sim_cycles" in record:
            if "duration_ns" in record:
                raise ValueError("choose sim_cycles or duration_ns, not both")
            cycles = numeric(record["sim_cycles"], "sim_cycles") - subtract_cycles
            if cycles < 0:
                raise ValueError("launch subtraction exceeds simulated cycles")
            clock = numeric(record.get("clock_mhz"), "clock_mhz", positive=True)
            duration = cycles / clock * 1000
        else:
            if subtract_cycles:
                raise ValueError("launch subtraction requires sim_cycles + clock_mhz")
            duration = numeric(record.get("duration_ns"), "duration_ns")
        result[identity] = duration
    return result


def fit_host(steps: list[dict]) -> dict:
    fits = {}
    for kind in ("prefill", "decode"):
        samples = [s for s in steps if not s["warmup"] and s["kind"] == kind]
        if not samples:
            continue
        xs = [len(s["kernels"]) for s in samples]
        ys = [s["host_ns"] for s in samples]
        xbar, ybar = mean(xs), mean(ys)
        variance = sum((x - xbar) ** 2 for x in xs)
        b = (
            sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / variance
            if variance
            else 0
        )
        fits[kind] = {
            "a_ns": ybar - b * xbar,
            "b_ns_per_kernel": b,
            "samples": len(xs),
            "slope_identifiable": bool(variance),
        }
    return fits


def derive_requests(data: dict) -> None:
    seen = set()
    for request in data["requests"]:
        if request.get("warmup"):
            continue
        repeat = request.get("repeat")
        if repeat in seen:
            raise ValueError("request derivation supports one non-warmup request per repeat")
        seen.add(repeat)
        steps = [s for s in data["steps"] if not s["warmup"] and s["repeat"] == repeat]
        if not steps:
            raise ValueError(f"no non-warmup steps for request repeat {repeat}")
        if any(s.get("num_reqs") != 1 for s in steps):
            raise ValueError("request derivation requires num_reqs=1 (no scheduler model)")
        prefill = [s["baseline_wall_ns"] / 1e6 for s in steps if s["kind"] == "prefill"]
        decode = [s["baseline_wall_ns"] / 1e6 for s in steps if s["kind"] == "decode"]
        if not prefill:
            raise ValueError("request derivation requires a prefill step")
        if "baseline_gap_before_ns" in steps[0]:
            derive_gap_request(request, steps)
            continue
        queue = numeric(request.get("queued_ms"), "queued_ms")
        request.update(
            ttft_ms=queue + sum(prefill),
            itl_ms_mean=mean(decode) if decode else None,
            itl_ms_p50=median(decode) if decode else None,
            e2e_ms=queue + sum(prefill) + sum(decode),
        )


def simulate(
    data: dict, durations: dict, host_model: str = "scaled-gaps", gap_model: str = "measured"
) -> dict:
    if host_model not in ("scaled-gaps", "parametric"):
        raise ValueError("unknown host model")
    result = copy.deepcopy(data)
    fits = fit_host(data["steps"]) if host_model == "parametric" else {}
    total_kernels = simulated_kernels = 0
    total_measured_busy = covered_measured_busy = 0.0
    total_unmodeled_busy = 0.0
    used = set()
    seq_counts = Counter(k["seq"] for s in data["steps"] if not s["warmup"] for k in s["kernels"])
    for repeat, _step_id, seq in durations:
        if repeat is None and seq_counts[seq] > 1:
            raise ValueError(f"ambiguous unscoped seq {seq}; provide repeat and step")
    per_step = []
    for step in result["steps"]:
        if step["warmup"]:
            continue
        identity = key(step)
        kernels = step["kernels"]
        host = step["host_ns"]
        if fits:
            fit = fits[step["kind"]]
            host = max(0, fit["a_ns"] + fit["b_ns_per_kernel"] * len(kernels))
        measured_busy = step["gpu_busy_ns"]
        unmodeled_busy = max(0, measured_busy - sum(k["duration_ns"] for k in kernels))
        count = covered = 0
        covered_busy = 0.0
        gaps = []
        old_end = 0.0
        measured_overlap = 0.0
        new_durations = []
        for kernel in kernels:
            gaps.append(max(0, kernel["start_offset_ns"] - old_end))
            kernel_end = kernel["start_offset_ns"] + kernel["duration_ns"]
            measured_overlap += max(0, min(old_end, kernel_end) - kernel["start_offset_ns"])
            old_end = max(old_end, kernel_end)
            scoped = (*identity, kernel["seq"])
            generic = (None, None, kernel["seq"])
            found = scoped if scoped in durations else generic
            if found in durations:
                used.add(found)
                covered += 1
                covered_busy += kernel["duration_ns"]
                new_durations.append(durations[found])
            else:
                new_durations.append(kernel["duration_ns"])
            count += 1
        if kernels:
            gaps.append(max(0, step["profiled_wall_ns"] - old_end))
            weight = sum(gaps)
            scaled = [g / weight * host if weight else host / len(gaps) for g in gaps]
            end = 0.0
            for i, (kernel, duration) in enumerate(zip(kernels, new_durations, strict=True)):
                kernel["start_offset_ns"] = end + scaled[i]
                kernel["duration_ns"] = duration
                end = kernel["start_offset_ns"] + duration
        busy = sum(new_durations) + unmodeled_busy
        wall = host + busy
        step.update(host_ns=host, gpu_busy_ns=busy, baseline_wall_ns=wall)
        # This output is a synthesized capture; never retain stale profiled timing fields.
        step["profiled_wall_ns"] = wall
        step["cuda_api_ns"] = None
        total_kernels += count
        simulated_kernels += covered
        total_measured_busy += measured_busy
        covered_measured_busy += covered_busy
        total_unmodeled_busy += unmodeled_busy
        per_step.append(
            {
                "repeat": identity[0],
                "step": identity[1],
                "kernels": count,
                "simulated_kernels": covered,
                "measured_gpu_ns": measured_busy,
                "simulated_coverage_gpu_ns": covered_busy,
                "unmodeled_gpu_ns": unmodeled_busy,
                "measured_kernel_overlap_ns": measured_overlap,
            }
        )
    unused = set(durations) - used
    if unused:
        raise ValueError(f"unmatched simulated duration keys: {sorted(unused, key=str)}")
    gap_info = apply_gaps(result, gap_model)
    derive_requests(result)
    result["summary"] = {
        "simulation": {
            "host_model": host_model,
            "host_fit": fits,
            "coverage": {
                "kernels": total_kernels,
                "simulated_kernels": simulated_kernels,
                "kernel_fraction": simulated_kernels / total_kernels if total_kernels else 0,
                "gpu_time_fraction": covered_measured_busy / total_measured_busy
                if total_measured_busy
                else 0,
                "measured_gpu_ns": total_measured_busy,
                "covered_measured_gpu_ns": covered_measured_busy,
                "unmodeled_gpu_ns": total_unmodeled_busy,
                "steps": per_step,
            },
            "request_assumption": "one request per repeat; queue + prefill/decode walls",
        },
        "per_kind": {
            kind: {
                "steps": len(selected),
                "wall_ns_mean": mean(s["baseline_wall_ns"] for s in selected),
                "gpu_busy_ns_mean": mean(s["gpu_busy_ns"] for s in selected),
            }
            for kind in ("prefill", "decode")
            if (selected := [s for s in result["steps"] if not s["warmup"] and s["kind"] == kind])
        },
    }
    if gap_info:
        result["summary"]["simulation"].update(gap_info)
        result["summary"]["simulation"]["request_assumption"] = (
            "one request per repeat; queued (or arrival) + pre/steps/inter-step gaps/post"
        )
        for kind, stats in gap_stats(result).items():
            result["summary"]["per_kind"][kind]["gap_before_ns"] = stats
    return result
