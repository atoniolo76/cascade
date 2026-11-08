from __future__ import annotations

import math
from statistics import mean

from cascade.sim.timeline import key, numeric


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)


def error(real: float, simulated: float) -> dict:
    delta = simulated - real
    return {
        "real": real,
        "simulated": simulated,
        "error": delta,
        "error_pct": delta / real * 100 if real else (0 if delta == 0 else None),
    }


def aggregate(errors: list[dict]) -> dict:
    deltas = [e["error"] for e in errors]
    absolute = [abs(d) for d in deltas]
    percents = [e["error_pct"] for e in errors if e["error_pct"] is not None]
    return {
        "count": len(errors),
        "mean_error": mean(deltas) if deltas else None,
        "mean_abs_error": mean(absolute) if absolute else None,
        "p50_abs_error": percentile(absolute, 0.5),
        "p95_abs_error": percentile(absolute, 0.95),
        "mean_error_pct": mean(percents) if percents else None,
        "mean_abs_error_pct": mean([abs(p) for p in percents]) if percents else None,
        "p50_abs_error_pct": percentile([abs(p) for p in percents], 0.5),
        "p95_abs_error_pct": percentile([abs(p) for p in percents], 0.95),
    }


def compare(real: dict, simulated: dict) -> dict:
    reference = {key(s): s for s in real["steps"] if not s["warmup"]}
    predicted = {key(s): s for s in simulated["steps"] if not s["warmup"]}
    if reference.keys() != predicted.keys():
        raise ValueError("real/simulated step keys differ")
    steps = []
    for identity, step in reference.items():
        other = predicted[identity]
        if step["kind"] != other["kind"]:
            raise ValueError("real/simulated step kinds differ")
        steps.append(
            {
                "repeat": identity[0],
                "step": identity[1],
                "kind": step["kind"],
                "wall_ns": error(step["baseline_wall_ns"], other["baseline_wall_ns"]),
                "gpu_busy_ns": error(step["gpu_busy_ns"], other["gpu_busy_ns"]),
            }
        )
    requests = []
    maps = []
    for data in (real, simulated):
        records = [r for r in data["requests"] if not r.get("warmup")]
        mapping = {r["repeat"]: r for r in records}
        if len(mapping) != len(records):
            raise ValueError("comparison supports one request per repeat")
        maps.append(mapping)
    if maps[0].keys() != maps[1].keys():
        raise ValueError("real/simulated request repeats differ")
    for repeat, request in maps[0].items():
        metrics = {}
        for field in ("ttft_ms", "itl_ms_mean", "itl_ms_p50", "e2e_ms"):
            actual, prediction = request.get(field), maps[1][repeat].get(field)
            if actual is None or prediction is None:
                metrics[field] = None
            else:
                metrics[field] = error(numeric(actual, field), numeric(prediction, field))
        requests.append({"repeat": repeat, **metrics})
    return {
        "steps": steps,
        "requests": requests,
        "summary": {
            field: aggregate([s[field] for s in steps]) for field in ("wall_ns", "gpu_busy_ns")
        },
    }


def table(report: dict) -> str:
    lines = ["Metric       N    mean abs error ns    p50 abs ns    p95 abs ns    mean abs %"]
    for field, label in (("wall_ns", "Step wall"), ("gpu_busy_ns", "GPU busy")):
        s = report["summary"][field]

        def display(value):
            return f"{value:.3f}" if value is not None else "n/a"

        lines.append(
            f"{label:<11} {s['count']:<4} "
            + "    ".join(
                display(s[k])
                for k in ("mean_abs_error", "p50_abs_error", "p95_abs_error", "mean_abs_error_pct")
            )
        )
    lines.append("Repeat    TTFT error ms    ITL mean error ms    ITL p50 error ms    e2e error ms")
    for request in report["requests"]:
        values = [
            f"{request[f]['error']:.6f}" if request[f] is not None else "n/a"
            for f in ("ttft_ms", "itl_ms_mean", "itl_ms_p50", "e2e_ms")
        ]
        lines.append(f"{request['repeat']:<9} " + "    ".join(values))
    return "\n".join(lines)


def perfetto(real: dict, simulated: dict) -> dict:
    events = []
    for pid, (name, data) in enumerate(
        (("Real unprofiled model", real), ("Simulated corrected-host timeline", simulated)), 1
    ):
        events.append({"ph": "M", "pid": pid, "name": "process_name", "args": {"name": name}})
        offsets = {}
        for step in data["steps"]:
            if step["warmup"]:
                continue
            repeat, step_id = key(step)
            start = offsets.get(repeat, 0)
            end = start + step["baseline_wall_ns"]
            for tid, label, begin, finish, args in [
                (repeat * 2 + 1, f"{step['kind']} step {step_id}", start, end, {"repeat": repeat})
            ] + [
                (
                    repeat * 2 + 2,
                    k["name"],
                    start + k["start_offset_ns"],
                    start + k["start_offset_ns"] + k["duration_ns"],
                    {"seq": k["seq"]},
                )
                for k in step["kernels"]
            ]:
                begin_ns, end_ns = round(begin), round(finish)
                if not math.isfinite(begin_ns) or end_ns < begin_ns:
                    raise ValueError("invalid trace bounds")
                events.append(
                    {
                        "ph": "X",
                        "pid": pid,
                        "tid": tid,
                        "name": label,
                        "ts": begin_ns / 1000,
                        "dur": (end_ns - begin_ns) / 1000,
                        "args": args,
                    }
                )
            offsets[repeat] = end
    return {"traceEvents": events, "displayTimeUnit": "ns"}
